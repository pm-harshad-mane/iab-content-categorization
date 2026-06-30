from __future__ import annotations

import argparse
import json
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests
from requests import HTTPError
from requests import ConnectionError as RequestsConnectionError
from requests import Timeout as RequestsTimeout

DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 180
DEFAULT_INPUT_DIR = "04_fetched_url_content_embedding_categories_files"
DEFAULT_EMBEDDING_INPUT_DIR = "03_fetched_url_content_embedding_files"
DEFAULT_OUTPUT_DIR = "05_fetched_url_content_embedding_categories_reranked_files"
DEFAULT_MODELS_ENDPOINT = "/v1/models"
DEFAULT_RERANK_ENDPOINTS = ("/v1/rerank", "/rerank", "/v2/rerank", "/v1/score", "/score")
DEFAULT_CONCURRENT_RECORDS = 500
DEFAULT_TOP_K = 5
DEFAULT_RERANK_QUERY_MAX_CHARS = 1800
DEFAULT_FAISS_DOC_LIMIT = 10

_thread_local = threading.local()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip()).strip("_") or "model"


def output_path_for_input(input_path: Path, output_dir: Path, reranker_model: str) -> Path:
    return output_dir / f"{input_path.stem}_reranked_{sanitize_model_name(reranker_model)}.jsonl"


def corresponding_embedding_path(faiss_input_path: Path, embedding_input_dir: Path) -> Path:
    stem = faiss_input_path.stem
    if stem.endswith("__faiss"):
        stem = stem[: -len("__faiss")]
    return embedding_input_dir / f"{stem}.jsonl"


def _get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def load_cached_url_hashes(path: Path) -> Set[str]:
    if not path.exists():
        return set()

    cached_hashes: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            url_hash = normalize_text(record.get("url_hash"))
            if url_hash:
                cached_hashes.add(url_hash)
    return cached_hashes


def load_jsonl_by_url_hash(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            url_hash = normalize_text(record.get("url_hash"))
            if url_hash:
                records[url_hash] = record
    return records


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            yield line_number, json.loads(payload)


def truncate_query_text(query_text: str, max_chars: int) -> str:
    text = normalize_text(query_text)
    if len(text) <= max_chars:
        return text
    suffix = " ... [truncated]"
    base = text[: max(0, max_chars - len(suffix))].rstrip()
    return (base + suffix)[:max_chars]


def make_rerank_documents(candidates: Sequence[Dict[str, Any]]) -> List[str]:
    docs: List[str] = []
    for candidate in candidates:
        path = normalize_text(candidate.get("path"))
        description = normalize_text(candidate.get("description"))
        parts: List[str] = []
        if path:
            parts.append(f"path: {path}")
        if description:
            parts.append(f"description: {description}")
        docs.append("\n".join(parts))
    return docs


def _select_model_metadata(payload: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, list):
        return None

    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("id") == model_name or item.get("root") == model_name:
            return {key: item.get(key) for key in ["id", "root", "max_model_len", "owned_by"]}
    return None


class HostedRerankerClient:
    def __init__(
        self,
        api_base: str,
        model_name: str,
        endpoints: Sequence[str],
        timeout: int,
        max_retries: int = 3,
        retry_backoff_ms: int = 250,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.endpoints = [ep if ep.startswith("/") else f"/{ep}" for ep in endpoints]
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_ms = max(0, int(retry_backoff_ms))
        self.models_endpoint = DEFAULT_MODELS_ENDPOINT
        self._working_endpoint: Optional[str] = None

    def _post_with_retries(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = _get_session().post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response
            except HTTPError as exc:
                last_error = exc
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                if status_code is None or status_code < 500 or attempt >= self.max_retries:
                    raise
            except (RequestsConnectionError, RequestsTimeout, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
            if self.retry_backoff_ms > 0:
                time.sleep((self.retry_backoff_ms * attempt) / 1000.0)
        if last_error is None:
            raise RuntimeError("HTTP request failed for unknown reason")
        raise last_error

    def _get_with_retries(self, url: str) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = _get_session().get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except HTTPError as exc:
                last_error = exc
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                if status_code is None or status_code < 500 or attempt >= self.max_retries:
                    raise
            except (RequestsConnectionError, RequestsTimeout, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
            if self.retry_backoff_ms > 0:
                time.sleep((self.retry_backoff_ms * attempt) / 1000.0)
        if last_error is None:
            raise RuntimeError("HTTP request failed for unknown reason")
        raise last_error

    def get_model_metadata(self) -> Dict[str, Any]:
        url = self.api_base + self.models_endpoint
        try:
            response = self._get_with_retries(url)
            payload = response.json()
            selected = _select_model_metadata(payload, self.model_name)
            return {
                "configured_model": self.model_name,
                "api_base": self.api_base,
                "endpoint_candidates": self.endpoints,
                "models_endpoint": self.models_endpoint,
                "resolved_model": selected,
            }
        except Exception as exc:
            return {
                "configured_model": self.model_name,
                "api_base": self.api_base,
                "endpoint_candidates": self.endpoints,
                "models_endpoint": self.models_endpoint,
                "resolved_model": None,
                "model_lookup_error": f"{type(exc).__name__}: {exc}",
            }

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> Tuple[List[float], float, str]:
        if not documents:
            return [], 0.0, self._working_endpoint or (self.endpoints[0] if self.endpoints else "")

        errors: List[str] = []
        candidate_endpoints = [self._working_endpoint] if self._working_endpoint else []
        candidate_endpoints.extend(ep for ep in self.endpoints if ep and ep != self._working_endpoint)

        for endpoint in candidate_endpoints:
            try:
                scores, elapsed_ms = self._rerank_once(endpoint, query, documents, top_k=top_k)
                self._working_endpoint = endpoint
                return scores, elapsed_ms, endpoint
            except Exception as exc:
                errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")

        raise RuntimeError("All reranker endpoints failed. " + " | ".join(errors))

    def _rerank_once(
        self,
        endpoint: str,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int],
    ) -> Tuple[List[float], float]:
        url = self.api_base + endpoint
        payload_variants = [
            {
                "model": self.model_name,
                "query": query,
                "documents": list(documents),
                "top_n": top_k or len(documents),
                "return_documents": False,
            },
            {
                "model": self.model_name,
                "query": query,
                "documents": [{"text": doc} for doc in documents],
                "top_n": top_k or len(documents),
                "return_documents": False,
            },
            {
                "model": self.model_name,
                "query": query,
                "texts": list(documents),
                "top_n": top_k or len(documents),
            },
            {
                "model": self.model_name,
                "query": query,
                "input": list(documents),
                "top_n": top_k or len(documents),
            },
        ]

        errors: List[str] = []
        for payload in payload_variants:
            t0 = time.perf_counter()
            try:
                response = self._post_with_retries(url, payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                body = response.json()
                scores = self._extract_scores(body, len(documents))
                return scores, round(elapsed_ms, 3)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        raise RuntimeError("All payload variants failed. " + " | ".join(errors))

    @staticmethod
    def _extract_scores(body: Dict[str, Any], num_docs: int) -> List[float]:
        data = body.get("results")
        if isinstance(data, list):
            scores_by_index: List[Optional[float]] = [None] * num_docs
            ordered_fallback: List[float] = []
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                score = item.get("relevance_score")
                if score is None:
                    score = item.get("score")
                if score is None:
                    continue
                score = float(score)
                doc_index = item.get("index")
                if isinstance(doc_index, int) and 0 <= doc_index < num_docs:
                    scores_by_index[doc_index] = score
                else:
                    ordered_fallback.append(score)
            if any(score is not None for score in scores_by_index):
                for i in range(num_docs):
                    if scores_by_index[i] is None:
                        scores_by_index[i] = 0.0
                return [float(score) for score in scores_by_index]
            if len(ordered_fallback) == num_docs:
                return ordered_fallback

        data = body.get("data")
        if isinstance(data, list):
            scores_by_index = [0.0] * num_docs
            found_any = False
            for item in data:
                if not isinstance(item, dict):
                    continue
                score = item.get("score")
                if score is None:
                    score = item.get("relevance_score")
                idx = item.get("index")
                if score is None:
                    continue
                score = float(score)
                if isinstance(idx, int) and 0 <= idx < num_docs:
                    scores_by_index[idx] = score
                    found_any = True
            if found_any:
                return scores_by_index

        raise RuntimeError(f"Could not parse reranker response: {body}")


def build_source_error_record(
    faiss_record: Dict[str, Any],
    reranker_model: str,
    reranker_model_details: Dict[str, Any],
    message: str,
    error_type: str = "SourceRecordError",
    error_code: str = "source_status_error",
) -> Dict[str, Any]:
    return {
        "input_url": normalize_text(faiss_record.get("input_url") or faiss_record.get("url")),
        "url": normalize_text(faiss_record.get("url") or faiss_record.get("input_url")),
        "url_hash": normalize_text(faiss_record.get("url_hash")),
        "status": "error",
        "source_status": normalize_text(faiss_record.get("status")),
        "embedding_model": normalize_text(faiss_record.get("embedding_model")),
        "reranker_model": reranker_model,
        "error_type": error_type,
        "error_code": error_code,
        "message": message,
        "retryable": False,
        "model_details": {
            "reranker_model": reranker_model_details,
            "faiss_model_details": faiss_record.get("model_details"),
        },
    }


def build_success_record(
    faiss_record: Dict[str, Any],
    reranker_model: str,
    reranker_model_details: Dict[str, Any],
    rerank_endpoint: str,
    rerank_query_chars_used: int,
    rerank_top_k: int,
    reranked_categories: List[Dict[str, Any]],
    rerank_api_ms: float,
    total_ms: float,
    ) -> Dict[str, Any]:
    return {
        "input_url": normalize_text(faiss_record.get("input_url") or faiss_record.get("url")),
        "url": normalize_text(faiss_record.get("url") or faiss_record.get("input_url")),
        "url_hash": normalize_text(faiss_record.get("url_hash")),
        "status": "ok",
        "source_status": normalize_text(faiss_record.get("source_status") or faiss_record.get("status")),
        "embedding_model": normalize_text(faiss_record.get("embedding_model")),
        "reranker_model": reranker_model,
        "domain": normalize_text(faiss_record.get("domain")),
        "title": normalize_text(faiss_record.get("title")),
        "faiss_top_k": int(faiss_record.get("faiss_top_k") or 0),
        "rerank_top_k": rerank_top_k,
        "faiss_top_categories": faiss_record.get("top_categories") or [],
        "reranked_top_categories": reranked_categories,
        "model_details": {
            "reranker_model": reranker_model_details,
            "faiss_model_details": faiss_record.get("model_details"),
            "rerank_endpoint": rerank_endpoint,
        },
        "timing_ms": {
            "rerank_api": round(rerank_api_ms, 3),
            "total": round(total_ms, 3),
        },
        "debug": {
            "rerank_query_chars_used": rerank_query_chars_used,
            "rerank_candidates_count": len(faiss_record.get("top_categories") or []),
        },
    }


def build_compact_reranked_categories(reranked_top: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact_rows: List[Dict[str, Any]] = []
    for reranker_rank, category in enumerate(reranked_top, start=1):
        compact_rows.append(
            {
                "unique_id": normalize_text(category.get("unique_id")),
                "parent_id": normalize_text(category.get("parent_id")),
                "path": normalize_text(category.get("path")),
                "faiss_score": round(float(category.get("faiss_score", 0.0)), 6),
                "rerank_score": round(float(category.get("rerank_score", 0.0)), 6),
                "faiss_rank": int(category.get("faiss_rank", 0)),
                "reranker_rank": reranker_rank,
            }
        )
    return compact_rows


def build_rerank_error_record(
    faiss_record: Dict[str, Any],
    reranker_model: str,
    reranker_model_details: Dict[str, Any],
    error: Exception,
    total_ms: float,
    rerank_query_chars_used: int,
) -> Dict[str, Any]:
    status_code = getattr(getattr(error, "response", None), "status_code", None)
    return {
        "input_url": normalize_text(faiss_record.get("input_url") or faiss_record.get("url")),
        "url": normalize_text(faiss_record.get("url") or faiss_record.get("input_url")),
        "url_hash": normalize_text(faiss_record.get("url_hash")),
        "status": "error",
        "source_status": normalize_text(faiss_record.get("source_status") or faiss_record.get("status")),
        "embedding_model": normalize_text(faiss_record.get("embedding_model")),
        "reranker_model": reranker_model,
        "domain": normalize_text(faiss_record.get("domain")),
        "title": normalize_text(faiss_record.get("title")),
        "error_type": type(error).__name__,
        "error_code": "rerank_http_error" if status_code else "rerank_request_error",
        "message": str(error),
        "retryable": status_code in {408, 425, 429, 500, 502, 503, 504},
        "status_code": status_code,
        "model_details": {
            "reranker_model": reranker_model_details,
            "faiss_model_details": faiss_record.get("model_details"),
        },
        "timing_ms": {
            "total": round(total_ms, 3),
        },
        "debug": {
            "rerank_query_chars_used": rerank_query_chars_used,
            "rerank_candidates_count": len(faiss_record.get("top_categories") or []),
        },
    }


def process_record(
    line_number: int,
    faiss_record: Dict[str, Any],
    embedding_records: Dict[str, Dict[str, Any]],
    rerank_client: HostedRerankerClient,
    reranker_model: str,
    reranker_model_details: Dict[str, Any],
    rerank_query_max_chars: int,
    rerank_top_k: int,
    faiss_doc_limit: int,
) -> Dict[str, Any]:
    started = time.perf_counter()

    if normalize_text(faiss_record.get("status")).lower() != "ok":
        return build_source_error_record(
            faiss_record,
            reranker_model,
            reranker_model_details,
            "FAISS input record status is not ok; reranking skipped.",
        )

    url_hash = normalize_text(faiss_record.get("url_hash"))
    embed_record = embedding_records.get(url_hash)
    if not embed_record:
        return build_source_error_record(
            faiss_record,
            reranker_model,
            reranker_model_details,
            "Matching embedding record not found for url_hash.",
            error_code="missing_embedding_record",
        )
    if normalize_text(embed_record.get("status")).lower() != "ok":
        return build_source_error_record(
            faiss_record,
            reranker_model,
            reranker_model_details,
            "Matching embedding record is not ok; reranking skipped.",
            error_code="embedding_source_status_error",
        )

    query_text = normalize_text(embed_record.get("query_text"))
    if not query_text:
        return build_source_error_record(
            faiss_record,
            reranker_model,
            reranker_model_details,
            "Matching embedding record has no query_text; reranking skipped.",
            error_code="missing_query_text",
        )

    faiss_candidates = list(faiss_record.get("top_categories") or [])[: max(1, int(faiss_doc_limit))]
    if not faiss_candidates:
        return build_source_error_record(
            faiss_record,
            reranker_model,
            reranker_model_details,
            "FAISS input record has no top categories to rerank.",
            error_code="missing_faiss_categories",
        )

    rerank_query = truncate_query_text(query_text, rerank_query_max_chars)
    rerank_docs = make_rerank_documents(faiss_candidates)

    try:
        scores, rerank_api_ms, rerank_endpoint = rerank_client.rerank(
            query=rerank_query,
            documents=rerank_docs,
            top_k=len(rerank_docs),
        )
    except Exception as exc:
        return build_rerank_error_record(
            faiss_record,
            reranker_model,
            reranker_model_details,
            error=exc,
            total_ms=(time.perf_counter() - started) * 1000.0,
            rerank_query_chars_used=len(rerank_query),
        )

    reranked: List[Dict[str, Any]] = []
    for faiss_rank, (candidate, score) in enumerate(zip(faiss_candidates, scores), start=1):
        row = dict(candidate)
        row["rerank_score"] = round(float(score), 6)
        row["faiss_rank"] = faiss_rank
        reranked.append(row)

    reranked_top = sorted(
        reranked,
        key=lambda item: (
            float(item.get("rerank_score", 0.0)),
            float(item.get("faiss_score", 0.0)),
        ),
        reverse=True,
    )[: max(1, int(rerank_top_k))]

    compact_reranked_top = build_compact_reranked_categories(reranked_top)

    return build_success_record(
        faiss_record=faiss_record,
        reranker_model=reranker_model,
        reranker_model_details=reranker_model_details,
        rerank_endpoint=rerank_endpoint,
        rerank_query_chars_used=len(rerank_query),
        rerank_top_k=rerank_top_k,
        reranked_categories=compact_reranked_top,
        rerank_api_ms=rerank_api_ms,
        total_ms=(time.perf_counter() - started) * 1000.0,
    )


def process_file(
    input_path: Path,
    embedding_input_dir: Path,
    output_path: Path,
    rerank_client: HostedRerankerClient,
    reranker_model: str,
    reranker_model_details: Dict[str, Any],
    concurrent_records: int,
    rerank_query_max_chars: int,
    rerank_top_k: int,
    faiss_doc_limit: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cached_hashes = load_cached_url_hashes(output_path)
    embedding_path = corresponding_embedding_path(input_path, embedding_input_dir)
    if not embedding_path.exists():
        raise FileNotFoundError(f"Matching embedding input file not found for {input_path}: {embedding_path}")
    embedding_records = load_jsonl_by_url_hash(embedding_path)

    pending: List[Tuple[int, Dict[str, Any]]] = []
    for line_number, record in iter_jsonl(input_path):
        url_hash = normalize_text(record.get("url_hash"))
        if url_hash and url_hash in cached_hashes:
            continue
        pending.append((line_number, record))

    if not pending:
        print(f"Nothing to do for {input_path}; output already up to date.")
        return

    with output_path.open("a", encoding="utf-8") as dst:
        with ThreadPoolExecutor(max_workers=max(1, int(concurrent_records))) as executor:
            future_map = {
                executor.submit(
                    process_record,
                    line_number,
                    record,
                    embedding_records,
                    rerank_client,
                    reranker_model,
                    reranker_model_details,
                    rerank_query_max_chars,
                    rerank_top_k,
                    faiss_doc_limit,
                ): (line_number, record)
                for line_number, record in pending
            }
            for future in as_completed(future_map):
                line_number, record = future_map[future]
                output_record = future.result()
                dst.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                dst.flush()
                print(
                    f"[{line_number}] {output_record['status']} "
                    f"{normalize_text(record.get('url') or record.get('input_url'))}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank FAISS taxonomy categories using a hosted reranker model on vLLM."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing FAISS category JSONL files.")
    parser.add_argument("--input-files", nargs="*", default=[], help="Optional explicit FAISS category JSONL files.")
    parser.add_argument(
        "--embedding-input-dir",
        default=DEFAULT_EMBEDDING_INPUT_DIR,
        help="Directory containing matching embedding JSONL files from stage 03.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write reranked category JSONL files.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Base URL for the hosted reranker API.")
    parser.add_argument("--reranker-model", required=True, help="Reranker model name hosted on vLLM.")
    parser.add_argument(
        "--endpoints",
        nargs="*",
        default=list(DEFAULT_RERANK_ENDPOINTS),
        help="Candidate rerank endpoints to try in order.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--concurrent-records",
        type=int,
        default=DEFAULT_CONCURRENT_RECORDS,
        help="Number of records to rerank concurrently.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of final reranked categories to keep.")
    parser.add_argument(
        "--rerank-query-max-chars",
        type=int,
        default=DEFAULT_RERANK_QUERY_MAX_CHARS,
        help="Maximum query text characters sent to the reranker.",
    )
    parser.add_argument(
        "--faiss-doc-limit",
        type=int,
        default=DEFAULT_FAISS_DOC_LIMIT,
        help="Maximum number of FAISS categories from the input file to send to the reranker.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    embedding_input_dir = Path(args.embedding_input_dir)
    output_dir = Path(args.output_dir)

    input_paths = [Path(path) for path in args.input_files] if args.input_files else sorted(input_dir.glob("*.jsonl"))
    if not input_paths:
        raise FileNotFoundError(f"No input JSONL files found in {input_dir}")

    rerank_client = HostedRerankerClient(
        api_base=args.api_base,
        model_name=args.reranker_model,
        endpoints=args.endpoints,
        timeout=args.timeout,
    )
    reranker_model_details = rerank_client.get_model_metadata()

    for input_path in input_paths:
        output_path = output_path_for_input(input_path, output_dir, args.reranker_model)
        print(f"Input: {input_path}")
        print(f"Embedding input: {corresponding_embedding_path(input_path, embedding_input_dir)}")
        print(f"Output: {output_path}")
        process_file(
            input_path=input_path,
            embedding_input_dir=embedding_input_dir,
            output_path=output_path,
            rerank_client=rerank_client,
            reranker_model=args.reranker_model,
            reranker_model_details=reranker_model_details,
            concurrent_records=args.concurrent_records,
            rerank_query_max_chars=args.rerank_query_max_chars,
            rerank_top_k=args.top_k,
            faiss_doc_limit=args.faiss_doc_limit,
        )


if __name__ == "__main__":
    main()
