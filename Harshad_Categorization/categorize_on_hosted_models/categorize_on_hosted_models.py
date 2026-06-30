#!/usr/bin/env python3
"""
Categorize page-content JSONL using hosted embedding and reranker models.

Pipeline:
1. Load taxonomy TSV and build taxonomy embeddings once from category descriptions.
2. Keep taxonomy embeddings in memory and build a FAISS inner-product index.
3. Read input JSONL containing page-content objects.
4. For each successful page object:
   - build one content query text
   - embed it with the hosted embedding model
   - retrieve top-N taxonomy candidates via FAISS
   - rerank those candidates with the hosted reranker model
5. Write one JSON object per input record to an output JSONL file.

Notes:
- Embeddings are normalized and searched with FAISS IndexFlatIP.
- Reranker API handling is intentionally flexible because hosted deployments vary.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
import requests
from requests import HTTPError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

DEFAULT_TAXONOMY_PATH = "taxonomy/Content_Taxonomy_3.1_2.tsv"
DEFAULT_INPUT_JSONL = "save_url_content_1000.jsonl"
DEFAULT_OUTPUT_JSONL = "categorized_on_hosted_models.jsonl"

DEFAULT_EMBED_API_BASE = "http://127.0.0.1:8000"
DEFAULT_RERANK_API_BASE = "http://127.0.0.1:8001"
DEFAULT_EMBED_ENDPOINT = "/v1/embeddings"
DEFAULT_RERANK_ENDPOINTS = ["/v1/rerank", "/rerank", "/score"]
DEFAULT_MODELS_ENDPOINT = "/v1/models"

DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

DEFAULT_FAISS_TOP_K = 10
DEFAULT_FINAL_TOP_K = 5
DEFAULT_CONCURRENT_RECORDS = 4
DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_EMBED_BATCH_SIZE = 128
DEFAULT_MAX_BODY_CHARS = 4000
DEFAULT_RERANK_QUERY_MAX_CHARS = 1800
DEFAULT_HTTP_MAX_RETRIES = 3
DEFAULT_HTTP_RETRY_BACKOFF_MS = 200

_thread_local = threading.local()
_print_lock = threading.Lock()


@dataclass
class PageContent:
    url: str
    domain: str
    title: str
    meta_description: str
    headings: List[str]
    body_text: str


@dataclass
class RetrievalCandidate:
    unique_id: int
    parent_id: Optional[int]
    tier1: str
    tier2: str
    tier3: str
    tier4: str
    path: str
    description: str
    faiss_score: float = 0.0
    rerank_score: float = 0.0


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def build_path_string(row: pd.Series) -> str:
    levels = [row.get("tier1", ""), row.get("tier2", ""), row.get("tier3", ""), row.get("tier4", "")]
    return " > ".join(part for part in (normalize_text(x) for x in levels) if part)


def load_taxonomy_tsv(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")

    normalized_cols = {}
    for col in df.columns:
        key = re.sub(r"\s+", " ", col.strip().lower())
        normalized_cols[col] = key
    df = df.rename(columns=normalized_cols)

    rename_map = {
        "unique id": "unique_id",
        "parent": "parent_id",
        "tier 1": "tier1",
        "tier 2": "tier2",
        "tier 3": "tier3",
        "tier 4": "tier4",
        "description": "description",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["unique_id", "parent_id", "tier1", "tier2", "description"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in taxonomy TSV: {missing}")

    for col in ["tier1", "tier2", "tier3", "tier4", "description"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(normalize_text)

    df["unique_id"] = pd.to_numeric(df["unique_id"], errors="coerce")
    df["parent_id"] = pd.to_numeric(df["parent_id"], errors="coerce")
    df = df[df["unique_id"].notna()].copy()
    df["unique_id"] = df["unique_id"].astype(int)
    df["path"] = df.apply(build_path_string, axis=1)

    return df[
        ["unique_id", "parent_id", "tier1", "tier2", "tier3", "tier4", "path", "description"]
    ].reset_index(drop=True)


def parse_page_record(record: Dict[str, Any]) -> PageContent:
    return PageContent(
        url=normalize_text(record.get("url") or record.get("input_url")),
        domain=normalize_text(record.get("domain")),
        title=normalize_text(record.get("title")),
        meta_description=normalize_text(record.get("meta_description")),
        headings=[normalize_text(x) for x in record.get("headings", []) if normalize_text(x)],
        body_text=normalize_text(record.get("body_text")),
    )


def build_page_query_text(page: PageContent, max_body_chars: int) -> str:
    parts: List[str] = []
    if page.title:
        parts.append(f"title: {page.title}")
    if page.meta_description:
        parts.append(f"description: {page.meta_description}")
    if page.headings:
        parts.append("headings: " + " | ".join(page.headings[:6]))
    if page.body_text:
        parts.append(f"content: {page.body_text[:max_body_chars]}")
    return " || ".join(parts)


def truncate_query_text(query_text: str, max_chars: int) -> str:
    text = normalize_text(query_text)
    if len(text) <= max_chars:
        return text
    if max_chars <= 16:
        return text[:max_chars]
    return text[: max_chars - 16].rstrip() + " ... [truncated]"


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (vectors / norms).astype("float32")


def _get_session() -> requests.Session:
    if not getattr(_thread_local, "session", None):
        _thread_local.session = requests.Session()
    return _thread_local.session


def _select_model_metadata(payload: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, list):
        return None

    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("id") == model_name or item.get("root") == model_name:
            return item

    for item in data:
        if not isinstance(item, dict):
            continue
        item_id = normalize_text(item.get("id"))
        item_root = normalize_text(item.get("root"))
        if model_name in {item_id, item_root}:
            return item
    return None


class HostedEmbeddingClient:
    def __init__(
        self,
        api_base: str,
        model_name: str,
        endpoint: str,
        timeout: int,
        max_retries: int,
        retry_backoff_ms: int,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_ms = max(0, int(retry_backoff_ms))
        self.models_endpoint = DEFAULT_MODELS_ENDPOINT

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
                "endpoint": self.endpoint,
                "models_endpoint": self.models_endpoint,
                "resolved_model": selected,
            }
        except Exception as exc:
            return {
                "configured_model": self.model_name,
                "api_base": self.api_base,
                "endpoint": self.endpoint,
                "models_endpoint": self.models_endpoint,
                "resolved_model": None,
                "model_lookup_error": f"{type(exc).__name__}: {exc}",
            }

    def embed_texts(self, texts: Sequence[str]) -> Tuple[np.ndarray, float]:
        if not texts:
            return np.empty((0, 0), dtype="float32"), 0.0

        payload = {
            "model": self.model_name,
            "input": list(texts),
            "encoding_format": "float",
        }
        url = self.api_base + self.endpoint

        start = time.perf_counter()
        response = self._post_with_retries(url, payload)
        body = response.json()
        data = body.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"Embedding API returned invalid response: {body}")

        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in ordered]
        if any(not isinstance(vec, list) for vec in vectors):
            raise RuntimeError(f"Embedding API response missing embeddings: {body}")

        embeddings = np.asarray(vectors, dtype="float32")
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return normalize_l2(embeddings), elapsed_ms


class HostedRerankerClient:
    def __init__(
        self,
        api_base: str,
        model_name: str,
        endpoints: Sequence[str],
        timeout: int,
        max_retries: int,
        retry_backoff_ms: int,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.endpoints = [ep if ep.startswith("/") else f"/{ep}" for ep in endpoints]
        self.timeout = timeout
        self._working_endpoint: Optional[str] = None
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_ms = max(0, int(retry_backoff_ms))
        self.models_endpoint = DEFAULT_MODELS_ENDPOINT

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
                "text": query,
                "texts": list(documents),
            },
        ]

        last_error: Optional[Exception] = None
        for payload in payload_variants:
            start = time.perf_counter()
            try:
                response = self._post_with_retries(url, payload)
                body = response.json()
                scores = self._parse_rerank_response(body, doc_count=len(documents))
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                return scores, elapsed_ms
            except Exception as exc:
                last_error = exc
        if last_error is None:
            raise RuntimeError("Reranker request failed for unknown reason.")
        raise last_error

    @staticmethod
    def _parse_rerank_response(body: Dict[str, Any], doc_count: int) -> List[float]:
        if isinstance(body.get("scores"), list):
            scores = [float(x) for x in body["scores"]]
            if len(scores) != doc_count:
                raise RuntimeError(f"Expected {doc_count} scores, got {len(scores)}")
            return scores

        for key in ("results", "data"):
            items = body.get(key)
            if not isinstance(items, list):
                continue
            scored_items: List[Tuple[int, float]] = []
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                score = item.get("relevance_score", item.get("score"))
                index = item.get("index", idx)
                if score is None:
                    continue
                scored_items.append((int(index), float(score)))
            if not scored_items:
                continue
            scores = [0.0] * doc_count
            used = set()
            for index, score in scored_items:
                if 0 <= index < doc_count:
                    scores[index] = score
                    used.add(index)
            if len(used) == doc_count or len(scored_items) == doc_count:
                return scores
            # Some APIs return only top_n items. Missing entries remain 0.
            return scores

        raise RuntimeError(f"Unsupported rerank response shape: {body}")


class FaissTaxonomyRetriever:
    def __init__(self, taxonomy_df: pd.DataFrame, taxonomy_embeddings: np.ndarray) -> None:
        if len(taxonomy_df) != len(taxonomy_embeddings):
            raise ValueError("taxonomy_df row count must match taxonomy_embeddings row count")
        self.df = taxonomy_df.reset_index(drop=True)
        self.embeddings = taxonomy_embeddings
        dim = int(taxonomy_embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(taxonomy_embeddings)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[RetrievalCandidate]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        scores, indices = self.index.search(query_vector.astype("float32"), int(top_k))

        results: List[RetrievalCandidate] = []
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) < 0:
                continue
            row = self.df.iloc[int(idx)]
            parent_id = None if pd.isna(row["parent_id"]) else int(row["parent_id"])
            results.append(
                RetrievalCandidate(
                    unique_id=int(row["unique_id"]),
                    parent_id=parent_id,
                    tier1=row["tier1"],
                    tier2=row["tier2"],
                    tier3=row["tier3"],
                    tier4=row["tier4"],
                    path=row["path"],
                    description=row["description"],
                    faiss_score=float(score),
                )
            )
        return results


def chunked(items: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_taxonomy_index(
    taxonomy_df: pd.DataFrame,
    embed_client: HostedEmbeddingClient,
    embed_batch_size: int,
) -> Tuple[FaissTaxonomyRetriever, Dict[str, Any]]:
    descriptions = taxonomy_df["description"].tolist()
    all_batches: List[np.ndarray] = []

    total_embed_ms = 0.0
    t0 = time.perf_counter()
    for batch in chunked(descriptions, embed_batch_size):
        vectors, batch_ms = embed_client.embed_texts(batch)
        all_batches.append(vectors)
        total_embed_ms += batch_ms

    if not all_batches:
        raise RuntimeError("No taxonomy embeddings were produced.")
    taxonomy_embeddings = np.vstack(all_batches).astype("float32")

    t_faiss0 = time.perf_counter()
    retriever = FaissTaxonomyRetriever(taxonomy_df=taxonomy_df, taxonomy_embeddings=taxonomy_embeddings)
    faiss_build_ms = round((time.perf_counter() - t_faiss0) * 1000, 2)

    return retriever, {
        "taxonomy_rows": len(taxonomy_df),
        "embedding_dimension": int(taxonomy_embeddings.shape[1]),
        "taxonomy_embedding_ms": round(total_embed_ms, 2),
        "faiss_index_build_ms": faiss_build_ms,
        "taxonomy_setup_total_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def candidate_to_dict(candidate: RetrievalCandidate) -> Dict[str, Any]:
    return {
        "unique_id": candidate.unique_id,
        "path": candidate.path,
        "description": candidate.description,
        "faiss_score": round(candidate.faiss_score, 6),
        "rerank_score": round(candidate.rerank_score, 6),
    }


def make_rerank_documents(candidates: Sequence[RetrievalCandidate]) -> List[str]:
    docs = []
    for candidate in candidates:
        docs.append(f"path: {candidate.path}\ndescription: {candidate.description}")
    return docs


def process_record(
    idx: int,
    record: Dict[str, Any],
    retriever: FaissTaxonomyRetriever,
    embed_client: HostedEmbeddingClient,
    rerank_client: HostedRerankerClient,
    faiss_top_k: int,
    final_top_k: int,
    max_body_chars: int,
    rerank_query_max_chars: int,
    shared_model_details: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    started = time.perf_counter()

    page_url = normalize_text(record.get("url") or record.get("input_url"))
    if normalize_text(record.get("status")).lower() != "ok":
        return idx, {
            "ok": False,
            "input_status": record.get("status"),
            "error": record.get("message") or "Input record is not status=ok",
            "page": {"url": page_url},
            "step_timings_ms": {
                "build_query_text": 0,
                "content_embedding": 0,
                "faiss_search": 0,
                "rerank": 0,
                "total": round((time.perf_counter() - started) * 1000),
            },
            "faiss_candidates": [],
            "final_ranked_categories": [],
        }

    page = parse_page_record(record)

    t_query0 = time.perf_counter()
    query_text = build_page_query_text(page, max_body_chars=max_body_chars)
    build_query_ms = round((time.perf_counter() - t_query0) * 1000, 2)
    if not query_text:
        return idx, {
            "ok": False,
            "input_status": record.get("status"),
            "error": "Input record has no usable text for embedding",
            "page": {
                "url": page.url,
                "domain": page.domain,
                "title": page.title,
            },
            "step_timings_ms": {
                "build_query_text": round(build_query_ms),
                "content_embedding": 0,
                "faiss_search": 0,
                "rerank": 0,
                "total": round((time.perf_counter() - started) * 1000),
            },
            "faiss_candidates": [],
            "final_ranked_categories": [],
        }

    embed_attempts: List[int] = []
    embed_error: Optional[str] = None
    query_text_for_embedding = query_text
    query_char_limits = [
        len(query_text),
        min(len(query_text), 2000),
        min(len(query_text), 1200),
        min(len(query_text), 800),
        min(len(query_text), 512),
    ]
    deduped_limits: List[int] = []
    for limit in query_char_limits:
        limit = max(64, int(limit))
        if limit not in deduped_limits:
            deduped_limits.append(limit)

    t_embed0 = time.perf_counter()
    query_vector: Optional[np.ndarray] = None
    embed_api_ms = 0.0
    for char_limit in deduped_limits:
        embed_attempts.append(char_limit)
        query_text_for_embedding = truncate_query_text(query_text, char_limit)
        try:
            query_vector, batch_ms = embed_client.embed_texts([query_text_for_embedding])
            embed_api_ms += batch_ms
            embed_error = None
            break
        except HTTPError as exc:
            embed_error = f"{type(exc).__name__}: {exc}"
            response = getattr(exc, "response", None)
            if response is None or response.status_code != 400:
                raise
        except Exception as exc:
            embed_error = f"{type(exc).__name__}: {exc}"
            raise

    if query_vector is None:
        return idx, {
            "ok": False,
            "input_status": record.get("status"),
            "error": embed_error or "Embedding failed after retries",
            "page": {
                "url": page.url,
                "domain": page.domain,
                "title": page.title,
            },
            "step_timings_ms": {
                "build_query_text": round(build_query_ms),
                "content_embedding": round((time.perf_counter() - t_embed0) * 1000),
                "content_embedding_api": round(embed_api_ms),
                "faiss_search": 0,
                "rerank": 0,
                "total": round((time.perf_counter() - started) * 1000),
            },
            "debug": {
                "query_text_chars": len(query_text),
                "embedding_char_limits_tried": embed_attempts,
            },
            "faiss_candidates": [],
            "final_ranked_categories": [],
        }

    content_embedding_ms = round((time.perf_counter() - t_embed0) * 1000, 2)

    t_faiss0 = time.perf_counter()
    faiss_candidates = retriever.search(query_vector=query_vector, top_k=faiss_top_k)
    faiss_search_ms = round((time.perf_counter() - t_faiss0) * 1000, 2)

    rerank_candidates = faiss_candidates[: max(1, final_top_k if final_top_k > faiss_top_k else faiss_top_k)]
    rerank_query = query_text[:rerank_query_max_chars]
    rerank_docs = make_rerank_documents(rerank_candidates)

    t_rerank0 = time.perf_counter()
    scores, rerank_api_ms, rerank_endpoint = rerank_client.rerank(
        query=rerank_query,
        documents=rerank_docs,
        top_k=len(rerank_docs),
    )
    rerank_ms = round((time.perf_counter() - t_rerank0) * 1000, 2)

    for candidate, score in zip(rerank_candidates, scores):
        candidate.rerank_score = float(score)

    final_ranked = sorted(
        rerank_candidates,
        key=lambda item: (item.rerank_score, item.faiss_score),
        reverse=True,
    )[:final_top_k]

    return idx, {
        "ok": True,
        "input_status": record.get("status"),
        "page": {
            "url": page.url,
            "domain": page.domain,
            "title": page.title,
            "meta_description": page.meta_description,
            "headings": page.headings[:6],
            "body_preview": page.body_text[:500],
        },
        "step_timings_ms": {
            "build_query_text": round(build_query_ms),
            "content_embedding": round(content_embedding_ms),
            "content_embedding_api": round(embed_api_ms),
            "faiss_search": round(faiss_search_ms),
            "rerank": round(rerank_ms),
            "rerank_api": round(rerank_api_ms),
            "total": round((time.perf_counter() - started) * 1000),
        },
        "model_details": {
            "embedding_model": shared_model_details["embedding_model"],
            "reranker_model": {
                **shared_model_details["reranker_model"],
                "resolved_endpoint_used": rerank_endpoint,
            },
            "faiss": {
                "backend": "IndexFlatIP",
                "distance": "inner_product",
                "normalized_embeddings": True,
            },
        },
        "debug": {
            "query_text_chars": len(query_text),
            "embedding_query_chars_used": len(query_text_for_embedding),
            "embedding_char_limits_tried": embed_attempts,
        },
        "faiss_candidates": [candidate_to_dict(candidate) for candidate in faiss_candidates],
        "final_ranked_categories": [candidate_to_dict(candidate) for candidate in final_ranked],
    }


def process_input_file(
    records: List[Dict[str, Any]],
    retriever: FaissTaxonomyRetriever,
    embed_client: HostedEmbeddingClient,
    rerank_client: HostedRerankerClient,
    faiss_top_k: int,
    final_top_k: int,
    concurrent_records: int,
    max_body_chars: int,
    rerank_query_max_chars: int,
    shared_model_details: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results: List[Optional[Dict[str, Any]]] = [None] * len(records)
    workers = max(1, int(concurrent_records))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(
                process_record,
                idx,
                record,
                retriever,
                embed_client,
                rerank_client,
                faiss_top_k,
                final_top_k,
                max_body_chars,
                rerank_query_max_chars,
                shared_model_details,
            ): idx
            for idx, record in enumerate(records)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                out_idx, payload = future.result()
            except Exception as exc:
                original = records[idx]
                payload = {
                    "ok": False,
                    "input_status": original.get("status"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "page": {"url": normalize_text(original.get('url') or original.get('input_url'))},
                    "step_timings_ms": None,
                    "faiss_candidates": [],
                    "final_ranked_categories": [],
                }
                out_idx = idx
            results[out_idx] = payload
            with _print_lock:
                print(f"[{out_idx + 1}/{len(records)}] ok={payload.get('ok')} url={payload.get('page', {}).get('url', '')}")

    final_results: List[Dict[str, Any]] = []
    for result in results:
        if result is None:
            raise RuntimeError("Internal error: missing result for one or more records")
        final_results.append(result)
    return final_results


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Categorize page-content JSONL via hosted embedding and reranker models.")
    parser.add_argument("--taxonomy", default=DEFAULT_TAXONOMY_PATH, help="Path to taxonomy TSV.")
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL, help="Input page-content JSONL path.")
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path.")

    parser.add_argument("--embed-api-base", default=DEFAULT_EMBED_API_BASE, help="Embedding service API base URL.")
    parser.add_argument("--embed-endpoint", default=DEFAULT_EMBED_ENDPOINT, help="Embedding endpoint path.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Hosted embedding model name.")
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE, help="Batch size for taxonomy embedding requests.")

    parser.add_argument("--rerank-api-base", default=DEFAULT_RERANK_API_BASE, help="Reranker service API base URL.")
    parser.add_argument(
        "--rerank-endpoints",
        nargs="+",
        default=DEFAULT_RERANK_ENDPOINTS,
        help="Reranker endpoint path candidates. First successful one is reused.",
    )
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL, help="Hosted reranker model name.")

    parser.add_argument("--faiss-top-k", type=int, default=DEFAULT_FAISS_TOP_K, help="How many FAISS candidates to retrieve.")
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K, help="How many reranked categories to save.")
    parser.add_argument("--concurrent-records", type=int, default=DEFAULT_CONCURRENT_RECORDS, help="How many input records to process concurrently.")
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument("--http-max-retries", type=int, default=DEFAULT_HTTP_MAX_RETRIES, help="Max retries for transient hosted-model HTTP failures.")
    parser.add_argument("--http-retry-backoff-ms", type=int, default=DEFAULT_HTTP_RETRY_BACKOFF_MS, help="Base backoff in milliseconds between transient HTTP retries.")
    parser.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY_CHARS, help="Max body chars used in the embedding query text.")
    parser.add_argument("--rerank-query-max-chars", type=int, default=DEFAULT_RERANK_QUERY_MAX_CHARS, help="Max chars sent as reranker query text.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of input rows to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    effective_workers = max(1, int(args.concurrent_records))

    records = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]
    if not records:
        raise SystemExit("No input records to process.")

    taxonomy_df = load_taxonomy_tsv(args.taxonomy)

    embed_client = HostedEmbeddingClient(
        api_base=args.embed_api_base,
        model_name=args.embed_model,
        endpoint=args.embed_endpoint,
        timeout=args.request_timeout,
        max_retries=args.http_max_retries,
        retry_backoff_ms=args.http_retry_backoff_ms,
    )
    rerank_client = HostedRerankerClient(
        api_base=args.rerank_api_base,
        model_name=args.rerank_model,
        endpoints=args.rerank_endpoints,
        timeout=args.request_timeout,
        max_retries=args.http_max_retries,
        retry_backoff_ms=args.http_retry_backoff_ms,
    )
    shared_model_details = {
        "embedding_model": embed_client.get_model_metadata(),
        "reranker_model": rerank_client.get_model_metadata(),
    }

    print(f"Loading taxonomy from {args.taxonomy}")
    print(f"Building taxonomy embeddings via {args.embed_api_base}{args.embed_endpoint}")
    retriever, taxonomy_stats = build_taxonomy_index(
        taxonomy_df=taxonomy_df,
        embed_client=embed_client,
        embed_batch_size=args.embed_batch_size,
    )

    results = process_input_file(
        records=records,
        retriever=retriever,
        embed_client=embed_client,
        rerank_client=rerank_client,
        faiss_top_k=args.faiss_top_k,
        final_top_k=args.final_top_k,
        concurrent_records=args.concurrent_records,
        max_body_chars=args.max_body_chars,
        rerank_query_max_chars=args.rerank_query_max_chars,
        shared_model_details=shared_model_details,
    )

    shared_metadata = {
        "run_metadata": {
            "input_jsonl": args.input_jsonl,
            "output_jsonl": args.output_jsonl,
            "taxonomy_tsv": args.taxonomy,
            "concurrent_records": effective_workers,
            "worker_details": {
                "executor_type": "ThreadPoolExecutor",
                "max_workers": effective_workers,
            },
            "faiss_top_k": int(args.faiss_top_k),
            "final_top_k": int(args.final_top_k),
            "embed_model": args.embed_model,
            "rerank_model": args.rerank_model,
            "http_max_retries": int(args.http_max_retries),
            "http_retry_backoff_ms": int(args.http_retry_backoff_ms),
            "model_details": shared_model_details,
            "taxonomy_setup": taxonomy_stats,
            "run_total_ms": None,
        }
    }

    output_rows = []
    for row in results:
        enriched = dict(shared_metadata)
        enriched.update(row)
        output_rows.append(enriched)

    total_run_ms = round((time.perf_counter() - run_started) * 1000)
    shared_metadata["run_metadata"]["run_total_ms"] = total_run_ms
    shared_metadata["run_metadata"]["record_count"] = len(records)
    shared_metadata["run_metadata"]["successful_records"] = sum(1 for row in output_rows if row.get("ok"))
    shared_metadata["run_metadata"]["failed_records"] = len(records) - shared_metadata["run_metadata"]["successful_records"]

    for row in output_rows:
        row["run_metadata"] = shared_metadata["run_metadata"]

    write_jsonl(args.output_jsonl, output_rows)

    ok_count = shared_metadata["run_metadata"]["successful_records"]
    print(
        f"Wrote {args.output_jsonl} with {ok_count}/{len(output_rows)} successful records "
        f"in {total_run_ms} ms"
    )


if __name__ == "__main__":
    main()
