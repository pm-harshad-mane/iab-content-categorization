from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_ENDPOINT = "/v1/embeddings"
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_BODY_CHARS = 4000
DEFAULT_WORKERS = 64
DEFAULT_INPUT_DIR = "fetched_url_content_files"
DEFAULT_OUTPUT_DIR = "fetched_url_content_embedding_files"

_thread_local = threading.local()


@dataclass
class PageContent:
    input_url: str
    url: str
    url_hash: str
    domain: str
    title: str
    meta_description: str
    headings: List[str]
    body_text: str


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip()).strip("_") or "model"


def output_path_for_input(input_path: Path, output_dir: Path, model: str) -> Path:
    return output_dir / f"{input_path.stem}__{sanitize_model_name(model)}.jsonl"


def output_path_for_input_shard(
    input_path: Path,
    output_dir: Path,
    model: str,
    shard_index: Optional[int],
    shard_count: int,
) -> Path:
    base = output_path_for_input(input_path, output_dir, model)
    if shard_index is None:
        return base
    return base.with_name(f"{base.stem}__shard{shard_index + 1:02d}of{shard_count:02d}{base.suffix}")


def record_belongs_to_shard(url_hash: str, shard_index: Optional[int], shard_count: int) -> bool:
    if shard_index is None:
        return True
    if not url_hash:
        return False
    return (int(url_hash[:16], 16) % shard_count) == shard_index


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

            url_hash = str(record.get("url_hash") or "").strip()
            if url_hash:
                cached_hashes.add(url_hash)
    return cached_hashes


def parse_page_record(record: Dict[str, Any]) -> PageContent:
    return PageContent(
        input_url=normalize_text(record.get("input_url") or record.get("url")),
        url=normalize_text(record.get("url") or record.get("input_url")),
        url_hash=normalize_text(record.get("url_hash")),
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


def _get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def embed_query(
    api_base: str,
    endpoint: str,
    model: str,
    query_text: str,
    timeout: int,
) -> Dict[str, Any]:
    session = _get_session()
    payload = {"model": model, "input": [query_text]}
    t0 = time.perf_counter()
    try:
        resp = session.post(
            api_base.rstrip("/") + endpoint,
            json=payload,
            timeout=timeout,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"Unexpected response shape: {body}")
        embedding = data[0].get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Unexpected embedding payload: {body}")
        return {
            "ok": True,
            "status_code": resp.status_code,
            "embedding": embedding,
            "embedding_dim": len(embedding),
            "embedding_api_ms": round(elapsed_ms, 3),
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return {
            "ok": False,
            "status_code": status_code,
            "embedding_api_ms": round(elapsed_ms, 3),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "retryable": status_code in {408, 425, 429, 500, 502, 503, 504},
        }


def build_source_error_record(record: Dict[str, Any], model: str) -> Dict[str, Any]:
    input_url = normalize_text(record.get("input_url") or record.get("url"))
    url = normalize_text(record.get("url") or record.get("input_url"))
    return {
        "input_url": input_url,
        "url": url,
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "error",
        "source_status": normalize_text(record.get("status")),
        "embedding_model": model,
        "error_type": "SourceRecordError",
        "error_code": "source_status_error",
        "message": "Source record status is not ok; embedding generation skipped.",
        "retryable": False,
    }


def build_no_text_record(page: PageContent, model: str) -> Dict[str, Any]:
    return {
        "input_url": page.input_url,
        "url": page.url,
        "url_hash": page.url_hash,
        "status": "error",
        "source_status": "ok",
        "embedding_model": model,
        "domain": page.domain,
        "title": page.title,
        "meta_description": page.meta_description,
        "headings": page.headings,
        "query_text": "",
        "query_text_chars": 0,
        "error_type": "ValueError",
        "error_code": "no_usable_text",
        "message": "Input record has no usable text for embedding.",
        "retryable": False,
    }


def build_success_record(
    page: PageContent,
    model: str,
    query_text: str,
    embedding_result: Dict[str, Any],
    total_ms: float,
) -> Dict[str, Any]:
    return {
        "input_url": page.input_url,
        "url": page.url,
        "url_hash": page.url_hash,
        "status": "ok",
        "source_status": "ok",
        "embedding_model": model,
        "domain": page.domain,
        "title": page.title,
        "meta_description": page.meta_description,
        "headings": page.headings,
        "query_text": query_text,
        "query_text_chars": len(query_text),
        "embedding_dim": embedding_result["embedding_dim"],
        "embedding": embedding_result["embedding"],
        "timing_ms": {
            "embedding_api": embedding_result["embedding_api_ms"],
            "total": round(total_ms, 3),
        },
    }


def build_embedding_error_record(
    page: PageContent,
    model: str,
    query_text: str,
    embedding_result: Dict[str, Any],
    total_ms: float,
) -> Dict[str, Any]:
    status_code = embedding_result.get("status_code")
    error_code = "embedding_http_error" if status_code else "embedding_request_error"
    return {
        "input_url": page.input_url,
        "url": page.url,
        "url_hash": page.url_hash,
        "status": "error",
        "source_status": "ok",
        "embedding_model": model,
        "domain": page.domain,
        "title": page.title,
        "meta_description": page.meta_description,
        "headings": page.headings,
        "query_text": query_text,
        "query_text_chars": len(query_text),
        "error_type": embedding_result.get("error_type", "RuntimeError"),
        "error_code": error_code,
        "message": embedding_result.get("message", "Embedding request failed."),
        "retryable": bool(embedding_result.get("retryable", False)),
        "status_code": status_code,
        "timing_ms": {
            "embedding_api": embedding_result["embedding_api_ms"],
            "total": round(total_ms, 3),
        },
    }


def process_source_record(
    record: Dict[str, Any],
    api_base: str,
    endpoint: str,
    model: str,
    timeout: int,
    max_body_chars: int,
) -> Tuple[str, Dict[str, Any]]:
    source_status = normalize_text(record.get("status"))
    input_url = normalize_text(record.get("input_url") or record.get("url"))
    url = normalize_text(record.get("url") or record.get("input_url"))
    url_hash = normalize_text(record.get("url_hash"))
    key = url_hash or url or input_url

    if source_status != "ok":
        return key, build_source_error_record(record, model)

    page = parse_page_record(record)
    query_text = build_page_query_text(page, max_body_chars=max_body_chars)
    if not query_text:
        return key, build_no_text_record(page, model)

    t0 = time.perf_counter()
    embedding_result = embed_query(api_base, endpoint, model, query_text, timeout)
    total_ms = (time.perf_counter() - t0) * 1000.0

    if embedding_result["ok"]:
        return key, build_success_record(page, model, query_text, embedding_result, total_ms)

    return key, build_embedding_error_record(page, model, query_text, embedding_result, total_ms)


def iter_source_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc


def process_file(
    input_path: Path,
    output_path: Path,
    api_base: str,
    endpoint: str,
    model: str,
    timeout: int,
    max_body_chars: int,
    workers: int,
    shard_index: Optional[int],
    shard_count: int,
) -> None:
    cached_hashes = load_cached_url_hashes(output_path)
    pending_records: List[Dict[str, Any]] = []
    total_source_records = 0

    for record in iter_source_records(input_path):
        url_hash = normalize_text(record.get("url_hash"))
        if not url_hash:
            raise ValueError(f"Input record in {input_path} is missing url_hash.")
        if not record_belongs_to_shard(url_hash, shard_index, shard_count):
            continue
        total_source_records += 1
        if url_hash in cached_hashes:
            continue
        pending_records.append(record)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Found {total_source_records} source records, {len(cached_hashes)} cached embeddings, {len(pending_records)} pending.")

    if not pending_records:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures: Dict[Future[Tuple[str, Dict[str, Any]]], str] = {
                executor.submit(
                    process_source_record,
                    record,
                    api_base,
                    endpoint,
                    model,
                    timeout,
                    max_body_chars,
                ): normalize_text(record.get("url_hash"))
                for record in pending_records
            }

            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()[1]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{index}/{len(pending_records)}] {record.get('status')} {record.get('url')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reusable URL-content embeddings via a vLLM embeddings endpoint.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY_CHARS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-files", nargs="*", default=None, help="Specific source JSONL files. Defaults to all JSONL files in --input-dir.")
    parser.add_argument("--shard-index", type=int, default=None, help="Optional 0-based shard index for splitting one input file across multiple embedding workers.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total number of shards when using --shard-index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if args.shard_index is not None and not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("--shard-index must be in [0, shard_count)")
    output_dir = Path(args.output_dir)

    if args.input_files:
        input_paths = [Path(p) for p in args.input_files]
    else:
        input_paths = sorted(Path(args.input_dir).glob("*.jsonl"))

    if not input_paths:
        raise SystemExit("No input JSONL files found.")

    for input_path in input_paths:
        process_file(
            input_path=input_path,
            output_path=output_path_for_input_shard(
                input_path=input_path,
                output_dir=output_dir,
                model=args.model,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            ),
            api_base=args.api_base,
            endpoint=args.endpoint,
            model=args.model,
            timeout=args.timeout,
            max_body_chars=args.max_body_chars,
            workers=args.workers,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )


if __name__ == "__main__":
    main()
