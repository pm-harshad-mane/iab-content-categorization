from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import faiss
import numpy as np
import requests

DEFAULT_TAXONOMY_PATH = "taxonomy/Content_Taxonomy_3.1_6.tsv"
DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_ENDPOINT = "/v1/embeddings"
DEFAULT_MODELS_ENDPOINT = "/v1/models"
DEFAULT_TIMEOUT = 180
DEFAULT_INPUT_DIR = "03_fetched_url_content_embedding_files"
DEFAULT_OUTPUT_DIR = "04_fetched_url_content_embedding_categories_files"
DEFAULT_TAXONOMY_BATCH_SIZE = 128
DEFAULT_TOP_K = 10
DEFAULT_CONCURRENT_RECORDS = 500

_thread_local = threading.local()


@dataclass
class TaxonomyRow:
    unique_id: str
    parent_id: str
    tier1: str
    tier2: str
    tier3: str
    tier4: str
    path: str
    description: str
    keywords: str
    negative_keywords: str
    index_text: str


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def output_path_for_input(input_path: Path, output_dir: Path, model: str) -> Path:
    return output_dir / f"{input_path.stem}__faiss.jsonl"


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


def chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    size = max(1, int(size))
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (vectors / norms).astype("float32")


def fetch_model_details(api_base: str, timeout: int, model_name: str) -> Dict[str, Any]:
    url = api_base.rstrip("/") + DEFAULT_MODELS_ENDPOINT
    try:
        response = _get_session().get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {"requested_model": model_name}

    data = payload.get("data")
    if not isinstance(data, list):
        return {"requested_model": model_name}

    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("id") == model_name or item.get("root") == model_name:
            result = {"requested_model": model_name}
            for key in ["id", "root", "max_model_len", "owned_by"]:
                if key in item:
                    result[key] = item.get(key)
            return result
    return {"requested_model": model_name}


def embed_text_batch(
    api_base: str,
    endpoint: str,
    model: str,
    texts: Sequence[str],
    timeout: int,
) -> Tuple[np.ndarray, float]:
    payload = {"model": model, "input": list(texts)}
    t0 = time.perf_counter()
    response = _get_session().post(
        api_base.rstrip("/") + endpoint,
        json=payload,
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    response.raise_for_status()
    body = response.json()
    data = body.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError(f"Unexpected embeddings response shape: {body}")

    ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
    vectors = np.asarray([item["embedding"] for item in ordered], dtype="float32")
    return normalize_l2(vectors), round(elapsed_ms, 3)


def load_taxonomy_rows(path: Path) -> List[TaxonomyRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: List[TaxonomyRow] = []
        required = {
            "Unique ID",
            "Parent",
            "Tier 1",
            "Tier 2",
            "Tier 3",
            "Tier 4",
            "Path",
            "Description",
            "Keywords",
        }
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing expected taxonomy columns: {missing}")

        for row in reader:
            path_text = normalize_text(row.get("Path"))
            description = normalize_text(row.get("Description"))
            keywords = normalize_text(row.get("Keywords"))
            index_parts: List[str] = []
            if path_text:
                index_parts.append(f"path: {path_text}")
            if description:
                index_parts.append(f"description: {description}")
            if keywords:
                index_parts.append(f"keywords: {keywords}")
            rows.append(
                TaxonomyRow(
                    unique_id=normalize_text(row.get("Unique ID")),
                    parent_id=normalize_text(row.get("Parent")),
                    tier1=normalize_text(row.get("Tier 1")),
                    tier2=normalize_text(row.get("Tier 2")),
                    tier3=normalize_text(row.get("Tier 3")),
                    tier4=normalize_text(row.get("Tier 4")),
                    path=path_text,
                    description=description,
                    keywords=keywords,
                    negative_keywords=normalize_text(row.get("Negative Keywords")),
                    index_text="\n".join(index_parts),
                )
            )
    return rows


def build_taxonomy_index(
    taxonomy_rows: Sequence[TaxonomyRow],
    api_base: str,
    endpoint: str,
    model: str,
    timeout: int,
    batch_size: int,
) -> Tuple[faiss.IndexFlatIP, np.ndarray, Dict[str, Any]]:
    texts = [row.index_text for row in taxonomy_rows]
    vectors_batches: List[np.ndarray] = []
    total_embed_ms = 0.0
    t0 = time.perf_counter()
    for batch in chunked(texts, batch_size):
        batch_vectors, batch_ms = embed_text_batch(
            api_base=api_base,
            endpoint=endpoint,
            model=model,
            texts=batch,
            timeout=timeout,
        )
        vectors_batches.append(batch_vectors)
        total_embed_ms += batch_ms

    if not vectors_batches:
        raise RuntimeError("No taxonomy embeddings were produced.")

    taxonomy_embeddings = np.vstack(vectors_batches).astype("float32")
    index = faiss.IndexFlatIP(int(taxonomy_embeddings.shape[1]))
    index.add(taxonomy_embeddings)

    details = {
        "taxonomy_rows": len(taxonomy_rows),
        "embedding_dimension": int(taxonomy_embeddings.shape[1]),
        "taxonomy_embedding_ms": round(total_embed_ms, 3),
        "taxonomy_setup_total_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "taxonomy_text_fields": ["Path", "Description", "Keywords"],
        "negative_keywords_used_in_index": False,
    }
    return index, taxonomy_embeddings, details


def extract_model_from_input(path: Path) -> Optional[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            record = json.loads(payload)
            model = normalize_text(record.get("embedding_model"))
            if model:
                return model
    return None


def build_success_record(
    record: Dict[str, Any],
    embedding_model: str,
    taxonomy_model_details: Dict[str, Any],
    top_k: int,
    categories: List[Dict[str, Any]],
    search_ms: float,
    total_ms: float,
) -> Dict[str, Any]:
    query_dim = len(record.get("embedding") or [])
    return {
        "input_url": normalize_text(record.get("input_url") or record.get("url")),
        "url": normalize_text(record.get("url") or record.get("input_url")),
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "ok",
        "source_status": normalize_text(record.get("source_status")),
        "embedding_model": embedding_model,
        "domain": normalize_text(record.get("domain")),
        "title": normalize_text(record.get("title")),
        "embedding_dim": query_dim,
        "faiss_top_k": int(top_k),
        "model_details": taxonomy_model_details,
        "top_categories": categories,
        "timing_ms": {
            "faiss_search": round(search_ms, 3),
            "total": round(total_ms, 3),
        },
    }


def build_error_record(
    record: Dict[str, Any],
    embedding_model: str,
    taxonomy_model_details: Dict[str, Any],
    error_type: str,
    error_code: str,
    message: str,
    retryable: bool = False,
    total_ms: Optional[float] = None,
) -> Dict[str, Any]:
    output = {
        "input_url": normalize_text(record.get("input_url") or record.get("url")),
        "url": normalize_text(record.get("url") or record.get("input_url")),
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "error",
        "source_status": normalize_text(record.get("source_status") or record.get("status")),
        "embedding_model": embedding_model,
        "domain": normalize_text(record.get("domain")),
        "title": normalize_text(record.get("title")),
        "model_details": taxonomy_model_details,
        "error_type": error_type,
        "error_code": error_code,
        "message": message,
        "retryable": retryable,
    }
    if total_ms is not None:
        output["timing_ms"] = {"total": round(total_ms, 3)}
    return output


def make_category_dict(row: TaxonomyRow, score: float) -> Dict[str, Any]:
    return {
        "unique_id": row.unique_id,
        "parent_id": row.parent_id,
        "tier1": row.tier1,
        "tier2": row.tier2,
        "tier3": row.tier3,
        "tier4": row.tier4,
        "path": row.path,
        "description": row.description,
        "keywords": row.keywords,
        "faiss_score": round(float(score), 6),
    }


def categorize_record(
    record: Dict[str, Any],
    embedding_model: str,
    taxonomy_model_details: Dict[str, Any],
    index: faiss.IndexFlatIP,
    taxonomy_rows: Sequence[TaxonomyRow],
    top_k: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    if normalize_text(record.get("status")) != "ok":
        return build_error_record(
            record=record,
            embedding_model=embedding_model,
            taxonomy_model_details=taxonomy_model_details,
            error_type="SourceEmbeddingRecordError",
            error_code="embedding_record_not_ok",
            message="Input embedding record status is not ok; FAISS categorization skipped.",
            retryable=False,
            total_ms=(time.perf_counter() - started) * 1000.0,
        )

    embedding = record.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        return build_error_record(
            record=record,
            embedding_model=embedding_model,
            taxonomy_model_details=taxonomy_model_details,
            error_type="ValueError",
            error_code="missing_embedding",
            message="Input record has no usable embedding vector.",
            retryable=False,
            total_ms=(time.perf_counter() - started) * 1000.0,
        )

    query = np.asarray([embedding], dtype="float32")
    if query.shape[1] != int(taxonomy_model_details["taxonomy_index"]["embedding_dimension"]):
        return build_error_record(
            record=record,
            embedding_model=embedding_model,
            taxonomy_model_details=taxonomy_model_details,
            error_type="ValueError",
            error_code="embedding_dim_mismatch",
            message=(
                f"Input embedding dimension {query.shape[1]} does not match "
                f"taxonomy embedding dimension {taxonomy_model_details['taxonomy_index']['embedding_dimension']}."
            ),
            retryable=False,
            total_ms=(time.perf_counter() - started) * 1000.0,
        )

    query = normalize_l2(query)
    t0 = time.perf_counter()
    scores, indices = index.search(query, top_k)
    search_ms = (time.perf_counter() - t0) * 1000.0

    categories: List[Dict[str, Any]] = []
    for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
        if idx < 0:
            continue
        categories.append(make_category_dict(taxonomy_rows[idx], score))

    return build_success_record(
        record=record,
        embedding_model=embedding_model,
        taxonomy_model_details=taxonomy_model_details,
        top_k=top_k,
        categories=categories,
        search_ms=search_ms,
        total_ms=(time.perf_counter() - started) * 1000.0,
    )


def process_input_file(
    input_path: Path,
    output_path: Path,
    embedding_model: str,
    taxonomy_model_details: Dict[str, Any],
    index: faiss.IndexFlatIP,
    taxonomy_rows: Sequence[TaxonomyRow],
    top_k: int,
    concurrent_records: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cached_hashes = load_cached_url_hashes(output_path)

    pending: List[Tuple[int, Dict[str, Any]]] = []
    with input_path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            payload = line.strip()
            if not payload:
                continue
            record = json.loads(payload)
            url_hash = normalize_text(record.get("url_hash"))
            if url_hash and url_hash in cached_hashes:
                continue
            pending.append((line_number, record))

    with output_path.open("a", encoding="utf-8") as dst:
        with ThreadPoolExecutor(max_workers=max(1, int(concurrent_records))) as executor:
            futures = {
                executor.submit(
                    categorize_record,
                    record,
                    embedding_model,
                    taxonomy_model_details,
                    index,
                    taxonomy_rows,
                    top_k,
                ): (line_number, record)
                for line_number, record in pending
            }
            for future in as_completed(futures):
                line_number, record = futures[future]
                output_record = future.result()
                dst.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                dst.flush()
                url_hash = normalize_text(record.get("url_hash"))
                if url_hash:
                    cached_hashes.add(url_hash)
                print(f"[{line_number}] {output_record['status']} {normalize_text(record.get('url') or record.get('input_url'))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a taxonomy FAISS index from Content_Taxonomy_3.1_6.tsv and retrieve top categories for embedding JSONL files."
    )
    parser.add_argument("--taxonomy-tsv", default=DEFAULT_TAXONOMY_PATH, help="Taxonomy TSV path.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Embedding API base URL.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Embedding endpoint path.")
    parser.add_argument("--model", default="", help="Embedding model name. If omitted, infer from each input file.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument("--taxonomy-batch-size", type=int, default=DEFAULT_TAXONOMY_BATCH_SIZE, help="Taxonomy embedding batch size.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-N FAISS categories to return per record.")
    parser.add_argument("--concurrent-records", type=int, default=DEFAULT_CONCURRENT_RECORDS, help="Number of embedding records to process concurrently.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing embedding JSONL files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write FAISS category JSONL files.")
    parser.add_argument("--input-files", nargs="*", default=None, help="Optional explicit embedding JSONL files to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy_rows = load_taxonomy_rows(Path(args.taxonomy_tsv))

    if args.input_files:
        input_paths = [Path(path) for path in args.input_files]
    else:
        input_paths = sorted(Path(args.input_dir).glob("*.jsonl"))

    if not input_paths:
        raise SystemExit("No input embedding JSONL files found.")

    taxonomy_cache: Dict[str, Tuple[faiss.IndexFlatIP, Dict[str, Any]]] = {}
    output_dir = Path(args.output_dir)

    for input_path in input_paths:
        embedding_model = normalize_text(args.model) or normalize_text(extract_model_from_input(input_path))
        if not embedding_model:
            raise RuntimeError(f"Could not determine embedding model for {input_path}")

        if embedding_model not in taxonomy_cache:
            model_details = fetch_model_details(args.api_base, args.timeout, embedding_model)
            index, _, taxonomy_index_details = build_taxonomy_index(
                taxonomy_rows=taxonomy_rows,
                api_base=args.api_base,
                endpoint=args.endpoint,
                model=embedding_model,
                timeout=args.timeout,
                batch_size=args.taxonomy_batch_size,
            )
            taxonomy_cache[embedding_model] = (
                index,
                {
                    "embedding_model": embedding_model,
                    "taxonomy_source": str(Path(args.taxonomy_tsv)),
                    "taxonomy_index": taxonomy_index_details,
                    "embedding_api_model_details": model_details,
                    "concurrent_records": int(args.concurrent_records),
                    "top_k": int(args.top_k),
                },
            )

        index, taxonomy_model_details = taxonomy_cache[embedding_model]
        output_path = output_path_for_input(input_path, output_dir, embedding_model)

        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        process_input_file(
            input_path=input_path,
            output_path=output_path,
            embedding_model=embedding_model,
            taxonomy_model_details=taxonomy_model_details,
            index=index,
            taxonomy_rows=taxonomy_rows,
            top_k=args.top_k,
            concurrent_records=args.concurrent_records,
        )


if __name__ == "__main__":
    main()
