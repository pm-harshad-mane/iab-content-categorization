from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_TAXONOMY_PATH = "taxonomy/Content_Taxonomy_3.1_6.tsv"
DEFAULT_INPUT_DIR = "02_fetched_url_content_files"
DEFAULT_OUTPUT_DIR = "07_fetched_url_content_categories_by_Gemma4"
DEFAULT_TOP_K = 5
DEFAULT_MAX_BODY_CHARS = 6000
DEFAULT_MAX_OUTPUT_TOKENS = 1000
DEFAULT_TIMEOUT = 300
DEFAULT_WORKERS = 32
DEFAULT_TEMPERATURE = 0.0

_thread_local = threading.local()


@dataclass
class TaxonomyRow:
    unique_id: str
    parent_id: str
    path: str
    description: str
    keywords: str


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", normalize_text(model)).strip("_") or "model"


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
    stem = base.stem
    suffix = base.suffix
    return base.with_name(f"{stem}__shard{shard_index + 1:02d}of{shard_count:02d}{suffix}")


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
            url_hash = normalize_text(record.get("url_hash"))
            if url_hash:
                cached_hashes.add(url_hash)
    return cached_hashes


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            yield line_number, json.loads(payload)


def load_taxonomy_rows(path: Path) -> List[TaxonomyRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Unique ID", "Parent", "Path", "Description", "Keywords"}
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing expected taxonomy columns: {missing}")

        rows: List[TaxonomyRow] = []
        for row in reader:
            rows.append(
                TaxonomyRow(
                    unique_id=normalize_text(row.get("Unique ID")),
                    parent_id=normalize_text(row.get("Parent")),
                    path=normalize_text(row.get("Path")),
                    description=normalize_text(row.get("Description")),
                    keywords=normalize_text(row.get("Keywords")),
                )
            )
    return rows


def taxonomy_lookup_by_id(rows: Sequence[TaxonomyRow]) -> Dict[str, TaxonomyRow]:
    return {row.unique_id: row for row in rows if row.unique_id}


def build_taxonomy_listing(rows: Sequence[TaxonomyRow]) -> str:
    lines: List[str] = []
    for row in rows:
        if not row.unique_id or not row.path:
            continue
        parts = [f"id={row.unique_id}", f"path={row.path}"]
        if row.description:
            parts.append(f"description={row.description}")
        if row.keywords:
            parts.append(f"keywords={row.keywords}")
        lines.append(" || ".join(parts))
    return "\n".join(lines)


def build_page_query_text(record: Dict[str, Any], max_body_chars: int) -> str:
    parts: List[str] = []
    url = normalize_text(record.get("url") or record.get("input_url"))
    domain = normalize_text(record.get("domain"))
    title = normalize_text(record.get("title"))
    meta_description = normalize_text(record.get("meta_description"))
    headings = [normalize_text(item) for item in (record.get("headings") or []) if normalize_text(item)]
    body_text = normalize_text(record.get("body_text"))

    if url:
        parts.append(f"url: {url}")
    if domain:
        parts.append(f"domain: {domain}")
    if title:
        parts.append(f"title: {title}")
    if meta_description:
        parts.append(f"description: {meta_description}")
    if headings:
        parts.append("headings: " + " | ".join(headings[:8]))
    if body_text:
        parts.append(f"content: {body_text[:max_body_chars]}")
    return " || ".join(parts)


def build_system_prompt(top_k: int, taxonomy_listing: str) -> str:
    return (
        "You categorize web pages into the provided content taxonomy.\n"
        "Rules:\n"
        f"- Return exactly {top_k} categories whenever the page has enough plausible taxonomy matches.\n"
        f"- If fewer than {top_k} categories are truly plausible, return as many as are justified, but prefer a fuller ranked list over stopping early.\n"
        "- Use only taxonomy unique_id values that appear in the taxonomy list below.\n"
        "- Prefer the most specific applicable category path.\n"
        "- Do not invent categories.\n"
        "- Avoid duplicates.\n"
        "- Scores must be floats in [0,1], sorted from best to worst.\n"
        "- If the page does not fit the taxonomy, return an empty array.\n"
        "- Output only minified JSON matching this exact shape: "
        '{"top_categories":[{"unique_id":"<id>","score":0.0}]}\n\n'
        "Taxonomy list:\n"
        f"{taxonomy_listing}"
    )


def build_user_prompt(record: Dict[str, Any], max_body_chars: int) -> str:
    page_text = build_page_query_text(record, max_body_chars=max_body_chars)
    return (
        "Classify this web page into the taxonomy.\n"
        "Return only the top taxonomy matches as JSON.\n\n"
        f"{page_text}"
    )


def _get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def extract_json_object(text: str) -> Dict[str, Any]:
    payload = normalize_text(text)
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(f"Could not find JSON object in model response: {text[:400]}")
    return json.loads(payload[start : end + 1])


def parse_usage(body: Dict[str, Any]) -> Dict[str, Any]:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return {}
    result: Dict[str, Any] = {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is not None:
        result["input_tokens"] = prompt_tokens
    if completion_tokens is not None:
        result["output_tokens"] = completion_tokens
    if total_tokens is not None:
        result["total_tokens"] = total_tokens
    return result


def parse_chat_completion(body: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Missing choices in vLLM response: {body}")
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        raise RuntimeError(f"Unexpected first choice shape: {body}")
    message = choice0.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Missing message in first choice: {body}")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Message content is not text: {body}")

    payload = extract_json_object(content)
    categories = payload.get("top_categories")
    if not isinstance(categories, list):
        raise RuntimeError(f"Model response missing top_categories array: {payload}")

    result: List[Dict[str, Any]] = []
    for item in categories:
        if not isinstance(item, dict):
            continue
        unique_id = normalize_text(item.get("unique_id"))
        if not unique_id:
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid score in model response: {item}") from exc
        result.append({"unique_id": unique_id, "score": round(score, 6)})
    return result, parse_usage(body)


def classify_record(
    api_base: str,
    endpoint: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    session = _get_session()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_output_tokens),
        "response_format": {"type": "json_object"},
    }
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
        categories, usage = parse_chat_completion(body)
        return {
            "ok": True,
            "status_code": resp.status_code,
            "request_ms": round(elapsed_ms, 3),
            "categories": categories,
            "usage": usage,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return {
            "ok": False,
            "status_code": status_code,
            "request_ms": round(elapsed_ms, 3),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "retryable": status_code in {408, 425, 429, 500, 502, 503, 504},
        }


def model_details(
    api_base: str,
    endpoint: str,
    model: str,
    taxonomy_tsv: str,
    top_k: int,
    max_body_chars: int,
    max_output_tokens: int,
    shard_index: Optional[int],
    shard_count: int,
) -> Dict[str, Any]:
    return {
        "provider": "vllm",
        "local_model": model,
        "api_base": api_base,
        "endpoint": endpoint,
        "taxonomy_source": taxonomy_tsv,
        "taxonomy_prompt_fields": ["Unique ID", "Path", "Description", "Keywords"],
        "top_k": int(top_k),
        "max_body_chars": int(max_body_chars),
        "max_output_tokens": int(max_output_tokens),
        "prompt_layout": "taxonomy_first_then_page_content",
        "shard_index": None if shard_index is None else int(shard_index),
        "shard_count": int(shard_count),
    }


def build_source_error_record(record: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input_url": normalize_text(record.get("input_url") or record.get("url")),
        "url": normalize_text(record.get("url") or record.get("input_url")),
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "error",
        "source_status": normalize_text(record.get("status")),
        "local_model": normalize_text(model_info.get("local_model")),
        "domain": normalize_text(record.get("domain")),
        "title": normalize_text(record.get("title")),
        "error_type": "SourceRecordError",
        "error_code": "source_status_error",
        "message": "Source content record status is not ok; Gemma categorization skipped.",
        "retryable": False,
        "model_details": model_info,
    }


def build_no_text_record(record: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input_url": normalize_text(record.get("input_url") or record.get("url")),
        "url": normalize_text(record.get("url") or record.get("input_url")),
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "error",
        "source_status": "ok",
        "local_model": normalize_text(model_info.get("local_model")),
        "domain": normalize_text(record.get("domain")),
        "title": normalize_text(record.get("title")),
        "error_type": "ValueError",
        "error_code": "no_usable_text",
        "message": "Input record has no usable text for local LLM categorization.",
        "retryable": False,
        "model_details": model_info,
    }


def build_success_record(
    record: Dict[str, Any],
    model_info: Dict[str, Any],
    taxonomy_by_id: Dict[str, TaxonomyRow],
    categories: Sequence[Dict[str, Any]],
    top_k: int,
    usage: Dict[str, Any],
    request_ms: float,
    total_ms: float,
) -> Dict[str, Any]:
    enriched: List[Dict[str, Any]] = []
    for rank, item in enumerate(categories, start=1):
        row = taxonomy_by_id.get(item["unique_id"])
        if row is None:
            continue
        enriched.append(
            {
                "unique_id": row.unique_id,
                "parent_id": row.parent_id,
                "path": row.path,
                "score": round(float(item["score"]), 6),
                "llm_rank": rank,
            }
        )

    enriched = enriched[:top_k]

    return {
        "input_url": normalize_text(record.get("input_url") or record.get("url")),
        "url": normalize_text(record.get("url") or record.get("input_url")),
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "ok",
        "source_status": "ok",
        "local_model": normalize_text(model_info.get("local_model")),
        "domain": normalize_text(record.get("domain")),
        "title": normalize_text(record.get("title")),
        "llm_top_k": len(enriched),
        "llm_top_categories": enriched,
        "model_details": model_info,
        "usage": usage,
        "timing_ms": {
            "llm_request": round(request_ms, 3),
            "total": round(total_ms, 3),
        },
    }


def build_llm_error_record(
    record: Dict[str, Any],
    model_info: Dict[str, Any],
    result: Dict[str, Any],
    total_ms: float,
) -> Dict[str, Any]:
    status_code = result.get("status_code")
    error_code = "llm_http_error" if status_code else "llm_request_error"
    return {
        "input_url": normalize_text(record.get("input_url") or record.get("url")),
        "url": normalize_text(record.get("url") or record.get("input_url")),
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "error",
        "source_status": "ok",
        "local_model": normalize_text(model_info.get("local_model")),
        "domain": normalize_text(record.get("domain")),
        "title": normalize_text(record.get("title")),
        "error_type": result.get("error_type", "RuntimeError"),
        "error_code": error_code,
        "message": result.get("message", "Local LLM categorization failed."),
        "retryable": bool(result.get("retryable", False)),
        "status_code": status_code,
        "model_details": model_info,
        "timing_ms": {
            "llm_request": round(float(result.get("request_ms") or 0.0), 3),
            "total": round(total_ms, 3),
        },
    }


def process_record(
    record: Dict[str, Any],
    api_base: str,
    endpoint: str,
    model: str,
    system_prompt: str,
    taxonomy_by_id: Dict[str, TaxonomyRow],
    taxonomy_tsv: str,
    top_k: int,
    max_body_chars: int,
    max_output_tokens: int,
    shard_index: Optional[int],
    shard_count: int,
    temperature: float,
    timeout: int,
) -> Tuple[str, Dict[str, Any]]:
    model_info = model_details(
        api_base=api_base,
        endpoint=endpoint,
        model=model,
        taxonomy_tsv=taxonomy_tsv,
        top_k=top_k,
        max_body_chars=max_body_chars,
        max_output_tokens=max_output_tokens,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    url_hash = normalize_text(record.get("url_hash"))
    t0 = time.perf_counter()

    if normalize_text(record.get("status")).lower() != "ok":
        return url_hash, build_source_error_record(record, model_info)

    user_prompt = build_user_prompt(record, max_body_chars=max_body_chars)
    if not user_prompt.strip():
        return url_hash, build_no_text_record(record, model_info)

    result = classify_record(
        api_base=api_base,
        endpoint=endpoint,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    if result["ok"]:
        return (
            url_hash,
            build_success_record(
                record=record,
                model_info=model_info,
                taxonomy_by_id=taxonomy_by_id,
                categories=result["categories"],
                top_k=top_k,
                usage=result["usage"],
                request_ms=result["request_ms"],
                total_ms=total_ms,
            ),
        )
    return url_hash, build_llm_error_record(record, model_info, result, total_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Categorize fetched-content JSONL files with a locally hosted Gemma 4 vLLM server."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing fetched-content JSONL files.")
    parser.add_argument("--input-files", nargs="*", default=None, help="Optional explicit fetched-content JSONL files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for Gemma category outputs.")
    parser.add_argument("--taxonomy-tsv", default=DEFAULT_TAXONOMY_PATH, help="Taxonomy TSV path.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="vLLM OpenAI-compatible API base URL.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="vLLM endpoint path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local model name served by vLLM.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of final categories to request from the model.")
    parser.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY_CHARS, help="Maximum page body characters to include.")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Maximum output tokens per request.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent requests to the local vLLM server.")
    parser.add_argument("--shard-index", type=int, default=None, help="Optional 0-based shard index for splitting one input file across multiple Gemma workers.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total number of shards when using --shard-index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if args.shard_index is not None and not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("--shard-index must be in [0, shard_count)")

    taxonomy_rows = load_taxonomy_rows(Path(args.taxonomy_tsv))
    taxonomy_by_id = taxonomy_lookup_by_id(taxonomy_rows)
    taxonomy_listing = build_taxonomy_listing(taxonomy_rows)
    system_prompt = build_system_prompt(top_k=args.top_k, taxonomy_listing=taxonomy_listing)

    input_paths = [Path(path) for path in args.input_files] if args.input_files else sorted(Path(args.input_dir).glob("*.jsonl"))
    if not input_paths:
        raise SystemExit("No input JSONL files found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        output_path = output_path_for_input_shard(
            input_path=input_path,
            output_dir=output_dir,
            model=args.model,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        cached_hashes = load_cached_url_hashes(output_path)

        pending_records: List[Dict[str, Any]] = []
        total_records = 0
        for _, record in iter_jsonl(input_path):
            url_hash = normalize_text(record.get("url_hash"))
            if not record_belongs_to_shard(url_hash, args.shard_index, args.shard_count):
                continue
            total_records += 1
            if url_hash and url_hash in cached_hashes:
                continue
            pending_records.append(record)

        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        print(f"Total input records: {total_records}")
        print(f"Cached records: {len(cached_hashes)}")
        print(f"Pending records: {len(pending_records)}")

        if not pending_records:
            continue

        completed = len(cached_hashes)
        ok_count = 0
        error_count = 0

        with output_path.open("a", encoding="utf-8") as dst, ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map: Dict[Future[Tuple[str, Dict[str, Any]]], Dict[str, Any]] = {}
            for record in pending_records:
                future = pool.submit(
                    process_record,
                    record,
                    args.api_base,
                    args.endpoint,
                    args.model,
                    system_prompt,
                    taxonomy_by_id,
                    args.taxonomy_tsv,
                    args.top_k,
                    args.max_body_chars,
                    args.max_output_tokens,
                    args.shard_index,
                    args.shard_count,
                    args.temperature,
                    args.timeout,
                )
                future_map[future] = record

            for future in as_completed(future_map):
                _, out_record = future.result()
                dst.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                dst.flush()
                completed += 1
                if normalize_text(out_record.get("status")).lower() == "ok":
                    ok_count += 1
                else:
                    error_count += 1
                print(
                    f"[{completed}/{total_records}] status={out_record.get('status')} "
                    f"url_hash={normalize_text(out_record.get('url_hash'))}"
                )

        print(
            f"Completed {input_path.name}: wrote {len(pending_records)} new records "
            f"({ok_count} ok, {error_count} error)"
        )


if __name__ == "__main__":
    main()
