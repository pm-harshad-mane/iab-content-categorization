from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import anthropic

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_TAXONOMY_PATH = "taxonomy/Content_Taxonomy_3.1_6.tsv"
DEFAULT_INPUT_DIR = "02_fetched_url_content_files"
DEFAULT_OUTPUT_DIR = "09_fetched_url_content_categories_by_Claude"
DEFAULT_TOP_K = 5
DEFAULT_MAX_BODY_CHARS = 6000
DEFAULT_MAX_OUTPUT_TOKENS = 1000
DEFAULT_TIMEOUT = 300
DEFAULT_WORKERS = 8
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 4

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


def build_page_text(record: Dict[str, Any], max_body_chars: int) -> str:
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


def build_system_prompt_blocks(top_k: int, taxonomy_listing: str) -> List[Dict[str, Any]]:
    instructions = (
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
        '{"top_categories":[{"unique_id":"<id>","score":0.0}]}\n'
        "- Do not include any prose, explanation, or markdown fences."
    )
    # The taxonomy listing is large and identical across every request, so it is
    # placed in its own cached system block to enable Anthropic prompt caching.
    return [
        {"type": "text", "text": instructions},
        {
            "type": "text",
            "text": "Taxonomy list:\n" + taxonomy_listing,
            "cache_control": {"type": "ephemeral"},
        },
    ]


def build_user_prompt(record: Dict[str, Any], max_body_chars: int) -> str:
    page_text = build_page_text(record, max_body_chars=max_body_chars)
    return (
        "Classify this web page into the taxonomy.\n"
        "Return only the top taxonomy matches as JSON.\n\n"
        f"{page_text}"
    )


def _get_client(api_key: Optional[str], timeout: int, max_retries: int) -> anthropic.Anthropic:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
            max_retries=max_retries,
        )
        _thread_local.client = client
    return client


def extract_json_object(text: str) -> Dict[str, Any]:
    payload = normalize_text(text)
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(f"Could not find JSON object in model response: {text[:400]}")
    return json.loads(payload[start : end + 1])


def message_text(message: Any) -> str:
    chunks: List[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            chunks.append(getattr(block, "text", "") or "")
    return "".join(chunks)


def parse_usage(message: Any) -> Dict[str, Any]:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    input_tokens = getattr(usage, "input_tokens", None) or 0
    output_tokens = getattr(usage, "output_tokens", None) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", None)
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    result: Dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_creation is not None:
        result["cache_creation_input_tokens"] = cache_creation
    if cache_read is not None:
        result["cache_read_input_tokens"] = cache_read
    return result


def parse_categories(message: Any) -> List[Dict[str, Any]]:
    payload = extract_json_object(message_text(message))
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
    return result


def classify_record(
    client: anthropic.Anthropic,
    model: str,
    system_blocks: List[Dict[str, Any]],
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        message = client.messages.create(
            model=model,
            max_tokens=int(max_output_tokens),
            temperature=float(temperature),
            system=system_blocks,
            messages=[{"role": "user", "content": user_prompt}],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        categories = parse_categories(message)
        return {
            "ok": True,
            "request_ms": round(elapsed_ms, 3),
            "categories": categories,
            "usage": parse_usage(message),
            "stop_reason": getattr(message, "stop_reason", None),
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        status_code = getattr(exc, "status_code", None)
        return {
            "ok": False,
            "status_code": status_code,
            "request_ms": round(elapsed_ms, 3),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "retryable": status_code in {408, 409, 429, 500, 502, 503, 504} if status_code else False,
        }


def model_details(
    model: str,
    taxonomy_tsv: str,
    top_k: int,
    max_body_chars: int,
    max_output_tokens: int,
) -> Dict[str, Any]:
    return {
        "provider": "anthropic",
        "cloud_model": model,
        "endpoint": "/v1/messages",
        "taxonomy_source": taxonomy_tsv,
        "taxonomy_prompt_fields": ["Unique ID", "Path", "Description", "Keywords"],
        "top_k": int(top_k),
        "max_body_chars": int(max_body_chars),
        "max_output_tokens": int(max_output_tokens),
        "prompt_layout": "taxonomy_first_then_page_content",
        "prompt_caching": "ephemeral_on_taxonomy_block",
    }


def score_gap(top_categories: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not top_categories:
        return None
    top1 = top_categories[0].get("score")
    if len(top_categories) == 1:
        return float(top1) if top1 is not None else None
    top2 = top_categories[1].get("score")
    if top1 is None or top2 is None:
        return None
    return round(float(top1) - float(top2), 6)


def confidence_bucket(top_categories: Sequence[Dict[str, Any]]) -> str:
    if not top_categories:
        return "discard"
    gap = score_gap(top_categories)
    count = len(top_categories)
    if gap is None:
        return "medium"
    if count >= 5 and gap >= 0.15:
        return "high"
    if count >= 3 and gap >= 0.08:
        return "medium"
    return "low"


def enrich_categories(
    categories: Sequence[Dict[str, Any]],
    taxonomy_by_id: Dict[str, TaxonomyRow],
    top_k: int,
) -> List[Dict[str, Any]]:
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
                "llm_rank": len(enriched) + 1,
            }
        )
    return enriched[:top_k]


def build_training_record(
    record: Dict[str, Any],
    source_file: str,
    label_file: str,
    model_info: Dict[str, Any],
    enriched: List[Dict[str, Any]],
    usage: Dict[str, Any],
    request_ms: float,
    total_ms: float,
    max_body_chars: int,
) -> Dict[str, Any]:
    top_category_ids = [item["unique_id"] for item in enriched]
    primary = enriched[0] if enriched else {}
    return {
        "source_file": source_file,
        "label_file": label_file,
        "input_url": record.get("input_url"),
        "url": record.get("url"),
        "url_hash": record.get("url_hash"),
        "domain": record.get("domain"),
        "title": record.get("title"),
        "meta_description": record.get("meta_description"),
        "headings": record.get("headings") or [],
        "body_text": normalize_text(record.get("body_text"))[:max_body_chars],
        "page_text": build_page_text(record, max_body_chars=max_body_chars),
        "teacher_model": model_info.get("cloud_model"),
        "teacher_top_k": len(enriched),
        "teacher_top_categories": enriched,
        "teacher_top_category_ids": top_category_ids,
        "teacher_primary_category_id": normalize_text(primary.get("unique_id")),
        "teacher_primary_category_path": normalize_text(primary.get("path")),
        "teacher_primary_score": primary.get("score"),
        "teacher_score_gap_top1_top2": score_gap(enriched),
        "confidence_bucket": confidence_bucket(enriched),
        "teacher_usage": usage,
        "teacher_model_details": model_info,
        "timing_ms": {
            "llm_request": round(request_ms, 3),
            "total": round(total_ms, 3),
        },
    }


def build_error_record(
    record: Dict[str, Any],
    source_file: str,
    label_file: str,
    model_info: Dict[str, Any],
    error_code: str,
    error_type: str,
    message: str,
    retryable: bool,
    status_code: Optional[int] = None,
    total_ms: Optional[float] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "source_file": source_file,
        "label_file": label_file,
        "input_url": record.get("input_url"),
        "url": record.get("url"),
        "url_hash": record.get("url_hash"),
        "domain": record.get("domain"),
        "title": record.get("title"),
        "status": "error",
        "source_status": normalize_text(record.get("status")),
        "teacher_model": model_info.get("cloud_model"),
        "error_type": error_type,
        "error_code": error_code,
        "message": message,
        "retryable": retryable,
        "teacher_model_details": model_info,
    }
    if status_code is not None:
        out["status_code"] = status_code
    if total_ms is not None:
        out["timing_ms"] = {"total": round(total_ms, 3)}
    return out


def process_record(
    record: Dict[str, Any],
    source_file: str,
    label_file: str,
    api_key: Optional[str],
    model: str,
    system_blocks: List[Dict[str, Any]],
    taxonomy_by_id: Dict[str, TaxonomyRow],
    taxonomy_tsv: str,
    top_k: int,
    max_body_chars: int,
    max_output_tokens: int,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> Tuple[str, Dict[str, Any]]:
    model_info = model_details(
        model=model,
        taxonomy_tsv=taxonomy_tsv,
        top_k=top_k,
        max_body_chars=max_body_chars,
        max_output_tokens=max_output_tokens,
    )
    url_hash = normalize_text(record.get("url_hash"))
    t0 = time.perf_counter()

    if normalize_text(record.get("status")).lower() != "ok":
        return url_hash, build_error_record(
            record, source_file, label_file, model_info,
            error_code="source_status_error",
            error_type="SourceRecordError",
            message="Source content record status is not ok; Claude categorization skipped.",
            retryable=False,
        )

    user_prompt = build_user_prompt(record, max_body_chars=max_body_chars)
    if not build_page_text(record, max_body_chars=max_body_chars).strip():
        return url_hash, build_error_record(
            record, source_file, label_file, model_info,
            error_code="no_usable_text",
            error_type="ValueError",
            message="Input record has no usable text for Claude categorization.",
            retryable=False,
        )

    client = _get_client(api_key=api_key, timeout=timeout, max_retries=max_retries)
    result = classify_record(
        client=client,
        model=model,
        system_blocks=system_blocks,
        user_prompt=user_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    total_ms = (time.perf_counter() - t0) * 1000.0

    if result["ok"]:
        enriched = enrich_categories(result["categories"], taxonomy_by_id, top_k)
        return url_hash, build_training_record(
            record=record,
            source_file=source_file,
            label_file=label_file,
            model_info=model_info,
            enriched=enriched,
            usage=result["usage"],
            request_ms=result["request_ms"],
            total_ms=total_ms,
            max_body_chars=max_body_chars,
        )

    return url_hash, build_error_record(
        record, source_file, label_file, model_info,
        error_code="claude_request_error",
        error_type=result.get("error_type", "RuntimeError"),
        message=result.get("message", "Claude categorization failed."),
        retryable=bool(result.get("retryable", False)),
        status_code=result.get("status_code"),
        total_ms=total_ms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Categorize fetched-content JSONL files with the Anthropic Claude API, "
        "writing records in the stage-08 training-dataset schema."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing fetched-content JSONL files.")
    parser.add_argument("--input-files", nargs="*", default=None, help="Explicit fetched-content JSONL files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for Claude category outputs.")
    parser.add_argument("--taxonomy-tsv", default=DEFAULT_TAXONOMY_PATH, help="Taxonomy TSV path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Anthropic model name.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of final categories to request.")
    parser.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY_CHARS, help="Maximum page body characters to include.")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Maximum output tokens per request.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent requests to the Anthropic API.")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="SDK-level retries for transient errors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Export it before running this script.")

    taxonomy_rows = load_taxonomy_rows(Path(args.taxonomy_tsv))
    taxonomy_by_id = taxonomy_lookup_by_id(taxonomy_rows)
    taxonomy_listing = build_taxonomy_listing(taxonomy_rows)
    system_blocks = build_system_prompt_blocks(top_k=args.top_k, taxonomy_listing=taxonomy_listing)

    input_paths = [Path(path) for path in args.input_files] if args.input_files else sorted(Path(args.input_dir).glob("*.jsonl"))
    if not input_paths:
        raise SystemExit("No input JSONL files found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        output_path = output_path_for_input(input_path, output_dir, args.model)
        cached_hashes = load_cached_url_hashes(output_path)

        pending_records: List[Dict[str, Any]] = []
        total_records = 0
        for _, record in iter_jsonl(input_path):
            total_records += 1
            url_hash = normalize_text(record.get("url_hash"))
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
                    input_path.name,
                    output_path.name,
                    api_key,
                    args.model,
                    system_blocks,
                    taxonomy_by_id,
                    args.taxonomy_tsv,
                    args.top_k,
                    args.max_body_chars,
                    args.max_output_tokens,
                    args.temperature,
                    args.timeout,
                    args.max_retries,
                )
                future_map[future] = record

            for future in as_completed(future_map):
                _, out_record = future.result()
                dst.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                dst.flush()
                completed += 1
                if out_record.get("status") == "error":
                    error_count += 1
                else:
                    ok_count += 1
                print(
                    f"[{completed}/{total_records}] "
                    f"{'error' if out_record.get('status') == 'error' else 'ok'} "
                    f"url_hash={normalize_text(out_record.get('url_hash'))}"
                )

        print(
            f"Completed {input_path.name}: wrote {len(pending_records)} new records "
            f"({ok_count} ok, {error_count} error)"
        )


if __name__ == "__main__":
    main()
