from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

DEFAULT_OPENAI_API_BASE = "https://api.openai.com"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_TAXONOMY_PATH = "taxonomy/Content_Taxonomy_3.1_6.tsv"
DEFAULT_INPUT_DIR = "02_fetched_url_content_files"
DEFAULT_OUTPUT_DIR = "06_fetched_url_content_categories_by_GPT"
DEFAULT_TRACKER_FILE = "batch_tracker.jsonl"
DEFAULT_REQUESTS_SUBDIR = "requests"
DEFAULT_DOWNLOADS_SUBDIR = "downloads"
DEFAULT_TOP_K = 5
DEFAULT_MAX_BODY_CHARS = 6000
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_OUTPUT_TOKENS = 220


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
    common_confusers: str
    negative_keywords: str


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", normalize_text(model)).strip("_") or "model"


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def enrich_tracker_timing_fields(row: Dict[str, Any]) -> None:
    created_at = maybe_float(row.get("created_at"))
    in_progress_at = maybe_float(row.get("in_progress_at"))
    completed_at = maybe_float(row.get("completed_at"))
    failed_at = maybe_float(row.get("failed_at"))
    cancelled_at = maybe_float(row.get("cancelled_at"))
    finalized_at = normalize_text(row.get("finalized_at"))

    if created_at is not None and in_progress_at is not None:
        row["queue_latency_seconds"] = round(in_progress_at - created_at, 3)
    if created_at is not None and completed_at is not None:
        row["batch_elapsed_seconds"] = round(completed_at - created_at, 3)
    elif created_at is not None and failed_at is not None:
        row["batch_elapsed_seconds"] = round(failed_at - created_at, 3)
    elif created_at is not None and cancelled_at is not None:
        row["batch_elapsed_seconds"] = round(cancelled_at - created_at, 3)

    if completed_at is not None and finalized_at:
        try:
            finalized_epoch = time.mktime(time.strptime(finalized_at, "%Y-%m-%dT%H:%M:%SZ"))
            row["post_completion_finalize_seconds"] = round(finalized_epoch - completed_at, 3)
        except ValueError:
            pass


def output_path_for_input(input_path: Path, output_dir: Path, model: str) -> Path:
    return output_dir / f"{input_path.stem}__{sanitize_model_name(model)}.jsonl"


def request_file_path_for_input(input_path: Path, output_dir: Path, model: str) -> Path:
    requests_dir = output_dir / DEFAULT_REQUESTS_SUBDIR
    return requests_dir / f"{input_path.stem}__{sanitize_model_name(model)}__batch_requests.jsonl"


def tracker_path_for_output_dir(output_dir: Path) -> Path:
    return output_dir / DEFAULT_TRACKER_FILE


def download_file_path(output_dir: Path, batch_id: str, label: str) -> Path:
    downloads_dir = output_dir / DEFAULT_DOWNLOADS_SUBDIR
    return downloads_dir / f"{batch_id}__{label}.jsonl"


def tracker_job_key(input_path: Path, model: str) -> str:
    return f"{str(input_path.resolve())}::{sanitize_model_name(model)}"


def build_prompt_cache_key(model: str, taxonomy_source: str) -> str:
    digest = hashlib.sha1(f"{model}|{taxonomy_source}|full-details|v1".encode("utf-8")).hexdigest()[:12]
    return f"tx-{sanitize_model_name(model)}-{digest}"


def custom_id_for_record(input_path: Path, line_number: int, record: Dict[str, Any]) -> str:
    url_hash = normalize_text(record.get("url_hash")) or f"line-{line_number}"
    return f"{input_path.stem}:{line_number}:{url_hash}"


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
            "Common Confusers",
            "Negative Keywords",
        }
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing expected taxonomy columns: {missing}")

        rows: List[TaxonomyRow] = []
        for row in reader:
            rows.append(
                TaxonomyRow(
                    unique_id=normalize_text(row.get("Unique ID")),
                    parent_id=normalize_text(row.get("Parent")),
                    tier1=normalize_text(row.get("Tier 1")),
                    tier2=normalize_text(row.get("Tier 2")),
                    tier3=normalize_text(row.get("Tier 3")),
                    tier4=normalize_text(row.get("Tier 4")),
                    path=normalize_text(row.get("Path")),
                    description=normalize_text(row.get("Description")),
                    keywords=normalize_text(row.get("Keywords")),
                    common_confusers=normalize_text(row.get("Common Confusers")),
                    negative_keywords=normalize_text(row.get("Negative Keywords")),
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
        parts: List[str] = [f"id={row.unique_id}", f"path={row.path}"]
        if row.description:
            parts.append(f"description={row.description}")
        if row.keywords:
            parts.append(f"keywords={row.keywords}")
        lines.append(" || ".join(parts))
    return "\n".join(lines)


def build_page_query_text(record: Dict[str, Any], max_body_chars: int) -> str:
    parts: List[str] = []
    title = normalize_text(record.get("title"))
    meta_description = normalize_text(record.get("meta_description"))
    headings = [normalize_text(item) for item in (record.get("headings") or []) if normalize_text(item)]
    body_text = normalize_text(record.get("body_text"))
    url = normalize_text(record.get("url") or record.get("input_url"))
    domain = normalize_text(record.get("domain"))

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
        f"- Return at most {top_k} categories.\n"
        "- Use only taxonomy unique_id values that appear in the taxonomy list below.\n"
        "- Prefer the most specific applicable category path.\n"
        "- Do not invent categories.\n"
        "- Avoid duplicates.\n"
        "- Scores must be floats in [0,1], sorted from best to worst.\n"
        "- If the page does not fit the taxonomy, return an empty array.\n"
        "- Output only valid JSON matching the requested schema.\n\n"
        "Taxonomy list:\n"
        f"{taxonomy_listing}"
    )


def build_user_prompt(record: Dict[str, Any], max_body_chars: int) -> str:
    page_text = build_page_query_text(record, max_body_chars=max_body_chars)
    return (
        "Classify this web page into the taxonomy.\n"
        "Return top taxonomy matches only.\n\n"
        f"{page_text}"
    )


def batch_response_schema(top_k: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "top_categories": {
                "type": "array",
                "maxItems": int(top_k),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "unique_id": {"type": "string"},
                        "score": {"type": "number"},
                    },
                    "required": ["unique_id", "score"],
                },
            }
        },
        "required": ["top_categories"],
    }


def build_batch_request_body(
    model: str,
    system_prompt: str,
    user_prompt: str,
    top_k: int,
    max_output_tokens: int,
    prompt_cache_key: str,
) -> Dict[str, Any]:
    return {
        "model": model,
        "prompt_cache_key": prompt_cache_key,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt,
                    }
                ],
            },
        ],
        "max_output_tokens": int(max_output_tokens),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "taxonomy_classification",
                "schema": batch_response_schema(top_k),
                "strict": True,
            }
        },
    }


def openai_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
    }


def openai_json_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def openai_upload_file(
    api_base: str,
    api_key: str,
    path: Path,
    timeout: int,
) -> Dict[str, Any]:
    with path.open("rb") as handle:
        response = requests.post(
            api_base.rstrip("/") + "/v1/files",
            headers=openai_headers(api_key),
            data={"purpose": "batch"},
            files={"file": (path.name, handle, "application/jsonl")},
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()


def openai_create_batch(
    api_base: str,
    api_key: str,
    input_file_id: str,
    completion_window: str,
    metadata: Dict[str, str],
    timeout: int,
) -> Dict[str, Any]:
    payload = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/responses",
        "completion_window": completion_window,
        "metadata": metadata,
    }
    response = requests.post(
        api_base.rstrip("/") + "/v1/batches",
        headers=openai_json_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def openai_get_batch(api_base: str, api_key: str, batch_id: str, timeout: int) -> Dict[str, Any]:
    response = requests.get(
        api_base.rstrip("/") + f"/v1/batches/{batch_id}",
        headers=openai_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def openai_download_file_content(
    api_base: str,
    api_key: str,
    file_id: str,
    timeout: int,
) -> str:
    response = requests.get(
        api_base.rstrip("/") + f"/v1/files/{file_id}/content",
        headers=openai_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def load_tracker(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                rows.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid tracker JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def save_tracker(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def upsert_tracker_row(rows: List[Dict[str, Any]], new_row: Dict[str, Any]) -> None:
    job_key = normalize_text(new_row.get("job_key"))
    for idx, row in enumerate(rows):
        if normalize_text(row.get("job_key")) == job_key:
            rows[idx] = new_row
            return
    rows.append(new_row)


def build_request_file(
    input_path: Path,
    request_path: Path,
    model: str,
    taxonomy_listing: str,
    taxonomy_source: str,
    top_k: int,
    max_body_chars: int,
    max_output_tokens: int,
) -> Dict[str, Any]:
    request_path.parent.mkdir(parents=True, exist_ok=True)
    system_prompt = build_system_prompt(top_k=top_k, taxonomy_listing=taxonomy_listing)
    total_records = 0
    submitted_records = 0
    source_error_records = 0
    prompt_cache_key = build_prompt_cache_key(model=model, taxonomy_source=taxonomy_source)

    with request_path.open("w", encoding="utf-8") as dst:
        for line_number, record in iter_jsonl(input_path):
            total_records += 1
            if normalize_text(record.get("status")).lower() != "ok":
                source_error_records += 1
                continue

            custom_id = custom_id_for_record(input_path, line_number, record)
            body = build_batch_request_body(
                model=model,
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(record, max_body_chars=max_body_chars),
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                prompt_cache_key=prompt_cache_key,
            )
            request_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            dst.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            submitted_records += 1

    return {
        "request_file": str(request_path),
        "total_input_records": total_records,
        "submitted_records": submitted_records,
        "source_error_records": source_error_records,
        "taxonomy_rows_in_prompt": taxonomy_listing.count("\n") + (1 if taxonomy_listing else 0),
        "taxonomy_prompt_fields": ["Unique ID", "Path", "Description", "Keywords"],
        "prompt_cache_key": prompt_cache_key,
    }


def parse_usage(body: Dict[str, Any]) -> Dict[str, Any]:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return {}
    result: Dict[str, Any] = {}
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        if key in usage:
            result[key] = usage.get(key)
    return result


def extract_response_text(body: Dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = body.get("output")
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for piece in content:
                if not isinstance(piece, dict):
                    continue
                if piece.get("type") in {"output_text", "text"} and isinstance(piece.get("text"), str):
                    parts.append(piece["text"])
        if parts:
            return "".join(parts).strip()

    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                if parts:
                    return "".join(parts).strip()

    raise RuntimeError(f"Could not extract output text from response body: {body}")


def parse_model_categories(body: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    text = extract_response_text(body)
    payload = json.loads(text)
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
        score = item.get("score")
        try:
            score_value = float(score)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid score in model response: {item}") from exc
        result.append({"unique_id": unique_id, "score": round(score_value, 6)})
    return result, parse_usage(body)


def load_batch_result_maps(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid downloaded batch JSON on line {line_number} of {path}: {exc}") from exc
            custom_id = normalize_text(row.get("custom_id"))
            if custom_id:
                rows[custom_id] = row
    return rows


def build_source_error_output_record(record: Dict[str, Any], model: str, job_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input_url": normalize_text(record.get("input_url") or record.get("url")),
        "url": normalize_text(record.get("url") or record.get("input_url")),
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "error",
        "source_status": normalize_text(record.get("status")),
        "cloud_model": model,
        "domain": normalize_text(record.get("domain")),
        "title": normalize_text(record.get("title")),
        "error_type": "SourceRecordError",
        "error_code": "source_status_error",
        "message": "Source content record status is not ok; GPT categorization skipped.",
        "retryable": False,
        "model_details": job_meta,
    }


def build_batch_error_output_record(
    record: Dict[str, Any],
    model: str,
    job_meta: Dict[str, Any],
    error_type: str,
    error_code: str,
    message: str,
    retryable: bool = False,
    status_code: Optional[int] = None,
) -> Dict[str, Any]:
    output = {
        "input_url": normalize_text(record.get("input_url") or record.get("url")),
        "url": normalize_text(record.get("url") or record.get("input_url")),
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "error",
        "source_status": normalize_text(record.get("status")),
        "cloud_model": model,
        "domain": normalize_text(record.get("domain")),
        "title": normalize_text(record.get("title")),
        "error_type": error_type,
        "error_code": error_code,
        "message": message,
        "retryable": retryable,
        "model_details": job_meta,
    }
    if status_code is not None:
        output["status_code"] = status_code
    return output


def build_success_output_record(
    record: Dict[str, Any],
    model: str,
    job_meta: Dict[str, Any],
    categories: Sequence[Dict[str, Any]],
    taxonomy_by_id: Dict[str, TaxonomyRow],
    usage: Dict[str, Any],
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
                "gpt_rank": rank,
            }
        )

    return {
        "input_url": normalize_text(record.get("input_url") or record.get("url")),
        "url": normalize_text(record.get("url") or record.get("input_url")),
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "ok",
        "source_status": "ok",
        "cloud_model": model,
        "domain": normalize_text(record.get("domain")),
        "title": normalize_text(record.get("title")),
        "gpt_top_k": len(enriched),
        "gpt_top_categories": enriched,
        "model_details": job_meta,
        "usage": usage,
    }


def tracker_model_details(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cloud_model": normalize_text(row.get("model")),
        "taxonomy_source": normalize_text(row.get("taxonomy_source")),
        "taxonomy_prompt_fields": row.get("taxonomy_prompt_fields") or [],
        "top_k": int(row.get("top_k") or 0),
        "max_body_chars": int(row.get("max_body_chars") or 0),
        "batch": {
            "batch_id": normalize_text(row.get("batch_id")),
            "status": normalize_text(row.get("batch_status") or row.get("status")),
            "input_file_id": normalize_text(row.get("openai_input_file_id")),
            "output_file_id": normalize_text(row.get("openai_output_file_id")),
            "error_file_id": normalize_text(row.get("openai_error_file_id")),
            "completion_window": normalize_text(row.get("completion_window")),
        },
    }


def command_submit(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for batch submission.")

    taxonomy_rows = load_taxonomy_rows(Path(args.taxonomy_tsv))
    taxonomy_listing = build_taxonomy_listing(taxonomy_rows)

    input_paths = [Path(path) for path in args.input_files] if args.input_files else sorted(Path(args.input_dir).glob("*.jsonl"))
    if not input_paths:
        raise SystemExit("No input JSONL files found.")

    output_dir = Path(args.output_dir)
    tracker_path = tracker_path_for_output_dir(output_dir)
    tracker_rows = load_tracker(tracker_path)
    tracker_by_key = {normalize_text(row.get("job_key")): row for row in tracker_rows}

    for input_path in input_paths:
        job_key = tracker_job_key(input_path, args.model)
        existing = tracker_by_key.get(job_key)
        if existing and not args.force_resubmit:
            status = normalize_text(existing.get("status"))
            if status and status not in {"failed", "cancelled", "expired"}:
                print(f"Skipping {input_path}; existing batch job {existing.get('batch_id')} is in status {status}.")
                continue

        request_path = request_file_path_for_input(input_path, output_dir, args.model)
        request_meta = build_request_file(
            input_path=input_path,
            request_path=request_path,
            model=args.model,
            taxonomy_listing=taxonomy_listing,
            taxonomy_source=str(Path(args.taxonomy_tsv)),
            top_k=args.top_k,
            max_body_chars=args.max_body_chars,
            max_output_tokens=args.max_output_tokens,
        )

        uploaded = openai_upload_file(
            api_base=args.api_base,
            api_key=api_key,
            path=request_path,
            timeout=args.timeout,
        )
        batch = openai_create_batch(
            api_base=args.api_base,
            api_key=api_key,
            input_file_id=normalize_text(uploaded.get("id")),
            completion_window=args.completion_window,
            metadata={
                "job_key": job_key,
                "input_file": str(input_path),
                "model": args.model,
            },
            timeout=args.timeout,
        )

        tracker_row = {
            "job_key": job_key,
            "input_file": str(input_path),
            "output_file": str(output_path_for_input(input_path, output_dir, args.model)),
            "model": args.model,
            "taxonomy_source": str(Path(args.taxonomy_tsv)),
            "taxonomy_prompt_fields": request_meta["taxonomy_prompt_fields"],
            "prompt_cache_key": request_meta["prompt_cache_key"],
            "top_k": int(args.top_k),
            "max_body_chars": int(args.max_body_chars),
            "max_output_tokens": int(args.max_output_tokens),
            "completion_window": args.completion_window,
            "request_file": request_meta["request_file"],
            "total_input_records": request_meta["total_input_records"],
            "submitted_records": request_meta["submitted_records"],
            "source_error_records": request_meta["source_error_records"],
            "taxonomy_rows_in_prompt": request_meta["taxonomy_rows_in_prompt"],
            "openai_input_file_id": normalize_text(uploaded.get("id")),
            "batch_id": normalize_text(batch.get("id")),
            "status": normalize_text(batch.get("status")),
            "batch_status": normalize_text(batch.get("status")),
            "openai_output_file_id": normalize_text(batch.get("output_file_id")),
            "openai_error_file_id": normalize_text(batch.get("error_file_id")),
            "submitted_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        upsert_tracker_row(tracker_rows, tracker_row)
        tracker_by_key[job_key] = tracker_row
        print(f"Submitted batch for {input_path}")
        print(f"Request file: {request_path}")
        print(f"OpenAI input file id: {tracker_row['openai_input_file_id']}")
        print(f"Batch id: {tracker_row['batch_id']}")
        print(f"Status: {tracker_row['status']}")

    save_tracker(tracker_path, tracker_rows)
    print(f"Tracker updated: {tracker_path}")


def command_sync(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for batch sync.")

    tracker_path = tracker_path_for_output_dir(Path(args.output_dir))
    tracker_rows = load_tracker(tracker_path)
    if not tracker_rows:
        raise SystemExit(f"No tracker rows found in {tracker_path}")

    target_batch_ids = {normalize_text(batch_id) for batch_id in (args.batch_ids or []) if normalize_text(batch_id)}
    changed = 0
    for row in tracker_rows:
        batch_id = normalize_text(row.get("batch_id"))
        if not batch_id:
            continue
        if target_batch_ids and batch_id not in target_batch_ids:
            continue
        batch = openai_get_batch(args.api_base, api_key, batch_id, args.timeout)
        row["status"] = normalize_text(batch.get("status"))
        row["batch_status"] = normalize_text(batch.get("status"))
        row["updated_at"] = utc_now_iso()
        row["openai_output_file_id"] = normalize_text(batch.get("output_file_id"))
        row["openai_error_file_id"] = normalize_text(batch.get("error_file_id"))
        row["openai_completion_file_id"] = normalize_text(batch.get("completion_file_id"))
        request_counts = batch.get("request_counts")
        if isinstance(request_counts, dict):
            row["request_counts"] = request_counts
        for key in ["created_at", "in_progress_at", "expires_at", "completed_at", "failed_at", "cancelled_at"]:
            if key in batch:
                row[key] = batch.get(key)
        enrich_tracker_timing_fields(row)
        changed += 1
        print(f"{batch_id}: {row['status']}")

    if changed == 0:
        print("No tracker rows matched for sync.")
        return
    save_tracker(tracker_path, tracker_rows)
    print(f"Tracker updated: {tracker_path}")


def command_finalize(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for batch finalize.")

    output_dir = Path(args.output_dir)
    tracker_path = tracker_path_for_output_dir(output_dir)
    tracker_rows = load_tracker(tracker_path)
    if not tracker_rows:
        raise SystemExit(f"No tracker rows found in {tracker_path}")

    taxonomy_rows = load_taxonomy_rows(Path(args.taxonomy_tsv))
    taxonomy_by_id = taxonomy_lookup_by_id(taxonomy_rows)

    target_batch_ids = {normalize_text(batch_id) for batch_id in (args.batch_ids or []) if normalize_text(batch_id)}

    changed = 0
    for row in tracker_rows:
        batch_id = normalize_text(row.get("batch_id"))
        if not batch_id:
            continue
        if target_batch_ids and batch_id not in target_batch_ids:
            continue

        status = normalize_text(row.get("batch_status") or row.get("status"))
        if status != "completed":
            print(f"Skipping {batch_id}; status is {status}.")
            continue

        error_file_id = normalize_text(row.get("openai_error_file_id"))
        output_file_id = normalize_text(row.get("openai_output_file_id"))
        if not output_file_id and not error_file_id:
            print(f"Skipping {batch_id}; neither output_file_id nor error_file_id is present.")
            continue

        output_download_path = download_file_path(output_dir, batch_id, "output")
        error_download_path = download_file_path(output_dir, batch_id, "error") if error_file_id else None

        output_download_path.parent.mkdir(parents=True, exist_ok=True)
        if output_file_id:
            output_download_path.write_text(
                openai_download_file_content(args.api_base, api_key, output_file_id, args.timeout),
                encoding="utf-8",
            )
        if error_file_id and error_download_path is not None:
            error_download_path.write_text(
                openai_download_file_content(args.api_base, api_key, error_file_id, args.timeout),
                encoding="utf-8",
            )

        success_map = load_batch_result_maps(output_download_path if output_file_id else None)
        error_map = load_batch_result_maps(error_download_path)

        input_path = Path(normalize_text(row.get("input_file")))
        final_output_path = Path(normalize_text(row.get("output_file")))
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        job_meta = tracker_model_details(row)

        with final_output_path.open("w", encoding="utf-8") as dst:
            for line_number, record in iter_jsonl(input_path):
                if normalize_text(record.get("status")).lower() != "ok":
                    out = build_source_error_output_record(record, row["model"], job_meta)
                else:
                    custom_id = custom_id_for_record(input_path, line_number, record)
                    success_row = success_map.get(custom_id)
                    error_row = error_map.get(custom_id)
                    if success_row is not None:
                        response = success_row.get("response")
                        body = response.get("body") if isinstance(response, dict) else None
                        status_code = response.get("status_code") if isinstance(response, dict) else None
                        if not isinstance(body, dict) or status_code != 200:
                            out = build_batch_error_output_record(
                                record,
                                row["model"],
                                job_meta,
                                error_type="BatchResponseError",
                                error_code="batch_response_not_ok",
                                message=f"Batch response status_code={status_code}",
                                retryable=False,
                                status_code=status_code if isinstance(status_code, int) else None,
                            )
                        else:
                            try:
                                categories, usage = parse_model_categories(body)
                                out = build_success_output_record(
                                    record=record,
                                    model=row["model"],
                                    job_meta=job_meta,
                                    categories=categories,
                                    taxonomy_by_id=taxonomy_by_id,
                                    usage=usage,
                                )
                            except Exception as exc:
                                out = build_batch_error_output_record(
                                    record,
                                    row["model"],
                                    job_meta,
                                    error_type=type(exc).__name__,
                                    error_code="response_parse_error",
                                    message=str(exc),
                                    retryable=False,
                                )
                    elif error_row is not None:
                        error_payload = error_row.get("error")
                        message = json.dumps(error_payload, ensure_ascii=False) if error_payload is not None else "OpenAI batch request failed."
                        out = build_batch_error_output_record(
                            record,
                            row["model"],
                            job_meta,
                            error_type="BatchRequestError",
                            error_code="batch_request_error",
                            message=message,
                            retryable=False,
                        )
                    else:
                        out = build_batch_error_output_record(
                            record,
                            row["model"],
                            job_meta,
                            error_type="MissingBatchResultError",
                            error_code="missing_batch_result",
                            message="No batch output or batch error row found for this record.",
                            retryable=False,
                        )
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")

        row["downloaded_output_path"] = str(output_download_path) if output_file_id else ""
        row["downloaded_error_path"] = str(error_download_path) if error_download_path else ""
        row["final_output_path"] = str(final_output_path)
        row["finalized_at"] = utc_now_iso()
        row["status"] = "finalized"
        row["updated_at"] = utc_now_iso()
        enrich_tracker_timing_fields(row)
        changed += 1
        print(f"Finalized {batch_id} -> {final_output_path}")

    if changed == 0:
        print("No completed batches were finalized.")
        return
    save_tracker(tracker_path, tracker_rows)
    print(f"Tracker updated: {tracker_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit, track, and finalize OpenAI Batch GPT taxonomy classification jobs for fetched-content JSONL files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="Create and submit a new OpenAI batch job for one or more content files.")
    submit.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing fetched-content JSONL files.")
    submit.add_argument("--input-files", nargs="*", default=None, help="Optional explicit fetched-content JSONL files.")
    submit.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for tracker, request files, downloads, and final outputs.")
    submit.add_argument("--taxonomy-tsv", default=DEFAULT_TAXONOMY_PATH, help="Taxonomy TSV path.")
    submit.add_argument("--api-base", default=DEFAULT_OPENAI_API_BASE, help="OpenAI API base URL.")
    submit.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name for batch categorization.")
    submit.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of final categories to request from the model.")
    submit.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY_CHARS, help="Maximum body text characters to include in the prompt.")
    submit.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Maximum output tokens per record.")
    submit.add_argument("--completion-window", default=DEFAULT_COMPLETION_WINDOW, help="OpenAI batch completion window.")
    submit.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    submit.add_argument("--force-resubmit", action="store_true", help="Allow resubmitting even if a tracker row already exists.")

    sync = subparsers.add_parser("sync", help="Refresh status for existing OpenAI batch jobs.")
    sync.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory containing batch_tracker.jsonl.")
    sync.add_argument("--api-base", default=DEFAULT_OPENAI_API_BASE, help="OpenAI API base URL.")
    sync.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    sync.add_argument("--batch-ids", nargs="*", default=None, help="Optional specific batch ids to sync.")

    finalize = subparsers.add_parser("finalize", help="Download completed batch results and write final enriched JSONL outputs.")
    finalize.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory containing batch_tracker.jsonl.")
    finalize.add_argument("--taxonomy-tsv", default=DEFAULT_TAXONOMY_PATH, help="Taxonomy TSV path.")
    finalize.add_argument("--api-base", default=DEFAULT_OPENAI_API_BASE, help="OpenAI API base URL.")
    finalize.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    finalize.add_argument("--batch-ids", nargs="*", default=None, help="Optional specific batch ids to finalize.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "submit":
        command_submit(args)
        return
    if args.command == "sync":
        command_sync(args)
        return
    if args.command == "finalize":
        command_finalize(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
