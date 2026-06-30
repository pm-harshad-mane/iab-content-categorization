"""
Fetch URLs from a text file and cache cleaned page content as JSONL.

The output file is append-only and resumable: if a URL already exists in the
JSONL file, it is skipped on subsequent runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
from urllib.parse import urlparse

from fetch_with_beautiful_soup_cffi import (
    DEFAULT_REQUEST_TIMEOUT,
    FetchFailure,
    fetch_page_content,
)

DEFAULT_URLS_FILE = "adserver_1000_urls.txt"
DEFAULT_OUTPUT_FILE = "save_url_content_1000.jsonl"
DEFAULT_WORKERS = 64


def read_urls(path: Path) -> List[str]:
    seen: Set[str] = set()
    urls: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw_url = line.strip()
            if not raw_url or raw_url.startswith("#") or raw_url in seen:
                continue
            seen.add(raw_url)
            urls.append(raw_url)
    return urls


def normalize_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.scheme:
        return raw_url
    return f"https://{raw_url}"


def build_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


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
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            url_hash = str(record.get("url_hash") or "").strip()
            if url_hash:
                cached_hashes.add(url_hash)
                continue

            cached_url = str(record.get("url") or record.get("input_url") or "").strip()
            if cached_url:
                cached_hashes.add(build_url_hash(normalize_url(cached_url)))

    return cached_hashes


def build_error_record(input_url: str, requested_url: str, exc: Exception) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "input_url": input_url,
        "url": requested_url,
        "url_hash": build_url_hash(requested_url),
        "status": "error",
        "error_type": type(exc).__name__,
        "message": str(exc),
    }

    if isinstance(exc, FetchFailure):
        record.update(
            {
                "error_code": exc.error_code,
                "retryable": exc.retryable,
                "status_code": exc.status_code,
                "final_url": exc.final_url,
                "attempt_count": exc.attempt_count,
            }
        )

    return record


def fetch_url_record(input_url: str, timeout: int) -> Tuple[str, Dict[str, Any]]:
    requested_url = normalize_url(input_url)
    try:
        page = fetch_page_content(requested_url, timeout=timeout)
        record: Dict[str, Any] = {
            "input_url": input_url,
            "url": requested_url,
            "url_hash": build_url_hash(requested_url),
            "status": "ok",
        }
        record.update(asdict(page))
    except Exception as exc:
        record = build_error_record(input_url, requested_url, exc)
    return requested_url, record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch URL content once and cache it in a JSONL file."
    )
    parser.add_argument(
        "--urls-file",
        default=DEFAULT_URLS_FILE,
        help="Text file containing one URL per line.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSONL cache file.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP timeout in seconds per URL.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of concurrent URL fetch workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls_path = Path(args.urls_file)
    output_path = Path(args.output)

    input_urls = read_urls(urls_path)
    cached_url_hashes = load_cached_url_hashes(output_path)
    pending_urls = [
        url
        for url in input_urls
        if build_url_hash(normalize_url(url)) not in cached_url_hashes
    ]

    print(f"Loaded {len(input_urls)} unique URLs from {urls_path}.")
    print(f"Found {len(cached_url_hashes)} cached URL hashes in {output_path}.")
    print(f"Fetching {len(pending_urls)} URLs with {args.workers} workers.")

    with output_path.open("a", encoding="utf-8") as handle:
        max_workers = max(1, int(args.workers))
        total = len(pending_urls)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_input_url: Dict[Future[Tuple[str, Dict[str, Any]]], str] = {
                executor.submit(fetch_url_record, input_url, args.timeout): input_url
                for input_url in pending_urls
            }

            for index, future in enumerate(as_completed(future_to_input_url), start=1):
                input_url = future_to_input_url[future]
                try:
                    requested_url, record = future.result()
                except Exception as exc:
                    requested_url = normalize_url(input_url)
                    record = build_error_record(input_url, requested_url, exc)

                print(f"[{index}/{total}] {requested_url}")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()


if __name__ == "__main__":
    main()
