#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

DEFAULT_INPUT_JSONL = "categorize_on_hosted_models_1000.jsonl"
DEFAULT_SOURCE_INPUT_JSONL = "save_url_content_1000.jsonl"
DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_ENDPOINT = "/v1/rerank"
DEFAULT_TIMEOUT = 180
DEFAULT_SAMPLE_INTERVAL = 1.0
DEFAULT_MAX_BODY_CHARS = 4000
DEFAULT_RERANK_QUERY_MAX_CHARS = 1800
DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]

_thread_local = threading.local()


@dataclass
class PageContent:
    url: str
    domain: str
    title: str
    meta_description: str
    headings: List[str]
    body_text: str


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


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


def build_page_query_from_result(page: Dict[str, Any]) -> str:
    parts: List[str] = []
    title = normalize_text(page.get("title"))
    meta_description = normalize_text(page.get("meta_description"))
    headings = [normalize_text(x) for x in page.get("headings", []) if normalize_text(x)]
    body_preview = normalize_text(page.get("body_preview"))
    if title:
        parts.append(f"title: {title}")
    if meta_description:
        parts.append(f"description: {meta_description}")
    if headings:
        parts.append("headings: " + " | ".join(headings[:6]))
    if body_preview:
        parts.append(f"content: {body_preview}")
    return " || ".join(parts)


def make_rerank_documents(candidates: Sequence[Dict[str, Any]]) -> List[str]:
    docs: List[str] = []
    for candidate in candidates:
        path = normalize_text(candidate.get("path"))
        description = normalize_text(candidate.get("description"))
        docs.append(f"path: {path}\ndescription: {description}")
    return docs


def load_source_queries(source_input_jsonl: str, max_body_chars: int, rerank_query_max_chars: int) -> Dict[str, str]:
    query_by_url: Dict[str, str] = {}
    with Path(source_input_jsonl).open() as handle:
        for line in handle:
            record = json.loads(line)
            if normalize_text(record.get("status")).lower() != "ok":
                continue
            page = parse_page_record(record)
            if not page.url:
                continue
            query_text = build_page_query_text(page, max_body_chars=max_body_chars)
            if query_text:
                query_by_url[page.url] = query_text[:rerank_query_max_chars]
    return query_by_url


def load_requests(
    input_jsonl: str,
    source_input_jsonl: str,
    max_body_chars: int,
    rerank_query_max_chars: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    source_query_by_url = load_source_queries(
        source_input_jsonl=source_input_jsonl,
        max_body_chars=max_body_chars,
        rerank_query_max_chars=rerank_query_max_chars,
    )
    with Path(input_jsonl).open() as handle:
        for line in handle:
            record = json.loads(line)
            if not record.get("ok"):
                continue
            page = record.get("page") or {}
            url = normalize_text(page.get("url"))
            query = source_query_by_url.get(url) or build_page_query_from_result(page)
            candidates = record.get("faiss_candidates") or []
            if not query or not candidates:
                continue
            docs = make_rerank_documents(candidates)
            if not docs:
                continue
            rows.append(
                {
                    "url": url,
                    "query_text": query,
                    "documents": docs,
                    "document_count": len(docs),
                    "query_chars": len(query),
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def repeat_requests(rows: List[Dict[str, Any]], repeat_factor: int) -> List[Dict[str, Any]]:
    if repeat_factor <= 1:
        return rows
    expanded: List[Dict[str, Any]] = []
    for rep in range(repeat_factor):
        for item in rows:
            clone = dict(item)
            clone["repeat_index"] = rep
            expanded.append(clone)
    return expanded


def _get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * p)))
    return ordered[idx]


class GpuSampler:
    def __init__(self, sample_interval_s: float, gpu_index: int) -> None:
        self.sample_interval_s = sample_interval_s
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str, Any]] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            ts = time.time()
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10)
                for line in out.splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) != 5:
                        continue
                    idx = int(parts[0])
                    if idx != self.gpu_index:
                        continue
                    self.samples.append(
                        {
                            "ts": ts,
                            "index": idx,
                            "name": parts[1],
                            "utilization_gpu": float(parts[2]),
                            "memory_used_mb": float(parts[3]),
                            "memory_total_mb": float(parts[4]),
                        }
                    )
            except Exception as exc:
                self.samples.append({"ts": ts, "error": f"{type(exc).__name__}: {exc}"})
            self._stop.wait(self.sample_interval_s)

    def summary(self) -> Dict[str, Any]:
        valid = [s for s in self.samples if "utilization_gpu" in s]
        if not valid:
            return {
                "sample_count": len(self.samples),
                "valid_sample_count": 0,
                "error_count": len([s for s in self.samples if "error" in s]),
            }
        utils = [s["utilization_gpu"] for s in valid]
        mems = [s["memory_used_mb"] for s in valid]
        return {
            "sample_count": len(self.samples),
            "valid_sample_count": len(valid),
            "error_count": len([s for s in self.samples if "error" in s]),
            "gpu_name": valid[0]["name"],
            "util_mean_pct": round(statistics.mean(utils), 3),
            "util_peak_pct": round(max(utils), 3),
            "memory_mean_mb": round(statistics.mean(mems), 3),
            "memory_peak_mb": round(max(mems), 3),
        }


def parse_rerank_response(body: Dict[str, Any], doc_count: int) -> List[float]:
    if isinstance(body.get("scores"), list):
        scores = [float(x) for x in body["scores"]]
        if len(scores) != doc_count:
            raise RuntimeError(f"Expected {doc_count} scores, got {len(scores)}")
        return scores
    for key in ("results", "data"):
        items = body.get(key)
        if not isinstance(items, list):
            continue
        scores = [0.0] * doc_count
        used = 0
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            score = item.get("relevance_score", item.get("score"))
            index = item.get("index", idx)
            if score is None:
                continue
            index = int(index)
            if 0 <= index < doc_count:
                scores[index] = float(score)
                used += 1
        if used:
            return scores
    raise RuntimeError(f"Unsupported rerank response shape: {body}")


def post_rerank(
    api_base: str,
    endpoint: str,
    model: str,
    query_text: str,
    documents: Sequence[str],
    timeout: int,
) -> Dict[str, Any]:
    session = _get_session()
    payload = {
        "model": model,
        "query": query_text,
        "documents": list(documents),
        "top_n": len(documents),
        "return_documents": False,
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
        scores = parse_rerank_response(body, doc_count=len(documents))
        return {
            "ok": True,
            "status_code": resp.status_code,
            "service_elapsed_ms": elapsed_ms,
            "document_count": len(scores),
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return {
            "ok": False,
            "status_code": status_code,
            "service_elapsed_ms": elapsed_ms,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_once(
    rows: List[Dict[str, Any]],
    api_base: str,
    endpoint: str,
    model: str,
    concurrency: int,
    timeout: int,
    gpu_index: int,
    sample_interval_s: float,
) -> Dict[str, Any]:
    sampler = GpuSampler(sample_interval_s=sample_interval_s, gpu_index=gpu_index)
    sampler.start()
    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_submit_ts = {}
        for item in rows:
            submit_ts = time.perf_counter()
            future = executor.submit(
                post_rerank,
                api_base,
                endpoint,
                model,
                item["query_text"],
                item["documents"],
                timeout,
            )
            future_to_submit_ts[future] = submit_ts
        for future in as_completed(future_to_submit_ts):
            result = future.result()
            done_ts = time.perf_counter()
            result["end_to_end_elapsed_ms"] = (done_ts - future_to_submit_ts[future]) * 1000.0
            results.append(result)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    sampler.stop()

    ok = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    service_latencies = [r["service_elapsed_ms"] for r in ok]
    end_to_end_latencies = [r["end_to_end_elapsed_ms"] for r in ok]
    error_counts: Dict[str, int] = {}
    for item in failures:
        key = item.get("error", "unknown")
        error_counts[key] = error_counts.get(key, 0) + 1

    return {
        "concurrency": concurrency,
        "request_count": len(rows),
        "successful_requests": len(ok),
        "failed_requests": len(failures),
        "success_rate_pct": round(100.0 * len(ok) / len(rows), 3) if rows else 0.0,
        "wall_time_ms": round(elapsed_ms, 3),
        "requests_per_second": round(len(rows) / (elapsed_ms / 1000.0), 3) if elapsed_ms > 0 else 0.0,
        "successful_requests_per_second": round(len(ok) / (elapsed_ms / 1000.0), 3) if elapsed_ms > 0 else 0.0,
        "service_time_mean_ms": round(statistics.mean(service_latencies), 3) if service_latencies else 0.0,
        "service_time_median_ms": round(statistics.median(service_latencies), 3) if service_latencies else 0.0,
        "service_time_p95_ms": round(percentile(service_latencies, 0.95), 3) if service_latencies else 0.0,
        "service_time_p99_ms": round(percentile(service_latencies, 0.99), 3) if service_latencies else 0.0,
        "end_to_end_time_mean_ms": round(statistics.mean(end_to_end_latencies), 3) if end_to_end_latencies else 0.0,
        "end_to_end_time_median_ms": round(statistics.median(end_to_end_latencies), 3) if end_to_end_latencies else 0.0,
        "end_to_end_time_p95_ms": round(percentile(end_to_end_latencies, 0.95), 3) if end_to_end_latencies else 0.0,
        "end_to_end_time_p99_ms": round(percentile(end_to_end_latencies, 0.99), 3) if end_to_end_latencies else 0.0,
        "error_counts": error_counts,
        "gpu_summary": sampler.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vLLM reranker concurrency using rerank requests from categorization output JSONL.")
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--source-input-jsonl", default=DEFAULT_SOURCE_INPUT_JSONL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY_CHARS)
    parser.add_argument("--rerank-query-max-chars", type=int, default=DEFAULT_RERANK_QUERY_MAX_CHARS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeat-factor", type=int, default=1)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-interval-s", type=float, default=DEFAULT_SAMPLE_INTERVAL)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=DEFAULT_CONCURRENCIES)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    rows = load_requests(
        input_jsonl=args.input_jsonl,
        source_input_jsonl=args.source_input_jsonl,
        max_body_chars=args.max_body_chars,
        rerank_query_max_chars=args.rerank_query_max_chars,
        limit=args.limit,
    )
    if not rows:
        raise SystemExit("No valid rerank requests found in input JSONL.")
    rows = repeat_requests(rows, args.repeat_factor)

    print(f"Loaded {len(rows)} rerank requests from {args.input_jsonl}")
    warmup = post_rerank(args.api_base, args.endpoint, args.model, rows[0]["query_text"], rows[0]["documents"], args.timeout)
    print(f"Warmup ok={warmup['ok']} service_latency_ms={round(warmup['service_elapsed_ms'], 3)}")
    if not warmup["ok"]:
        raise SystemExit(f"Warmup request failed: {warmup.get('error')}")

    results = []
    for concurrency in args.concurrencies:
        for trial in range(1, args.trials + 1):
            result = run_once(
                rows=rows,
                api_base=args.api_base,
                endpoint=args.endpoint,
                model=args.model,
                concurrency=concurrency,
                timeout=args.timeout,
                gpu_index=args.gpu_index,
                sample_interval_s=args.sample_interval_s,
            )
            result["trial"] = trial
            results.append(result)
            print(
                json.dumps(
                    {
                        "concurrency": result["concurrency"],
                        "trial": trial,
                        "success_rate_pct": result["success_rate_pct"],
                        "wall_time_ms": result["wall_time_ms"],
                        "requests_per_second": result["requests_per_second"],
                        "service_time_p95_ms": result["service_time_p95_ms"],
                        "end_to_end_time_p95_ms": result["end_to_end_time_p95_ms"],
                        "gpu_summary": result["gpu_summary"],
                        "error_counts": result["error_counts"],
                    }
                )
            )

    output = {
        "model": args.model,
        "api_base": args.api_base,
        "endpoint": args.endpoint,
        "input_jsonl": args.input_jsonl,
        "source_input_jsonl": args.source_input_jsonl,
        "request_count": len(rows),
        "repeat_factor": args.repeat_factor,
        "max_body_chars": args.max_body_chars,
        "rerank_query_max_chars": args.rerank_query_max_chars,
        "concurrencies": args.concurrencies,
        "trials": args.trials,
        "results": results,
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
