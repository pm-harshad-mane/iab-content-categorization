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
from typing import Any, Dict, List, Optional

import requests
from transformers import AutoTokenizer

DEFAULT_INPUT_JSONL = "save_url_content_1000.jsonl"
DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_ENDPOINT = "/pooling"
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_BODY_CHARS = 4000
DEFAULT_MAX_MODEL_TOKENS = 512
DEFAULT_SAMPLE_INTERVAL = 1.0
DEFAULT_CONCURRENCIES = [32, 64, 128, 256, 512, 1024]

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
    return re.sub(r"\s+", " ", text.strip())


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


@dataclass
class TokenTruncationResult:
    text: str
    original_tokens: int
    used_tokens: int
    max_model_tokens: int
    reserved_special_tokens: int
    content_token_limit: int
    truncated: bool


class TokenAwareQueryTruncator:
    def __init__(self, model_name: str, max_model_tokens: int) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=True,
            )
        self.model_name = model_name
        self.max_model_tokens = max(1, int(max_model_tokens))
        self.reserved_special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        self.content_token_limit = max(1, self.max_model_tokens - self.reserved_special_tokens)

    def truncate(self, text: str) -> TokenTruncationResult:
        normalized = normalize_text(text)
        if not normalized:
            return TokenTruncationResult("", 0, 0, self.max_model_tokens, self.reserved_special_tokens, self.content_token_limit, False)

        encoded = self.tokenizer(
            normalized,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded.get("offset_mapping")
        token_count = len(input_ids)
        if token_count <= self.content_token_limit:
            return TokenTruncationResult(
                text=normalized,
                original_tokens=token_count,
                used_tokens=token_count,
                max_model_tokens=self.max_model_tokens,
                reserved_special_tokens=self.reserved_special_tokens,
                content_token_limit=self.content_token_limit,
                truncated=False,
            )

        if not offsets or len(offsets) < self.content_token_limit:
            truncated_text = self.tokenizer.decode(
                input_ids[: self.content_token_limit],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            end_char = int(offsets[self.content_token_limit - 1][1])
            truncated_text = normalized[:end_char].rstrip()

        used_tokens = len(
            self.tokenizer(
                truncated_text,
                add_special_tokens=False,
                truncation=False,
                verbose=False,
            )["input_ids"]
        )
        return TokenTruncationResult(
            text=truncated_text,
            original_tokens=token_count,
            used_tokens=used_tokens,
            max_model_tokens=self.max_model_tokens,
            reserved_special_tokens=self.reserved_special_tokens,
            content_token_limit=self.content_token_limit,
            truncated=True,
        )


def load_queries(
    input_jsonl: str,
    max_body_chars: int,
    truncator: TokenAwareQueryTruncator,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(input_jsonl).open() as handle:
        for line in handle:
            record = json.loads(line)
            if normalize_text(record.get("status")).lower() != "ok":
                continue
            page = parse_page_record(record)
            query = build_page_query_text(page, max_body_chars=max_body_chars)
            if not query:
                continue
            trunc = truncator.truncate(query)
            rows.append(
                {
                    "url": page.url,
                    "query_text": trunc.text,
                    "query_chars": len(trunc.text),
                    "original_tokens": trunc.original_tokens,
                    "used_tokens": trunc.used_tokens,
                    "truncated": trunc.truncated,
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def repeat_queries(queries: List[Dict[str, Any]], repeat_factor: int) -> List[Dict[str, Any]]:
    if repeat_factor <= 1:
        return queries
    expanded: List[Dict[str, Any]] = []
    for rep in range(repeat_factor):
        for item in queries:
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


def post_pooling(api_base: str, endpoint: str, model: str, query_text: str, timeout: int) -> Dict[str, Any]:
    session = _get_session()
    payload = {"model": model, "input": [query_text], "encoding_format": "float"}
    t0 = time.perf_counter()
    try:
        resp = session.post(api_base.rstrip("/") + endpoint, json=payload, timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"Unexpected response shape: {body}")
        item = data[0]
        raw = item.get("embedding")
        if raw is None and isinstance(item.get("data"), list):
            raw = item.get("data")
        if not raw or not isinstance(raw, list):
            raise RuntimeError(f"Missing token vectors: {body}")
        if not isinstance(raw[0], list):
            raise RuntimeError("Expected per-token vectors from /pooling, got pooled 1-D embedding.")
        token_count = len(raw)
        dim = len(raw[0]) if token_count else 0
        return {
            "ok": True,
            "status_code": resp.status_code,
            "service_elapsed_ms": elapsed_ms,
            "token_count": token_count,
            "embedding_dim": dim,
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
    queries: List[Dict[str, Any]],
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
        for item in queries:
            submit_ts = time.perf_counter()
            future = executor.submit(post_pooling, api_base, endpoint, model, item["query_text"], timeout)
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
    token_counts = [r["token_count"] for r in ok if r.get("token_count") is not None]
    dims = sorted({r["embedding_dim"] for r in ok if r.get("embedding_dim") is not None})
    error_counts: Dict[str, int] = {}
    for item in failures:
        key = item.get("error", "unknown")
        error_counts[key] = error_counts.get(key, 0) + 1

    return {
        "concurrency": concurrency,
        "request_count": len(queries),
        "successful_requests": len(ok),
        "failed_requests": len(failures),
        "success_rate_pct": round(100.0 * len(ok) / len(queries), 3) if queries else 0.0,
        "wall_time_ms": round(elapsed_ms, 3),
        "requests_per_second": round(len(queries) / (elapsed_ms / 1000.0), 3) if elapsed_ms > 0 else 0.0,
        "successful_requests_per_second": round(len(ok) / (elapsed_ms / 1000.0), 3) if elapsed_ms > 0 else 0.0,
        "service_time_mean_ms": round(statistics.mean(service_latencies), 3) if service_latencies else 0.0,
        "service_time_median_ms": round(statistics.median(service_latencies), 3) if service_latencies else 0.0,
        "service_time_p95_ms": round(percentile(service_latencies, 0.95), 3) if service_latencies else 0.0,
        "end_to_end_time_mean_ms": round(statistics.mean(end_to_end_latencies), 3) if end_to_end_latencies else 0.0,
        "end_to_end_time_median_ms": round(statistics.median(end_to_end_latencies), 3) if end_to_end_latencies else 0.0,
        "end_to_end_time_p95_ms": round(percentile(end_to_end_latencies, 0.95), 3) if end_to_end_latencies else 0.0,
        "token_count_mean": round(statistics.mean(token_counts), 3) if token_counts else 0.0,
        "token_count_p95": round(percentile(token_counts, 0.95), 3) if token_counts else 0.0,
        "embedding_dimensions": dims,
        "error_counts": error_counts,
        "gpu_summary": sampler.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ColBERT /pooling concurrency using token-aware page queries.")
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default="colbert-ir/colbertv2.0")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY_CHARS)
    parser.add_argument("--max-model-tokens", type=int, default=DEFAULT_MAX_MODEL_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeat-factor", type=int, default=1)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-interval-s", type=float, default=DEFAULT_SAMPLE_INTERVAL)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=DEFAULT_CONCURRENCIES)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    truncator = TokenAwareQueryTruncator(model_name=args.model, max_model_tokens=args.max_model_tokens)
    queries = load_queries(args.input_jsonl, args.max_body_chars, truncator, args.limit)
    if not queries:
        raise SystemExit("No valid queries found in input JSONL.")
    queries = repeat_queries(queries, args.repeat_factor)

    print(f"Loaded {len(queries)} ColBERT requests from {args.input_jsonl}")
    warmup = post_pooling(args.api_base, args.endpoint, args.model, queries[0]["query_text"], args.timeout)
    print(f"Warmup ok={warmup['ok']} service_latency_ms={round(warmup['service_elapsed_ms'], 3)}")
    if not warmup["ok"]:
        raise SystemExit(f"Warmup request failed: {warmup.get('error')}")

    results = []
    for concurrency in args.concurrencies:
        for trial in range(1, args.trials + 1):
            result = run_once(
                queries=queries,
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
                        "token_count_mean": result["token_count_mean"],
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
        "request_count": len(queries),
        "repeat_factor": args.repeat_factor,
        "max_body_chars": args.max_body_chars,
        "max_model_tokens": args.max_model_tokens,
        "concurrencies": args.concurrencies,
        "trials": args.trials,
        "results": results,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
