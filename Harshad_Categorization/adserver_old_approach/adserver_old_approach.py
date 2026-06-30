#!/usr/bin/env python3
"""
Workflow (adserver "old approach" — full taxonomy in one LLM call)

1. Read URLs from a text file (one URL per line).
2. For each URL, fetch and extract cleaned page text via ``fetch_with_beautiful_soup.fetch_page_content``.
3. Load the full Content Taxonomy 3.1 TSV (raw + id→path map) once at startup.
4. Call the vLLM OpenAI-compatible ``/v1/chat/completions`` endpoint with:
   - the page text (truncated if over a max length), and
   - the entire taxonomy TSV body,
   asking the model for the top K taxonomy leaf/category rows with a relevance score each.
5. Parse the model JSON, validate ``unique_id`` against the taxonomy, normalize ``path`` from the TSV.
6. Append one JSON object per URL to a JSONL output file.
7. Each output record includes per-step timings in milliseconds: fetch, vLLM, parse/validate, total.

Dependencies: requests, beautifulsoup4 (same as fetch_with_beautiful_soup).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from fetch_with_beautiful_soup import PageContent, fetch_page_content

# ---------------------------------------------------------------------------
# Thread-local HTTP session for vLLM
# ---------------------------------------------------------------------------

_thread_local = threading.local()
_print_lock = threading.Lock()

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def _normalize_uid(raw: Union[int, str, None]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None


def _json_uid(uid_str: str) -> Union[int, str]:
    if uid_str.isdigit():
        return int(uid_str)
    return uid_str


def load_taxonomy_tsv(tsv_path: str) -> Tuple[str, Dict[str, str]]:
    """
    Read the taxonomy file for (1) embedding in prompts as raw TSV text, and
    (2) a map Unique ID (string key) -> canonical path "Tier1 > Tier2 > ...".
    """
    path = Path(tsv_path)
    raw = path.read_text(encoding="utf-8")
    id_to_path: Dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for _ in range(1):
            next(f)
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            uid = _normalize_uid(row.get("Unique ID"))
            if not uid:
                continue
            parts = []
            for col in ("Tier 1", "Tier 2", "Tier 3", "Tier 4"):
                v = (row.get(col) or "").strip()
                if v:
                    parts.append(v)
            if parts:
                id_to_path[uid] = " > ".join(parts)

    return raw, id_to_path


def page_content_to_text(pc: PageContent, max_chars: Optional[int]) -> str:
    lines = [
        f"URL: {pc.url}",
        f"Title: {pc.title}",
        f"Meta: {pc.meta_description}",
    ]
    if pc.headings:
        lines.append("Headings: " + " | ".join(pc.headings))
    lines.append("Body:\n" + (pc.body_text or ""))
    text = "\n".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text


def _call_vllm_chat(
    api_base: str,
    model: str,
    prompt: str,
    temperature: float = 0,
    timeout: int = 180,
) -> Tuple[str, float, Optional[str]]:
    url = api_base.rstrip("/") + CHAT_COMPLETIONS_PATH
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }

    if not getattr(_thread_local, "session", None):
        _thread_local.session = requests.Session()
    session = _thread_local.session

    start = time.perf_counter()
    try:
        response = session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        choices = result.get("choices")
        if not choices:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return "", elapsed_ms, "API returned no choices"
        content = choices[0].get("message") or choices[0]
        generated = (content.get("content") or content.get("text") or "").strip()
        if not generated:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return "", elapsed_ms, "API returned empty content"
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return generated, elapsed_ms, None
    except requests.exceptions.HTTPError as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        err = f"HTTP {e.response.status_code}: {(e.response.text or str(e))[:500]}"
        return "", elapsed_ms, err
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return "", elapsed_ms, f"{type(e).__name__}: {e}"


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(text[i : j + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def build_categorization_prompt(
    page_text: str,
    taxonomy_tsv_raw: str,
    top_k: int,
) -> str:
    return f"""You are a content taxonomy classifier. Your task is to map web page content to the BEST matching categories from the official taxonomy below.

The taxonomy is a TSV table. Columns include: Unique ID, Parent, Name, Tier 1, Tier 2, Tier 3, Tier 4.
Each row is one category. The full category path is the non-empty Tier 1 through Tier 4 values joined with " > " (same order as in the file). Use the exact Unique ID string from the file (numeric or alphanumeric).

Rules:
- Select at most {top_k} categories, ranked by relevance to the page.
- Scores must be floats between 0 and 1 (higher = stronger match). They do not need to sum to 1.
- Only use Unique IDs that appear in the taxonomy TSV below. Paths must match the taxonomy exactly for that Unique ID.
- Prefer the most specific applicable row (deepest tier) when it clearly fits.

Output ONLY a single JSON object, no markdown, no commentary. Schema:
{{"paths":[{{"unique_id": <number or string as in TSV>, "path": "<exact path string>", "score": <float>}}, ...]}}

PAGE TEXT:
{page_text}

TAXONOMY TSV:
{taxonomy_tsv_raw}
"""


def parse_and_validate_paths(
    raw_llm: str,
    id_to_path: Dict[str, str],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    obj = _extract_json_object(raw_llm)
    if not obj:
        return [], "Could not parse JSON from model output"
    paths_in = obj.get("paths")
    if not isinstance(paths_in, list):
        return [], 'Missing or invalid "paths" array'

    out: List[Dict[str, Any]] = []
    for item in paths_in:
        if not isinstance(item, dict):
            continue
        uid = _normalize_uid(item.get("unique_id"))
        if uid is None or uid not in id_to_path:
            continue
        score = item.get("score")
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        score_f = max(0.0, min(1.0, score_f))
        out.append(
            {
                "unique_id": _json_uid(uid),
                "path": id_to_path[uid],
                "score": score_f,
            }
        )

    out.sort(key=lambda x: x["score"], reverse=True)
    out = out[:top_k]
    return out, None


def process_one_url(
    url: str,
    taxonomy_tsv_rel: str,
    taxonomy_raw: str,
    id_to_path: Dict[str, str],
    model: str,
    api_base: str,
    top_k: int,
    fetch_timeout: int,
    max_page_chars: Optional[int],
    vllm_timeout: int,
) -> Dict[str, Any]:
    t_total0 = time.perf_counter()
    timing: Dict[str, float] = {}

    t0 = time.perf_counter()
    fetch_error: Optional[str] = None
    page_text = ""
    try:
        pc = fetch_page_content(url, timeout=fetch_timeout)
        page_text = page_content_to_text(pc, max_page_chars)
    except Exception as e:
        fetch_error = f"{type(e).__name__}: {e}"
    timing["fetch_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if fetch_error:
        timing["vllm_ms"] = 0.0
        timing["parse_validate_ms"] = 0.0
        timing["total_ms"] = round((time.perf_counter() - t_total0) * 1000, 2)
        return {
            "url": url,
            "model_name": model,
            "taxonomy_tsv": taxonomy_tsv_rel,
            "paths": [],
            "error": fetch_error,
            "timing_ms": timing,
        }

    prompt = build_categorization_prompt(page_text, taxonomy_raw, top_k)
    raw, vllm_ms, vllm_err = _call_vllm_chat(
        api_base, model, prompt, temperature=0, timeout=vllm_timeout
    )
    timing["vllm_ms"] = vllm_ms

    t1 = time.perf_counter()
    paths, parse_err = parse_and_validate_paths(raw, id_to_path, top_k)
    timing["parse_validate_ms"] = round((time.perf_counter() - t1) * 1000, 2)

    timing["total_ms"] = round((time.perf_counter() - t_total0) * 1000, 2)

    record: Dict[str, Any] = {
        "url": url,
        "model_name": model,
        "taxonomy_tsv": taxonomy_tsv_rel,
        "paths": paths,
        "timing_ms": timing,
    }
    if vllm_err:
        record["error"] = vllm_err
    if parse_err and not vllm_err:
        record["parse_warning"] = parse_err
    return record


def read_urls(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u and not u.startswith("#"):
                urls.append(u)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adserver old approach: fetch URLs with BeautifulSoup, categorize via vLLM with full taxonomy TSV."
    )
    parser.add_argument(
        "--urls-file",
        default="adserver_1000_urls.txt",
        help="Text file with one URL per line.",
    )
    parser.add_argument(
        "--taxonomy",
        default="taxonomy/Content_Taxonomy_3.1.tsv",
        help="Path to Content Taxonomy 3.1 TSV.",
    )
    parser.add_argument(
        "--output",
        default="adserver_old_approach_2.jsonl",
        help="Output JSONL path (one JSON object per line).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of taxonomy paths to return per URL.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="Model name as served by vLLM.",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
        help="vLLM OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--fetch-timeout",
        type=int,
        default=15,
        help="HTTP timeout (seconds) for BeautifulSoup fetch.",
    )
    parser.add_argument(
        "--vllm-timeout",
        type=int,
        default=180,
        help="HTTP timeout (seconds) for chat completions.",
    )
    parser.add_argument(
        "--max-page-chars",
        type=int,
        default=120_000,
        help="Max characters of page text sent to the model (omit or use 0 for no limit).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="If > 0, only process the first N URLs (for testing).",
    )
    args = parser.parse_args()

    max_chars: Optional[int] = args.max_page_chars
    if max_chars is not None and max_chars <= 0:
        max_chars = None

    taxonomy_path = str(Path(args.taxonomy).resolve())
    taxonomy_tsv_rel = args.taxonomy.replace("\\", "/")
    with _print_lock:
        print(f"Loading taxonomy from {taxonomy_path} ...")
    taxonomy_raw, id_to_path = load_taxonomy_tsv(taxonomy_path)
    with _print_lock:
        print(f"  {len(id_to_path)} taxonomy rows with paths.")

    urls = read_urls(args.urls_file)
    if args.limit and args.limit > 0:
        urls = urls[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _print_lock:
        print(f"Processing {len(urls)} URLs -> {out_path}")

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, url in enumerate(urls, start=1):
            rec = process_one_url(
                url=url,
                taxonomy_tsv_rel=taxonomy_tsv_rel,
                taxonomy_raw=taxonomy_raw,
                id_to_path=id_to_path,
                model=args.model,
                api_base=args.api_base,
                top_k=args.top_k,
                fetch_timeout=args.fetch_timeout,
                max_page_chars=max_chars,
                vllm_timeout=args.vllm_timeout,
            )
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            tm = rec.get("timing_ms") or {}
            with _print_lock:
                print(
                    f"[{i}/{len(urls)}] {url[:80]}..."
                    if len(url) > 80
                    else f"[{i}/{len(urls)}] {url}",
                    f"fetch={tm.get('fetch_ms')}ms vllm={tm.get('vllm_ms')}ms "
                    f"parse={tm.get('parse_validate_ms')}ms total={tm.get('total_ms')}ms "
                    f"paths={len(rec.get('paths') or [])}",
                )


if __name__ == "__main__":
    main()

