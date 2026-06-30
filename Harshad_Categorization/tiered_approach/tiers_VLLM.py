#!/usr/bin/env python3
"""
Workflow: tiered content categorization with vLLM (T1 → T2 → T3 → rerank)

1. Read URLs from a text file (one URL per line).
2. Process each URL concurrently (thread pool): fetch HTML → cleaned text via BeautifulSoup.
3. Load Content Taxonomy 3.1.2 TSV once. Tree: Tier 1 = rows with empty Parent; Tier 2 = rows
   whose Parent is a T1 Unique ID; Tier 3 = rows whose Parent is a T2 Unique ID.
4. vLLM call #1: page text + all T1 candidates (Unique ID, path, Description from TSV).
   Same prompt template asks for top --tier-top-k categories as JSON array.
5. Collect T2 rows whose Parent is in the set of Unique IDs returned at T1.
6. vLLM call #2: page text + those T2 candidates (with descriptions). Same prompt, top --tier-top-k.
7. Collect T3 rows whose Parent is in the set of Unique IDs returned at T2.
8. vLLM call #3: page text + those T3 candidates (with descriptions). Same prompt, top --tier-top-k.
9. vLLM call #4 (rerank): page text + union of selected T1/T2/T3 rows (deduped by Unique ID, with
   descriptions). Separate prompt; top --rerank-top-k.
10. Emit JSONL: one object per URL with model_name, taxonomy path, T1/T2/T3/reranked arrays,
    timing_ms (fetch, t1, t2, t3, rerank, total).

Output row order matches the input URL file order.
"""

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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import requests

from fetch_with_beautiful_soup import PageContent, fetch_page_content

_thread_local = threading.local()
_print_lock = threading.Lock()

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


@dataclass
class TaxonomyRow:
    unique_id: str
    parent: str
    tier1: str
    tier2: str
    tier3: str
    tier4: str
    description: str
    path: str


def _normalize_uid(raw: Union[int, str, None]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None


def _json_uid(uid_str: str) -> Union[int, str]:
    if uid_str.isdigit():
        return int(uid_str)
    return uid_str


def _row_path(row: Dict[str, str]) -> str:
    parts = []
    for col in ("Tier 1", "Tier 2", "Tier 3", "Tier 4"):
        v = (row.get(col) or "").strip()
        if v:
            parts.append(v)
    return " > ".join(parts)


def load_taxonomy(tsv_path: str) -> Tuple[List[TaxonomyRow], Dict[str, TaxonomyRow]]:
    """Load TSV into rows and uid -> row map (tree edges use Parent → Unique ID)."""
    path = Path(tsv_path)
    rows: List[TaxonomyRow] = []
    by_id: Dict[str, TaxonomyRow] = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            uid = _normalize_uid(row.get("Unique ID"))
            if not uid:
                continue
            parent = _normalize_uid(row.get("Parent")) or ""
            tr = TaxonomyRow(
                unique_id=uid,
                parent=parent,
                tier1=(row.get("Tier 1") or "").strip(),
                tier2=(row.get("Tier 2") or "").strip(),
                tier3=(row.get("Tier 3") or "").strip(),
                tier4=(row.get("Tier 4") or "").strip(),
                description=(row.get("Description") or "").strip(),
                path=_row_path(row),
            )
            rows.append(tr)
            by_id[uid] = tr

    return rows, by_id


def t1_rows(all_rows: List[TaxonomyRow]) -> List[TaxonomyRow]:
    return [r for r in all_rows if not r.parent]


def children_rows(parent_uids: Set[str], by_id: Dict[str, TaxonomyRow], all_rows: List[TaxonomyRow]) -> List[TaxonomyRow]:
    if not parent_uids:
        return []
    return [r for r in all_rows if r.parent in parent_uids]


def format_candidates_block(candidates: List[TaxonomyRow]) -> str:
    lines = []
    for r in candidates:
        lines.append(f"- unique_id: {r.unique_id}\n  path: {r.path}\n  description: {r.description}")
    return "\n".join(lines)


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


def build_tier_categorization_prompt(
    page_text: str,
    candidates_block: str,
    tier_top_k: int,
) -> str:
    return f"""
        You are a content taxonomy classifier.

        Below are candidate categories from an official taxonomy. Each entry has unique_id (use exactly as given), path (canonical label chain), and description (extra context for matching).

        Rules:
        - Pick at most {tier_top_k} categories from the candidate list only. Rank by relevance to the page.
        - Scores are floats from 0 to 1 (higher = stronger match). Scores need not sum to 1.
        - Every unique_id and path must match a candidate entry exactly.

        Output ONLY a JSON array (no markdown fences, no commentary). Example shape:
        [{{"unique_id": 390, "path": "Science > Weather", "score": 0.95}}]

        CANDIDATES:
        {candidates_block}

        PAGE TEXT:
        {page_text}
    """


def build_rerank_prompt(page_text: str, candidates_block: str, rerank_top_k: int) -> str:
    return f"""
    You are a content taxonomy classifier performing a final rerank.

    Earlier steps proposed categories at different depths. Below is the combined shortlist (unique_id, path, description). Pick the single best-fitting categories for this page.

    Rules:
    - Return at most {rerank_top_k} items from the shortlist only.
    - Scores are floats from 0 to 1 (higher = stronger match).
    - unique_id and path must match a shortlist entry exactly.

    Output ONLY a JSON array (no markdown fences, no commentary). Example:
    [{{"unique_id": 390, "path": "Science > Weather", "score": 0.95}}]

    SHORTLIST:
    {candidates_block}

    PAGE TEXT:
    {page_text}
    """


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


def _extract_json_array(text: str) -> Optional[List[Any]]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "paths" in data and isinstance(data["paths"], list):
            return data["paths"]
    except json.JSONDecodeError:
        pass
    i = text.find("[")
    j = text.rfind("]")
    if i >= 0 and j > i:
        try:
            data = json.loads(text[i : j + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            return None
    return None


def parse_and_validate_array(
    raw_llm: str,
    allowed_ids: Set[str],
    id_to_path: Dict[str, str],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    arr = _extract_json_array(raw_llm)
    if arr is None:
        return [], "Could not parse JSON array from model output"

    out: List[Dict[str, Any]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        uid = _normalize_uid(item.get("unique_id"))
        if uid is None or uid not in allowed_ids:
            continue
        path_final = id_to_path.get(uid, "").strip()
        if not path_final:
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
                "path": path_final,
                "score": score_f,
            }
        )

    out.sort(key=lambda x: x["score"], reverse=True)
    out = out[:top_k]
    return out, None


def process_one_url(
    url: str,
    taxonomy_tsv_rel: str,
    all_rows: List[TaxonomyRow],
    by_id: Dict[str, TaxonomyRow],
    model: str,
    api_base: str,
    tier_top_k: int,
    rerank_top_k: int,
    fetch_timeout: int,
    max_page_chars: Optional[int],
    vllm_timeout: int,
) -> Dict[str, Any]:
    t_total0 = time.perf_counter()
    timing: Dict[str, float] = {}

    id_to_path = {r.unique_id: r.path for r in all_rows}

    t0 = time.perf_counter()
    fetch_error: Optional[str] = None
    page_text = ""
    try:
        pc = fetch_page_content(url, timeout=fetch_timeout)
        page_text = page_content_to_text(pc, max_page_chars)
    except Exception as e:
        fetch_error = f"{type(e).__name__}: {e}"
    timing["fetch_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    empty_timing_tail = {
        "t1_ms": 0.0,
        "t2_ms": 0.0,
        "t3_ms": 0.0,
        "rerank_ms": 0.0,
        "total_ms": round((time.perf_counter() - t_total0) * 1000, 2),
    }

    if fetch_error:
        timing.update(empty_timing_tail)
        return {
            "url": url,
            "model_name": model,
            "taxonomy": taxonomy_tsv_rel,
            "T1": [],
            "T2": [],
            "T3": [],
            "reranked": [],
            "error": fetch_error,
            "timing_ms": timing,
        }

    errors: List[str] = []
    t1_out: List[Dict[str, Any]] = []
    t2_out: List[Dict[str, Any]] = []
    t3_out: List[Dict[str, Any]] = []
    reranked_out: List[Dict[str, Any]] = []

    # --- T1 ---
    c1 = t1_rows(all_rows)
    if not c1:
        timing["t1_ms"] = 0.0
        errors.append("No T1 rows in taxonomy")
    else:
        p1 = build_tier_categorization_prompt(page_text, format_candidates_block(c1), tier_top_k)
        raw1, ms1, e1 = _call_vllm_chat(api_base, model, p1, temperature=0, timeout=vllm_timeout)
        timing["t1_ms"] = ms1
        allowed_t1 = {r.unique_id for r in c1}
        t1_out, w1 = parse_and_validate_array(raw1, allowed_t1, id_to_path, tier_top_k)
        if e1:
            errors.append(f"T1 vLLM: {e1}")
        if w1:
            errors.append(f"T1 parse: {w1}")

    selected_t1 = {str(x["unique_id"]) for x in t1_out}

    # --- T2 ---
    c2 = children_rows(selected_t1, by_id, all_rows)
    if not selected_t1 or not c2:
        timing["t2_ms"] = 0.0
    else:
        p2 = build_tier_categorization_prompt(page_text, format_candidates_block(c2), tier_top_k)
        raw2, ms2, e2 = _call_vllm_chat(api_base, model, p2, temperature=0, timeout=vllm_timeout)
        timing["t2_ms"] = ms2
        allowed_t2 = {r.unique_id for r in c2}
        t2_out, w2 = parse_and_validate_array(raw2, allowed_t2, id_to_path, tier_top_k)
        if e2:
            errors.append(f"T2 vLLM: {e2}")
        if w2:
            errors.append(f"T2 parse: {w2}")

    selected_t2 = {str(x["unique_id"]) for x in t2_out}

    # --- T3 ---
    c3 = children_rows(selected_t2, by_id, all_rows)
    if not selected_t2 or not c3:
        timing["t3_ms"] = 0.0
    else:
        p3 = build_tier_categorization_prompt(page_text, format_candidates_block(c3), tier_top_k)
        raw3, ms3, e3 = _call_vllm_chat(api_base, model, p3, temperature=0, timeout=vllm_timeout)
        timing["t3_ms"] = ms3
        allowed_t3 = {r.unique_id for r in c3}
        t3_out, w3 = parse_and_validate_array(raw3, allowed_t3, id_to_path, tier_top_k)
        if e3:
            errors.append(f"T3 vLLM: {e3}")
        if w3:
            errors.append(f"T3 parse: {w3}")

    # --- Rerank: union of selected uids from T1, T2, T3 ---
    t_r0 = time.perf_counter()
    union_uids: Set[str] = set()
    for x in t1_out:
        union_uids.add(str(x["unique_id"]))
    for x in t2_out:
        union_uids.add(str(x["unique_id"]))
    for x in t3_out:
        union_uids.add(str(x["unique_id"]))

    rerank_rows = [by_id[u] for u in union_uids if u in by_id]
    rerank_rows.sort(key=lambda r: r.unique_id)

    if not rerank_rows:
        timing["rerank_ms"] = 0.0
    else:
        pr = build_rerank_prompt(page_text, format_candidates_block(rerank_rows), rerank_top_k)
        raw_r, ms_r, er = _call_vllm_chat(api_base, model, pr, temperature=0, timeout=vllm_timeout)
        timing["rerank_ms"] = ms_r
        allowed_r = {r.unique_id for r in rerank_rows}
        reranked_out, wr = parse_and_validate_array(raw_r, allowed_r, id_to_path, rerank_top_k)
        if er:
            errors.append(f"rerank vLLM: {er}")
        if wr:
            errors.append(f"rerank parse: {wr}")

    timing["total_ms"] = round((time.perf_counter() - t_total0) * 1000, 2)

    record: Dict[str, Any] = {
        "url": url,
        "model_name": model,
        "taxonomy": taxonomy_tsv_rel,
        "T1": t1_out,
        "T2": t2_out,
        "T3": t3_out,
        "reranked": reranked_out,
        "timing_ms": timing,
    }
    if errors:
        record["errors"] = errors
    return record


def _process_one_url_safe(
    index: int,
    url: str,
    taxonomy_tsv_rel: str,
    all_rows: List[TaxonomyRow],
    by_id: Dict[str, TaxonomyRow],
    model: str,
    api_base: str,
    tier_top_k: int,
    rerank_top_k: int,
    fetch_timeout: int,
    max_page_chars: Optional[int],
    vllm_timeout: int,
) -> Tuple[int, Dict[str, Any]]:
    try:
        rec = process_one_url(
            url=url,
            taxonomy_tsv_rel=taxonomy_tsv_rel,
            all_rows=all_rows,
            by_id=by_id,
            model=model,
            api_base=api_base,
            tier_top_k=tier_top_k,
            rerank_top_k=rerank_top_k,
            fetch_timeout=fetch_timeout,
            max_page_chars=max_page_chars,
            vllm_timeout=vllm_timeout,
        )
        return index, rec
    except Exception as e:
        t0 = time.perf_counter()
        rec = {
            "url": url,
            "model_name": model,
            "taxonomy": taxonomy_tsv_rel,
            "T1": [],
            "T2": [],
            "T3": [],
            "reranked": [],
            "error": f"{type(e).__name__}: {e}",
            "timing_ms": {
                "fetch_ms": 0.0,
                "t1_ms": 0.0,
                "t2_ms": 0.0,
                "t3_ms": 0.0,
                "rerank_ms": 0.0,
                "total_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        }
        return index, rec


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
        description="Tiered T1→T2→T3 vLLM categorization with final rerank; parallel URL processing."
    )
    parser.add_argument(
        "--urls-file",
        default="adserver_1000_urls.txt",
        help="Text file with one URL per line.",
    )
    parser.add_argument(
        "--taxonomy",
        default="taxonomy/Content_Taxonomy_3.1_2.tsv",
        help="Path to Content Taxonomy TSV.",
    )
    parser.add_argument(
        "--output",
        default="tiers_VLLM_gemma3_1000.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--tier-top-k",
        type=int,
        default=10,
        help="Max categories per tier (T1, T2, T3).",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=5,
        help="Max categories after final rerank.",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-4b-it",
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
        help="Max characters of page text sent to the model (0 = no limit).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="If > 0, only process the first N URLs (for testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max concurrent URLs (thread pool size).",
    )
    args = parser.parse_args()

    max_chars: Optional[int] = args.max_page_chars
    if max_chars is not None and max_chars <= 0:
        max_chars = None

    workers = max(1, args.workers)

    taxonomy_path = str(Path(args.taxonomy).resolve())
    taxonomy_tsv_rel = args.taxonomy.replace("\\", "/")
    all_rows, by_id = load_taxonomy(taxonomy_path)

    with _print_lock:
        print(f"Loaded {len(all_rows)} taxonomy rows from {taxonomy_path}")

    urls = read_urls(args.urls_file)
    if args.limit and args.limit > 0:
        urls = urls[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _print_lock:
        print(f"Processing {len(urls)} URLs with {workers} workers -> {out_path}")

    results: List[Optional[Dict[str, Any]]] = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                _process_one_url_safe,
                i,
                url,
                taxonomy_tsv_rel,
                all_rows,
                by_id,
                args.model,
                args.api_base,
                args.tier_top_k,
                args.rerank_top_k,
                args.fetch_timeout,
                max_chars,
                args.vllm_timeout,
            ): i
            for i, url in enumerate(urls)
        }
        completed = 0
        for fut in as_completed(future_to_index):
            idx, rec = fut.result()
            results[idx] = rec
            completed += 1
            tm = rec.get("timing_ms") or {}
            u = rec.get("url", urls[idx])
            line = f"[{completed}/{len(urls)}] {u[:80]}..." if len(u) > 80 else f"[{completed}/{len(urls)}] {u}"
            with _print_lock:
                print(
                    line,
                    f"fetch={tm.get('fetch_ms')}ms t1={tm.get('t1_ms')}ms t2={tm.get('t2_ms')}ms "
                    f"t3={tm.get('t3_ms')}ms rerank={tm.get('rerank_ms')}ms total={tm.get('total_ms')}ms",
                )

    with out_path.open("w", encoding="utf-8") as out_f:
        for rec in results:
            assert rec is not None
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

