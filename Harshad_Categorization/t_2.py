#!/usr/bin/env python3
"""
Content and Ad Product Categorization (Tier 1 → Tier 2 → Tier 3) using Ollama.

- Content Taxonomy 3.1: categorizes content into Tier1 → Tier2 → Tier3.
- Ad Product Taxonomy 2.0: same flow for ad-product taxonomy.

Loads each taxonomy TSV once and keeps Tiers, associations, and Unique IDs in memory.
Output: one JSON Lines file (one line per URL+model) with content_taxonomy and ad_product_taxonomy clearly separated.
"""

import csv
import json
import re
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from fetch_url import fetch_url_content

# ---------------------------------------------------------------------------
# Taxonomy: loaded once per path, reused for all categorize_content calls (Option A)
# ---------------------------------------------------------------------------

_TAXONOMY_CACHE: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
_TAXONOMY_CACHE_LOCK = threading.Lock()

# Thread-local HTTP session for connection reuse (avoids new connection per request)
_thread_local = threading.local()

# Serialize print output when running models in parallel (avoids interleaved lines)
_print_lock = threading.Lock()

# Upper bound on concurrent model workers to avoid overwhelming Ollama
MAX_CONCURRENT_MODEL_WORKERS = 8


def _parse_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def load_taxonomy_once(
    tsv_path: str,
    parent_col: str = "Parent",
    skip_lines: int = 1,
) -> Dict[str, Any]:
    """
    Load the taxonomy TSV once per path. Builds in-memory structures for T1/T2/T3 and Unique IDs.
    Result is cached by (tsv_path, parent_col, skip_lines).
    - Content Taxonomy 3.1: parent_col="Parent", skip_lines=1.
    - Ad Product Taxonomy 2.0: parent_col="Parent ID", skip_lines=0.
    """
    global _TAXONOMY_CACHE
    cache_key = (tsv_path, parent_col, skip_lines)
    with _TAXONOMY_CACHE_LOCK:
        if cache_key in _TAXONOMY_CACHE:
            return _TAXONOMY_CACHE[cache_key]

    tier1_names = set()
    tier1_name_to_id: Dict[str, int] = {}
    # t1_id -> list of (tier2_name, unique_id of row that defines Tier2)
    tier2_by_tier1_id: Dict[int, List[Tuple[str, int]]] = {}
    # t2_id -> list of (tier3_name, unique_id of row that defines Tier3)
    tier3_by_tier2_id: Dict[int, List[Tuple[str, int]]] = {}
    # Row metadata for path building: unique_id -> (tier1_name, tier2_name, tier3_name or None)
    row_id_to_names: Dict[int, Tuple[str, str, Optional[str]]] = {}

    with open(tsv_path, "r", encoding="utf-8") as f:
        for _ in range(skip_lines):
            next(f)
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            uid = _parse_int(row.get("Unique ID", ""))
            parent = _parse_int(row.get(parent_col, ""))
            t1 = (row.get("Tier 1") or "").strip()
            t2 = (row.get("Tier 2") or "").strip()
            t3 = (row.get("Tier 3") or "").strip()
            if uid is None:
                continue

            # Tier1 root: no parent, or parent equals self (Ad Product style)
            if (not parent or parent == uid) and t1:
                # Tier1 root row
                tier1_names.add(t1)
                tier1_name_to_id[t1] = uid
                row_id_to_names[uid] = (t1, "", None)
            elif t1 and t2 and not t3:
                # Tier2 row (no Tier3 on this row)
                t1_id = tier1_name_to_id.get(t1)
                if t1_id is not None:
                    tier2_by_tier1_id.setdefault(t1_id, []).append((t2, uid))
                row_id_to_names[uid] = (t1, t2, None)
            elif t1 and t2 and t3:
                # Tier3 row: parent is Tier2 row id
                if parent is not None:
                    tier3_by_tier2_id.setdefault(parent, []).append((t3, uid))
                row_id_to_names[uid] = (t1, t2, t3)

    tax = {
        "tier1_names": tier1_names,
        "tier1_name_to_id": tier1_name_to_id,
        "tier2_by_tier1_id": tier2_by_tier1_id,
        "tier3_by_tier2_id": tier3_by_tier2_id,
        "row_id_to_names": row_id_to_names,
    }
    with _TAXONOMY_CACHE_LOCK:
        _TAXONOMY_CACHE[cache_key] = tax
    return tax


def _call_llm_categories(
    content: str,
    category_label: str,
    allowed_names: List[str],
    ollama_url: str,
    model: str,
    max_content_length: int = 10000,
    max_category_display_len: int = 500,
) -> Tuple[List[str], List[float], List[str], float]:
    """
    Ask LLM to return applicable category IDs, reason, and confidence (0-1 decimal) per category.
    Categories are sent as "id: label" to save tokens; LLM returns id, we map back to name.
    Returns (valid_names, confidence_scores, reasons, time_ms).
    """
    if not allowed_names:
        return [], [], [], 0.0

    # Indexed list: "0: Label0, 1: Label1, ..." (optionally truncate labels to save tokens)
    labels = [
        (n[:max_category_display_len] + ("..." if len(n) > max_category_display_len else ""))
        for n in allowed_names
    ]
    category_list = ", ".join(f"{i}: {lab}" for i, lab in enumerate(labels))
    n = len(allowed_names)

    prompt = f"""You are a content categorization expert. Pick applicable {category_label} categories by ID (0 to {n - 1}) with a confidence score each.

Categories (reply with id 0-{n - 1}): {category_list}

Content:
{content[:max_content_length]}

Return ONLY a JSON array of objects: "id" (integer 0-{n - 1}), "reason" (string, max 20 words: why this category fits the content), and "confidence" (0.05-1.0). Omit non-applicable. If none apply, return [].
Example: [{{"id": 0, "reason": "content discusses theme parks and attractions", "confidence": 0.8}}, {{"id": 2, "reason": "mentions family travel and destinations", "confidence": 0.6}}]"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",  # Keep model loaded 30 min to avoid cold start on next request
        "options": {"temperature": 0},
    }
    allowed_set = set(allowed_names)

    if not hasattr(_thread_local, "session") or _thread_local.session is None:
        _thread_local.session = requests.Session()
    session = _thread_local.session

    start = time.perf_counter()
    try:
        response = session.post(
            ollama_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        generated = (result.get("response") or "").strip()
    except Exception:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return [], [], [], elapsed_ms

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    raw_list = []
    json_match = re.search(r"\[.*?\]", generated, re.DOTALL)
    if json_match:
        try:
            raw_list = json.loads(json_match.group())
        except json.JSONDecodeError:
            try:
                cleaned = json_match.group().replace("\n", " ").strip()
                raw_list = json.loads(cleaned)
            except json.JSONDecodeError:
                return [], [], [], elapsed_ms
    else:
        try:
            raw_list = json.loads(generated)
        except json.JSONDecodeError:
            return [], [], [], elapsed_ms

    if not isinstance(raw_list, list):
        return [], [], [], elapsed_ms

    MIN_CONFIDENCE = 0.05
    valid_names = []
    confidence_scores = []
    reasons = []
    for item in raw_list:
        reason = ""
        if isinstance(item, dict):
            reason = (item.get("reason") or "").strip() if isinstance(item.get("reason"), str) else ""
            conf = item.get("confidence")
            score = 0.5
            if isinstance(conf, (int, float)) and 0 <= conf <= 1:
                score = max(float(conf), MIN_CONFIDENCE)
            # Prefer id (saves tokens); fall back to category/name
            raw_id = item.get("id")
            if isinstance(raw_id, int) and 0 <= raw_id < len(allowed_names):
                valid_names.append(allowed_names[raw_id])
                confidence_scores.append(score)
                reasons.append(reason)
                continue
            name = item.get("category") or item.get("name")
            if isinstance(name, str) and name in allowed_set:
                valid_names.append(name)
                confidence_scores.append(score)
                reasons.append(reason)
        elif isinstance(item, str) and item in allowed_set:
            valid_names.append(item)
            confidence_scores.append(0.5)
            reasons.append("")
    return valid_names, confidence_scores, reasons, elapsed_ms


def categorize_content(
    content: str,
    taxonomy_path: str = "taxonomy/Content Taxonomy 3.1.tsv",
    model: str = "llama3.2:3b",
    ollama_url: str = "http://localhost:11434/api/generate",
    include_tier_names: bool = True,
    include_tier3: bool = True,
    parent_col: str = "Parent",
    skip_lines: int = 1,
) -> Dict[str, Any]:
    """
    Categorize content into Tier1, then Tier2 (per Tier1), then Tier3 (per Tier2).
    Returns full paths with unique_id and timing keys T1, T2_<t1_id>, T3_<t1_id>_<t2_id>.
    If include_tier_names is False, each path in the response contains only unique_id (no tier1/tier2/tier3 names).
    If include_tier3 is False, Tier 3 categorization is skipped; paths stop at Tier 2 (tier3 null).
    """
    tax = load_taxonomy_once(taxonomy_path, parent_col=parent_col, skip_lines=skip_lines)
    tier1_names = tax["tier1_names"]
    tier1_name_to_id = tax["tier1_name_to_id"]
    tier2_by_tier1_id = tax["tier2_by_tier1_id"]
    tier3_by_tier2_id = tax["tier3_by_tier2_id"]
    row_id_to_names = tax["row_id_to_names"]

    timing: Dict[str, float] = {}
    paths: List[Dict[str, Any]] = []
    errors: List[str] = []

    # ---- Tier 1 ----
    tier1_list = sorted(tier1_names)
    selected_t1_names, t1_confidences, t1_reasons, t1_ms = _call_llm_categories(
        content, "Tier 1", tier1_list, ollama_url, model
    )
    timing["T1"] = t1_ms

    selected_t1 = [
        (name, tier1_name_to_id[name], t1_confidences[i] if i < len(t1_confidences) else 0.5)
        for i, name in enumerate(selected_t1_names)
        if name in tier1_name_to_id
    ]

    for t1_name, t1_id, _ in selected_t1:
        tier2_options = tier2_by_tier1_id.get(t1_id, [])
        if not tier2_options:
            continue

        tier2_names = [t[0] for t in tier2_options]
        selected_t2_names, t2_confidences, t2_reasons, t2_ms = _call_llm_categories(
            content, "Tier 2", tier2_names, ollama_url, model
        )
        timing[f"T2_{t1_id}"] = t2_ms

        t2_name_to_id = {t[0]: t[1] for t in tier2_options}
        for i, t2_name in enumerate(selected_t2_names):
            t2_id = t2_name_to_id.get(t2_name)
            if t2_id is None:
                continue
            t2_conf = t2_confidences[i] if i < len(t2_confidences) else 0.5
            t2_reason = t2_reasons[i] if i < len(t2_reasons) else ""

            tier3_options = tier3_by_tier2_id.get(t2_id, []) if include_tier3 else []
            if not tier3_options:
                # Path with Tier1 + Tier2 only (tier3 null); unique_id = t2 row id
                names = row_id_to_names.get(t2_id, (t1_name, t2_name, None))
                path = {"unique_id": t2_id, "confidence_score": t2_conf, "reason": t2_reason}
                if include_tier_names:
                    path["tier1"] = names[0]
                    path["tier2"] = names[1]
                    path["tier3"] = None
                paths.append(path)
                continue

            tier3_names = [t[0] for t in tier3_options]
            selected_t3_names, t3_confidences, t3_reasons, t3_ms = _call_llm_categories(
                content, "Tier 3", tier3_names, ollama_url, model
            )
            timing[f"T3_{t1_id}_{t2_id}"] = t3_ms

            t3_name_to_id = {t[0]: t[1] for t in tier3_options}
            for j, t3_name in enumerate(selected_t3_names):
                t3_id = t3_name_to_id.get(t3_name)
                if t3_id is None:
                    continue
                t3_conf = t3_confidences[j] if j < len(t3_confidences) else 0.5
                t3_reason = t3_reasons[j] if j < len(t3_reasons) else ""
                names = row_id_to_names.get(t3_id, (t1_name, t2_name, t3_name))
                path = {"unique_id": t3_id, "confidence_score": t3_conf, "reason": t3_reason}
                if include_tier_names:
                    path["tier1"] = names[0]
                    path["tier2"] = names[1]
                    path["tier3"] = names[2]
                paths.append(path)

    timing["total_ms"] = round(sum(timing.values()), 2)

    result: Dict[str, Any] = {
        "paths": paths,
        "count": len(paths),
        "timing": timing,
    }
    if errors:
        result["errors"] = errors
    return result


# Default paths for taxonomies
CONTENT_TAXONOMY_PATH = "taxonomy/Content_Taxonomy_3.1.tsv"
AD_PRODUCT_TAXONOMY_PATH = "taxonomy/Ad_Product_Taxonomy_2.0.tsv"


def categorize_ad_product(
    content: str,
    taxonomy_path: str = AD_PRODUCT_TAXONOMY_PATH,
    model: str = "llama3.2:3b",
    ollama_url: str = "http://localhost:11434/api/generate",
    include_tier_names: bool = True,
    include_tier3: bool = True,
) -> Dict[str, Any]:
    """
    Categorize content using the Ad Product Taxonomy (same Tier1 → Tier2 → Tier3 logic).
    Uses Ad Product Taxonomy 2.0 TSV format (Parent ID column, no header skip).
    """
    return categorize_content(
        content,
        taxonomy_path=taxonomy_path,
        model=model,
        ollama_url=ollama_url,
        include_tier_names=include_tier_names,
        include_tier3=include_tier3,
        parent_col="Parent ID",
        skip_lines=0,
    )


def _path_to_category_str(path: Dict[str, Any]) -> str:
    """Build 'Tier1 > Tier2 > Tier3' string for a path (tier3 may be None)."""
    t1 = path.get("tier1") or ""
    t2 = path.get("tier2") or ""
    t3 = path.get("tier3")
    base = f"{t1} > {t2}".rstrip(" >")
    return f"{base} > {t3}" if t3 else base


def _categorize_one_model(
    content: str,
    model: str,
    ollama_url: str = "http://localhost:11434/api/generate",
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Run content + ad-product categorization for one model (for parallel execution).
    Returns (model, content_result, ad_product_result).
    """
    content_result = categorize_content(
        content,
        taxonomy_path=CONTENT_TAXONOMY_PATH,
        model=model,
        ollama_url=ollama_url,
        parent_col="Parent",
        skip_lines=1,
    )
    ad_product_result = categorize_ad_product(content, model=model, ollama_url=ollama_url)
    return (model, content_result, ad_product_result)


def _build_log_entry(
    url: str,
    model: str,
    content_result: Dict[str, Any],
    ad_product_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one JSON Lines record for (url, model)."""
    return {
        "url": url,
        "model": model,
        "content_taxonomy": {
            "taxonomy": "Content Taxonomy 3.1",
            "paths": content_result["paths"],
            "count": content_result["count"],
            "timing_ms": content_result["timing"]["total_ms"],
        },
        "ad_product_taxonomy": {
            "taxonomy": "Ad Product Taxonomy 2.0",
            "paths": ad_product_result["paths"],
            "count": ad_product_result["count"],
            "timing_ms": ad_product_result["timing"]["total_ms"],
        },
    }


def _run_models_for_url(
    content: str,
    models: List[str],
    parallel: bool,
    ollama_url: str = "http://localhost:11434/api/generate",
) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Run all models for one URL. Returns {model: (content_result, ad_product_result)}.
    If parallel is True, runs models in parallel; otherwise sequentially.
    """
    error_fallback = (
        {"paths": [], "count": 0, "timing": {"total_ms": 0}},
        {"paths": [], "count": 0, "timing": {"total_ms": 0}},
    )
    results: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    if parallel:
        max_workers = min(len(models), MAX_CONCURRENT_MODEL_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(_categorize_one_model, content, m, ollama_url): m for m in models
            }
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    _, content_result, ad_product_result = future.result()
                    results[model] = (content_result, ad_product_result)
                    with _print_lock:
                        print(f"  Completed: {model}")
                except Exception as e:
                    with _print_lock:
                        print(f"  Error for model {model}: {e}")
                    results[model] = (
                        {**error_fallback[0], "errors": [str(e)]},
                        {**error_fallback[1], "errors": [str(e)]},
                    )
    else:
        for model in models:
            print(f"  Using model: {model}")
            try:
                _, content_result, ad_product_result = _categorize_one_model(content, model, ollama_url)
                results[model] = (content_result, ad_product_result)
                print("  Content taxonomy:", json.dumps(content_result, indent=2))
                print("  Ad Product taxonomy:", json.dumps(ad_product_result, indent=2))
            except Exception as e:
                print(f"  Error for model {model}: {e}")
                results[model] = (
                    {**error_fallback[0], "errors": [str(e)]},
                    {**error_fallback[1], "errors": [str(e)]},
                )
            print()

    return results


def _run_pipeline(parallel: bool) -> None:
    """Load taxonomies, process URLs with all models, write JSON Lines. parallel=True uses ThreadPoolExecutor."""
    OUTPUT_FILE = "outputs/categorization_log_14.jsonl"
    MAX_URLS = 1
    # models = ["llama3.2:3b", "ministral-3:3b", "gemma3:4b"]
    models = ["gemma3:4b"]
    ollama_url = "http://localhost:11434/api/generate"
    #ollama_url = "http://10.1.64.15:11434/api/generate"

    run_start = time.perf_counter()
    load_taxonomy_once(CONTENT_TAXONOMY_PATH, parent_col="Parent", skip_lines=1)
    load_taxonomy_once(AD_PRODUCT_TAXONOMY_PATH, parent_col="Parent ID", skip_lines=0)
    print("Content and Ad Product taxonomies loaded.\n")

    urls_file = Path("new_urls.txt")
    urls = urls_file.read_text(encoding="utf-8").strip().splitlines() if urls_file.exists() else []
    print(f"Found {len(urls)} URLs to process")

    log_path = Path(OUTPUT_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(urls, 1):
        if i > MAX_URLS:
            break
        print(f"--- URL {i}: {url}")
        content = fetch_url_content(url)
        if content is None:
            print("  Failed to fetch. Skipping.\n")
            continue
        print(f"  Content preview: {content.strip()}...\n")

        results_by_model = _run_models_for_url(content, models, parallel=parallel, ollama_url=ollama_url)

        with open(log_path, "a", encoding="utf-8") as f:
            for model in models:
                if model not in results_by_model:
                    continue
                content_result, ad_product_result = results_by_model[model]
                if parallel:
                    print(f"  Model {model} - Content taxonomy:", json.dumps(content_result, indent=2))
                    print(f"  Model {model} - Ad Product taxonomy:", json.dumps(ad_product_result, indent=2))
                log_entry = _build_log_entry(url, model, content_result, ad_product_result)
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        if parallel:
            print()

    total_seconds = time.perf_counter() - run_start
    print(f"Log written to {log_path} (one line per URL+model).")
    print(f"Total time: {total_seconds:.2f} seconds")


def main() -> None:
    parallel = True
    """Sequential: one model after another per URL."""
    """Parallel: all models run concurrently per URL; results written in fixed order."""
    _run_pipeline(parallel)

if __name__ == "__main__":
    main()

