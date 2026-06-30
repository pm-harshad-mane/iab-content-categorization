#!/usr/bin/env python3
"""
Content Categorization (Tier 1 → Tier 2 → Tier 3) using FAISS + sentence-transformers.

Classifies content locally without LLMs:
- faiss-cpu: in-memory vector index
- sentence-transformers: text → embeddings
- pandas: load IAB Content Taxonomy TSV

Same tiering as t_3.py: first T1, then T2 per T1, then T3 per T2.
Output: same JSON Lines style (one line per URL).

Run: python faiss_categorize.py   (script is named faiss_categorize.py to avoid
shadowing the 'faiss' package when importing it.)
"""

# Reduce segfault risk on macOS when using faiss-cpu + sentence-transformers (see e.g.
# https://github.com/UKPLab/sentence-transformers/issues/2291). Set before importing
# numpy/faiss/torch.
import os
# Limit OpenMP threads (used by numpy/scipy and many C libs) to avoid oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Limit OpenBLAS threads (BLAS implementation used by some numpy builds).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# Limit Intel MKL threads (BLAS/LAPACK used when numpy is built with MKL).
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from fetch_url import fetch_url_content

# Import faiss after sentence_transformers so native libs load in an order that
# avoids segfaults on macOS when both are used.
import faiss

# ---------------------------------------------------------------------------
# Taxonomy: load with pandas, build T1/T2/T3 structures (same logic as t_3)
# ---------------------------------------------------------------------------

CONTENT_TAXONOMY_PATH = "taxonomy/Content_Taxonomy_3.1.tsv"
PARENT_COL = "Parent"
SKIP_LINES = 1  # Content Taxonomy 3.1 has one header line to skip


def _parse_int(s) -> Optional[int]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def load_taxonomy_pandas(tsv_path: str) -> Dict[str, Any]:
    """
    Load taxonomy TSV with pandas. Build tier1, tier2_by_tier1_id, tier3_by_tier2_id, row_id_to_names.
    Content Taxonomy 3.1: skip first line, use Parent column.
    """
    with open(tsv_path, "r", encoding="utf-8") as f:
        for _ in range(SKIP_LINES):
            next(f)
        df = pd.read_csv(f, delimiter="\t", dtype=str)

    df = df.fillna("")
    tier1_names = set()
    tier1_name_to_id: Dict[str, int] = {}
    tier2_by_tier1_id: Dict[int, List[Tuple[str, int]]] = {}
    tier3_by_tier2_id: Dict[int, List[Tuple[str, int]]] = {}
    row_id_to_names: Dict[int, Tuple[str, str, Optional[str]]] = {}

    for _, row in df.iterrows():
        uid = _parse_int(row.get("Unique ID", ""))
        parent = _parse_int(row.get(PARENT_COL, ""))
        t1 = (row.get("Tier 1") or "").strip()
        t2 = (row.get("Tier 2") or "").strip()
        t3 = (row.get("Tier 3") or "").strip()
        if uid is None:
            continue

        if (not parent or parent == uid) and t1:
            tier1_names.add(t1)
            tier1_name_to_id[t1] = uid
            row_id_to_names[uid] = (t1, "", None)
        elif t1 and t2 and not t3:
            t1_id = tier1_name_to_id.get(t1)
            if t1_id is not None:
                tier2_by_tier1_id.setdefault(t1_id, []).append((t2, uid))
            row_id_to_names[uid] = (t1, t2, None)
        elif t1 and t2 and t3:
            if parent is not None:
                tier3_by_tier2_id.setdefault(parent, []).append((t3, uid))
            row_id_to_names[uid] = (t1, t2, t3)

    return {
        "tier1_names": tier1_names,
        "tier1_name_to_id": tier1_name_to_id,
        "tier2_by_tier1_id": tier2_by_tier1_id,
        "tier3_by_tier2_id": tier3_by_tier2_id,
        "row_id_to_names": row_id_to_names,
    }


# ---------------------------------------------------------------------------
# Embedding model and FAISS indices (built once, reused)
# ---------------------------------------------------------------------------

def _normalize_l2(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows so that IndexFlatIP gives cosine similarity."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return x.astype(np.float32) / norms


class FAISSCategorizer:
    """Build T1/T2/T3 FAISS indices from taxonomy and run tiered search."""

    def __init__(
        self,
        taxonomy_path: str = CONTENT_TAXONOMY_PATH,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        top_k_t1: int = 5,
        top_k_t2: int = 5,
        top_k_t3: int = 5,
        include_tier3: bool = True,
    ):
        self.taxonomy_path = taxonomy_path
        self.model_name = model_name
        self.top_k_t1 = top_k_t1
        self.top_k_t2 = top_k_t2
        self.top_k_t3 = top_k_t3
        self.include_tier3 = include_tier3

        self.tax = load_taxonomy_pandas(taxonomy_path)
        self.tier1_names = self.tax["tier1_names"]
        self.tier1_name_to_id = self.tax["tier1_name_to_id"]
        self.tier2_by_tier1_id = self.tax["tier2_by_tier1_id"]
        self.tier3_by_tier2_id = self.tax["tier3_by_tier2_id"]
        self.row_id_to_names = self.tax["row_id_to_names"]

        # Model load options: GTE (and similar) need trust_remote_code + CPU to avoid MPS bugs.
        # Default all-MiniLM-L6-v2 is most reliable; GTE (Alibaba-NLP/gte-base-en-v1.5) can hit
        # position_ids/IndexError in its custom RoPE code on some envs - use --model to try it.
        _is_gte_or_custom = (
            "gte" in model_name.lower()
            or "Alibaba-NLP" in model_name
        )
        _model_kwargs = {"trust_remote_code": True} if _is_gte_or_custom else {}
        if _is_gte_or_custom:
            _model_kwargs["device"] = "cpu"  # avoid MPS AcceleratorError on macOS
        self.model = SentenceTransformer(model_name, **_model_kwargs)
        dim = self.model.get_sentence_embedding_dimension()

        # T1 index: one vector per Tier 1 name
        tier1_list = sorted(self.tier1_names)
        self.tier1_list = tier1_list
        t1_emb = self.model.encode(tier1_list, convert_to_numpy=True, show_progress_bar=False)
        t1_emb = _normalize_l2(t1_emb)
        self.index_t1 = faiss.IndexFlatIP(dim)
        self.index_t1.add(t1_emb)
        self.t1_index_to_uid = [self.tier1_name_to_id[n] for n in tier1_list]

        # T2 indices: one index per T1 (only for T1s that have T2 children)
        self.t2_indices: Dict[int, Tuple[faiss.Index, List[Tuple[str, int]]]] = {}
        for t1_id, t2_options in self.tier2_by_tier1_id.items():
            t2_names = [t[0] for t in t2_options]
            # Embed with T1 context for better discrimination
            t1_name = next((n for n, i in self.tier1_name_to_id.items() if i == t1_id), "")
            texts = [f"{t1_name} {n}" for n in t2_names]
            t2_emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            t2_emb = _normalize_l2(t2_emb)
            idx = faiss.IndexFlatIP(dim)
            idx.add(t2_emb)
            self.t2_indices[t1_id] = (idx, t2_options)

        # T3 indices: one index per T2 (only for T2s that have T3 children)
        self.t3_indices: Dict[int, Tuple[faiss.Index, List[Tuple[str, int]]]] = {}
        for t2_id, t3_options in self.tier3_by_tier2_id.items():
            t3_names = [t[0] for t in t3_options]
            t1_name, t2_name, _ = self.row_id_to_names.get(t2_id, ("", "", None))
            texts = [f"{t1_name} {t2_name} {n}" for n in t3_names]
            t3_emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            t3_emb = _normalize_l2(t3_emb)
            idx = faiss.IndexFlatIP(dim)
            idx.add(t3_emb)
            self.t3_indices[t2_id] = (idx, t3_options)

    def _similarity_to_confidence(self, score: float) -> float:
        """Map cosine similarity (roughly [-1,1] for IP on normalized vecs) to [0.5, 1]."""
        # return max(0.5, min(1.0, float(score)))
        return float(score)

    def categorize(
        self,
        content: str,
        max_content_chars: int = 8000,
        include_tier_names: bool = True,
    ) -> Dict[str, Any]:
        """
        Categorize content: T1 → T2 → T3 using FAISS search. No LLM.
        Returns same shape as t_3 categorize_content: paths, count, timing.
        """
        timing: Dict[str, float] = {}
        paths: List[Dict[str, Any]] = []

        if not (content or "").strip():
            return {"paths": [], "count": 0, "timing": {"T1": 0, "total_ms": 0}}

        text = content.strip()[:max_content_chars]
        t0 = time.perf_counter()
        content_emb = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        content_emb = _normalize_l2(content_emb)

        # ---- Tier 1 ----
        t1_start = time.perf_counter()
        k1 = min(self.top_k_t1, self.index_t1.ntotal)
        scores_t1, indices_t1 = self.index_t1.search(content_emb, k1)
        timing["T1"] = round((time.perf_counter() - t1_start) * 1000, 2)

        selected_t1: List[Tuple[str, int, float]] = []
        for j, idx in enumerate(indices_t1[0]):
            if idx < 0:
                continue
            uid = self.t1_index_to_uid[idx]
            score = float(scores_t1[0][j])
            name = self.row_id_to_names.get(uid, ("", "", None))[0]
            if name:
                selected_t1.append((name, uid, self._similarity_to_confidence(score)))

        # ---- Tier 2 (and T3) per selected T1 ----
        for t1_name, t1_id, t1_conf in selected_t1:
            t2_options = self.tier2_by_tier1_id.get(t1_id)
            if not t2_options:
                continue
            idx_t2, t2_list = self.t2_indices[t1_id]
            k2 = min(self.top_k_t2, idx_t2.ntotal)
            t2_start = time.perf_counter()
            scores_t2, indices_t2 = idx_t2.search(content_emb, k2)
            timing[f"T2_{t1_id}"] = round((time.perf_counter() - t2_start) * 1000, 2)

            for j, idx in enumerate(indices_t2[0]):
                if idx < 0 or idx >= len(t2_list):
                    continue
                t2_name, t2_id = t2_list[idx]
                t2_conf = self._similarity_to_confidence(float(scores_t2[0][j]))
                tier3_options = self.tier3_by_tier2_id.get(t2_id, []) if self.include_tier3 else []

                if not tier3_options:
                    if t2_conf >= 0:
                        path = {
                            "unique_id": t2_id,
                            "confidence_score": round(max(0.0, t2_conf), 2),
                            "reason": "",
                        }
                        if include_tier_names:
                            path["tier1"] = t1_name
                            path["tier2"] = t2_name
                            path["tier3"] = None
                        paths.append(path)
                    continue

                # T3 search for this T2
                idx_t3, t3_list = self.t3_indices[t2_id]
                k3 = min(self.top_k_t3, idx_t3.ntotal)
                t3_start = time.perf_counter()
                scores_t3, indices_t3 = idx_t3.search(content_emb, k3)
                timing[f"T3_{t1_id}_{t2_id}"] = round((time.perf_counter() - t3_start) * 1000, 2)

                for k, i3 in enumerate(indices_t3[0]):
                    if i3 < 0 or i3 >= len(t3_list):
                        continue
                    t3_name, t3_id = t3_list[i3]
                    t3_conf = self._similarity_to_confidence(float(scores_t3[0][k]))
                    if t3_conf < 0:
                        continue
                    names = self.row_id_to_names.get(t3_id, (t1_name, t2_name, t3_name))
                    path = {
                        "unique_id": t3_id,
                        "confidence_score": round(max(0.0, t3_conf), 2),
                        "reason": "",
                    }
                    if include_tier_names:
                        path["tier1"] = names[0]
                        path["tier2"] = names[1]
                        path["tier3"] = names[2]
                    paths.append(path)

        timing["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "paths": paths,
            "count": len(paths),
            "timing": timing,
        }


# ---------------------------------------------------------------------------
# Output: same style as t_3 (one JSON line per URL)
# ---------------------------------------------------------------------------


def _round_floats_for_json(obj: Any, ndigits: int = 2) -> Any:
    """Recursively round floats so JSON output has at most ndigits after decimal."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats_for_json(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats_for_json(x, ndigits) for x in obj]
    return obj


def build_log_entry(url: str, content_result: Dict[str, Any]) -> Dict[str, Any]:
    """One JSON Lines record per URL (model = 'faiss', no ad product)."""
    paths = sorted(
        content_result["paths"],
        key=lambda p: p.get("confidence_score", 0),
        reverse=True,
    )
    return {
        "url": url,
        "model": "faiss",
        "content_taxonomy": {
            "taxonomy": "Content Taxonomy 3.1",
            "paths": paths,
            "count": content_result["count"],
            "timing_ms": content_result["timing"]["total_ms"],
        },
        "ad_product_taxonomy": {
            "taxonomy": "Ad Product Taxonomy 2.0",
            "paths": [],
            "count": 0,
            "timing_ms": 0,
        },
    }


def run_pipeline(
    urls_file: str = "new_urls.txt",
    output_file: str = "outputs/faiss_categorization_log.jsonl",
    taxonomy_path: str = CONTENT_TAXONOMY_PATH,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    top_k_t1: int = 5,
    top_k_t2: int = 5,
    top_k_t3: int = 5,
    max_urls: Optional[int] = None,
) -> None:
    """Load taxonomy, build FAISS indices, process URLs, write JSONL."""
    run_start = time.perf_counter()
    print("Loading taxonomy and building FAISS indices...")
    categorizer = FAISSCategorizer(
        taxonomy_path=taxonomy_path,
        model_name=model_name,
        top_k_t1=top_k_t1,
        top_k_t2=top_k_t2,
        top_k_t3=top_k_t3,
    )
    print("Done.\n")

    urls_path = Path(urls_file)
    urls = urls_path.read_text(encoding="utf-8").strip().splitlines() if urls_path.exists() else []
    print(f"Found {len(urls)} URLs to process.")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(urls, 1):
        if max_urls is not None and i > max_urls:
            break
        print(f"--- URL {i}: {url}")
        content = fetch_url_content(url)
        if content is None:
            print("  Failed to fetch. Skipping.\n")
            continue
        print(f"  Content length: {len(content)} chars.")
        result = categorizer.categorize(content, include_tier_names=True)
        print(f"  Paths: {result['count']}, timing_ms: {result['timing']['total_ms']}")
        log_entry = build_log_entry(url, result)
        log_entry = _round_floats_for_json(log_entry, 2)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print()

    total_seconds = time.perf_counter() - run_start
    print(f"Log written to {out_path} (one line per URL).")
    print(f"Total time: {total_seconds:.2f} seconds.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Content Categorization (T1→T2→T3) using FAISS + sentence-transformers (no LLM)."
    )
    parser.add_argument("--urls", default="new_urls.txt", help="Path to file with one URL per line.")
    parser.add_argument("--output", default="outputs/faiss_categorization_log.jsonl", help="Output JSONL path.")
    parser.add_argument("--taxonomy", default=CONTENT_TAXONOMY_PATH, help="Content taxonomy TSV path.")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="sentence-transformers model. Options: all-mpnet-base-v2 (default), BAAI/bge-small-en-v1.5, all-MiniLM-L6-v2.",
    )
    parser.add_argument("--top-k-t1", type=int, default=3, help="Top-k Tier 1 categories.")
    parser.add_argument("--top-k-t2", type=int, default=3, help="Top-k Tier 2 per T1.")
    parser.add_argument("--top-k-t3", type=int, default=3, help="Top-k Tier 3 per T2.")
    parser.add_argument("--max-urls", type=int, default=15, help="Max URLs to process (default: all).")
    args = parser.parse_args()

    model_name = "BAAI/bge-small-en-v1.5"
    # model_name = "all-MiniLM-L6-v2"
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_name = "Alibaba-NLP/gte-base-en-v1.5" # doesn't work with faiss-cpu
    output_file = f"outputs_2/faiss_categorization_{model_name.replace('/', '_')}_log.jsonl"
    print(f"Output file: {output_file}")
    run_pipeline(
        urls_file=args.urls,
        output_file=output_file,
        taxonomy_path=args.taxonomy,
        model_name=model_name,
        # adjusting below temprature decides how many values to return
        top_k_t1=args.top_k_t1,
        top_k_t2=args.top_k_t2,
        top_k_t3=args.top_k_t3,
        max_urls=args.max_urls,
    )


if __name__ == "__main__":
    main()

