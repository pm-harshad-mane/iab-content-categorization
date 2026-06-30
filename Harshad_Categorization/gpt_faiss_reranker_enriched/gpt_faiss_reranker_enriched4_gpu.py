import json
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from fetch_with_beautiful_soup import PageContent, fetch_page_content


TSV_PATH = "taxonomy/Content_Taxonomy_3.1_2.tsv"
URL_LIST_PATH = "new_urls.txt"
OUTPUT_JSON_PATH = "gpt_faiss_reranker_enriched4_gpu.json"

# Dense retriever model
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Reranker model
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_PER_QUERY = 10
FUSED_TOP_K = 10
FINAL_TOP_K = 5
REQUEST_TIMEOUT = 15

# GPU knobs
EMBED_BATCH_SIZE = 128
RERANK_BATCH_SIZE = 64
USE_FP16 = True
FAISS_USE_ALL_GPUS = False  # set True if your FAISS build supports multi-GPU and index is large enough


@dataclass
class RetrievalCandidate:
    unique_id: int
    parent_id: Optional[int]
    tier1: str
    tier2: str
    tier3: str
    tier4: str
    path: str
    description: str
    faiss_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0


def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def build_path_string(row: pd.Series) -> str:
    levels = [
        row.get("tier1", ""),
        row.get("tier2", ""),
        row.get("tier3", ""),
        row.get("tier4", ""),
    ]
    return " > ".join([normalize_text(x) for x in levels if normalize_text(x)])


def load_taxonomy_tsv(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")

    normalized_cols = {}
    for c in df.columns:
        key = c.strip().lower()
        key = re.sub(r"\s+", " ", key)
        normalized_cols[c] = key
    df = df.rename(columns=normalized_cols)

    rename_map = {
        "unique id": "unique_id",
        "parent": "parent_id",
        "tier 1": "tier1",
        "tier 2": "tier2",
        "tier 3": "tier3",
        "tier 4": "tier4",
        "description": "description",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["unique_id", "parent_id", "tier1", "tier2", "description"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in TSV: {missing}")

    if "tier3" not in df.columns:
        df["tier3"] = ""
    if "tier4" not in df.columns:
        df["tier4"] = ""

    for col in ["tier1", "tier2", "tier3", "tier4", "description"]:
        df[col] = df[col].map(normalize_text)

    df["unique_id"] = pd.to_numeric(df["unique_id"], errors="coerce")
    df["parent_id"] = pd.to_numeric(df["parent_id"], errors="coerce")
    df = df[df["unique_id"].notna()].copy()
    df["unique_id"] = df["unique_id"].astype(int)

    df["path"] = df.apply(build_path_string, axis=1)

    return df[
        ["unique_id", "parent_id", "tier1", "tier2", "tier3", "tier4", "path", "description"]
    ].reset_index(drop=True)


def load_urls_from_file(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def get_torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model_kwargs() -> Dict[str, Any]:
    device = get_torch_device()
    kwargs: Dict[str, Any] = {"device": device}
    if device == "cuda" and USE_FP16:
        # sentence-transformers forwards model_kwargs to the HF model
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return kwargs


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)

    if faiss.get_num_gpus() <= 0:
        cpu_index.add(embeddings)
        print("FAISS running on CPU (no FAISS GPU detected).")
        return cpu_index

    try:
        if FAISS_USE_ALL_GPUS and hasattr(faiss, "index_cpu_to_all_gpus"):
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
            gpu_index.add(embeddings)
            print(f"FAISS running on all available GPUs ({faiss.get_num_gpus()}).")
            return gpu_index

        if hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            gpu_index.add(embeddings)
            print("FAISS running on GPU 0.")
            return gpu_index

        cpu_index.add(embeddings)
        print("FAISS GPU libraries not available in this faiss build. Falling back to CPU index.")
        return cpu_index
    except Exception as e:
        cpu_index.add(embeddings)
        print(f"FAISS GPU setup failed, falling back to CPU index: {e}")
        return cpu_index


class FaissTaxonomyRetriever:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.device = get_torch_device()
        self.model = SentenceTransformer(model_name, **get_model_kwargs())
        self.df: Optional[pd.DataFrame] = None
        self.index: Optional[faiss.Index] = None

    def build_index(self, taxonomy_df: pd.DataFrame) -> None:
        self.df = taxonomy_df.copy()
        docs = self.df["description"].tolist()

        embeddings = self.model.encode(
            docs,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        self.index = build_faiss_index(embeddings)

    def search(self, query_text: str, top_k: int = TOP_K_PER_QUERY) -> List[RetrievalCandidate]:
        if self.df is None or self.index is None:
            raise RuntimeError("Index not built.")

        query_vec = self.model.encode(
            [query_text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) < 0:
                continue
            row = self.df.iloc[int(idx)]
            parent_id = None if pd.isna(row["parent_id"]) else int(row["parent_id"])

            results.append(
                RetrievalCandidate(
                    unique_id=int(row["unique_id"]),
                    parent_id=parent_id,
                    tier1=row["tier1"],
                    tier2=row["tier2"],
                    tier3=row["tier3"],
                    tier4=row["tier4"],
                    path=row["path"],
                    description=row["description"],
                    faiss_score=float(score),
                )
            )
        return results


class TaxonomyReranker:
    def __init__(self, model_name: str = RERANK_MODEL_NAME):
        self.device = get_torch_device()
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            automodel_args={"torch_dtype": torch.float16} if self.device == "cuda" and USE_FP16 else None,
        )

    def build_rerank_query(self, page: PageContent, max_chars: int = 1800) -> str:
        parts = []
        if page.title:
            parts.append(f"title: {page.title}")
        if page.meta_description:
            parts.append(f"description: {page.meta_description}")
        if page.headings:
            parts.append("headings: " + " | ".join(page.headings[:6]))
        if page.body_text:
            parts.append(f"content: {page.body_text[:max_chars]}")
        return " || ".join(parts)

    def rerank(
        self,
        page: PageContent,
        candidates: List[RetrievalCandidate],
        final_top_k: int = FINAL_TOP_K,
    ) -> List[RetrievalCandidate]:
        if not candidates:
            return []

        query = self.build_rerank_query(page)
        pairs = [(c.description, query) for c in candidates]

        scores = self.model.predict(
            pairs,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        ranked = []
        for cand, score in zip(candidates, scores):
            cand.rerank_score = float(score)
            ranked.append(cand)

        ranked.sort(key=lambda x: (x.rerank_score, x.fused_score, x.faiss_score), reverse=True)
        return ranked[:final_top_k]


def extract_page_content_from_url(url: str) -> PageContent:
    return fetch_page_content(url, timeout=REQUEST_TIMEOUT)


def build_multi_queries(page: PageContent, max_body_chars: int = 1800) -> List[str]:
    body = page.body_text[:max_body_chars]
    headings_joined = " | ".join(page.headings[:6])

    queries = []
    q5 = []
    if page.title:
        q5.append(f"title: {page.title}")
    if page.meta_description:
        q5.append(f"description: {page.meta_description}")
    if headings_joined:
        q5.append(f"headings: {headings_joined}")
    if body:
        q5.append(f"content: {body}")
    if q5:
        queries = [" || ".join(q5)]

    return dedupe_preserve_order([q for q in queries if normalize_text(q)])


def reciprocal_rank_fusion(
    per_query_results: List[List[RetrievalCandidate]],
    final_top_k: int = FUSED_TOP_K,
    k: int = 60,
) -> List[RetrievalCandidate]:
    merged: Dict[int, Dict[str, Any]] = {}

    for results in per_query_results:
        for rank, cand in enumerate(results, start=1):
            if cand.unique_id not in merged:
                merged[cand.unique_id] = {
                    "candidate": cand,
                    "rrf_score": 0.0,
                    "best_faiss_score": cand.faiss_score,
                    "appearances": 0,
                }

            merged[cand.unique_id]["rrf_score"] += 1.0 / (k + rank)
            merged[cand.unique_id]["best_faiss_score"] = max(
                merged[cand.unique_id]["best_faiss_score"], cand.faiss_score
            )
            merged[cand.unique_id]["appearances"] += 1

    fused = []
    for item in merged.values():
        cand = item["candidate"]
        cand.fused_score = item["rrf_score"]
        cand.faiss_score = item["best_faiss_score"]
        fused.append(cand)

    fused.sort(key=lambda x: (x.fused_score, x.faiss_score), reverse=True)
    return fused[:final_top_k]


def candidate_to_dict(c: RetrievalCandidate) -> Dict[str, Any]:
    return {
        "unique_id": c.unique_id,
        "path": c.path,
        "faiss_score": round(c.faiss_score, 6),
        "fused_score": round(c.fused_score, 6),
        "rerank_score": round(c.rerank_score, 6),
    }


def get_ranked_taxonomy_categories(
    url: str,
    retriever: FaissTaxonomyRetriever,
    reranker: TaxonomyReranker,
    top_k_per_query: int = TOP_K_PER_QUERY,
    fused_top_k: int = FUSED_TOP_K,
    final_top_k: int = FINAL_TOP_K,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    t_fetch0 = time.perf_counter()
    page = fetch_page_content(url, timeout=REQUEST_TIMEOUT)
    fetch_seconds = time.perf_counter() - t_fetch0

    clean_seconds = 0.0

    t_faiss0 = time.perf_counter()
    queries = build_multi_queries(page)
    per_query_results = []
    for query in queries:
        per_query_results.append(retriever.search(query, top_k=top_k_per_query))
    faiss_seconds = time.perf_counter() - t_faiss0

    t_fuse0 = time.perf_counter()
    fused_candidates = reciprocal_rank_fusion(
        per_query_results,
        final_top_k=fused_top_k,
    )
    fuse_seconds = time.perf_counter() - t_fuse0

    t_rerank0 = time.perf_counter()
    final_ranked = reranker.rerank(
        page=page,
        candidates=fused_candidates,
        final_top_k=final_top_k,
    )
    rerank_seconds = time.perf_counter() - t_rerank0

    ranking_time_seconds = time.perf_counter() - t0
    fetch_ms = round(fetch_seconds * 1000)
    total_ms = round(ranking_time_seconds * 1000)

    step_timings_ms = {
        "fetch_url": fetch_ms,
        "clean_content": round(clean_seconds * 1000),
        "faiss_categorization": round(faiss_seconds * 1000),
        "fuse": round(fuse_seconds * 1000),
        "rerank": round(rerank_seconds * 1000),
        "total": total_ms,
        "total_without_fetch_url": total_ms - fetch_ms,
    }

    return {
        "page": {
            "url": page.url,
            "domain": page.domain,
            "title": page.title,
            "meta_description": page.meta_description,
            "headings": page.headings[:6],
            "body_preview": page.body_text[:500],
        },
        "step_timings_ms": step_timings_ms,
        "fused_candidates": [candidate_to_dict(c) for c in fused_candidates],
        "final_ranked_categories": [candidate_to_dict(c) for c in final_ranked],
    }


if __name__ == "__main__":
    MAX_URLS_TO_FETCH: Optional[int] = 1

    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"torch device = {get_torch_device()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"faiss.get_num_gpus() = {faiss.get_num_gpus()}")

    urls = load_urls_from_file(URL_LIST_PATH)
    if not urls:
        raise SystemExit(f"No URLs found in {URL_LIST_PATH}")

    if MAX_URLS_TO_FETCH is not None:
        n = max(0, int(MAX_URLS_TO_FETCH))
        urls = urls[:n]
        if not urls:
            raise SystemExit(
                f"No URLs to process after MAX_URLS_TO_FETCH={MAX_URLS_TO_FETCH} "
                f"(file had entries but limit is 0)"
            )

    taxonomy_df = load_taxonomy_tsv(TSV_PATH)

    retriever = FaissTaxonomyRetriever(EMBED_MODEL_NAME)
    print("Building FAISS index over category descriptions (once)...")
    retriever.build_index(taxonomy_df)

    reranker = TaxonomyReranker(RERANK_MODEL_NAME)

    results: List[Dict[str, Any]] = []
    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] {url}")
        try:
            result = get_ranked_taxonomy_categories(
                url=url,
                retriever=retriever,
                reranker=reranker,
                top_k_per_query=TOP_K_PER_QUERY,
                fused_top_k=FUSED_TOP_K,
                final_top_k=FINAL_TOP_K,
            )
            result["ok"] = True
            result["error"] = None
            results.append(result)
        except Exception as e:
            results.append({
                "ok": False,
                "error": str(e),
                "page": {"url": url},
                "step_timings_ms": None,
                "fused_candidates": [],
                "final_ranked_categories": [],
            })

    out = {
        "source_file": URL_LIST_PATH,
        "taxonomy_tsv": TSV_PATH,
        "url_count": len(urls),
        "results": results,
    }
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in results if r.get("ok"))
    print(f"Wrote {OUTPUT_JSON_PATH} ({ok}/{len(results)} succeeded)")

