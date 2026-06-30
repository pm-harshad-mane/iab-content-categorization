"""
Content categorization pipeline (FAISS retrieval + cross-encoder rerank on GPU only).

This variant is based on gpt_faiss_reranker_enriched5_gpu.py, but requires both:
- CUDA-enabled torch for embedding + reranker inference
- FAISS GPU support for index build/search

If either requirement is unavailable, the script exits instead of silently falling
back to CPU.
"""

import json
import os
import re
import site
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm

from fetch_with_beautiful_soup import PageContent, fetch_page_content

TSV_PATH = "taxonomy/Content_Taxonomy_3.1_2.tsv"
URL_LIST_PATH = "adserver_1000_urls.txt"
OUTPUT_JSON_PATH = "gpt_faiss_reranker_enriched7_gpu_only_3000.json"

# Parallelism: how many URLs to process concurrently (threads).
MAX_URL_WORKERS = 1

# Dense retriever model
#EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_MODEL_NAME = "nvidia/NV-Embed-v2"
#EMBED_MODEL_NAME = "Alibaba-NLP/gte-modernbert-base"

# Reranker model
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

TOP_K_PER_QUERY = 10
FUSED_TOP_K = 10
FINAL_TOP_K = 5
REQUEST_TIMEOUT = 15

# GPU knobs
CUDA_DEVICE = 0
EMBED_BATCH_SIZE = 128
RERANK_BATCH_SIZE = 64
USE_FP16 = True
NVIDIA_EMBED_BATCH_SIZE = 8
NVIDIA_EMBED_MAX_SEQ_LENGTH = 256

_gpu_lock = threading.Lock()
_print_lock = threading.Lock()


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


@dataclass
class FaissRuntimeInfo:
    module_path: str
    version: str
    gpu_api_available: bool
    gpu_count: int
    imported_from_user_site: bool

    @property
    def summary(self) -> str:
        bits = [
            f"module={self.module_path}",
            f"version={self.version}",
            f"gpu_api_available={self.gpu_api_available}",
            f"gpu_count={self.gpu_count}",
        ]
        return ", ".join(bits)


def get_faiss_runtime_info() -> FaissRuntimeInfo:
    module_path = getattr(faiss, "__file__", "<unknown>")
    version = str(getattr(faiss, "__version__", "unknown"))
    gpu_api_available = hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")
    gpu_count = safe_faiss_gpu_count()

    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = ""

    imported_from_user_site = bool(user_site) and os.path.abspath(module_path).startswith(os.path.abspath(user_site))
    return FaissRuntimeInfo(
        module_path=module_path,
        version=version,
        gpu_api_available=gpu_api_available,
        gpu_count=gpu_count,
        imported_from_user_site=imported_from_user_site,
    )


def explain_faiss_gpu_failure(info: FaissRuntimeInfo) -> str:
    lines = [
        "FAISS GPU support is required for this script.",
        f"Detected FAISS runtime: {info.summary}",
        f"Python executable: {sys.executable}",
    ]

    if not info.gpu_api_available:
        lines.append("This FAISS build is CPU-only and does not expose GPU APIs such as StandardGpuResources.")
        if info.imported_from_user_site:
            lines.append(
                "The active interpreter is importing FAISS from your user site-packages (~/.local), which is often a CPU wheel shadowing a GPU-capable environment."
            )
        lines.append(
            "Fix: run the script from a Python environment that has a GPU-enabled FAISS build, or replace the current faiss-cpu install with a CUDA-matching faiss-gpu package."
        )
        return "\n".join(lines)

    lines.append(
        "This FAISS build exposes GPU APIs, but faiss.get_num_gpus() returned 0 for the current process."
    )
    lines.append(
        "Fix: verify that this exact Python environment can see the NVIDIA driver/CUDA runtime, and that the FAISS GPU wheel matches the installed CUDA stack."
    )
    return "\n".join(lines)


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


def safe_faiss_gpu_count() -> int:
    if not hasattr(faiss, "get_num_gpus"):
        return 0
    try:
        return max(0, int(faiss.get_num_gpus()))
    except Exception:
        return 0


def require_cuda_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but torch.cuda.is_available() is False.")

    gpu_count = torch.cuda.device_count()
    if CUDA_DEVICE < 0 or CUDA_DEVICE >= gpu_count:
        raise RuntimeError(
            f"Configured CUDA_DEVICE={CUDA_DEVICE} is out of range for {gpu_count} visible GPU(s)."
        )

    return f"cuda:{CUDA_DEVICE}"


def is_nvidia_embed_model(model_name: str) -> bool:
    return "nvidia/nv-embed" in model_name.lower()


def get_model_kwargs(model_name: str) -> Dict[str, Any]:
    device = require_cuda_device()
    kwargs: Dict[str, Any] = {"device": device}
    if is_nvidia_embed_model(model_name):
        kwargs["trust_remote_code"] = True
    if USE_FP16:
        dtype_key = "dtype" if is_nvidia_embed_model(model_name) else "torch_dtype"
        kwargs["model_kwargs"] = {dtype_key: torch.float16}
    return kwargs


def get_effective_embed_batch_size(model_name: str) -> int:
    if is_nvidia_embed_model(model_name):
        return NVIDIA_EMBED_BATCH_SIZE
    return EMBED_BATCH_SIZE


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def apply_nvidia_embed_compat(model: SentenceTransformer, model_name: str) -> None:
    if not is_nvidia_embed_model(model_name):
        return

    if hasattr(model, "max_seq_length"):
        model.max_seq_length = min(int(model.max_seq_length), NVIDIA_EMBED_MAX_SEQ_LENGTH)

    for module in model._modules.values():
        auto_model = getattr(module, "auto_model", None)
        if auto_model is None:
            continue
        if hasattr(auto_model, "config"):
            auto_model.config.use_cache = False
        embedding_model = getattr(auto_model, "embedding_model", None)
        if embedding_model is None:
            continue
        if not hasattr(embedding_model, "rotary_emb") or not hasattr(embedding_model, "layers"):
            continue

        def _forward_compat(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Any] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Any:
            from transformers.modeling_outputs import BaseModelOutputWithPast
            from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # NV-Embed-v2's remote code targets an older transformers Mistral API.
            # Disable cache usage for embedding inference and provide explicit rotary embeddings.
            use_cache = False
            output_attentions = False

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
            if input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                position_ids = position_ids.view(-1, seq_length).long()

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            all_hidden_states = () if output_hidden_states else None
            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )

            hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if not return_dict:
                return tuple(v for v in (hidden_states, None, all_hidden_states, None) if v is not None)

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=None,
                hidden_states=all_hidden_states,
                attentions=None,
            )

        embedding_model.forward = types.MethodType(_forward_compat, embedding_model)


class FaissTaxonomyRetriever:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self.device = require_cuda_device()
        self.embed_batch_size = get_effective_embed_batch_size(model_name)
        self.model = SentenceTransformer(model_name, **get_model_kwargs(model_name))
        apply_nvidia_embed_compat(self.model, model_name)
        self.df: Optional[pd.DataFrame] = None
        self.index: Optional[faiss.Index] = None
        self.index_backend = "uninitialized"
        self._faiss_gpu_resources: Optional[Any] = None
        self._faiss_search_lock = threading.Lock()

    def _get_nvidia_auto_model(self) -> Any:
        for module in self.model._modules.values():
            auto_model = getattr(module, "auto_model", None)
            if auto_model is not None and hasattr(auto_model, "encode"):
                return auto_model
        raise RuntimeError("Failed to locate NV-Embed auto_model.encode().")

    def _encode_texts(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")

        if not is_nvidia_embed_model(self.model_name):
            return self.model.encode(
                texts,
                batch_size=self.embed_batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

        auto_model = self._get_nvidia_auto_model()
        batches: List[np.ndarray] = []
        iterator = range(0, len(texts), self.embed_batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, total=(len(texts) + self.embed_batch_size - 1) // self.embed_batch_size, desc="NV-Embed batches")
        torch.cuda.empty_cache()
        for start in iterator:
            batch = texts[start : start + self.embed_batch_size]
            batch_embeddings = auto_model.encode(
                batch,
                max_length=NVIDIA_EMBED_MAX_SEQ_LENGTH,
                return_numpy=True,
            )
            if torch.is_tensor(batch_embeddings):
                batch_embeddings = batch_embeddings.detach().cpu().numpy()
            batch_embeddings = np.asarray(batch_embeddings, dtype="float32")
            batches.append(normalize_l2(batch_embeddings))

        return np.concatenate(batches, axis=0)

    def build_index(self, taxonomy_df: pd.DataFrame) -> None:
        self.df = taxonomy_df.copy()
        docs = self.df["description"].tolist()

        embeddings = self._encode_texts(docs, show_progress_bar=True)

        faiss_info = get_faiss_runtime_info()
        if faiss_info.gpu_count <= 0 or not faiss_info.gpu_api_available:
            raise RuntimeError(explain_faiss_gpu_failure(faiss_info))

        dim = embeddings.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)
        self._faiss_gpu_resources = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self._faiss_gpu_resources, CUDA_DEVICE, cpu_index)
        self.index.add(embeddings)
        self.index_backend = "gpu"
        print(f"FAISS running on GPU {CUDA_DEVICE}.")

    def search(self, query_text: str, top_k: int = TOP_K_PER_QUERY) -> List[RetrievalCandidate]:
        if self.df is None or self.index is None:
            raise RuntimeError("Index not built.")

        query_vec = self._encode_texts([query_text]).astype("float32")

        with self._faiss_search_lock:
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
        self.model_name = model_name
        self.device = require_cuda_device()
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            automodel_args={"torch_dtype": torch.float16} if USE_FP16 else None,
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


def build_model_details(
    retriever: FaissTaxonomyRetriever,
    reranker: TaxonomyReranker,
) -> Dict[str, Any]:
    return {
        "embedding_model": {
            "name": retriever.model_name,
            "device": retriever.device,
            "batch_size": EMBED_BATCH_SIZE,
            "normalize_embeddings": True,
            "torch_dtype": "float16" if USE_FP16 else "float32",
        },
        "reranker_model": {
            "name": reranker.model_name,
            "device": reranker.device,
            "batch_size": RERANK_BATCH_SIZE,
            "torch_dtype": "float16" if USE_FP16 else "float32",
        },
        "faiss": {
            "backend": retriever.index_backend,
            "cuda_device": CUDA_DEVICE,
            "gpu_count": safe_faiss_gpu_count(),
        },
    }


def get_ranked_taxonomy_categories(
    url: str,
    retriever: FaissTaxonomyRetriever,
    reranker: TaxonomyReranker,
    top_k_per_query: int = TOP_K_PER_QUERY,
    fused_top_k: int = FUSED_TOP_K,
    final_top_k: int = FINAL_TOP_K,
    gpu_lock: Optional[threading.Lock] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    model_details = build_model_details(retriever, reranker)

    t_fetch0 = time.perf_counter()
    page = fetch_page_content(url, timeout=REQUEST_TIMEOUT)
    fetch_seconds = time.perf_counter() - t_fetch0

    clean_seconds = 0.0

    def run_gpu_pipeline() -> Tuple[List[RetrievalCandidate], List[RetrievalCandidate], float, float, float]:
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
        return fused_candidates, final_ranked, faiss_seconds, fuse_seconds, rerank_seconds

    if gpu_lock is not None:
        with gpu_lock:
            fused_candidates, final_ranked, faiss_seconds, fuse_seconds, rerank_seconds = run_gpu_pipeline()
    else:
        fused_candidates, final_ranked, faiss_seconds, fuse_seconds, rerank_seconds = run_gpu_pipeline()

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
        "model_details": model_details,
        "step_timings_ms": step_timings_ms,
        "fused_candidates": [candidate_to_dict(c) for c in fused_candidates],
        "final_ranked_categories": [candidate_to_dict(c) for c in final_ranked],
    }


def _process_one_url(
    idx: int,
    url: str,
    total: int,
    retriever: FaissTaxonomyRetriever,
    reranker: TaxonomyReranker,
) -> Tuple[int, Dict[str, Any]]:
    with _print_lock:
        print(f"[{idx + 1}/{total}] {url}")
    try:
        result = get_ranked_taxonomy_categories(
            url=url,
            retriever=retriever,
            reranker=reranker,
            top_k_per_query=TOP_K_PER_QUERY,
            fused_top_k=FUSED_TOP_K,
            final_top_k=FINAL_TOP_K,
            gpu_lock=_gpu_lock,
        )
        result["ok"] = True
        result["error"] = None
        return idx, result
    except Exception as e:
        return idx, {
            "ok": False,
            "error": str(e),
            "page": {"url": url},
            "model_details": build_model_details(retriever, reranker),
            "step_timings_ms": None,
            "fused_candidates": [],
            "final_ranked_categories": [],
        }


if __name__ == "__main__":
    MAX_URLS_TO_FETCH: Optional[int] = 1000

    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    faiss_info = get_faiss_runtime_info()
    print(f"faiss.get_num_gpus() = {faiss_info.gpu_count}")
    print(f"faiss module = {faiss_info.module_path}")
    print(f"faiss version = {faiss_info.version}")
    print(f"faiss GPU APIs available = {faiss_info.gpu_api_available}")
    print(f"CUDA_DEVICE = {CUDA_DEVICE}")
    print(f"MAX_URL_WORKERS = {MAX_URL_WORKERS}")

    require_cuda_device()
    if faiss_info.gpu_count <= 0 or not faiss_info.gpu_api_available:
        raise SystemExit(explain_faiss_gpu_failure(faiss_info))

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
    print("Building FAISS index over category descriptions on GPU (once)...")
    retriever.build_index(taxonomy_df)

    reranker = TaxonomyReranker(RERANK_MODEL_NAME)

    results: List[Optional[Dict[str, Any]]] = [None] * len(urls)

    workers = max(1, int(MAX_URL_WORKERS))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_meta = {
            executor.submit(
                _process_one_url,
                idx,
                url,
                len(urls),
                retriever,
                reranker,
            ): (idx, url)
            for idx, url in enumerate(urls)
        }
        for fut in as_completed(future_to_meta):
            idx, _url = future_to_meta[fut]
            out_idx, payload = fut.result()
            results[out_idx] = payload

    results_ordered: List[Dict[str, Any]] = []
    for r in results:
        if r is None:
            raise RuntimeError("Internal error: missing result for some URL indices")
        results_ordered.append(r)

    model_details = build_model_details(retriever, reranker)
    out = {
        "source_file": URL_LIST_PATH,
        "taxonomy_tsv": TSV_PATH,
        "url_count": len(urls),
        "max_url_workers": workers,
        "model_details": model_details,
        "results": results_ordered,
    }
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in results_ordered if r.get("ok"))
    print(f"Wrote {OUTPUT_JSON_PATH} ({ok}/{len(results_ordered)} succeeded)")
