#!/usr/bin/env python3
"""
CPU-only categorization pipeline for page-content JSONL.

Differences from categorize_on_hosted_models_colbert_maxsim.py:
- No vLLM / hosted-model calls.
- Local CPU inference only.
- ProcessPoolExecutor instead of ThreadPoolExecutor.
- Each worker process builds and owns its own taxonomy FAISS index and ColBERT store.
- No batching during per-record processing; each worker handles one URL at a time.

The goal of this variant is to test whether fully local CPU inference with
multiple worker processes can improve concurrent URL processing throughput on
this machine.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoTokenizer, BertModel


# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_TAXONOMY_PATH             = "taxonomy/Content_Taxonomy_3.1_2.tsv"
DEFAULT_INPUT_JSONL               = "save_url_content_1000.jsonl"
DEFAULT_OUTPUT_JSONL              = "categorize_on_hosted_models_colbert_maxsim_cpu.jsonl"

DEFAULT_EMBED_MODEL               = "BAAI/bge-m3"
DEFAULT_COLBERT_MODEL             = "colbert-ir/colbertv2.0"

DEFAULT_FAISS_TOP_K               = 10
DEFAULT_FINAL_TOP_K               = 5
DEFAULT_CONCURRENT_RECORDS        = 4
DEFAULT_MAX_BODY_CHARS            = 4000

DEFAULT_TORCH_NUM_THREADS         = 1
DEFAULT_TORCH_NUM_INTEROP_THREADS = 1
DEFAULT_FAISS_NUM_THREADS         = 1


_WORKER_STATE: Dict[str, Any] = {}


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class PageContent:
    url: str
    domain: str
    title: str
    meta_description: str
    headings: List[str]
    body_text: str


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
    colbert_score: float = 0.0


@dataclass
class TokenTruncationResult:
    text: str
    original_tokens: int
    used_tokens: int
    max_model_tokens: int
    reserved_special_tokens: int
    content_token_limit: int
    truncated: bool


# ── text utilities ────────────────────────────────────────────────────────────

def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip())


def build_path_string(row: pd.Series) -> str:
    levels = [row.get("tier1", ""), row.get("tier2", ""),
              row.get("tier3", ""), row.get("tier4", "")]
    return " > ".join(part for part in (normalize_text(x) for x in levels) if part)


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


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype("float32")


def normalize_token_vecs(vecs: np.ndarray, min_norm: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1)
    mask = norms >= min_norm
    vecs = vecs[mask]
    norms = norms[mask]
    if vecs.shape[0] == 0:
        return np.empty((0, vecs.shape[1]), dtype="float32")
    return (vecs / norms[:, np.newaxis]).astype("float32")


def colbert_maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    if query_vecs.shape[0] == 0 or doc_vecs.shape[0] == 0:
        return 0.0
    sim = query_vecs @ doc_vecs.T
    return float(sim.max(axis=1).sum())


def resolve_local_model_path(model_name: str, local_files_only: bool) -> str:
    if os.path.isdir(model_name):
        return model_name
    return snapshot_download(
        repo_id=model_name,
        local_files_only=local_files_only,
    )


class TokenAwareQueryTruncator:
    def __init__(self, model_name: str, max_model_tokens: int, local_files_only: bool) -> None:
        model_path = resolve_local_model_path(model_name, local_files_only=local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model_name = model_name
        self.max_model_tokens = max(1, int(max_model_tokens))
        self.reserved_special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        self.content_token_limit = max(1, self.max_model_tokens - self.reserved_special_tokens)

    def truncate(self, text: str) -> TokenTruncationResult:
        normalized = normalize_text(text)
        if not normalized:
            return TokenTruncationResult(
                text="",
                original_tokens=0,
                used_tokens=0,
                max_model_tokens=self.max_model_tokens,
                reserved_special_tokens=self.reserved_special_tokens,
                content_token_limit=self.content_token_limit,
                truncated=False,
            )

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
            self.tokenizer(truncated_text, add_special_tokens=False, verbose=False)["input_ids"]
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


def resolve_local_model_max_length(model_name: str, default: int, local_files_only: bool) -> int:
    model_path = resolve_local_model_path(model_name, local_files_only=local_files_only)
    try:
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        for key in ("max_model_len", "max_position_embeddings"):
            value = getattr(config, key, None)
            if value is not None:
                return max(1, int(value))
    except Exception:
        pass
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        value = getattr(tokenizer, "model_max_length", None)
        if value and int(value) < 10**12:
            return max(1, int(value))
    except Exception:
        pass
    return max(1, int(default))


# ── taxonomy loading / page parsing ───────────────────────────────────────────

def load_taxonomy_tsv(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    normalized_cols = {col: re.sub(r"\s+", " ", col.strip().lower()) for col in df.columns}
    df = df.rename(columns=normalized_cols)
    rename_map = {
        "unique id": "unique_id", "parent": "parent_id",
        "tier 1": "tier1", "tier 2": "tier2", "tier 3": "tier3", "tier 4": "tier4",
        "description": "description",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    required = ["unique_id", "parent_id", "tier1", "tier2", "description"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in taxonomy TSV: {missing}")
    for col in ["tier1", "tier2", "tier3", "tier4", "description"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(normalize_text)
    df["unique_id"] = pd.to_numeric(df["unique_id"], errors="coerce")
    df["parent_id"] = pd.to_numeric(df["parent_id"], errors="coerce")
    df = df[df["unique_id"].notna()].copy()
    df["unique_id"] = df["unique_id"].astype(int)
    df["path"] = df.apply(build_path_string, axis=1)
    return df[
        ["unique_id", "parent_id", "tier1", "tier2", "tier3", "tier4", "path", "description"]
    ].reset_index(drop=True)


def parse_page_record(record: Dict[str, Any]) -> PageContent:
    return PageContent(
        url=normalize_text(record.get("url") or record.get("input_url")),
        domain=normalize_text(record.get("domain")),
        title=normalize_text(record.get("title")),
        meta_description=normalize_text(record.get("meta_description")),
        headings=[normalize_text(x) for x in record.get("headings", []) if normalize_text(x)],
        body_text=normalize_text(record.get("body_text")),
    )


# ── local model wrappers ──────────────────────────────────────────────────────

class LocalBiEncoder:
    def __init__(self, model_name: str, local_files_only: bool) -> None:
        self.model_name = model_name
        self.model_path = resolve_local_model_path(model_name, local_files_only=local_files_only)
        self.model = SentenceTransformer(
            self.model_path,
            device="cpu",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "configured_model": self.model_name,
            "mode": "local-cpu",
            "backend": "sentence-transformers",
            "embedding_dimension": self.dimension,
        }

    def embed_text(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vec, dtype="float32")


class LocalColBERTEncoder:
    def __init__(self, model_name: str, local_files_only: bool) -> None:
        self.model_name = model_name
        self.model_path = resolve_local_model_path(model_name, local_files_only=local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=local_files_only,
        )
        self.bert = BertModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        weights_path = os.path.join(self.model_path, "model.safetensors")
        if not weights_path:
            raise RuntimeError(f"Could not resolve model.safetensors for {model_name}")
        state = load_safetensors(weights_path, device="cpu")
        linear_weight = state["linear.weight"].to(dtype=self.bert.dtype)
        self.projection = torch.nn.Linear(
            self.bert.config.hidden_size,
            linear_weight.shape[0],
            bias=False,
        )
        with torch.no_grad():
            self.projection.weight.copy_(linear_weight)
        self.bert.eval()
        self.projection.eval()
        self.dimension = int(linear_weight.shape[0])
        self.max_model_tokens = int(getattr(self.bert.config, "max_position_embeddings", 512))

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "configured_model": self.model_name,
            "mode": "local-cpu",
            "backend": "transformers+BertModel",
            "embedding_dimension": self.dimension,
            "max_model_len": self.max_model_tokens,
            "reranking_method": "ColBERT MaxSim",
        }

    def embed_text(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_tokens,
            return_special_tokens_mask=True,
        )
        model_inputs = {
            key: value
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.inference_mode():
            outputs = self.bert(**model_inputs)
            projected = self.projection(outputs.last_hidden_state)
        mask = encoded["attention_mask"].bool()
        special_tokens_mask = encoded.get("special_tokens_mask")
        if special_tokens_mask is not None:
            mask = mask & (~special_tokens_mask.bool())
        vecs = projected[0][mask[0]].cpu().numpy().astype("float32")
        if vecs.size == 0:
            return np.empty((0, self.dimension), dtype="float32")
        return normalize_token_vecs(vecs)


# ── in-memory indexes ─────────────────────────────────────────────────────────

class ColBERTCategoryStore:
    def __init__(self) -> None:
        self._store: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._store)

    def add(self, unique_id: int, token_vecs: np.ndarray) -> None:
        self._store[unique_id] = token_vecs

    def score(self, query_token_vecs: np.ndarray, unique_id: int) -> float:
        doc_vecs = self._store.get(unique_id)
        if doc_vecs is None or doc_vecs.shape[0] == 0:
            return 0.0
        return colbert_maxsim(query_token_vecs, doc_vecs)

    def rerank(
        self,
        query_token_vecs: np.ndarray,
        candidates: Sequence[RetrievalCandidate],
    ) -> List[float]:
        return [self.score(query_token_vecs, c.unique_id) for c in candidates]


class FaissTaxonomyRetriever:
    def __init__(self, taxonomy_df: pd.DataFrame, taxonomy_embeddings: np.ndarray) -> None:
        if len(taxonomy_df) != len(taxonomy_embeddings):
            raise ValueError("taxonomy_df row count must match taxonomy_embeddings row count")
        self.df = taxonomy_df.reset_index(drop=True)
        self.embeddings = taxonomy_embeddings
        dim = int(taxonomy_embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(taxonomy_embeddings)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[RetrievalCandidate]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        scores, indices = self.index.search(query_vector.astype("float32"), int(top_k))
        results: List[RetrievalCandidate] = []
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) < 0:
                continue
            row = self.df.iloc[int(idx)]
            parent_id = None if pd.isna(row["parent_id"]) else int(row["parent_id"])
            results.append(RetrievalCandidate(
                unique_id=int(row["unique_id"]),
                parent_id=parent_id,
                tier1=row["tier1"],
                tier2=row["tier2"],
                tier3=row["tier3"],
                tier4=row["tier4"],
                path=row["path"],
                description=row["description"],
                faiss_score=float(score),
            ))
        return results


# ── local taxonomy builders ───────────────────────────────────────────────────

def build_taxonomy_index_local(
    taxonomy_df: pd.DataFrame,
    embed_model: LocalBiEncoder,
) -> Tuple[FaissTaxonomyRetriever, Dict[str, Any]]:
    descriptions = taxonomy_df["description"].tolist()
    vectors: List[np.ndarray] = []
    total_embed_ms = 0.0
    t0 = time.perf_counter()

    for text in descriptions:
        started = time.perf_counter()
        vectors.append(embed_model.embed_text(text))
        total_embed_ms += (time.perf_counter() - started) * 1000

    if not vectors:
        raise RuntimeError("No taxonomy embeddings were produced.")
    taxonomy_embeddings = np.vstack(vectors).astype("float32")

    t_faiss0 = time.perf_counter()
    retriever = FaissTaxonomyRetriever(taxonomy_df, taxonomy_embeddings)
    faiss_build_ms = round((time.perf_counter() - t_faiss0) * 1000, 2)
    return retriever, {
        "taxonomy_rows": len(taxonomy_df),
        "embedding_dimension": int(taxonomy_embeddings.shape[1]),
        "taxonomy_embedding_ms": round(total_embed_ms, 2),
        "faiss_index_build_ms": faiss_build_ms,
        "taxonomy_setup_total_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def build_colbert_taxonomy_store_local(
    taxonomy_df: pd.DataFrame,
    colbert_model: LocalColBERTEncoder,
) -> Tuple[ColBERTCategoryStore, Dict[str, Any]]:
    texts = [
        f"path: {row['path']}\ndescription: {row['description']}"
        for _, row in taxonomy_df.iterrows()
    ]
    unique_ids = taxonomy_df["unique_id"].tolist()

    store = ColBERTCategoryStore()
    total_embed_ms = 0.0
    total_tokens = 0
    t0 = time.perf_counter()

    for uid, text in zip(unique_ids, texts):
        started = time.perf_counter()
        token_vecs = colbert_model.embed_text(text)
        total_embed_ms += (time.perf_counter() - started) * 1000
        store.add(int(uid), token_vecs)
        total_tokens += token_vecs.shape[0]

    if len(store) == 0:
        raise RuntimeError("ColBERT category store is empty after building.")
    return store, {
        "taxonomy_rows": len(taxonomy_df),
        "total_stored_tokens": total_tokens,
        "avg_tokens_per_category": round(total_tokens / max(1, len(store)), 1),
        "colbert_taxonomy_embedding_ms": round(total_embed_ms, 2),
        "colbert_store_build_total_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ── worker initialization ─────────────────────────────────────────────────────

def _init_worker(config: Dict[str, Any]) -> None:
    global _WORKER_STATE
    worker_started = time.perf_counter()

    torch.set_num_threads(max(1, int(config["torch_num_threads"])))
    torch.set_num_interop_threads(max(1, int(config["torch_num_interop_threads"])))
    faiss.omp_set_num_threads(max(1, int(config["faiss_num_threads"])))

    taxonomy_df = load_taxonomy_tsv(config["taxonomy"])
    embed_model = LocalBiEncoder(
        model_name=config["embed_model"],
        local_files_only=bool(config["local_files_only"]),
    )
    colbert_model = LocalColBERTEncoder(
        model_name=config["colbert_model"],
        local_files_only=bool(config["local_files_only"]),
    )

    embed_max_tokens = resolve_local_model_max_length(
        config["embed_model"],
        default=8192,
        local_files_only=bool(config["local_files_only"]),
    )
    colbert_max_tokens = resolve_local_model_max_length(
        config["colbert_model"],
        default=512,
        local_files_only=bool(config["local_files_only"]),
    )
    embed_truncator = TokenAwareQueryTruncator(
        model_name=config["embed_model"],
        max_model_tokens=embed_max_tokens,
        local_files_only=bool(config["local_files_only"]),
    )
    colbert_truncator = TokenAwareQueryTruncator(
        model_name=config["colbert_model"],
        max_model_tokens=colbert_max_tokens,
        local_files_only=bool(config["local_files_only"]),
    )

    retriever, faiss_stats = build_taxonomy_index_local(taxonomy_df, embed_model)
    colbert_store, colbert_stats = build_colbert_taxonomy_store_local(taxonomy_df, colbert_model)

    _WORKER_STATE = {
        "config": config,
        "taxonomy_df": taxonomy_df,
        "embed_model": embed_model,
        "colbert_model": colbert_model,
        "embed_truncator": embed_truncator,
        "colbert_truncator": colbert_truncator,
        "retriever": retriever,
        "colbert_store": colbert_store,
        "faiss_stats": faiss_stats,
        "colbert_stats": colbert_stats,
        "shared_model_details": {
            "embedding_model": {
                **embed_model.get_model_metadata(),
                "max_model_len": embed_max_tokens,
            },
            "colbert_model": {
                **colbert_model.get_model_metadata(),
                "max_model_len": colbert_max_tokens,
            },
        },
    }

    print(
        f"[worker pid={os.getpid()}] ready: "
        f"taxonomy_rows={faiss_stats['taxonomy_rows']} "
        f"faiss_dim={faiss_stats['embedding_dimension']} "
        f"stored_tokens={colbert_stats['total_stored_tokens']}"
    )
    ready_queue = config.get("worker_ready_queue")
    if ready_queue is not None:
        ready_queue.put({
            "pid": os.getpid(),
            "worker_init_ms": round((time.perf_counter() - worker_started) * 1000),
            "taxonomy_rows": int(faiss_stats["taxonomy_rows"]),
            "faiss_dim": int(faiss_stats["embedding_dimension"]),
            "stored_tokens": int(colbert_stats["total_stored_tokens"]),
        })


def _worker_ping() -> int:
    return os.getpid()


# ── output helpers ────────────────────────────────────────────────────────────

def candidate_to_dict(candidate: RetrievalCandidate) -> Dict[str, Any]:
    return {
        "unique_id": candidate.unique_id,
        "path": candidate.path,
        "description": candidate.description,
        "faiss_score": round(candidate.faiss_score, 6),
        "colbert_maxsim_score": round(candidate.colbert_score, 6),
    }


def _empty_timings() -> Dict[str, int]:
    return {
        "build_query_text": 0,
        "content_embedding": 0,
        "faiss_search": 0,
        "colbert_embed": 0,
        "maxsim_cpu": 0,
        "total": 0,
    }


# ── per-record processing ─────────────────────────────────────────────────────

def process_record_cpu(
    idx: int,
    record: Dict[str, Any],
    faiss_top_k: int,
    final_top_k: int,
    max_body_chars: int,
) -> Tuple[int, Dict[str, Any]]:
    started = time.perf_counter()
    state = _WORKER_STATE
    page_url = normalize_text(record.get("url") or record.get("input_url"))

    if normalize_text(record.get("status")).lower() != "ok":
        return idx, {
            "ok": False,
            "input_status": record.get("status"),
            "error": record.get("message") or "Input record is not status=ok",
            "page": {"url": page_url},
            "step_timings_ms": _empty_timings(),
            "faiss_candidates": [],
            "final_ranked_categories": [],
        }

    page = parse_page_record(record)

    t_query0 = time.perf_counter()
    query_text = build_page_query_text(page, max_body_chars=max_body_chars)
    build_query_ms = round((time.perf_counter() - t_query0) * 1000, 2)

    if not query_text:
        return idx, {
            "ok": False,
            "input_status": record.get("status"),
            "error": "Input record has no usable text for embedding",
            "page": {"url": page.url, "domain": page.domain, "title": page.title},
            "step_timings_ms": {**_empty_timings(), "build_query_text": round(build_query_ms),
                                "total": round((time.perf_counter() - started) * 1000)},
            "faiss_candidates": [],
            "final_ranked_categories": [],
        }

    t_embed0 = time.perf_counter()
    embed_query = state["embed_truncator"].truncate(query_text)
    try:
        query_vector = state["embed_model"].embed_text(embed_query.text)
    except Exception as exc:
        return idx, {
            "ok": False,
            "input_status": record.get("status"),
            "error": f"{type(exc).__name__}: {exc}",
            "page": {"url": page.url, "domain": page.domain, "title": page.title},
            "step_timings_ms": {
                "build_query_text": round(build_query_ms),
                "content_embedding": round((time.perf_counter() - t_embed0) * 1000),
                "faiss_search": 0,
                "colbert_embed": 0,
                "maxsim_cpu": 0,
                "total": round((time.perf_counter() - started) * 1000),
            },
            "debug": {
                "query_text_chars": len(query_text),
                "embedding_query_chars_used": len(embed_query.text),
                "embedding_query_tokens_original": embed_query.original_tokens,
                "embedding_query_tokens_used": embed_query.used_tokens,
                "embedding_model_max_tokens": embed_query.max_model_tokens,
            },
            "faiss_candidates": [],
            "final_ranked_categories": [],
        }
    content_embedding_ms = round((time.perf_counter() - t_embed0) * 1000, 2)

    t_faiss0 = time.perf_counter()
    faiss_candidates = state["retriever"].search(query_vector=query_vector, top_k=faiss_top_k)
    faiss_search_ms = round((time.perf_counter() - t_faiss0) * 1000, 2)
    rerank_candidates = faiss_candidates[: max(1, faiss_top_k)]

    t_colbert0 = time.perf_counter()
    colbert_query = state["colbert_truncator"].truncate(query_text)
    try:
        query_token_vecs = state["colbert_model"].embed_text(colbert_query.text)
    except Exception as exc:
        return idx, {
            "ok": False,
            "input_status": record.get("status"),
            "error": f"{type(exc).__name__}: {exc}",
            "page": {"url": page.url, "domain": page.domain, "title": page.title},
            "step_timings_ms": {
                "build_query_text": round(build_query_ms),
                "content_embedding": round(content_embedding_ms),
                "faiss_search": round(faiss_search_ms),
                "colbert_embed": round((time.perf_counter() - t_colbert0) * 1000),
                "maxsim_cpu": 0,
                "total": round((time.perf_counter() - started) * 1000),
            },
            "debug": {
                "query_text_chars": len(query_text),
                "embedding_query_chars_used": len(embed_query.text),
                "embedding_query_tokens_original": embed_query.original_tokens,
                "embedding_query_tokens_used": embed_query.used_tokens,
                "embedding_model_max_tokens": embed_query.max_model_tokens,
                "colbert_query_chars_used": len(colbert_query.text),
                "colbert_query_tokens_original": colbert_query.original_tokens,
                "colbert_query_tokens_used": colbert_query.used_tokens,
                "colbert_model_max_tokens": colbert_query.max_model_tokens,
            },
            "faiss_candidates": [candidate_to_dict(c) for c in faiss_candidates],
            "final_ranked_categories": [],
        }
    colbert_embed_ms = round((time.perf_counter() - t_colbert0) * 1000, 2)

    if query_token_vecs.shape[0] == 0:
        return idx, {
            "ok": False,
            "input_status": record.get("status"),
            "error": "ColBERT embedding returned zero usable tokens",
            "page": {"url": page.url, "domain": page.domain, "title": page.title},
            "step_timings_ms": {
                "build_query_text": round(build_query_ms),
                "content_embedding": round(content_embedding_ms),
                "faiss_search": round(faiss_search_ms),
                "colbert_embed": round(colbert_embed_ms),
                "maxsim_cpu": 0,
                "total": round((time.perf_counter() - started) * 1000),
            },
            "debug": {
                "query_text_chars": len(query_text),
                "embedding_query_chars_used": len(embed_query.text),
                "embedding_query_tokens_original": embed_query.original_tokens,
                "embedding_query_tokens_used": embed_query.used_tokens,
                "embedding_model_max_tokens": embed_query.max_model_tokens,
                "colbert_query_chars_used": len(colbert_query.text),
                "colbert_query_tokens_original": colbert_query.original_tokens,
                "colbert_query_tokens_used": colbert_query.used_tokens,
                "colbert_model_max_tokens": colbert_query.max_model_tokens,
            },
            "faiss_candidates": [candidate_to_dict(c) for c in faiss_candidates],
            "final_ranked_categories": [],
        }

    t_maxsim0 = time.perf_counter()
    maxsim_scores = state["colbert_store"].rerank(query_token_vecs, rerank_candidates)
    maxsim_cpu_ms = round((time.perf_counter() - t_maxsim0) * 1000, 2)

    for candidate, score in zip(rerank_candidates, maxsim_scores):
        candidate.colbert_score = score

    final_ranked = sorted(
        rerank_candidates,
        key=lambda c: (c.colbert_score, c.faiss_score),
        reverse=True,
    )[:final_top_k]

    return idx, {
        "ok": True,
        "input_status": record.get("status"),
        "page": {
            "url": page.url,
            "domain": page.domain,
            "title": page.title,
            "meta_description": page.meta_description,
            "headings": page.headings[:6],
            "body_preview": page.body_text[:500],
        },
        "step_timings_ms": {
            "build_query_text": round(build_query_ms),
            "content_embedding": round(content_embedding_ms),
            "faiss_search": round(faiss_search_ms),
            "colbert_embed": round(colbert_embed_ms),
            "maxsim_cpu": round(maxsim_cpu_ms),
            "total": round((time.perf_counter() - started) * 1000),
        },
        "model_details": {
            "embedding_model": state["shared_model_details"]["embedding_model"],
            "colbert_model": {
                **state["shared_model_details"]["colbert_model"],
                "query_token_count": int(query_token_vecs.shape[0]),
            },
            "faiss": {
                "backend": "IndexFlatIP",
                "distance": "inner_product",
                "normalized_embeddings": True,
            },
        },
        "debug": {
            "worker_pid": os.getpid(),
            "query_text_chars": len(query_text),
            "embedding_query_chars_used": len(embed_query.text),
            "embedding_query_tokens_original": embed_query.original_tokens,
            "embedding_query_tokens_used": embed_query.used_tokens,
            "embedding_model_max_tokens": embed_query.max_model_tokens,
            "colbert_query_chars_used": len(colbert_query.text),
            "colbert_query_tokens_original": colbert_query.original_tokens,
            "colbert_query_tokens_used": colbert_query.used_tokens,
            "colbert_model_max_tokens": colbert_query.max_model_tokens,
        },
        "faiss_candidates": [candidate_to_dict(c) for c in faiss_candidates],
        "final_ranked_categories": [candidate_to_dict(c) for c in final_ranked],
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU-only categorization via local BGE retrieval + local ColBERT MaxSim."
    )
    p.add_argument("--taxonomy", default=DEFAULT_TAXONOMY_PATH, help="Path to taxonomy TSV.")
    p.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL, help="Input page-content JSONL.")
    p.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path.")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Local BGE model name.")
    p.add_argument("--colbert-model", default=DEFAULT_COLBERT_MODEL, help="Local ColBERT model.")
    p.add_argument("--faiss-top-k", type=int, default=DEFAULT_FAISS_TOP_K,
                   help="How many FAISS candidates to retrieve.")
    p.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K,
                   help="How many MaxSim-reranked categories to keep.")
    p.add_argument("--concurrent-records", type=int, default=DEFAULT_CONCURRENT_RECORDS,
                   help="Number of worker processes.")
    p.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY_CHARS,
                   help="Max body chars used in the query text.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional max number of input rows to process.")
    p.add_argument("--torch-num-threads", type=int, default=DEFAULT_TORCH_NUM_THREADS,
                   help="Torch intra-op threads per worker process.")
    p.add_argument("--torch-num-interop-threads", type=int,
                   default=DEFAULT_TORCH_NUM_INTEROP_THREADS,
                   help="Torch inter-op threads per worker process.")
    p.add_argument("--faiss-num-threads", type=int, default=DEFAULT_FAISS_NUM_THREADS,
                   help="FAISS OpenMP threads per worker process.")
    p.add_argument("--allow-model-downloads", action="store_true",
                   help="Allow downloading models from Hugging Face if not cached locally.")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    effective_workers = max(1, int(args.concurrent_records))

    records = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]
    if not records:
        raise SystemExit("No input records to process.")

    taxonomy_df = load_taxonomy_tsv(args.taxonomy)

    worker_init_config = {
        "taxonomy": args.taxonomy,
        "embed_model": args.embed_model,
        "colbert_model": args.colbert_model,
        "local_files_only": not bool(args.allow_model_downloads),
        "torch_num_threads": int(args.torch_num_threads),
        "torch_num_interop_threads": int(args.torch_num_interop_threads),
        "faiss_num_threads": int(args.faiss_num_threads),
    }

    results: List[Optional[Dict[str, Any]]] = [None] * len(records)
    ctx = mp.get_context("spawn")
    worker_ready_queue = ctx.Queue()
    worker_init_config["worker_ready_queue"] = worker_ready_queue
    bootstrap_started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(worker_init_config,),
    ) as executor:
        warmup_futures = [executor.submit(_worker_ping) for _ in range(effective_workers)]
        ready_workers: Dict[int, Dict[str, Any]] = {}
        while len(ready_workers) < effective_workers:
            msg = worker_ready_queue.get()
            ready_workers[int(msg["pid"])] = msg
        for future in warmup_futures:
            future.result()

        bootstrap_total_ms = round((time.perf_counter() - bootstrap_started) * 1000)
        processing_started = time.perf_counter()
        future_to_idx = {
            executor.submit(
                process_record_cpu,
                idx,
                record,
                args.faiss_top_k,
                args.final_top_k,
                args.max_body_chars,
            ): idx
            for idx, record in enumerate(records)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                out_idx, payload = future.result()
            except Exception as exc:
                original = records[idx]
                payload = {
                    "ok": False,
                    "input_status": original.get("status"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "page": {"url": normalize_text(original.get("url") or original.get("input_url"))},
                    "step_timings_ms": None,
                    "faiss_candidates": [],
                    "final_ranked_categories": [],
                }
                out_idx = idx
            results[out_idx] = payload
            print(
                f"[{out_idx + 1}/{len(records)}] "
                f"ok={payload.get('ok')} "
                f"url={payload.get('page', {}).get('url', '')}"
            )
        processing_only_ms = round((time.perf_counter() - processing_started) * 1000)

    output_rows: List[Dict[str, Any]] = []
    for row in results:
        if row is None:
            raise RuntimeError("Internal error: missing result for one or more records")
        output_rows.append(row)

    total_run_ms = round((time.perf_counter() - run_started) * 1000)
    shared_metadata = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "taxonomy_tsv": args.taxonomy,
        "resource_mode": "local_cpu",
        "reranking_method": "ColBERT MaxSim",
        "record_count": len(records),
        "successful_records": sum(1 for r in output_rows if r.get("ok")),
        "failed_records": sum(1 for r in output_rows if not r.get("ok")),
        "concurrent_records": effective_workers,
        "worker_details": {
            "executor_type": "ProcessPoolExecutor",
            "max_workers": effective_workers,
            "mp_start_method": "spawn",
            "taxonomy_replication": "per_worker_process",
            "torch_num_threads": int(args.torch_num_threads),
            "torch_num_interop_threads": int(args.torch_num_interop_threads),
            "faiss_num_threads": int(args.faiss_num_threads),
        },
        "faiss_top_k": int(args.faiss_top_k),
        "final_top_k": int(args.final_top_k),
        "embed_model": args.embed_model,
        "colbert_model": args.colbert_model,
        "model_downloads_allowed": bool(args.allow_model_downloads),
        "taxonomy_rows": int(len(taxonomy_df)),
        "worker_bootstrap_total_ms": bootstrap_total_ms,
        "processing_only_ms": processing_only_ms,
        "avg_worker_init_ms": round(
            sum(msg["worker_init_ms"] for msg in ready_workers.values()) / max(1, len(ready_workers))
        ),
        "max_worker_init_ms": max(msg["worker_init_ms"] for msg in ready_workers.values()),
        "run_total_ms": total_run_ms,
    }

    for row in output_rows:
        row["run_metadata"] = shared_metadata

    write_jsonl(args.output_jsonl, output_rows)
    print(
        f"Wrote {args.output_jsonl} with "
        f"{shared_metadata['successful_records']}/{len(output_rows)} "
        f"successful records in {total_run_ms} ms"
    )
    print(
        f"Bootstrap={bootstrap_total_ms} ms, "
        f"processing_only={processing_only_ms} ms, "
        f"avg_worker_init={shared_metadata['avg_worker_init_ms']} ms, "
        f"max_worker_init={shared_metadata['max_worker_init_ms']} ms"
    )


if __name__ == "__main__":
    main()
