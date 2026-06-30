#!/usr/bin/env python3
"""
Categorize page-content JSONL using hosted bi-encoder (FAISS) +
ColBERT MaxSim late-interaction reranking.

Pipeline:
1. Load taxonomy TSV and build two indexes at startup:
   a. Bi-encoder dense embeddings → FAISS IndexFlatIP  (retrieval stage, unchanged).
   b. ColBERT per-token embeddings → ColBERTCategoryStore  (replaces cross-encoder reranker).
      All 700 category token vectors are pre-computed once and held in memory.
2. Read input JSONL containing page-content objects.
3. For each successful record:
   a. Build content query text.
   b. Bi-encode  → FAISS top-N candidates  (unchanged).
   c. ColBERT-encode query content → per-token vectors  (ONE HTTP call per record).
   d. MaxSim score each FAISS candidate against its cached token vectors (pure numpy).
   e. Return top-K by ColBERT MaxSim score.

ColBERT MaxSim:
  score(q, d) = Σ_i  max_j  (q_i · d_j)
  q_i : L2-normalised query token vectors   [Q, dim]
  d_j : L2-normalised document token vectors [D, dim]
  (dot product == cosine when both sides are L2-normalised)

Why this is faster than a cross-encoder reranker:
  OLD: N HTTP calls to reranker per URL (one per FAISS candidate, or 1 batched call
       that still runs N joint forward passes inside the model).
  NEW: 1 HTTP call (ColBERT encode query) + cheap in-process numpy MaxSim.
       Document token vectors are pre-computed once at startup – zero HTTP cost at
       rerank time.

vLLM setup for the ColBERT model:
  vllm serve <colbert-model> --task embed --trust-remote-code
  Models known to work: colbert-ir/colbertv2.0
                        answerdotai/answerai-colbert-small-v1
  The default ColBERT serving path in this script is /pooling, which must return
  token vectors as a nested list (shape [n_tokens, dim]) under `data` or
  `embedding`, not a flat 1-D pooled embedding.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
import requests
from requests import HTTPError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout
from transformers import AutoTokenizer

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_TAXONOMY_PATH             = "taxonomy/Content_Taxonomy_3.1_2.tsv"
DEFAULT_INPUT_JSONL               = "save_url_content_1000.jsonl"
DEFAULT_OUTPUT_JSONL              = "categorize_on_hosted_models_colbert_maxsim.jsonl"

# Bi-encoder for FAISS retrieval (port 8000, unchanged from original)
DEFAULT_EMBED_API_BASE            = "http://127.0.0.1:8000"
DEFAULT_EMBED_ENDPOINT            = "/v1/embeddings"
DEFAULT_EMBED_MODEL               = "BAAI/bge-m3"
DEFAULT_EMBED_BATCH_SIZE          = 128

# ColBERT multi-vector encoder – replaces the cross-encoder reranker (port 8001)
DEFAULT_COLBERT_API_BASE          = "http://127.0.0.1:8001"
DEFAULT_COLBERT_ENDPOINT          = "/pooling"
DEFAULT_COLBERT_MODEL             = "colbert-ir/colbertv2.0"
DEFAULT_COLBERT_BATCH_SIZE        = 32          # smaller: each text → many token vecs
DEFAULT_COLBERT_QUERY_MAX_CHARS   = 1800

DEFAULT_MODELS_ENDPOINT           = "/v1/models"
DEFAULT_FAISS_TOP_K               = 10
DEFAULT_FINAL_TOP_K               = 5
DEFAULT_CONCURRENT_RECORDS        = 4
DEFAULT_REQUEST_TIMEOUT           = 180
DEFAULT_MAX_BODY_CHARS            = 4000
DEFAULT_HTTP_MAX_RETRIES          = 3
DEFAULT_HTTP_RETRY_BACKOFF_MS     = 200

_thread_local = threading.local()
_print_lock   = threading.Lock()


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
    colbert_score: float = 0.0      # ColBERT MaxSim replaces rerank_score


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


def truncate_query_text(query_text: str, max_chars: int) -> str:
    text = normalize_text(query_text)
    if len(text) <= max_chars:
        return text
    if max_chars <= 16:
        return text[:max_chars]
    return text[: max_chars - 16].rstrip() + " ... [truncated]"


def resolve_model_max_length(model_metadata: Dict[str, Any], default: int = 512) -> int:
    resolved = model_metadata.get("resolved_model")
    if isinstance(resolved, dict):
        for key in ("max_model_len", "max_model_length"):
            value = resolved.get(key)
            if value is None:
                continue
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue
    return max(1, int(default))


class TokenAwareQueryTruncator:
    """
    Truncate text by tokenizer token count instead of raw character count.

    This avoids 400 responses from the hosted model when a character-based cap
    still tokenizes to more than the model's maximum sequence length.
    """

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

        # Re-tokenise the final slice so debug output reflects what will actually be sent.
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


# ── taxonomy loading ──────────────────────────────────────────────────────────

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


# ── page record parsing ───────────────────────────────────────────────────────

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


# ── vector math ───────────────────────────────────────────────────────────────

def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """Batch L2-normalise row vectors (for bi-encoder / FAISS)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype("float32")


def normalize_token_vecs(vecs: np.ndarray, min_norm: float = 1e-6) -> np.ndarray:
    """
    L2-normalise each token vector and discard near-zero-norm rows (padding tokens).
    Input:  [n_tokens, dim]
    Output: [n_kept_tokens, dim]  (rows are unit vectors)
    """
    norms = np.linalg.norm(vecs, axis=1)        # [n_tokens]
    mask  = norms >= min_norm
    vecs, norms = vecs[mask], norms[mask]
    if vecs.shape[0] == 0:
        return np.empty((0, vecs.shape[1]), dtype="float32")
    return (vecs / norms[:, np.newaxis]).astype("float32")


def colbert_maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """
    ColBERT MaxSim score.

      score = Σ_i  max_j  (q_i · d_j)

    Both inputs must be L2-normalised per token so dot product == cosine similarity.
    query_vecs : [Q, dim]
    doc_vecs   : [D, dim]
    Returns a scalar: sum of per-query-token maximum alignments to any document token.
    """
    if query_vecs.shape[0] == 0 or doc_vecs.shape[0] == 0:
        return 0.0
    sim = query_vecs @ doc_vecs.T        # [Q, D]  – all cosine sims
    return float(sim.max(axis=1).sum())  # max over D for each query token, then sum


# ── HTTP infrastructure ───────────────────────────────────────────────────────

def _get_session() -> requests.Session:
    if not getattr(_thread_local, "session", None):
        _thread_local.session = requests.Session()
    return _thread_local.session


def _select_model_metadata(
    payload: Dict[str, Any], model_name: str
) -> Optional[Dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("id") == model_name or item.get("root") == model_name:
            return item
    for item in data:
        if not isinstance(item, dict):
            continue
        if model_name in {normalize_text(item.get("id")), normalize_text(item.get("root"))}:
            return item
    return None


class _BaseHttpClient:
    """
    Shared HTTP retry / session logic for all hosted-model clients.
    Subclasses inherit _post_with_retries, _get_with_retries, and get_model_metadata.
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        timeout: int,
        max_retries: int,
        retry_backoff_ms: int,
    ) -> None:
        self.api_base         = api_base.rstrip("/")
        self.model_name       = model_name
        self.timeout          = timeout
        self.max_retries      = max(1, int(max_retries))
        self.retry_backoff_ms = max(0, int(retry_backoff_ms))
        self.models_endpoint  = DEFAULT_MODELS_ENDPOINT

    def _post_with_retries(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = _get_session().post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp
            except HTTPError as exc:
                last_error = exc
                code = getattr(getattr(exc, "response", None), "status_code", None)
                if code is None or code < 500 or attempt >= self.max_retries:
                    raise
            except (RequestsConnectionError, RequestsTimeout, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
            if self.retry_backoff_ms > 0:
                time.sleep((self.retry_backoff_ms * attempt) / 1000.0)
        raise last_error or RuntimeError("HTTP POST failed for unknown reason")

    def _get_with_retries(self, url: str) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = _get_session().get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp
            except HTTPError as exc:
                last_error = exc
                code = getattr(getattr(exc, "response", None), "status_code", None)
                if code is None or code < 500 or attempt >= self.max_retries:
                    raise
            except (RequestsConnectionError, RequestsTimeout, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
            if self.retry_backoff_ms > 0:
                time.sleep((self.retry_backoff_ms * attempt) / 1000.0)
        raise last_error or RuntimeError("HTTP GET failed for unknown reason")

    def get_model_metadata(self) -> Dict[str, Any]:
        url = self.api_base + self.models_endpoint
        try:
            body = self._get_with_retries(url).json()
            return {
                "configured_model": self.model_name,
                "api_base": self.api_base,
                "models_endpoint": self.models_endpoint,
                "resolved_model": _select_model_metadata(body, self.model_name),
            }
        except Exception as exc:
            return {
                "configured_model": self.model_name,
                "api_base": self.api_base,
                "models_endpoint": self.models_endpoint,
                "resolved_model": None,
                "model_lookup_error": f"{type(exc).__name__}: {exc}",
            }


# ── bi-encoder client (FAISS retrieval – logic unchanged from original) ────────

class HostedEmbeddingClient(_BaseHttpClient):
    """
    Calls the embedding endpoint and returns one L2-normalised dense vector per text.
    Used exclusively for FAISS candidate retrieval.
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        endpoint: str,
        timeout: int,
        max_retries: int,
        retry_backoff_ms: int,
    ) -> None:
        super().__init__(api_base, model_name, timeout, max_retries, retry_backoff_ms)
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"

    def get_model_metadata(self) -> Dict[str, Any]:
        meta = super().get_model_metadata()
        meta["endpoint"] = self.endpoint
        return meta

    def embed_texts(self, texts: Sequence[str]) -> Tuple[np.ndarray, float]:
        if not texts:
            return np.empty((0, 0), dtype="float32"), 0.0
        payload = {
            "model": self.model_name,
            "input": list(texts),
            "encoding_format": "float",
        }
        url   = self.api_base + self.endpoint
        start = time.perf_counter()
        body  = self._post_with_retries(url, payload).json()
        data  = body.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"Embedding API returned invalid response: {body}")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in ordered]
        if any(not isinstance(v, list) for v in vectors):
            raise RuntimeError(f"Embedding API response missing embeddings: {body}")
        embeddings = np.asarray(vectors, dtype="float32")
        return normalize_l2(embeddings), round((time.perf_counter() - start) * 1000, 2)


# ── ColBERT multi-vector client (replaces HostedRerankerClient) ───────────────

class ColBERTTokenEmbeddingClient(_BaseHttpClient):
    """
    Calls a ColBERT pooling endpoint served via vLLM and returns per-token vectors.

    vLLM serving command:
      vllm serve <colbert-model> --task embed --trust-remote-code

    The /pooling response is expected to return per-token vectors as a
    **list-of-lists** (shape [n_tokens, dim]) under `data` or `embedding`.
    A flat 1-D list indicates a pooled sequence embedding response instead –
    this client raises a descriptive RuntimeError in that case.

    Returned matrices are L2-normalised per token; near-zero-norm rows (padding /
    masked tokens) are silently dropped.
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        endpoint: str,
        timeout: int,
        max_retries: int,
        retry_backoff_ms: int,
    ) -> None:
        super().__init__(api_base, model_name, timeout, max_retries, retry_backoff_ms)
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"

    def get_model_metadata(self) -> Dict[str, Any]:
        meta = super().get_model_metadata()
        meta["endpoint"] = self.endpoint
        meta["mode"]     = "per-token (ColBERT MaxSim)"
        return meta

    def embed_texts(self, texts: Sequence[str]) -> Tuple[List[np.ndarray], float]:
        """
        Returns a list of per-token matrices aligned with the input texts.
        Each matrix: shape [n_kept_tokens, dim], rows are L2-unit vectors.

        Supports both:
        - OpenAI-style embedding responses with `embedding`
        - vLLM token-embed /pooling responses with `data`
        """
        if not texts:
            return [], 0.0
        payload = {
            "model": self.model_name,
            "input": list(texts),
            "encoding_format": "float",
        }
        url   = self.api_base + self.endpoint
        start = time.perf_counter()
        body  = self._post_with_retries(url, payload).json()
        data  = body.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"ColBERT API returned invalid response: {body}")

        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        result: List[np.ndarray] = []
        for item in ordered:
            raw = item.get("embedding")
            if raw is None and isinstance(item.get("data"), list):
                raw = item.get("data")
            if not raw:
                raise RuntimeError(
                    "ColBERT API response missing token vectors under "
                    f"'embedding' or 'data': {item}"
                )
            # Detect pooled (1-D) vs per-token (2-D) response
            if not isinstance(raw[0], list):
                raise RuntimeError(
                    "ColBERT endpoint returned a 1-D (pooled) embedding instead of "
                    "per-token vectors.  Ensure the ColBERT model is served with "
                    "multi-vector output (vllm serve --task embed --trust-remote-code).\n"
                    f"  model    : {self.model_name}\n"
                    f"  endpoint : {url}"
                )
            vecs = normalize_token_vecs(np.asarray(raw, dtype="float32"))
            result.append(vecs)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return result, elapsed_ms


# ── ColBERT category store ────────────────────────────────────────────────────

class ColBERTCategoryStore:
    """
    In-memory store of pre-computed ColBERT token vectors for every taxonomy category.

    Built once at startup via build_colbert_taxonomy_store().
    At rerank time, score() / rerank() perform pure numpy MaxSim – zero HTTP calls.

    Memory estimate: 700 categories × ~30 tokens × 128 dim × 4 bytes ≈ 10 MB.
    """

    def __init__(self) -> None:
        self._store: Dict[int, np.ndarray] = {}   # unique_id → [n_tokens, dim]

    def __len__(self) -> int:
        return len(self._store)

    def add(self, unique_id: int, token_vecs: np.ndarray) -> None:
        self._store[unique_id] = token_vecs

    def score(self, query_token_vecs: np.ndarray, unique_id: int) -> float:
        """MaxSim score between query token vectors and a stored category."""
        doc_vecs = self._store.get(unique_id)
        if doc_vecs is None or doc_vecs.shape[0] == 0:
            return 0.0
        return colbert_maxsim(query_token_vecs, doc_vecs)

    def rerank(
        self,
        query_token_vecs: np.ndarray,
        candidates: Sequence[RetrievalCandidate],
    ) -> List[float]:
        """Return a ColBERT MaxSim score for each candidate, in candidate order."""
        return [self.score(query_token_vecs, c.unique_id) for c in candidates]


# ── FAISS retriever (unchanged from original) ─────────────────────────────────

class FaissTaxonomyRetriever:
    def __init__(
        self, taxonomy_df: pd.DataFrame, taxonomy_embeddings: np.ndarray
    ) -> None:
        if len(taxonomy_df) != len(taxonomy_embeddings):
            raise ValueError("taxonomy_df row count must match taxonomy_embeddings row count")
        self.df         = taxonomy_df.reset_index(drop=True)
        self.embeddings = taxonomy_embeddings
        dim             = int(taxonomy_embeddings.shape[1])
        self.index      = faiss.IndexFlatIP(dim)
        self.index.add(taxonomy_embeddings)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[RetrievalCandidate]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        scores, indices = self.index.search(query_vector.astype("float32"), int(top_k))
        results: List[RetrievalCandidate] = []
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) < 0:
                continue
            row       = self.df.iloc[int(idx)]
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


# ── index / store builders ────────────────────────────────────────────────────

def chunked(items: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(items), size):
        yield items[start: start + size]


def build_taxonomy_index(
    taxonomy_df: pd.DataFrame,
    embed_client: HostedEmbeddingClient,
    embed_batch_size: int,
) -> Tuple[FaissTaxonomyRetriever, Dict[str, Any]]:
    """Build the FAISS index from bi-encoder embeddings.  Unchanged from original."""
    descriptions   = taxonomy_df["description"].tolist()
    all_batches: List[np.ndarray] = []
    total_embed_ms = 0.0
    t0             = time.perf_counter()

    for batch in chunked(descriptions, embed_batch_size):
        vectors, batch_ms = embed_client.embed_texts(batch)
        all_batches.append(vectors)
        total_embed_ms += batch_ms

    if not all_batches:
        raise RuntimeError("No taxonomy embeddings were produced.")
    taxonomy_embeddings = np.vstack(all_batches).astype("float32")

    t_faiss0       = time.perf_counter()
    retriever      = FaissTaxonomyRetriever(taxonomy_df, taxonomy_embeddings)
    faiss_build_ms = round((time.perf_counter() - t_faiss0) * 1000, 2)

    return retriever, {
        "taxonomy_rows":         len(taxonomy_df),
        "embedding_dimension":   int(taxonomy_embeddings.shape[1]),
        "taxonomy_embedding_ms": round(total_embed_ms, 2),
        "faiss_index_build_ms":  faiss_build_ms,
        "taxonomy_setup_total_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def build_colbert_taxonomy_store(
    taxonomy_df: pd.DataFrame,
    colbert_client: ColBERTTokenEmbeddingClient,
    colbert_batch_size: int,
) -> Tuple[ColBERTCategoryStore, Dict[str, Any]]:
    """
    Pre-compute ColBERT per-token embeddings for every taxonomy category.

    Document text format mirrors the original reranker:
      "path: <path>\\ndescription: <description>"

    Returns a ColBERTCategoryStore keyed by unique_id.
    """
    texts = [
        f"path: {row['path']}\ndescription: {row['description']}"
        for _, row in taxonomy_df.iterrows()
    ]
    unique_ids = taxonomy_df["unique_id"].tolist()

    store          = ColBERTCategoryStore()
    total_embed_ms = 0.0
    total_tokens   = 0
    t0             = time.perf_counter()

    for batch_start in range(0, len(texts), colbert_batch_size):
        batch_texts = texts[batch_start: batch_start + colbert_batch_size]
        batch_ids   = unique_ids[batch_start: batch_start + colbert_batch_size]

        token_vecs_list, batch_ms = colbert_client.embed_texts(batch_texts)
        total_embed_ms += batch_ms

        for uid, token_vecs in zip(batch_ids, token_vecs_list):
            store.add(int(uid), token_vecs)
            total_tokens += token_vecs.shape[0]

    if len(store) == 0:
        raise RuntimeError("ColBERT category store is empty after building.")

    return store, {
        "taxonomy_rows":                 len(taxonomy_df),
        "total_stored_tokens":           total_tokens,
        "avg_tokens_per_category":       round(total_tokens / max(1, len(store)), 1),
        "colbert_taxonomy_embedding_ms": round(total_embed_ms, 2),
        "colbert_store_build_total_ms":  round((time.perf_counter() - t0) * 1000, 2),
    }


# ── output helpers ────────────────────────────────────────────────────────────

def candidate_to_dict(candidate: RetrievalCandidate) -> Dict[str, Any]:
    return {
        "unique_id":            candidate.unique_id,
        "path":                 candidate.path,
        "description":          candidate.description,
        "faiss_score":          round(candidate.faiss_score, 6),
        "colbert_maxsim_score": round(candidate.colbert_score, 6),
    }


# ── per-record processing ─────────────────────────────────────────────────────

def _empty_timings() -> Dict[str, int]:
    return {
        "build_query_text": 0, "content_embedding": 0, "faiss_search": 0,
        "colbert_embed": 0, "maxsim_cpu": 0, "total": 0,
    }


def process_record(
    idx: int,
    record: Dict[str, Any],
    retriever: FaissTaxonomyRetriever,
    embed_client: HostedEmbeddingClient,
    colbert_client: ColBERTTokenEmbeddingClient,
    colbert_store: ColBERTCategoryStore,
    colbert_query_truncator: TokenAwareQueryTruncator,
    faiss_top_k: int,
    final_top_k: int,
    max_body_chars: int,
    shared_model_details: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    started  = time.perf_counter()
    page_url = normalize_text(record.get("url") or record.get("input_url"))

    # ── skip non-ok input records ───────────────────────────────────────────
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

    # ── build query text ────────────────────────────────────────────────────
    t_query0      = time.perf_counter()
    query_text    = build_page_query_text(page, max_body_chars=max_body_chars)
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

    # ── bi-encode for FAISS retrieval ───────────────────────────────────────
    embed_attempts: List[int] = []
    embed_error: Optional[str] = None
    query_text_for_embedding   = query_text
    query_char_limits = [
        len(query_text),
        min(len(query_text), 2000),
        min(len(query_text), 1200),
        min(len(query_text), 800),
        min(len(query_text), 512),
    ]
    deduped_limits: List[int] = []
    for limit in query_char_limits:
        limit = max(64, int(limit))
        if limit not in deduped_limits:
            deduped_limits.append(limit)

    t_embed0       = time.perf_counter()
    query_vector: Optional[np.ndarray] = None
    embed_api_ms   = 0.0
    for char_limit in deduped_limits:
        embed_attempts.append(char_limit)
        query_text_for_embedding = truncate_query_text(query_text, char_limit)
        try:
            query_vector, batch_ms = embed_client.embed_texts([query_text_for_embedding])
            embed_api_ms += batch_ms
            embed_error   = None
            break
        except HTTPError as exc:
            embed_error = f"{type(exc).__name__}: {exc}"
            if getattr(getattr(exc, "response", None), "status_code", None) != 400:
                raise
        except Exception as exc:
            embed_error = f"{type(exc).__name__}: {exc}"
            raise

    content_embedding_ms = round((time.perf_counter() - t_embed0) * 1000, 2)

    if query_vector is None:
        return idx, {
            "ok": False,
            "input_status": record.get("status"),
            "error": embed_error or "Embedding failed after retries",
            "page": {"url": page.url, "domain": page.domain, "title": page.title},
            "step_timings_ms": {
                "build_query_text":      round(build_query_ms),
                "content_embedding":     round(content_embedding_ms),
                "content_embedding_api": round(embed_api_ms),
                "faiss_search": 0, "colbert_embed": 0, "maxsim_cpu": 0,
                "total": round((time.perf_counter() - started) * 1000),
            },
            "debug": {
                "query_text_chars": len(query_text),
                "embedding_char_limits_tried": embed_attempts,
            },
            "faiss_candidates": [],
            "final_ranked_categories": [],
        }

    # ── FAISS candidate retrieval ───────────────────────────────────────────
    t_faiss0         = time.perf_counter()
    faiss_candidates = retriever.search(query_vector=query_vector, top_k=faiss_top_k)
    faiss_search_ms  = round((time.perf_counter() - t_faiss0) * 1000, 2)

    rerank_candidates = faiss_candidates[: max(1, faiss_top_k)]

    # ── ColBERT encode query (ONE HTTP call per record) ─────────────────────
    colbert_query = colbert_query_truncator.truncate(query_text)
    colbert_text = colbert_query.text
    colbert_embed_error: Optional[str] = None
    query_token_vecs: Optional[np.ndarray] = None
    colbert_api_ms = 0.0

    t_colbert0 = time.perf_counter()
    try:
        vecs_list, batch_ms = colbert_client.embed_texts([colbert_text])
        query_token_vecs = vecs_list[0]
        colbert_api_ms += batch_ms
    except Exception as exc:
        colbert_embed_error = f"{type(exc).__name__}: {exc}"

    colbert_embed_ms = round((time.perf_counter() - t_colbert0) * 1000, 2)

    if query_token_vecs is None or query_token_vecs.shape[0] == 0:
        return idx, {
            "ok": False,
            "input_status": record.get("status"),
            "error": colbert_embed_error or "ColBERT embedding returned zero usable tokens",
            "page": {"url": page.url, "domain": page.domain, "title": page.title},
            "step_timings_ms": {
                "build_query_text":      round(build_query_ms),
                "content_embedding":     round(content_embedding_ms),
                "content_embedding_api": round(embed_api_ms),
                "faiss_search":          round(faiss_search_ms),
                "colbert_embed":         round(colbert_embed_ms),
                "colbert_embed_api":     round(colbert_api_ms),
                "maxsim_cpu": 0,
                "total": round((time.perf_counter() - started) * 1000),
            },
            "debug": {
                "query_text_chars":            len(query_text),
                "embedding_char_limits_tried": embed_attempts,
                "colbert_query_chars_used":      len(colbert_text),
                "colbert_request_chars_tried":   [len(colbert_text)],
                "colbert_query_tokens_original": colbert_query.original_tokens,
                "colbert_query_tokens_used":     colbert_query.used_tokens,
                "colbert_request_tokens_tried":  [colbert_query.used_tokens],
                "colbert_model_max_tokens":      colbert_query.max_model_tokens,
                "colbert_reserved_special_tokens": colbert_query.reserved_special_tokens,
                "colbert_content_token_limit":   colbert_query.content_token_limit,
                "colbert_token_truncated":       colbert_query.truncated,
            },
            "faiss_candidates": [candidate_to_dict(c) for c in faiss_candidates],
            "final_ranked_categories": [],
        }

    # ── MaxSim scoring (pure numpy – no HTTP call) ──────────────────────────
    t_maxsim0      = time.perf_counter()
    maxsim_scores  = colbert_store.rerank(query_token_vecs, rerank_candidates)
    maxsim_cpu_ms  = round((time.perf_counter() - t_maxsim0) * 1000, 2)

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
            "url":              page.url,
            "domain":           page.domain,
            "title":            page.title,
            "meta_description": page.meta_description,
            "headings":         page.headings[:6],
            "body_preview":     page.body_text[:500],
        },
        "step_timings_ms": {
            "build_query_text":      round(build_query_ms),
            "content_embedding":     round(content_embedding_ms),
            "content_embedding_api": round(embed_api_ms),
            "faiss_search":          round(faiss_search_ms),
            # colbert_embed = 1 HTTP call to get query token vecs (replaces N reranker calls)
            "colbert_embed":         round(colbert_embed_ms),
            "colbert_embed_api":     round(colbert_api_ms),
            # maxsim_cpu = pure numpy; should be <1 ms for 10 candidates
            "maxsim_cpu":            round(maxsim_cpu_ms),
            "total":                 round((time.perf_counter() - started) * 1000),
        },
        "model_details": {
            "embedding_model": shared_model_details["embedding_model"],
            "colbert_model": {
                **shared_model_details["colbert_model"],
                "reranking_method":  "ColBERT MaxSim",
                "query_token_count": int(query_token_vecs.shape[0]),
            },
            "faiss": {
                "backend":              "IndexFlatIP",
                "distance":             "inner_product",
                "normalized_embeddings": True,
            },
        },
        "debug": {
            "query_text_chars":            len(query_text),
            "embedding_query_chars_used":  len(query_text_for_embedding),
            "embedding_char_limits_tried": embed_attempts,
            "colbert_query_chars_used":    len(colbert_text),
            "colbert_request_chars_tried": [len(colbert_text)],
            "colbert_query_tokens_original": colbert_query.original_tokens,
            "colbert_query_tokens_used":     colbert_query.used_tokens,
            "colbert_request_tokens_tried":  [colbert_query.used_tokens],
            "colbert_model_max_tokens":      colbert_query.max_model_tokens,
            "colbert_reserved_special_tokens": colbert_query.reserved_special_tokens,
            "colbert_content_token_limit":   colbert_query.content_token_limit,
            "colbert_token_truncated":       colbert_query.truncated,
        },
        "faiss_candidates":        [candidate_to_dict(c) for c in faiss_candidates],
        "final_ranked_categories": [candidate_to_dict(c) for c in final_ranked],
    }


# ── batch processing ──────────────────────────────────────────────────────────

def process_input_file(
    records: List[Dict[str, Any]],
    retriever: FaissTaxonomyRetriever,
    embed_client: HostedEmbeddingClient,
    colbert_client: ColBERTTokenEmbeddingClient,
    colbert_store: ColBERTCategoryStore,
    colbert_query_truncator: TokenAwareQueryTruncator,
    faiss_top_k: int,
    final_top_k: int,
    concurrent_records: int,
    max_body_chars: int,
    shared_model_details: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results: List[Optional[Dict[str, Any]]] = [None] * len(records)
    workers = max(1, int(concurrent_records))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(
                process_record,
                idx, record,
                retriever, embed_client, colbert_client, colbert_store,
                colbert_query_truncator,
                faiss_top_k, final_top_k, max_body_chars,
                shared_model_details,
            ): idx
            for idx, record in enumerate(records)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                out_idx, payload = future.result()
            except Exception as exc:
                original = records[idx]
                payload  = {
                    "ok": False,
                    "input_status": original.get("status"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "page": {"url": normalize_text(
                        original.get("url") or original.get("input_url"))},
                    "step_timings_ms": None,
                    "faiss_candidates": [],
                    "final_ranked_categories": [],
                }
                out_idx = idx
            results[out_idx] = payload
            with _print_lock:
                print(f"[{out_idx + 1}/{len(records)}] "
                      f"ok={payload.get('ok')} "
                      f"url={payload.get('page', {}).get('url', '')}")

    final: List[Dict[str, Any]] = []
    for r in results:
        if r is None:
            raise RuntimeError("Internal error: missing result for one or more records")
        final.append(r)
    return final


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
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc
    return rows


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Categorize page-content JSONL via bi-encoder FAISS retrieval "
            "+ ColBERT MaxSim late-interaction reranking."
        )
    )
    p.add_argument("--taxonomy",     default=DEFAULT_TAXONOMY_PATH,
                   help="Path to taxonomy TSV.")
    p.add_argument("--input-jsonl",  default=DEFAULT_INPUT_JSONL,
                   help="Input page-content JSONL path.")
    p.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL,
                   help="Output JSONL path.")

    # ── bi-encoder (FAISS retrieval) ────────────────────────────────────────
    p.add_argument("--embed-api-base",   default=DEFAULT_EMBED_API_BASE,
                   help="Bi-encoder vLLM service base URL.")
    p.add_argument("--embed-endpoint",   default=DEFAULT_EMBED_ENDPOINT,
                   help="Bi-encoder endpoint path.")
    p.add_argument("--embed-model",      default=DEFAULT_EMBED_MODEL,
                   help="Hosted bi-encoder model name.")
    p.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE,
                   help="Batch size for taxonomy bi-encoder embedding requests.")

    # ── ColBERT (MaxSim reranker) ────────────────────────────────────────────
    p.add_argument("--colbert-api-base",  default=DEFAULT_COLBERT_API_BASE,
                   help="ColBERT vLLM service base URL  "
                        "(vllm serve <model> --task embed --trust-remote-code).")
    p.add_argument("--colbert-endpoint",  default=DEFAULT_COLBERT_ENDPOINT,
                   help="ColBERT token-embedding endpoint path (default: /pooling).")
    p.add_argument("--colbert-model",     default=DEFAULT_COLBERT_MODEL,
                   help="Hosted ColBERT model name.")
    p.add_argument("--colbert-batch-size", type=int, default=DEFAULT_COLBERT_BATCH_SIZE,
                   help="Batch size for taxonomy ColBERT pre-computation at startup.")
    p.add_argument("--colbert-query-max-chars", type=int,
                   default=DEFAULT_COLBERT_QUERY_MAX_CHARS,
                   help="Deprecated fallback knob from the old char-based truncation path.")

    # ── pipeline tuning ──────────────────────────────────────────────────────
    p.add_argument("--faiss-top-k",           type=int, default=DEFAULT_FAISS_TOP_K,
                   help="How many FAISS candidates to retrieve.")
    p.add_argument("--final-top-k",           type=int, default=DEFAULT_FINAL_TOP_K,
                   help="How many MaxSim-reranked categories to save.")
    p.add_argument("--concurrent-records",    type=int, default=DEFAULT_CONCURRENT_RECORDS,
                   help="Concurrent input records to process.")
    p.add_argument("--request-timeout",       type=int, default=DEFAULT_REQUEST_TIMEOUT,
                   help="HTTP timeout in seconds.")
    p.add_argument("--http-max-retries",      type=int, default=DEFAULT_HTTP_MAX_RETRIES,
                   help="Max retries for transient HTTP failures.")
    p.add_argument("--http-retry-backoff-ms", type=int,
                   default=DEFAULT_HTTP_RETRY_BACKOFF_MS,
                   help="Base backoff in ms between HTTP retries.")
    p.add_argument("--max-body-chars",        type=int, default=DEFAULT_MAX_BODY_CHARS,
                   help="Max body chars used in the bi-encoder query text.")
    p.add_argument("--limit",                 type=int, default=None,
                   help="Optional max number of input rows to process.")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args             = parse_args()
    run_started      = time.perf_counter()
    effective_workers = max(1, int(args.concurrent_records))

    records = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]
    if not records:
        raise SystemExit("No input records to process.")

    taxonomy_df = load_taxonomy_tsv(args.taxonomy)

    embed_client = HostedEmbeddingClient(
        api_base=args.embed_api_base,
        model_name=args.embed_model,
        endpoint=args.embed_endpoint,
        timeout=args.request_timeout,
        max_retries=args.http_max_retries,
        retry_backoff_ms=args.http_retry_backoff_ms,
    )
    colbert_client = ColBERTTokenEmbeddingClient(
        api_base=args.colbert_api_base,
        model_name=args.colbert_model,
        endpoint=args.colbert_endpoint,
        timeout=args.request_timeout,
        max_retries=args.http_max_retries,
        retry_backoff_ms=args.http_retry_backoff_ms,
    )
    shared_model_details = {
        "embedding_model": embed_client.get_model_metadata(),
        "colbert_model":   colbert_client.get_model_metadata(),
    }
    colbert_max_model_tokens = resolve_model_max_length(
        shared_model_details["colbert_model"],
        default=512,
    )
    print(f"Loading ColBERT tokenizer for token-aware truncation "
          f"(model: {args.colbert_model}, max_model_tokens={colbert_max_model_tokens})")
    colbert_query_truncator = TokenAwareQueryTruncator(
        model_name=args.colbert_model,
        max_model_tokens=colbert_max_model_tokens,
    )

    print(f"Loading taxonomy from {args.taxonomy}")
    print(f"Building FAISS index   via {args.embed_api_base}{args.embed_endpoint} "
          f"(model: {args.embed_model})")
    retriever, faiss_stats = build_taxonomy_index(
        taxonomy_df=taxonomy_df,
        embed_client=embed_client,
        embed_batch_size=args.embed_batch_size,
    )
    print(f"FAISS index ready: {faiss_stats['taxonomy_rows']} categories, "
          f"dim={faiss_stats['embedding_dimension']}, "
          f"built in {faiss_stats['taxonomy_setup_total_ms']} ms")

    print(f"Pre-computing ColBERT store via {args.colbert_api_base}{args.colbert_endpoint} "
          f"(model: {args.colbert_model})")
    colbert_store, colbert_stats = build_colbert_taxonomy_store(
        taxonomy_df=taxonomy_df,
        colbert_client=colbert_client,
        colbert_batch_size=args.colbert_batch_size,
    )
    print(f"ColBERT store ready: {len(colbert_store)} categories, "
          f"{colbert_stats['total_stored_tokens']} total tokens "
          f"(avg {colbert_stats['avg_tokens_per_category']} per category), "
          f"built in {colbert_stats['colbert_store_build_total_ms']} ms")

    results = process_input_file(
        records=records,
        retriever=retriever,
        embed_client=embed_client,
        colbert_client=colbert_client,
        colbert_store=colbert_store,
        colbert_query_truncator=colbert_query_truncator,
        faiss_top_k=args.faiss_top_k,
        final_top_k=args.final_top_k,
        concurrent_records=args.concurrent_records,
        max_body_chars=args.max_body_chars,
        shared_model_details=shared_model_details,
    )

    shared_metadata = {
        "run_metadata": {
            "input_jsonl":        args.input_jsonl,
            "output_jsonl":       args.output_jsonl,
            "taxonomy_tsv":       args.taxonomy,
            "reranking_method":   "ColBERT MaxSim",
            "concurrent_records": effective_workers,
            "worker_details": {
                "executor_type": "ThreadPoolExecutor",
                "max_workers":   effective_workers,
            },
            "faiss_top_k":             int(args.faiss_top_k),
            "final_top_k":             int(args.final_top_k),
            "embed_model":             args.embed_model,
            "colbert_model":           args.colbert_model,
            "colbert_query_truncation": {
                "mode": "token-aware",
                "max_model_tokens": colbert_max_model_tokens,
                "reserved_special_tokens": colbert_query_truncator.reserved_special_tokens,
                "content_token_limit": colbert_query_truncator.content_token_limit,
                "legacy_colbert_query_max_chars": int(args.colbert_query_max_chars),
            },
            "http_max_retries":        int(args.http_max_retries),
            "http_retry_backoff_ms":   int(args.http_retry_backoff_ms),
            "model_details":           shared_model_details,
            "taxonomy_faiss_setup":    faiss_stats,
            "taxonomy_colbert_setup":  colbert_stats,
            "run_total_ms":            None,
        }
    }

    output_rows = []
    for row in results:
        enriched = dict(shared_metadata)
        enriched.update(row)
        output_rows.append(enriched)

    total_run_ms = round((time.perf_counter() - run_started) * 1000)
    shared_metadata["run_metadata"]["run_total_ms"]       = total_run_ms
    shared_metadata["run_metadata"]["record_count"]       = len(records)
    shared_metadata["run_metadata"]["successful_records"] = sum(
        1 for r in output_rows if r.get("ok")
    )
    shared_metadata["run_metadata"]["failed_records"] = (
        len(records) - shared_metadata["run_metadata"]["successful_records"]
    )

    for row in output_rows:
        row["run_metadata"] = shared_metadata["run_metadata"]

    write_jsonl(args.output_jsonl, output_rows)

    ok_count = shared_metadata["run_metadata"]["successful_records"]
    print(
        f"Wrote {args.output_jsonl} with {ok_count}/{len(output_rows)} "
        f"successful records in {total_run_ms} ms"
    )


if __name__ == "__main__":
    main()
