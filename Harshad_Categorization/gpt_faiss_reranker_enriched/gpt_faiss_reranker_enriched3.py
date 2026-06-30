import json
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import faiss
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from sentence_transformers import SentenceTransformer, CrossEncoder


TSV_PATH = "taxonomy/Content_Taxonomy_3.1_2.tsv"
#URL_LIST_PATH = "new_urls.txt"
URL_LIST_PATH = "adserver_1000_urls.txt"

OUTPUT_JSON_PATH = "gpt_faiss_reranker_enriched3_adserver_1000.json"

# Dense retriever model
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Reranker model
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" # did not work well

# TOP_K_PER_QUERY = 25
TOP_K_PER_QUERY = 10
# FUSED_TOP_K = 40
FUSED_TOP_K = 10
FINAL_TOP_K = 5
REQUEST_TIMEOUT = 15


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
class PageContent:
    url: str
    domain: str
    title: str
    meta_description: str
    headings: List[str]
    body_text: str


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
    """
    Loads the new TSV with columns:
    Unique ID, Parent, Tier 1, Tier 2, Tier 3, Tier 4, Description

    This loader is robust to:
    - spaces in column names
    - tabs vs minor formatting variation
    - blank parent values
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")

    # Normalize column names
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

    # Optional tier columns
    if "tier3" not in df.columns:
        df["tier3"] = ""
    if "tier4" not in df.columns:
        df["tier4"] = ""

    # Clean fields
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


class FaissTaxonomyRetriever:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.df: Optional[pd.DataFrame] = None
        self.index: Optional[faiss.Index] = None

    def build_index(self, taxonomy_df: pd.DataFrame) -> None:
        self.df = taxonomy_df.copy()

        # Use ONLY description for semantic retrieval
        docs = self.df["description"].tolist()

        embeddings = self.model.encode(
            docs,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query_text: str, top_k: int = TOP_K_PER_QUERY) -> List[RetrievalCandidate]:
        if self.df is None or self.index is None:
            raise RuntimeError("Index not built.")

        query_vec = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
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
        self.model = CrossEncoder(model_name)

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

        # Use ONLY category description for reranking
        pairs = [(c.description, query) for c in candidates]

        scores = self.model.predict(pairs)

        ranked = []
        for cand, score in zip(candidates, scores):
            cand.rerank_score = float(score)
            ranked.append(cand)

        ranked.sort(key=lambda x: (x.rerank_score, x.fused_score, x.faiss_score), reverse=True)
        return ranked[:final_top_k]


def fetch_url_html(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def get_meta_content(soup: BeautifulSoup, attrs: Dict[str, str]) -> str:
    tag = soup.find("meta", attrs=attrs)
    if tag and tag.get("content"):
        return normalize_text(tag.get("content"))
    return ""


def remove_noise_nodes(soup: BeautifulSoup) -> None:
    selectors = [
        "script", "style", "noscript", "svg", "iframe", "canvas", "form",
        "nav", "footer", "header", "aside"
    ]
    for selector in selectors:
        for tag in soup.select(selector):
            tag.decompose()

    for tag in soup.find_all(True):
        if tag is None or getattr(tag, "attrs", None) is None:
            continue
        classes = " ".join(tag.get("class", [])) if tag.get("class") else ""
        id_ = tag.get("id", "")
        marker = f"{classes} {id_}".lower()
        if any(x in marker for x in [
            "cookie", "consent", "newsletter", "subscribe", "promo",
            "advert", "ad-", "ads", "banner", "breadcrumb", "related",
            "social-share", "share", "outbrain", "taboola", "recommended"
        ]):
            tag.decompose()


def extract_best_text_container(soup: BeautifulSoup) -> Optional[Tag]:
    priority_selectors = [
        "article",
        "main",
        "[role='main']",
        ".article",
        ".post",
        ".entry-content",
        ".article-content",
        ".post-content",
        ".story-body",
        ".content",
    ]
    for selector in priority_selectors:
        node = soup.select_one(selector)
        if node:
            return node

    body = soup.body
    if not body:
        return None

    best_node = None
    best_score = -1

    for node in body.find_all(["div", "section"], recursive=True):
        text = normalize_text(node.get_text(" ", strip=True))
        if not text:
            continue

        p_count = len(node.find_all("p"))
        heading_count = len(node.find_all(["h1", "h2", "h3"]))
        text_len = len(text)
        score = text_len + (p_count * 200) + (heading_count * 100)

        if score > best_score:
            best_score = score
            best_node = node

    return best_node or body


def extract_headings(root: Tag, limit: int = 8) -> List[str]:
    headings = []
    for tag in root.find_all(["h1", "h2", "h3"]):
        txt = normalize_text(tag.get_text(" ", strip=True))
        if txt and len(txt) > 2:
            headings.append(txt)
    return dedupe_preserve_order(headings)[:limit]


def extract_body_text(root: Tag) -> str:
    paras = []
    for p in root.find_all(["p", "li"]):
        txt = normalize_text(p.get_text(" ", strip=True))
        if txt and len(txt) >= 40:
            paras.append(txt)

    if not paras:
        return normalize_text(root.get_text(" ", strip=True))

    return "\n".join(dedupe_preserve_order(paras))


def strip_noise(text: str) -> str:
    text = normalize_text(text)
    noise_patterns = [
        r"\bprivacy policy\b",
        r"\bterms\s*(and|&)\s*conditions\b",
        r"\bcontact us\b",
        r"\bfollow us\b",
        r"\bdownload app\b",
        r"\badvertisement\b",
        r"\ball rights reserved\b",
        r"\bnews archive\b",
        r"\btopics archive\b",
        r"\bread more\b",
        r"\bclick here\b",
    ]
    for pat in noise_patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def build_page_content_from_html(html: str, url: str) -> PageContent:
    soup = BeautifulSoup(html, "html.parser")
    remove_noise_nodes(soup)

    title = (
        get_meta_content(soup, {"property": "og:title"})
        or get_meta_content(soup, {"name": "twitter:title"})
        or normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    )

    meta_description = (
        get_meta_content(soup, {"property": "og:description"})
        or get_meta_content(soup, {"name": "description"})
        or get_meta_content(soup, {"name": "twitter:description"})
    )

    root = extract_best_text_container(soup)
    if root is None:
        headings = []
        body_text = ""
    else:
        headings = extract_headings(root)
        body_text = extract_body_text(root)

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    return PageContent(
        url=url,
        domain=domain,
        title=strip_noise(title),
        meta_description=strip_noise(meta_description),
        headings=[strip_noise(h) for h in headings if strip_noise(h)],
        body_text=strip_noise(body_text),
    )


def extract_page_content_from_url(url: str) -> PageContent:
    html = fetch_url_html(url)
    return build_page_content_from_html(html, url)


def build_multi_queries(page: PageContent, max_body_chars: int = 1800) -> List[str]:
    body = page.body_text[:max_body_chars]
    headings_joined = " | ".join(page.headings[:6])

    queries = []

    q1 = []
    if page.title:
        q1.append(f"title: {page.title}")
    if page.meta_description:
        q1.append(f"description: {page.meta_description}")
    if q1:
        queries.append(" || ".join(q1))

    q2 = []
    if page.title:
        q2.append(f"title: {page.title}")
    if headings_joined:
        q2.append(f"headings: {headings_joined}")
    if body:
        q2.append(f"content: {body[:700]}")
    if q2:
        queries.append(" || ".join(q2))

    q3 = []
    if page.title:
        q3.append(f"title: {page.title}")
    if page.meta_description:
        q3.append(f"description: {page.meta_description}")
    if body:
        q3.append(f"content: {body}")
    if q3:
        queries.append(" || ".join(q3))

    if body:
        first_chunk = body[:500]
        middle_start = max(0, min(len(body) // 2, max(0, len(body) - 500)))
        middle_chunk = body[middle_start:middle_start + 500]

        q4 = []
        if page.title:
            q4.append(f"title: {page.title}")
        q4.append(f"lead: {first_chunk}")
        if middle_chunk and middle_chunk != first_chunk:
            q4.append(f"middle: {middle_chunk}")
        queries.append(" || ".join(q4))


    # create new entry for q5 with title, meta description, headings_joined and body text
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
        # pass only q5 in queries
        queries = []
        queries.append(" || ".join(q5))

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
    html = fetch_url_html(url)
    fetch_seconds = time.perf_counter() - t_fetch0

    t_clean0 = time.perf_counter()
    page = build_page_content_from_html(html, url)
    clean_seconds = time.perf_counter() - t_clean0

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
    # How many URLs to process from the file (first N). None = all entries in URL_LIST_PATH.
    MAX_URLS_TO_FETCH: Optional[int] = 1000

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
    print(f"Wrote {OUTPUT_JSON_PATH} ({ok}/{len(urls)} succeeded)")

