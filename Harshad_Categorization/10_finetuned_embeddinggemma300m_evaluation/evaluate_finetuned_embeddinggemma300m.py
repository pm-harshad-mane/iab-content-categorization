from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


DEFAULT_MODEL_PATH = Path(
    "08_training_dataset_using_Gemma4/runs/embeddinggemma300m_high_medium_triplet_v2_bs8"
)
DEFAULT_INPUT_FILE = Path("02_fetched_url_content_files/new_urls.jsonl")
DEFAULT_TAXONOMY_PATH = Path("taxonomy/Content_Taxonomy_3.1_6.tsv")
DEFAULT_EXISTING_STAGE04 = Path(
    "04_fetched_url_content_embedding_categories_files/new_urls__google_embeddinggemma-300m__faiss.jsonl"
)
DEFAULT_GPT_BASELINE = Path(
    "06_fetched_url_content_categories_by_GPT/new_urls__gpt-5.4.jsonl"
)
DEFAULT_ALT_BASELINE = Path(
    "07_fetched_url_content_categories_by_Gemma4/new_urls__google_gemma-4-E4B-it.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("10_finetuned_embeddinggemma300m_evaluation")
DEFAULT_TOP_K = 10
DEFAULT_MAX_BODY_CHARS = 4000
DEFAULT_COMPARE_TOP_K = 5


@dataclass
class TaxonomyRow:
    unique_id: str
    parent_id: str
    tier1: str
    tier2: str
    tier3: str
    tier4: str
    path: str
    description: str
    keywords: str
    negative_keywords: str
    index_text: str


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                yield json.loads(payload)


def build_page_query_text(record: Dict[str, Any], max_body_chars: int) -> str:
    parts: List[str] = []
    title = normalize_text(record.get("title"))
    meta_description = normalize_text(record.get("meta_description"))
    headings = [normalize_text(x) for x in record.get("headings", []) if normalize_text(x)]
    body_text = normalize_text(record.get("body_text"))
    if title:
        parts.append(f"title: {title}")
    if meta_description:
        parts.append(f"description: {meta_description}")
    if headings:
        parts.append("headings: " + " | ".join(headings[:6]))
    if body_text:
        parts.append(f"content: {body_text[:max_body_chars]}")
    return " || ".join(parts)


def load_taxonomy_rows(path: Path) -> List[TaxonomyRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: List[TaxonomyRow] = []
        for row in reader:
            path_text = normalize_text(row.get("Path"))
            description = normalize_text(row.get("Description"))
            keywords = normalize_text(row.get("Keywords"))
            index_parts: List[str] = []
            if path_text:
                index_parts.append(f"path: {path_text}")
            if description:
                index_parts.append(f"description: {description}")
            if keywords:
                index_parts.append(f"keywords: {keywords}")
            rows.append(
                TaxonomyRow(
                    unique_id=normalize_text(row.get("Unique ID")),
                    parent_id=normalize_text(row.get("Parent")),
                    tier1=normalize_text(row.get("Tier 1")),
                    tier2=normalize_text(row.get("Tier 2")),
                    tier3=normalize_text(row.get("Tier 3")),
                    tier4=normalize_text(row.get("Tier 4")),
                    path=path_text,
                    description=description,
                    keywords=keywords,
                    negative_keywords=normalize_text(row.get("Negative Keywords")),
                    index_text="\n".join(index_parts),
                )
            )
    return rows


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (vectors / norms).astype("float32")


def encode_texts(model: SentenceTransformer, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype="float32")


def output_path_for_input(input_path: Path, output_dir: Path, model_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}__{model_dir.name}__faiss.jsonl"


def summary_path_for_input(input_path: Path, output_dir: Path, model_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}__{model_dir.name}__comparison.md"


def json_path_for_input(input_path: Path, output_dir: Path, model_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}__{model_dir.name}__comparison.json"


def safe_label(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "baseline"


def build_result_record(
    record: Dict[str, Any],
    model_name: str,
    taxonomy_path: Path,
    embedding_dim: int,
    top_k: int,
    categories: List[Dict[str, Any]],
    search_ms: float,
    total_ms: float,
) -> Dict[str, Any]:
    return {
        "input_url": normalize_text(record.get("input_url") or record.get("url")),
        "url": normalize_text(record.get("url") or record.get("input_url")),
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "ok",
        "source_status": "ok",
        "embedding_model": model_name,
        "domain": normalize_text(record.get("domain")),
        "title": normalize_text(record.get("title")),
        "embedding_dim": embedding_dim,
        "faiss_top_k": top_k,
        "model_details": {
            "embedding_model": model_name,
            "taxonomy_source": str(taxonomy_path),
            "taxonomy_text_fields": ["Path", "Description", "Keywords"],
            "negative_keywords_used_in_index": False,
        },
        "top_categories": categories,
        "timing_ms": {
            "faiss_search": round(search_ms, 3),
            "total": round(total_ms, 3),
        },
    }


def build_error_record(record: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    return {
        "input_url": normalize_text(record.get("input_url") or record.get("url")),
        "url": normalize_text(record.get("url") or record.get("input_url")),
        "url_hash": normalize_text(record.get("url_hash")),
        "status": "error",
        "source_status": normalize_text(record.get("status")),
        "embedding_model": model_name,
        "error_type": "SourceRecordError",
        "error_code": "source_status_error",
        "message": "Source record status is not ok; retrieval evaluation skipped.",
        "retryable": False,
    }


def load_top_ids(path: Path, top_field: str, compare_top_k: int) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for record in iter_jsonl(path):
        if normalize_text(record.get("status")) != "ok":
            continue
        url_hash = normalize_text(record.get("url_hash"))
        if not url_hash:
            continue
        categories = record.get(top_field) or []
        ids = [normalize_text(item.get("unique_id")) for item in categories if normalize_text(item.get("unique_id"))]
        mapping[url_hash] = ids[:compare_top_k]
    return mapping


def infer_top_field(path: Path) -> str:
    path_str = str(path)
    if "06_fetched_url_content_categories_by_GPT" in path_str:
        return "gpt_top_categories"
    if "07_fetched_url_content_categories_by_Gemma4" in path_str:
        return "llm_top_categories"
    return "top_categories"


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def dcg_for_prediction(predicted_ids: Sequence[str], gpt_ids: Sequence[str]) -> float:
    rank_map = {uid: idx + 1 for idx, uid in enumerate(gpt_ids)}
    score = 0.0
    for pred_rank, uid in enumerate(predicted_ids, start=1):
        gpt_rank = rank_map.get(uid)
        if gpt_rank is None:
            continue
        rel = 1.0 / math.log2(gpt_rank + 1)
        score += rel / math.log2(pred_rank + 1)
    return score


def ideal_dcg(gpt_ids: Sequence[str]) -> float:
    score = 0.0
    for rank, _uid in enumerate(gpt_ids, start=1):
        rel = 1.0 / math.log2(rank + 1)
        score += rel / math.log2(rank + 1)
    return score


def positional_credit(predicted_ids: Sequence[str], gpt_ids: Sequence[str]) -> float:
    rank_map = {uid: idx + 1 for idx, uid in enumerate(gpt_ids)}
    total = 0.0
    for pred_rank, uid in enumerate(predicted_ids, start=1):
        gpt_rank = rank_map.get(uid)
        if gpt_rank is None:
            continue
        total += 1.0 / (1.0 + abs(pred_rank - gpt_rank))
    return total / max(1, len(gpt_ids))


def compare_against_gpt(
    predicted: Dict[str, List[str]],
    gpt: Dict[str, List[str]],
    compare_top_k: int,
) -> Dict[str, Any]:
    common = sorted(set(predicted) & set(gpt))
    if not common:
        return {"common_rows": 0}

    top1_matches = 0
    gpt_top1_in_pred = 0
    pred_top1_in_gpt = 0
    overlap_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0
    ndcg_total = 0.0
    positional_total = 0.0
    overlap_hist: Dict[int, int] = {}

    for key in common:
        pred_ids = predicted[key][:compare_top_k]
        gpt_ids = gpt[key][:compare_top_k]
        if pred_ids and gpt_ids and pred_ids[0] == gpt_ids[0]:
            top1_matches += 1
        if gpt_ids and gpt_ids[0] in pred_ids:
            gpt_top1_in_pred += 1
        if pred_ids and pred_ids[0] in gpt_ids:
            pred_top1_in_gpt += 1

        overlap = len(set(pred_ids) & set(gpt_ids))
        overlap_total += overlap
        overlap_hist[overlap] = overlap_hist.get(overlap, 0) + 1

        precision = overlap / max(1, len(pred_ids))
        recall = overlap / max(1, len(gpt_ids))
        precision_total += precision
        recall_total += recall
        if precision + recall:
            f1_total += (2 * precision * recall) / (precision + recall)

        ideal = ideal_dcg(gpt_ids)
        if ideal > 0:
            ndcg_total += dcg_for_prediction(pred_ids, gpt_ids) / ideal
        positional_total += positional_credit(pred_ids, gpt_ids)

    count = len(common)
    ndcg_score = 100.0 * ndcg_total / count
    f1_score = 100.0 * f1_total / count
    positional_score = 100.0 * positional_total / count
    heuristic = 0.6 * ndcg_score + 0.3 * f1_score + 0.1 * positional_score
    return {
        "common_rows": count,
        "top1_match_rate": top1_matches / count,
        "gpt_top1_in_pred_rate": gpt_top1_in_pred / count,
        "pred_top1_in_gpt_rate": pred_top1_in_gpt / count,
        "avg_overlap_count": overlap_total / count,
        "avg_precision": precision_total / count,
        "avg_recall": recall_total / count,
        "avg_f1": f1_total / count,
        "ndcg_at_k": ndcg_total / count,
        "ndcg_score_100": ndcg_score,
        "positional_score_100": positional_score,
        "heuristic_score_100": heuristic,
        "overlap_histogram": overlap_hist,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned embeddinggemma-300m in the FAISS retrieval path.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--taxonomy-path", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--existing-stage04", type=Path, default=DEFAULT_EXISTING_STAGE04)
    parser.add_argument("--gpt-baseline", type=Path, default=DEFAULT_GPT_BASELINE)
    parser.add_argument("--baseline-file", type=Path, default=None)
    parser.add_argument("--baseline-top-field", type=str, default=None)
    parser.add_argument("--baseline-name", type=str, default="gpt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--compare-top-k", type=int, default=DEFAULT_COMPARE_TOP_K)
    parser.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY_CHARS)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_rows = load_taxonomy_rows(args.taxonomy_path)
    taxonomy_texts = [row.index_text for row in taxonomy_rows]

    device = resolve_device(args.device)
    model = SentenceTransformer(str(args.model_path), local_files_only=True, device=device)

    t0 = time.perf_counter()
    taxonomy_embeddings = encode_texts(model, taxonomy_texts, batch_size=32)
    taxonomy_embeddings = normalize_l2(taxonomy_embeddings)
    index = faiss.IndexFlatIP(int(taxonomy_embeddings.shape[1]))
    index.add(taxonomy_embeddings)
    taxonomy_setup_ms = (time.perf_counter() - t0) * 1000.0

    output_path = output_path_for_input(args.input_file, args.output_dir, args.model_path)
    results: List[Dict[str, Any]] = []

    for record in iter_jsonl(args.input_file):
        if normalize_text(record.get("status")) != "ok":
            results.append(build_error_record(record, args.model_path.name))
            continue

        query_text = build_page_query_text(record, args.max_body_chars)
        if not query_text:
            error = build_error_record(record, args.model_path.name)
            error["message"] = "No usable text after query-text construction."
            results.append(error)
            continue

        row_t0 = time.perf_counter()
        query_embedding = encode_texts(model, [query_text], batch_size=1)
        query_embedding = normalize_l2(query_embedding)
        search_t0 = time.perf_counter()
        scores, indices = index.search(query_embedding, args.top_k)
        search_ms = (time.perf_counter() - search_t0) * 1000.0
        categories: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            tax = taxonomy_rows[int(idx)]
            categories.append(
                {
                    "unique_id": tax.unique_id,
                    "parent_id": tax.parent_id,
                    "tier1": tax.tier1,
                    "tier2": tax.tier2,
                    "tier3": tax.tier3,
                    "tier4": tax.tier4,
                    "path": tax.path,
                    "description": tax.description,
                    "keywords": tax.keywords,
                    "faiss_score": round(float(score), 6),
                    "faiss_rank": rank,
                }
            )
        results.append(
            build_result_record(
                record=record,
                model_name=args.model_path.name,
                taxonomy_path=args.taxonomy_path,
                embedding_dim=int(query_embedding.shape[1]),
                top_k=args.top_k,
                categories=categories,
                search_ms=search_ms,
                total_ms=(time.perf_counter() - row_t0) * 1000.0,
            )
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    baseline_file = args.baseline_file or args.gpt_baseline
    baseline_top_field = args.baseline_top_field or infer_top_field(baseline_file)
    baseline_name = args.baseline_name if args.baseline_file else "gpt"
    baseline_key = safe_label(baseline_name)

    trained_mapping = load_top_ids(output_path, "top_categories", args.compare_top_k)
    existing_mapping = load_top_ids(args.existing_stage04, "top_categories", args.compare_top_k)
    baseline_mapping = load_top_ids(baseline_file, baseline_top_field, args.compare_top_k)

    trained_vs_baseline = compare_against_gpt(trained_mapping, baseline_mapping, args.compare_top_k)
    existing_vs_baseline = compare_against_gpt(existing_mapping, baseline_mapping, args.compare_top_k)

    comparison = {
        "input_file": str(args.input_file),
        "trained_model_path": str(args.model_path),
        "taxonomy_source": str(args.taxonomy_path),
        "taxonomy_rows": len(taxonomy_rows),
        "taxonomy_setup_ms": round(taxonomy_setup_ms, 3),
        "device": device,
        "top_k": args.top_k,
        "compare_top_k": args.compare_top_k,
        "trained_output_file": str(output_path),
        "existing_stage04_file": str(args.existing_stage04),
        "baseline_file": str(baseline_file),
        "baseline_top_field": baseline_top_field,
        "baseline_name": baseline_name,
        f"trained_vs_{baseline_key}": trained_vs_baseline,
        f"existing_vs_{baseline_key}": existing_vs_baseline,
    }

    json_path = json_path_for_input(args.input_file, args.output_dir, args.model_path)
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_lines = [
        "# Fine-Tuned embeddinggemma-300m Evaluation",
        "",
        f"- Input file: `{args.input_file}`",
        f"- Trained model: `{args.model_path}`",
        f"- Output file: `{output_path}`",
        f"- Taxonomy source: `{args.taxonomy_path}`",
        f"- Taxonomy rows: `{len(taxonomy_rows)}`",
        f"- Taxonomy setup ms: `{round(taxonomy_setup_ms, 3)}`",
        f"- Device: `{device}`",
        f"- Existing stage-04 baseline: `{args.existing_stage04}`",
        f"- Comparison baseline: `{baseline_file}` (`{baseline_name}`, field `{baseline_top_field}`)",
        "",
        f"## Trained vs {baseline_name}",
        "",
        f"- Common rows: `{trained_vs_baseline.get('common_rows', 0)}`",
        f"- Top-1 match rate: `{trained_vs_baseline.get('top1_match_rate', 0.0):.4f}`",
        f"- {baseline_name} top-1 in trained top-{args.compare_top_k}: `{trained_vs_baseline.get('gpt_top1_in_pred_rate', 0.0):.4f}`",
        f"- Avg overlap count: `{trained_vs_baseline.get('avg_overlap_count', 0.0):.4f}`",
        f"- Avg F1: `{trained_vs_baseline.get('avg_f1', 0.0):.4f}`",
        f"- NDCG@{args.compare_top_k}: `{trained_vs_baseline.get('ndcg_at_k', 0.0):.4f}`",
        f"- Heuristic score / 100: `{trained_vs_baseline.get('heuristic_score_100', 0.0):.2f}`",
        "",
        f"## Existing embeddinggemma-300m vs {baseline_name}",
        "",
        f"- Common rows: `{existing_vs_baseline.get('common_rows', 0)}`",
        f"- Top-1 match rate: `{existing_vs_baseline.get('top1_match_rate', 0.0):.4f}`",
        f"- {baseline_name} top-1 in existing top-{args.compare_top_k}: `{existing_vs_baseline.get('gpt_top1_in_pred_rate', 0.0):.4f}`",
        f"- Avg overlap count: `{existing_vs_baseline.get('avg_overlap_count', 0.0):.4f}`",
        f"- Avg F1: `{existing_vs_baseline.get('avg_f1', 0.0):.4f}`",
        f"- NDCG@{args.compare_top_k}: `{existing_vs_baseline.get('ndcg_at_k', 0.0):.4f}`",
        f"- Heuristic score / 100: `{existing_vs_baseline.get('heuristic_score_100', 0.0):.2f}`",
        "",
    ]
    summary_path = summary_path_for_input(args.input_file, args.output_dir, args.model_path)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
