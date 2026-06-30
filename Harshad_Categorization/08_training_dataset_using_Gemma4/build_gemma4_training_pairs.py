from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_INPUT_DIR = Path("08_training_dataset_using_Gemma4")
DEFAULT_TAXONOMY_TSV = Path("taxonomy/Content_Taxonomy_3.1_6.tsv")
DEFAULT_OUTPUT_DIR = Path("08_training_dataset_using_Gemma4")
DEFAULT_MANIFEST_JSON = "gemma4_training_pairs_manifest.json"
DEFAULT_NEGATIVE_COUNT = 4

DATASET_FAMILIES = ("all_usable", "high_medium", "high_only")
SPLITS = ("train", "valid", "test")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                yield json.loads(payload)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def build_taxonomy_index(path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]], List[str]]:
    by_id: Dict[str, Dict[str, str]] = {}
    children_by_parent: Dict[str, List[str]] = {}
    all_ids: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            unique_id = normalize_text(row.get("Unique ID"))
            parent_id = normalize_text(row.get("Parent"))
            path_text = normalize_text(row.get("Path"))
            description = normalize_text(row.get("Description"))
            keywords = normalize_text(row.get("Keywords"))
            taxonomy_text_parts = []
            if path_text:
                taxonomy_text_parts.append(f"path: {path_text}")
            if description:
                taxonomy_text_parts.append(f"description: {description}")
            if keywords:
                taxonomy_text_parts.append(f"keywords: {keywords}")
            by_id[unique_id] = {
                "unique_id": unique_id,
                "parent_id": parent_id,
                "path": path_text,
                "description": description,
                "keywords": keywords,
                "taxonomy_text": " || ".join(taxonomy_text_parts),
            }
            children_by_parent.setdefault(parent_id, []).append(unique_id)
            all_ids.append(unique_id)
    return by_id, children_by_parent, all_ids


def deterministic_pick(candidates: List[str], key: str, limit: int) -> List[str]:
    if not candidates or limit <= 0:
        return []
    scored = []
    for candidate in candidates:
        digest = hashlib.sha1(f"{key}:{candidate}".encode("utf-8")).hexdigest()
        scored.append((digest, candidate))
    scored.sort()
    return [candidate for _, candidate in scored[:limit]]


def choose_negative_ids(
    record: Dict[str, Any],
    taxonomy_by_id: Dict[str, Dict[str, str]],
    children_by_parent: Dict[str, List[str]],
    all_taxonomy_ids: List[str],
    negative_count: int,
) -> List[str]:
    positive_id = normalize_text(record.get("teacher_primary_category_id"))
    ranked_teacher_ids = [
        normalize_text(category.get("unique_id"))
        for category in (record.get("teacher_top_categories") or [])
        if normalize_text(category.get("unique_id"))
    ]

    chosen: List[str] = []
    seen = {positive_id}

    for candidate_id in ranked_teacher_ids[1:]:
        if candidate_id and candidate_id not in seen and candidate_id in taxonomy_by_id:
            chosen.append(candidate_id)
            seen.add(candidate_id)
            if len(chosen) >= negative_count:
                return chosen

    parent_id = taxonomy_by_id.get(positive_id, {}).get("parent_id", "")
    sibling_candidates = [
        candidate_id
        for candidate_id in children_by_parent.get(parent_id, [])
        if candidate_id not in seen and candidate_id in taxonomy_by_id
    ]
    for candidate_id in deterministic_pick(sibling_candidates, key=record["url_hash"], limit=negative_count - len(chosen)):
        chosen.append(candidate_id)
        seen.add(candidate_id)
        if len(chosen) >= negative_count:
            return chosen

    global_candidates = [
        candidate_id
        for candidate_id in all_taxonomy_ids
        if candidate_id not in seen and candidate_id in taxonomy_by_id
    ]
    for candidate_id in deterministic_pick(global_candidates, key=record["url_hash"], limit=negative_count - len(chosen)):
        chosen.append(candidate_id)
        seen.add(candidate_id)
        if len(chosen) >= negative_count:
            return chosen

    return chosen


def build_pair_record(
    record: Dict[str, Any],
    taxonomy_by_id: Dict[str, Dict[str, str]],
    children_by_parent: Dict[str, List[str]],
    all_taxonomy_ids: List[str],
    negative_count: int,
) -> Dict[str, Any]:
    positive_id = normalize_text(record.get("teacher_primary_category_id"))
    positive = taxonomy_by_id[positive_id]
    negative_ids = choose_negative_ids(
        record=record,
        taxonomy_by_id=taxonomy_by_id,
        children_by_parent=children_by_parent,
        all_taxonomy_ids=all_taxonomy_ids,
        negative_count=negative_count,
    )
    hard_negatives = [taxonomy_by_id[candidate_id] for candidate_id in negative_ids]
    return {
        "url_hash": record.get("url_hash"),
        "source_file": record.get("source_file"),
        "label_file": record.get("label_file"),
        "input_url": record.get("input_url"),
        "url": record.get("url"),
        "query_text": record.get("page_text"),
        "confidence_bucket": record.get("confidence_bucket"),
        "teacher_model": record.get("teacher_model"),
        "teacher_primary_score": record.get("teacher_primary_score"),
        "teacher_score_gap_top1_top2": record.get("teacher_score_gap_top1_top2"),
        "teacher_top_category_ids": record.get("teacher_top_category_ids") or [],
        "positive": positive,
        "hard_negatives": hard_negatives,
    }


def process_split(
    input_path: Path,
    output_path: Path,
    taxonomy_by_id: Dict[str, Dict[str, str]],
    children_by_parent: Dict[str, List[str]],
    all_taxonomy_ids: List[str],
    negative_count: int,
) -> Dict[str, Any]:
    written = 0
    confidence_counts: Dict[str, int] = {}
    skipped_missing_positive = 0
    skipped_missing_query = 0

    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_jsonl(input_path):
            positive_id = normalize_text(record.get("teacher_primary_category_id"))
            query_text = normalize_text(record.get("page_text"))
            if not query_text:
                skipped_missing_query += 1
                continue
            if not positive_id or positive_id not in taxonomy_by_id:
                skipped_missing_positive += 1
                continue
            pair_record = build_pair_record(
                record=record,
                taxonomy_by_id=taxonomy_by_id,
                children_by_parent=children_by_parent,
                all_taxonomy_ids=all_taxonomy_ids,
                negative_count=negative_count,
            )
            out.write(json.dumps(pair_record, ensure_ascii=False) + "\n")
            written += 1
            bucket = normalize_text(record.get("confidence_bucket"))
            confidence_counts[bucket] = confidence_counts.get(bucket, 0) + 1

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "rows_written": written,
        "skipped_missing_query": skipped_missing_query,
        "skipped_missing_positive": skipped_missing_positive,
        "confidence_counts": confidence_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pair/triplet-style training data from Gemma split datasets.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--taxonomy-tsv", type=Path, default=DEFAULT_TAXONOMY_TSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--negative-count", type=int, default=DEFAULT_NEGATIVE_COUNT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / args.manifest_json

    taxonomy_by_id, children_by_parent, all_taxonomy_ids = build_taxonomy_index(args.taxonomy_tsv)
    results: Dict[str, Dict[str, Any]] = {}

    for dataset_name in DATASET_FAMILIES:
        results[dataset_name] = {}
        for split_name in SPLITS:
            input_path = args.input_dir / f"{dataset_name}_{split_name}.jsonl"
            output_path = args.output_dir / f"{dataset_name}_{split_name}_pairs.jsonl"
            summary = process_split(
                input_path=input_path,
                output_path=output_path,
                taxonomy_by_id=taxonomy_by_id,
                children_by_parent=children_by_parent,
                all_taxonomy_ids=all_taxonomy_ids,
                negative_count=args.negative_count,
            )
            results[dataset_name][split_name] = summary

    manifest = {
        "taxonomy_tsv": str(args.taxonomy_tsv),
        "negative_count": args.negative_count,
        "datasets": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Manifest: {manifest_path}")
    for dataset_name in DATASET_FAMILIES:
        totals = {
            split_name: results[dataset_name][split_name]["rows_written"]
            for split_name in SPLITS
        }
        print(
            f"{dataset_name}: "
            f"train={totals['train']} "
            f"valid={totals['valid']} "
            f"test={totals['test']}"
        )


if __name__ == "__main__":
    main()
