from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_STAGE02_DIR = Path("02_fetched_url_content_files")
DEFAULT_STAGE07_DIR = Path("07_fetched_url_content_categories_by_Gemma4")
DEFAULT_OUTPUT_DIR = Path("08_training_dataset_using_Gemma4")
DEFAULT_OUTPUT_JSONL = "gemma4_training_dataset.jsonl"
DEFAULT_MANIFEST_JSON = "gemma4_training_manifest.json"
DEFAULT_MODEL_NAME = "google/gemma-4-E4B-it"
DEFAULT_BODY_CHARS = 6000


@dataclass(frozen=True)
class CanonicalGemmaFile:
    stage02_name: str
    stage07_path: Path
    shard_index: Optional[int]
    shard_count: int


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                yield json.loads(payload)


def is_ignored_gemma_file(path: Path) -> bool:
    name = path.name
    blocked_substrings = [
        ".restart_backup.",
        ".sandbox_failed.",
        ".failed_w32_timeout300.",
        ".top5_server_down_failed.",
        ".top8_before_prompt_fix.",
        ".top8_prompt_fixed.",
        ".top5.",
        "31B",
    ]
    return any(part in name for part in blocked_substrings)


def parse_stage07_file(path: Path, model_name: str) -> Optional[CanonicalGemmaFile]:
    if path.suffix != ".jsonl" or is_ignored_gemma_file(path):
        return None
    sanitized_model_name = sanitize_model_name(model_name)
    marker = f"__{sanitized_model_name}"
    if marker not in path.stem:
        return None

    stem = path.stem
    if "__shard" in stem:
        with_model, shard_part = stem.split("__shard", 1)
        if marker not in with_model:
            return None
        base = with_model[: with_model.index(marker)]
        match = re.fullmatch(r"(\d+)of(\d+)", shard_part)
        if not match:
            return None
        shard_index = int(match.group(1)) - 1
        shard_count = int(match.group(2))
    else:
        base = stem[: stem.index(marker)]
        shard_index = None
        shard_count = 1

    if "__" in base:
        return None

    return CanonicalGemmaFile(
        stage02_name=f"{base}.jsonl",
        stage07_path=path,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def build_canonical_file_list(stage07_dir: Path, model_name: str) -> List[CanonicalGemmaFile]:
    grouped: Dict[str, List[CanonicalGemmaFile]] = {}
    for path in sorted(stage07_dir.glob("*.jsonl")):
        parsed = parse_stage07_file(path, model_name)
        if parsed is not None:
            grouped.setdefault(parsed.stage02_name, []).append(parsed)

    files: List[CanonicalGemmaFile] = []
    for stage02_name in sorted(grouped):
        candidates = grouped[stage02_name]
        sharded = [item for item in candidates if item.shard_index is not None]
        if sharded:
            files.extend(sorted(sharded, key=lambda item: item.shard_index or 0))
        else:
            files.extend(sorted(candidates, key=lambda item: item.stage07_path.name))
    return files


def record_belongs_to_shard(url_hash: str, shard_index: Optional[int], shard_count: int) -> bool:
    if shard_index is None:
        return True
    if not url_hash:
        return False
    return (int(url_hash[:16], 16) % shard_count) == shard_index


def build_page_text(record: Dict[str, Any], max_body_chars: int) -> str:
    parts: List[str] = []
    url = normalize_text(record.get("url") or record.get("input_url"))
    domain = normalize_text(record.get("domain"))
    title = normalize_text(record.get("title"))
    meta_description = normalize_text(record.get("meta_description"))
    headings = [normalize_text(item) for item in (record.get("headings") or []) if normalize_text(item)]
    body_text = normalize_text(record.get("body_text"))

    if url:
        parts.append(f"url: {url}")
    if domain:
        parts.append(f"domain: {domain}")
    if title:
        parts.append(f"title: {title}")
    if meta_description:
        parts.append(f"description: {meta_description}")
    if headings:
        parts.append("headings: " + " | ".join(headings[:8]))
    if body_text:
        parts.append(f"content: {body_text[:max_body_chars]}")
    return " || ".join(parts)


def score_gap(top_categories: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not top_categories:
        return None
    top1 = top_categories[0].get("score")
    if len(top_categories) == 1:
        return float(top1) if top1 is not None else None
    top2 = top_categories[1].get("score")
    if top1 is None or top2 is None:
        return None
    return float(top1) - float(top2)


def confidence_bucket(top_categories: Sequence[Dict[str, Any]]) -> str:
    if not top_categories:
        return "discard"
    gap = score_gap(top_categories)
    count = len(top_categories)
    if gap is None:
        return "medium"
    if count >= 5 and gap >= 0.15:
        return "high"
    if count >= 3 and gap >= 0.08:
        return "medium"
    return "low"


def build_training_record(
    source_record: Dict[str, Any],
    gemma_record: Dict[str, Any],
    source_file: str,
    label_file: str,
    max_body_chars: int,
) -> Dict[str, Any]:
    top_categories = gemma_record.get("llm_top_categories") or []
    top_category_ids = [normalize_text(item.get("unique_id")) for item in top_categories if normalize_text(item.get("unique_id"))]
    primary = top_categories[0] if top_categories else {}
    return {
        "source_file": source_file,
        "label_file": label_file,
        "input_url": source_record.get("input_url"),
        "url": source_record.get("url"),
        "url_hash": source_record.get("url_hash"),
        "domain": source_record.get("domain"),
        "title": source_record.get("title"),
        "meta_description": source_record.get("meta_description"),
        "headings": source_record.get("headings") or [],
        "body_text": normalize_text(source_record.get("body_text"))[:max_body_chars],
        "page_text": build_page_text(source_record, max_body_chars=max_body_chars),
        "teacher_model": gemma_record.get("local_model"),
        "teacher_top_k": gemma_record.get("llm_top_k"),
        "teacher_top_categories": top_categories,
        "teacher_top_category_ids": top_category_ids,
        "teacher_primary_category_id": normalize_text(primary.get("unique_id")),
        "teacher_primary_category_path": normalize_text(primary.get("path")),
        "teacher_primary_score": primary.get("score"),
        "teacher_score_gap_top1_top2": score_gap(top_categories),
        "confidence_bucket": confidence_bucket(top_categories),
        "teacher_usage": gemma_record.get("usage") or {},
        "teacher_model_details": gemma_record.get("model_details") or {},
    }


def process_one_file(
    stage02_dir: Path,
    canonical: CanonicalGemmaFile,
    output_handle,
    max_body_chars: int,
) -> Dict[str, Any]:
    stage02_path = stage02_dir / canonical.stage02_name
    if not stage02_path.exists():
        raise FileNotFoundError(f"Missing stage-02 file for {canonical.stage07_path.name}: {stage02_path}")

    gemma_by_hash: Dict[str, Dict[str, Any]] = {}
    gemma_total = 0
    gemma_ok = 0
    for record in iter_jsonl(canonical.stage07_path):
        gemma_total += 1
        if record.get("status") != "ok":
            continue
        url_hash = normalize_text(record.get("url_hash"))
        if url_hash:
            gemma_by_hash[url_hash] = record
            gemma_ok += 1

    written = 0
    confidence_counts: Dict[str, int] = {}
    score_gaps: List[float] = []
    for source_record in iter_jsonl(stage02_path):
        if source_record.get("status") != "ok":
            continue
        url_hash = normalize_text(source_record.get("url_hash"))
        if not record_belongs_to_shard(url_hash, canonical.shard_index, canonical.shard_count):
            continue
        gemma_record = gemma_by_hash.get(url_hash)
        if gemma_record is None:
            continue
        training_record = build_training_record(
            source_record=source_record,
            gemma_record=gemma_record,
            source_file=canonical.stage02_name,
            label_file=canonical.stage07_path.name,
            max_body_chars=max_body_chars,
        )
        output_handle.write(json.dumps(training_record, ensure_ascii=False) + "\n")
        written += 1
        bucket = training_record["confidence_bucket"]
        confidence_counts[bucket] = confidence_counts.get(bucket, 0) + 1
        gap = training_record["teacher_score_gap_top1_top2"]
        if gap is not None:
            score_gaps.append(float(gap))

    return {
        "stage02_file": canonical.stage02_name,
        "stage07_file": canonical.stage07_path.name,
        "shard_index": canonical.shard_index,
        "shard_count": canonical.shard_count,
        "gemma_rows_total": gemma_total,
        "gemma_rows_ok": gemma_ok,
        "training_rows_written": written,
        "confidence_counts": confidence_counts,
        "avg_score_gap_top1_top2": round(mean(score_gaps), 6) if score_gaps else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean training dataset from stage-02 content and stage-07 Gemma labels.")
    parser.add_argument("--stage02-dir", type=Path, default=DEFAULT_STAGE02_DIR)
    parser.add_argument("--stage07-dir", type=Path, default=DEFAULT_STAGE07_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-body-chars", type=int, default=DEFAULT_BODY_CHARS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / args.output_jsonl
    manifest_json = args.output_dir / args.manifest_json

    canonical_files = build_canonical_file_list(args.stage07_dir, args.model_name)
    per_file: List[Dict[str, Any]] = []

    with output_jsonl.open("w", encoding="utf-8") as out:
        for canonical in canonical_files:
            summary = process_one_file(
                stage02_dir=args.stage02_dir,
                canonical=canonical,
                output_handle=out,
                max_body_chars=args.max_body_chars,
            )
            per_file.append(summary)

    total_rows = 0
    confidence_totals: Dict[str, int] = {}
    for item in per_file:
        total_rows += item["training_rows_written"]
        for bucket, count in item["confidence_counts"].items():
            confidence_totals[bucket] = confidence_totals.get(bucket, 0) + count

    manifest = {
        "model_name": args.model_name,
        "stage02_dir": str(args.stage02_dir),
        "stage07_dir": str(args.stage07_dir),
        "output_jsonl": str(output_jsonl),
        "total_training_rows": total_rows,
        "confidence_totals": confidence_totals,
        "canonical_files": per_file,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Canonical Gemma files: {len(canonical_files)}")
    print(f"Output dataset: {output_jsonl}")
    print(f"Manifest: {manifest_json}")
    print(f"Total training rows: {total_rows}")
    print(f"Confidence totals: {json.dumps(confidence_totals, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
