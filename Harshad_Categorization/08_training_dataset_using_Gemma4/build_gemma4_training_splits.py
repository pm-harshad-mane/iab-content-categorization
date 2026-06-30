from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_INPUT_JSONL = Path("08_training_dataset_using_Gemma4/gemma4_training_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path("08_training_dataset_using_Gemma4")
DEFAULT_MANIFEST_JSON = "gemma4_training_splits_manifest.json"

SPLIT_DEFINITIONS = {
    "all_usable": {"high", "medium", "low"},
    "high_medium": {"high", "medium"},
    "high_only": {"high"},
}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                yield json.loads(payload)


def choose_split(url_hash: str) -> str:
    digest = hashlib.sha1(url_hash.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "valid"
    return "test"


def build_output_paths(output_dir: Path, prefix: str) -> Dict[str, Path]:
    return {
        split_name: output_dir / f"{prefix}_{split_name}.jsonl"
        for split_name in ("train", "valid", "test")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train/valid/test splits from the Gemma4 training dataset.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-json", default=DEFAULT_MANIFEST_JSON)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / args.manifest_json

    handles: Dict[Tuple[str, str], Any] = {}
    paths: Dict[str, Dict[str, Path]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    confidence_counts: Dict[str, Dict[str, Dict[str, int]]] = {}

    try:
        for dataset_name in SPLIT_DEFINITIONS:
            paths[dataset_name] = build_output_paths(args.output_dir, dataset_name)
            counts[dataset_name] = {"train": 0, "valid": 0, "test": 0}
            confidence_counts[dataset_name] = {
                "train": {},
                "valid": {},
                "test": {},
            }
            for split_name, path in paths[dataset_name].items():
                handles[(dataset_name, split_name)] = path.open("w", encoding="utf-8")

        total_rows = 0
        skipped_missing_hash = 0
        skipped_discard = 0

        for record in iter_jsonl(args.input_jsonl):
            total_rows += 1
            url_hash = str(record.get("url_hash") or "").strip()
            if not url_hash:
                skipped_missing_hash += 1
                continue

            bucket = str(record.get("confidence_bucket") or "").strip()
            if bucket == "discard":
                skipped_discard += 1
                continue

            split_name = choose_split(url_hash)
            for dataset_name, allowed_buckets in SPLIT_DEFINITIONS.items():
                if bucket not in allowed_buckets:
                    continue
                handles[(dataset_name, split_name)].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[dataset_name][split_name] += 1
                bucket_counts = confidence_counts[dataset_name][split_name]
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "total_input_rows": total_rows,
        "skipped_missing_hash": skipped_missing_hash,
        "skipped_discard": skipped_discard,
        "split_rule": {
            "train": "sha1(url_hash) % 100 < 90",
            "valid": "90 <= sha1(url_hash) % 100 < 95",
            "test": "sha1(url_hash) % 100 >= 95",
        },
        "datasets": {},
    }

    for dataset_name in SPLIT_DEFINITIONS:
        dataset_total = sum(counts[dataset_name].values())
        manifest["datasets"][dataset_name] = {
            "allowed_confidence_buckets": sorted(SPLIT_DEFINITIONS[dataset_name]),
            "total_rows": dataset_total,
            "files": {split_name: str(path) for split_name, path in paths[dataset_name].items()},
            "counts": counts[dataset_name],
            "confidence_counts": confidence_counts[dataset_name],
        }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Input dataset: {args.input_jsonl}")
    print(f"Manifest: {manifest_path}")
    for dataset_name in ("all_usable", "high_medium", "high_only"):
        dataset = manifest["datasets"][dataset_name]
        print(
            f"{dataset_name}: total={dataset['total_rows']} "
            f"train={dataset['counts']['train']} "
            f"valid={dataset['counts']['valid']} "
            f"test={dataset['counts']['test']}"
        )


if __name__ == "__main__":
    main()
