from __future__ import annotations

import argparse
import json
from pathlib import Path

from save_url_content import build_url_hash, normalize_url


def backfill_file(path: Path) -> tuple[int, int]:
    updated = 0
    total = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            payload = line.strip()
            if not payload:
                continue

            try:
                record = json.loads(payload)
            except json.JSONDecodeError as exc:
                tmp_path.unlink(missing_ok=True)
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc

            total += 1
            if not str(record.get("url_hash") or "").strip():
                url = str(record.get("url") or record.get("input_url") or "").strip()
                if url:
                    record["url_hash"] = build_url_hash(normalize_url(url))
                    updated += 1

            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp_path.replace(path)
    return total, updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill url_hash into existing JSONL records.")
    parser.add_argument("paths", nargs="+", help="JSONL files to rewrite in place.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for raw_path in args.paths:
        path = Path(raw_path)
        total, updated = backfill_file(path)
        print(f"{path}: total={total} updated={updated}")


if __name__ == "__main__":
    main()
