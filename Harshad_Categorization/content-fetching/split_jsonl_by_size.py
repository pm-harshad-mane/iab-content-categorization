#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MAX_BYTES = 50_000_000


def chunk_path_for(input_path: Path, chunk_index: int) -> Path:
    return input_path.with_name(f"{input_path.stem}.chunk-{chunk_index:04d}{input_path.suffix}")


def remove_existing_chunks(input_path: Path) -> None:
    pattern = f"{input_path.stem}.chunk-*{input_path.suffix}"
    for path in input_path.parent.glob(pattern):
        path.unlink()


def split_jsonl_file(input_path: Path, max_bytes: int) -> list[dict[str, int | str]]:
    remove_existing_chunks(input_path)

    chunk_index = 0
    chunk_size = 0
    chunk_line_count = 0
    total_line_count = 0
    chunk_file = None
    chunks: list[dict[str, int | str]] = []

    try:
        with input_path.open("rb") as source:
            for raw_line in source:
                line_size = len(raw_line)
                if line_size > max_bytes:
                    raise ValueError(
                        f"{input_path} contains a line of {line_size} bytes, larger than the chunk limit of {max_bytes} bytes"
                    )

                if chunk_file is None or chunk_size + line_size > max_bytes:
                    if chunk_file is not None:
                        chunk_file.close()
                        chunks.append(
                            {
                                "path": str(chunk_path_for(input_path, chunk_index).relative_to(input_path.parent.parent.parent)),
                                "bytes": chunk_size,
                                "lines": chunk_line_count,
                            }
                        )

                    chunk_index += 1
                    chunk_size = 0
                    chunk_line_count = 0
                    chunk_output_path = chunk_path_for(input_path, chunk_index)
                    chunk_file = chunk_output_path.open("wb")

                chunk_file.write(raw_line)
                chunk_size += line_size
                chunk_line_count += 1
                total_line_count += 1

        if chunk_file is not None:
            chunk_file.close()
            chunks.append(
                {
                    "path": str(chunk_path_for(input_path, chunk_index).relative_to(input_path.parent.parent.parent)),
                    "bytes": chunk_size,
                    "lines": chunk_line_count,
                }
            )
    finally:
        if chunk_file is not None and not chunk_file.closed:
            chunk_file.close()

    manifest_path = input_path.with_name(f"{input_path.stem}.chunks.json")
    manifest = {
        "source_file": str(input_path.relative_to(input_path.parent.parent.parent)),
        "max_bytes": max_bytes,
        "total_lines": total_line_count,
        "chunks": chunks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split JSONL files into chunks capped by byte size.")
    parser.add_argument("files", nargs="+", type=Path, help="JSONL files to split")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"Maximum chunk size in bytes (default: {DEFAULT_MAX_BYTES})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for input_path in args.files:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing file: {input_path}")
        if input_path.suffix != ".jsonl":
            raise ValueError(f"Expected a .jsonl file: {input_path}")
        chunks = split_jsonl_file(input_path, args.max_bytes)
        print(f"{input_path}: wrote {len(chunks)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
