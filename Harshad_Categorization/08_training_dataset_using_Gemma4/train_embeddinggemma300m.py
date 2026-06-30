from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from torch.utils.data import DataLoader

from sentence_transformers import InputExample, SentenceTransformer, evaluation, losses


DEFAULT_MODEL_PATH = Path(
    "/home/harshad.mane/.cache/huggingface/hub/models--google--embeddinggemma-300m/"
    "snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
)
DEFAULT_DATA_DIR = Path("08_training_dataset_using_Gemma4")
DEFAULT_TRAIN_PAIRS = "high_medium_train_pairs.jsonl"
DEFAULT_VALID_PAIRS = "high_medium_valid_pairs.jsonl"
DEFAULT_TEST_PAIRS = "high_medium_test_pairs.jsonl"
DEFAULT_OUTPUT_DIR = Path("08_training_dataset_using_Gemma4/runs/embeddinggemma300m_high_medium_triplet_v1")
DEFAULT_EPOCHS = 1
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MARGIN = 0.2
DEFAULT_NEGATIVES_PER_ROW = 2


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


def triplets_from_pairs(path: Path, negatives_per_row: int, limit_rows: Optional[int] = None) -> List[InputExample]:
    triplets: List[InputExample] = []
    rows_seen = 0
    for record in iter_jsonl(path):
        query_text = normalize_text(record.get("query_text"))
        positive_text = normalize_text((record.get("positive") or {}).get("taxonomy_text"))
        negatives = record.get("hard_negatives") or []
        if not query_text or not positive_text or not negatives:
            continue
        for negative in negatives[:negatives_per_row]:
            negative_text = normalize_text(negative.get("taxonomy_text"))
            if not negative_text:
                continue
            triplets.append(InputExample(texts=[query_text, positive_text, negative_text]))
        rows_seen += 1
        if limit_rows is not None and rows_seen >= limit_rows:
            break
    return triplets


def evaluator_from_pairs(path: Path, limit_rows: Optional[int] = None) -> evaluation.TripletEvaluator:
    anchors: List[str] = []
    positives: List[str] = []
    negatives: List[str] = []
    rows_seen = 0
    for record in iter_jsonl(path):
        query_text = normalize_text(record.get("query_text"))
        positive_text = normalize_text((record.get("positive") or {}).get("taxonomy_text"))
        hard_negatives = record.get("hard_negatives") or []
        if not query_text or not positive_text or not hard_negatives:
            continue
        negative_text = normalize_text(hard_negatives[0].get("taxonomy_text"))
        if not negative_text:
            continue
        anchors.append(query_text)
        positives.append(positive_text)
        negatives.append(negative_text)
        rows_seen += 1
        if limit_rows is not None and rows_seen >= limit_rows:
            break
    return evaluation.TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name=path.stem,
    )


@dataclass
class RecipeSummary:
    model_path: str
    train_pairs: str
    valid_pairs: str
    test_pairs: str
    output_dir: str
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    warmup_ratio: float
    warmup_steps: int
    weight_decay: float
    margin: float
    negatives_per_row: int
    train_triplets: int
    valid_triplets: int
    test_triplets: int
    valid_eval_rows: int
    test_eval_rows: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune embeddinggemma-300m from Gemma-labeled pair data.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-pairs", default=DEFAULT_TRAIN_PAIRS)
    parser.add_argument("--valid-pairs", default=DEFAULT_VALID_PAIRS)
    parser.add_argument("--test-pairs", default=DEFAULT_TEST_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--negatives-per-row", type=int, default=DEFAULT_NEGATIVES_PER_ROW)
    parser.add_argument("--limit-train-rows", type=int, default=None)
    parser.add_argument("--limit-valid-rows", type=int, default=None)
    parser.add_argument("--limit-test-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_pairs_path = args.data_dir / args.train_pairs
    valid_pairs_path = args.data_dir / args.valid_pairs
    test_pairs_path = args.data_dir / args.test_pairs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = triplets_from_pairs(
        train_pairs_path,
        negatives_per_row=args.negatives_per_row,
        limit_rows=args.limit_train_rows,
    )
    valid_examples = triplets_from_pairs(
        valid_pairs_path,
        negatives_per_row=args.negatives_per_row,
        limit_rows=args.limit_valid_rows,
    )
    test_examples = triplets_from_pairs(
        test_pairs_path,
        negatives_per_row=args.negatives_per_row,
        limit_rows=args.limit_test_rows,
    )

    valid_evaluator = evaluator_from_pairs(valid_pairs_path, limit_rows=args.limit_valid_rows)
    test_evaluator = evaluator_from_pairs(test_pairs_path, limit_rows=args.limit_test_rows)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.train_batch_size)
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * args.warmup_ratio)

    summary = RecipeSummary(
        model_path=str(args.model_path),
        train_pairs=str(train_pairs_path),
        valid_pairs=str(valid_pairs_path),
        test_pairs=str(test_pairs_path),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        margin=args.margin,
        negatives_per_row=args.negatives_per_row,
        train_triplets=len(train_examples),
        valid_triplets=len(valid_examples),
        test_triplets=len(test_examples),
        valid_eval_rows=len(valid_evaluator.anchors),
        test_eval_rows=len(test_evaluator.anchors),
    )

    recipe_path = args.output_dir / "training_recipe.json"
    recipe_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    model = SentenceTransformer(str(args.model_path), local_files_only=True)
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=args.margin,
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=valid_evaluator,
        epochs=args.epochs,
        optimizer_params={"lr": args.learning_rate},
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        output_path=str(args.output_dir),
        use_amp=True,
        checkpoint_path=str(args.output_dir / "checkpoints"),
        checkpoint_save_steps=max(500, len(train_dataloader) // 2),
        checkpoint_save_total_limit=2,
        show_progress_bar=True,
    )

    test_score = test_evaluator(model, output_path=str(args.output_dir))
    (args.output_dir / "test_score.json").write_text(
        json.dumps({"triplet_evaluator_score": test_score}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"test_triplet_score={test_score}")


if __name__ == "__main__":
    main()
