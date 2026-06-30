# Gemma4 Training Dataset

This folder contains the first consolidated silver-label training dataset built from:

- stage 02 page content in [`02_fetched_url_content_files`](/home/harshad.mane/Harshad_Categorization/02_fetched_url_content_files)
- stage 07 Gemma category labels in [`07_fetched_url_content_categories_by_Gemma4`](/home/harshad.mane/Harshad_Categorization/07_fetched_url_content_categories_by_Gemma4)

The goal is to use `google/gemma-4-E4B-it` category outputs as teacher labels to improve `google/embeddinggemma-300m`.

## Files

[`build_gemma4_training_dataset.py`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/build_gemma4_training_dataset.py)

- Builds the consolidated training dataset from stage 02 and stage 07.
- Keeps only canonical `google/gemma-4-E4B-it` label files.
- Excludes failed and non-canonical files such as:
  - `restart_backup`
  - `sandbox_failed`
  - `failed_w32_timeout300`
  - old `top5` / `top8` experiment files
  - `31B` outputs
- Prefers shard files when both merged and shard outputs exist.
- Joins records by `url_hash`.
- Writes:
  - [`gemma4_training_dataset.jsonl`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/gemma4_training_dataset.jsonl)
  - [`gemma4_training_manifest.json`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/gemma4_training_manifest.json)

Run command:

```bash
python3 08_training_dataset_using_Gemma4/build_gemma4_training_dataset.py
```

[`gemma4_training_dataset.jsonl`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/gemma4_training_dataset.jsonl)

- Main training dataset.
- One JSON object per usable page.
- Current size:
  - `133890` rows
  - about `966M`

Important fields:

- `source_file`
- `label_file`
- `url_hash`
- `page_text`
- `teacher_model`
- `teacher_top_categories`
- `teacher_primary_category_id`
- `teacher_primary_category_path`
- `teacher_primary_score`
- `teacher_score_gap_top1_top2`
- `confidence_bucket`

[`gemma4_training_manifest.json`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/gemma4_training_manifest.json)

- Build summary and audit file.
- Contains:
  - total training row count
  - confidence totals
  - per-source-file row counts
  - which canonical Gemma files were used

Current totals:

- canonical Gemma files used: `16`
- total training rows: `133890`
- confidence totals:
  - `high`: `7424`
  - `medium`: `100824`
  - `low`: `25494`
  - `discard`: `148`

[`build_gemma4_training_splits.py`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/build_gemma4_training_splits.py)

- Creates deterministic `train` / `valid` / `test` splits from [`gemma4_training_dataset.jsonl`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/gemma4_training_dataset.jsonl).
- Splits by `url_hash`, so the same page cannot land in multiple splits.
- Produces three filtered split families:
  - `all_usable`
  - `high_medium`
  - `high_only`
- Also writes a split manifest:
  - [`gemma4_training_splits_manifest.json`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/gemma4_training_splits_manifest.json)

Run command:

```bash
python3 08_training_dataset_using_Gemma4/build_gemma4_training_splits.py
```

[`build_gemma4_training_pairs.py`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/build_gemma4_training_pairs.py)

- Converts split datasets into pair/triplet-style training records.
- Uses taxonomy text from [`Content_Taxonomy_3.1_6.tsv`](/home/harshad.mane/Harshad_Categorization/taxonomy/Content_Taxonomy_3.1_6.tsv).
- Each output row contains:
  - `query_text`
  - one positive taxonomy target
  - a curated set of hard negatives
- Hard negatives are chosen in this order:
  - lower-ranked teacher categories
  - sibling categories from the taxonomy
  - deterministic global fallback categories
- Writes `*_pairs.jsonl` files for:
  - `all_usable`
  - `high_medium`
  - `high_only`
- Also writes:
  - [`gemma4_training_pairs_manifest.json`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/gemma4_training_pairs_manifest.json)

Run command:

```bash
python3 08_training_dataset_using_Gemma4/build_gemma4_training_pairs.py
```

[`train_embeddinggemma300m.py`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/train_embeddinggemma300m.py)

- First fine-tuning recipe for `embeddinggemma-300m`.
- Uses the local cached model snapshot by default.
- Uses `high_medium` pairs by default.
- Expands each pair row into triplets:
  - anchor = `query_text`
  - positive = teacher primary taxonomy target
  - negative = hard negative taxonomy target
- Uses `sentence-transformers` `TripletLoss` with cosine distance.
- Uses `2` hard negatives per row by default for the first experiment.
- Writes:
  - the fine-tuned model directory
  - `training_recipe.json`
  - `test_score.json`

Default recipe:

- dataset family: `high_medium`
- epochs: `1`
- train batch size: `32`
- learning rate: `2e-5`
- weight decay: `0.01`
- triplet margin: `0.2`
- negatives per row: `2`

Dry-run command:

```bash
python3 08_training_dataset_using_Gemma4/train_embeddinggemma300m.py --dry-run
```

Small smoke test:

```bash
python3 08_training_dataset_using_Gemma4/train_embeddinggemma300m.py \
  --output-dir 08_training_dataset_using_Gemma4/runs/smoke_train_embeddinggemma300m \
  --limit-train-rows 100 \
  --limit-valid-rows 20 \
  --limit-test-rows 20
```

First real training run:

```bash
python3 08_training_dataset_using_Gemma4/train_embeddinggemma300m.py
```

Detached full run on GPU(s):

```bash
setsid env CUDA_VISIBLE_DEVICES=0,1 python3 -u \
  08_training_dataset_using_Gemma4/train_embeddinggemma300m.py \
  > 08_training_dataset_using_Gemma4/embeddinggemma300m_high_medium_triplet_v1.nohup.log 2>&1 < /dev/null &
```

Check the training log:

```bash
sed -n '1,120p' 08_training_dataset_using_Gemma4/embeddinggemma300m_high_medium_triplet_v1.nohup.log
```

Check GPU usage during training:

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

Dry-run summary for the current default recipe:

- train triplets: `195060`
- valid triplets: `10700`
- test triplets: `10736`
- valid evaluator rows: `5350`
- test evaluator rows: `5368`
- warmup steps: `610`

Operational note:

- the training script uses the local cached `embeddinggemma-300m` snapshot by default
- with multiple visible GPUs, `sentence-transformers` may use DataParallel automatically
- if you want to force one GPU, launch with:

```bash
CUDA_VISIBLE_DEVICES=0 python3 08_training_dataset_using_Gemma4/train_embeddinggemma300m.py
```

Observed finding from the first full run:

- the initial full recipe with:
  - `CUDA_VISIBLE_DEVICES=0,1`
  - `train_batch_size=32`
  - `TripletLoss`
- failed with `torch.OutOfMemoryError` on the very first training step
- the failure happened after model load and after entering the training loop
- practical takeaway:
  - the first recipe is too large for this machine at batch size `32`
  - `DataParallel` across 2 GPUs did not avoid the memory issue

Recommended retry:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u 08_training_dataset_using_Gemma4/train_embeddinggemma300m.py \
  --train-batch-size 8 \
  --output-dir 08_training_dataset_using_Gemma4/runs/embeddinggemma300m_high_medium_triplet_v2_bs8
```

Why this retry:

- lower batch size is the first lever to reduce activation memory
- single-GPU avoids DataParallel overhead for the retry
- this is the cleanest next step before changing the loss or truncating inputs

## How The Builder Works

The builder:

1. Scans [`07_fetched_url_content_categories_by_Gemma4`](/home/harshad.mane/Harshad_Categorization/07_fetched_url_content_categories_by_Gemma4) for canonical `google/gemma-4-E4B-it` outputs.
2. Removes failed, backup, and experimental files.
3. Maps each stage-07 file back to its matching stage-02 content file.
4. Keeps only `status="ok"` label rows from Gemma outputs.
5. Keeps only `status="ok"` content rows from stage 02.
6. For sharded label files, keeps only the rows belonging to that shard.
7. Builds a compact `page_text` field from:
   - URL
   - domain
   - title
   - meta description
   - headings
   - truncated body text
8. Computes:
   - `teacher_score_gap_top1_top2`
   - `confidence_bucket`

Current confidence logic:

- `high`
  - at least `5` returned categories and top1-top2 gap `>= 0.15`
- `medium`
  - at least `3` returned categories and top1-top2 gap `>= 0.08`
- `low`
  - usable, but weaker teacher confidence
- `discard`
  - no usable category list

## Intended Use

This dataset is for training or distilling a better taxonomy-retrieval model around `embeddinggemma-300m`.

Important caveat:

- these are **silver labels**, not gold labels
- they come from `google/gemma-4-E4B-it`, not GPT
- they are useful for training, but should not be treated as perfect truth

## Plan Of Action

The plan to improve `embeddinggemma-300m` is:

1. Create filtered datasets from [`gemma4_training_dataset.jsonl`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4/gemma4_training_dataset.jsonl)
   - `high` only
   - `high + medium`
   - optional full silver set for comparison

2. Create train / validation / test splits
   - split by `url_hash`
   - avoid leakage across splits

3. Build the taxonomy side consistently from [`Content_Taxonomy_3.1_6.tsv`](/home/harshad.mane/Harshad_Categorization/taxonomy/Content_Taxonomy_3.1_6.tsv)
   - embed only:
     - `Path`
     - `Description`
     - `Keywords`
   - do not embed:
     - `Common Confusers`
     - `Negative Keywords`

4. Define the first training objective
   - positive target:
     - `page_text` -> `teacher_primary_category`
   - hard negatives:
     - lower-ranked teacher categories
     - optionally stage-05 reranked alternatives later

5. Train a first small experiment on `high + medium`
   - keep validation fixed
   - compare against current `embeddinggemma-300m`

6. Evaluate on the existing categorization pipeline
   - stage 03 embeddings
   - stage 04 FAISS retrieval
   - stage 05 reranking compatibility
   - compare on `new_urls` first, then larger files

## Recommended Next Script

Current recommended first training recipe:

- start with `high_medium`
- use `TripletLoss`
- use `2` hard negatives per row
- train for `1` epoch first
- evaluate on the fixed `high_medium_valid` and `high_medium_test` splits

After that first run, the next comparison should be:

1. `high_medium` vs `high_only`
2. `1` negative per row vs `2` negatives per row
3. current baseline `embeddinggemma-300m` vs fine-tuned model on the existing categorization pipeline
