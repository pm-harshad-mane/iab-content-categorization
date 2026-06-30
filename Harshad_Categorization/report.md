# Project Report

## Executive Summary

- A complete taxonomy-based URL categorization pipeline was built across stages `01` to `10`, covering content fetching, embeddings, FAISS retrieval, reranking, GPT/Gemma teacher labeling, training-data creation, model fine-tuning, and evaluation.
- The taxonomy was progressively enriched and cleaned, and [`Content_Taxonomy_3.1_6.tsv`](./taxonomy/Content_Taxonomy_3.1_6.tsv) is the final recommended ANN-friendly taxonomy for this project.
- GPT-based labeling worked as the highest-quality benchmark, but the projected cost for large-scale labeling was too high, so GPT was kept only as a limited reference baseline.
- Among local Gemma models, `google/gemma-4-E4B-it` was the only practical teacher model on this machine; larger Gemma 4 variants were not usable enough for this workload.
- A large Gemma-labeled silver dataset was built and used to fine-tune `embeddinggemma-300m`, but the first fine-tuned checkpoint did **not** beat the original `embeddinggemma-300m` retrieval baseline.
- The project is therefore closed at the current proof-of-concept stage: the pipeline is complete, the first research cycle is complete, and future work should resume only with a revised training strategy rather than continuing the same recipe.

## Purpose

This repository was used to build and evaluate a taxonomy-based content categorization pipeline for large URL datasets.

The main goals were:

- fetch page content from URL lists
- generate dense embeddings with multiple embedding models
- retrieve top taxonomy categories with FAISS
- rerank retrieved categories with a stronger reranker
- build higher-quality teacher labels with GPT and Gemma 4
- use Gemma 4 labels to fine-tune `google/embeddinggemma-300m`
- check whether the fine-tuned model improves retrieval quality

This report captures what was built, where to find it, what was learned, and where the work was intentionally paused.

## Final Project Status

This project is effectively **closed at the current proof-of-concept stage**.

The main reason is simple:

- the first fine-tuned `embeddinggemma-300m` checkpoint did **not** beat the original `embeddinggemma-300m` baseline in retrieval evaluation
- direct GPT labeling is too expensive for large-scale teacher-label generation
- Gemma 4 E4B was practical as a local teacher, but still only a moderate approximation to GPT

So the current state is:

- pipeline and datasets are built
- taxonomy work is complete for the current iteration
- local teacher-label generation with Gemma 4 E4B is working
- one full fine-tuning experiment was completed
- evaluation was completed
- the first fine-tuned model did not justify continuing blindly

This is a good stopping point for the first research cycle.

## High-Level Workflow

The main pipeline is organized into numbered folders:

1. [`01_plain_urls_files`](./01_plain_urls_files)
2. [`02_fetched_url_content_files`](./02_fetched_url_content_files)
3. [`03_fetched_url_content_embedding_files`](./03_fetched_url_content_embedding_files)
4. [`04_fetched_url_content_embedding_categories_files`](./04_fetched_url_content_embedding_categories_files)
5. [`05_fetched_url_content_embedding_categories_reranked_files`](./05_fetched_url_content_embedding_categories_reranked_files)
6. [`06_fetched_url_content_categories_by_GPT`](./06_fetched_url_content_categories_by_GPT)
7. [`07_fetched_url_content_categories_by_Gemma4`](./07_fetched_url_content_categories_by_Gemma4)
8. [`08_training_dataset_using_Gemma4`](./08_training_dataset_using_Gemma4)
9. [`09_fetched_url_content_categories_by_Claude`](./09_fetched_url_content_categories_by_Claude)
10. [`10_finetuned_embeddinggemma300m_evaluation`](./10_finetuned_embeddinggemma300m_evaluation)

The project’s operational pipeline documentation lives in:

- [`content-fetching/readme.md`](./content-fetching/readme.md)

## Dataset Coverage

The main named datasets used in this repo are:

- `new_urls`
- `adserver_1000_urls`
- `urls_may_14_26`
- `HarshardData_New_1M_01` through `HarshardData_New_1M_06`

Practical completion summary:

| Dataset | Stage 02 Content | Stage 03 Embeddings | Stage 04 FAISS | Stage 05 Reranked | Stage 06 GPT | Stage 07 Gemma4 | Step 10 Evaluation |
|---|---|---|---|---|---|---|---|
| `new_urls` | yes | yes | yes | yes | yes | yes | yes |
| `adserver_1000_urls` | yes | yes | yes | yes | no | yes | yes |
| `urls_may_14_26` | yes | yes | yes | yes | no | yes | no |
| `HarshardData_New_1M_01` | yes | yes | yes | yes | no | yes | no |
| `HarshardData_New_1M_02` | yes | yes | yes | yes | no | yes | no |
| `HarshardData_New_1M_03` | yes | yes | yes | yes | no | yes | no |
| `HarshardData_New_1M_04` | yes | yes | yes | yes | no | yes | no |
| `HarshardData_New_1M_05` | yes | yes | yes | yes | no | yes | no |
| `HarshardData_New_1M_06` | yes | yes | yes | yes | no | yes | no |

Interpretation:

- GPT was used only on a small benchmark slice because of cost
- Gemma4 was used for broader local silver-label generation
- the final fine-tuned embedding evaluation was run only on small benchmark slices, not on the full large datasets

## What Was Built

### 1. Content fetching pipeline

Built under:

- [`content-fetching/01_save_url_content.py`](./content-fetching/01_save_url_content.py)

This stage:

- reads plain URL files
- fetches content
- writes JSONL rows
- preserves `url_hash` as a stable ID

### 2. Embedding generation pipeline

Built under:

- [`content-fetching/02_generate_url_embeddings.py`](./content-fetching/02_generate_url_embeddings.py)

Embedding models used:

- `BAAI/bge-m3`
- `google/embeddinggemma-300m`

This stage writes stage-03 embedding JSONL files.

### 3. FAISS taxonomy retrieval

Built under:

- [`content-fetching/03_generate_faiss_taxonomy_categories.py`](./content-fetching/03_generate_faiss_taxonomy_categories.py)

This stage:

- loads the taxonomy
- embeds taxonomy rows
- builds a FAISS index
- finds top-N categories per content embedding

### 4. Reranking stage

Built under:

- [`content-fetching/04_generate_reranked_taxonomy_categories.py`](./content-fetching/04_generate_reranked_taxonomy_categories.py)

Reranker used in the main pipeline:

- `mixedbread-ai/mxbai-rerank-base-v2`

This stage:

- consumes stage-04 FAISS candidates
- reranks them
- writes stage-05 JSONL files

### 5. GPT batch categorization

Built under:

- [`content-fetching/05_generate_categories_by_GPT.py`](./content-fetching/05_generate_categories_by_GPT.py)

This stage supports:

- `submit`
- `sync`
- `finalize`

It was used for a limited benchmark set because of cost.

### 6. Gemma 4 categorization

Built under:

- [`content-fetching/06_generate_categories_by_Gemma4.py`](./content-fetching/06_generate_categories_by_Gemma4.py)

The practical local teacher model became:

- `google/gemma-4-E4B-it`

This stage was used to generate local silver-label data at scale.

### 7. Claude categorization

Built under:

- [`content-fetching/07_generate_categories_by_Claude.py`](./content-fetching/07_generate_categories_by_Claude.py)

This was a side benchmark / comparison path rather than the main training path.

Practical status:

- Claude outputs exist in [`09_fetched_url_content_categories_by_Claude`](./09_fetched_url_content_categories_by_Claude)
- they were not used as the main teacher-label source for model training
- Gemma4 and GPT were the main reference systems used in this project

### 8. Gemma-based training dataset prep

Built under:

- [`08_training_dataset_using_Gemma4/build_gemma4_training_dataset.py`](./08_training_dataset_using_Gemma4/build_gemma4_training_dataset.py)
- [`08_training_dataset_using_Gemma4/build_gemma4_training_splits.py`](./08_training_dataset_using_Gemma4/build_gemma4_training_splits.py)
- [`08_training_dataset_using_Gemma4/build_gemma4_training_pairs.py`](./08_training_dataset_using_Gemma4/build_gemma4_training_pairs.py)

These scripts:

- consolidate canonical Gemma labels
- filter unusable rows
- create train/valid/test splits
- create triplet-style training records

### 9. Fine-tuning `embeddinggemma-300m`

Built under:

- [`08_training_dataset_using_Gemma4/train_embeddinggemma300m.py`](./08_training_dataset_using_Gemma4/train_embeddinggemma300m.py)

The successful run directory is:

- [`08_training_dataset_using_Gemma4/runs/embeddinggemma300m_high_medium_triplet_v2_bs8`](./08_training_dataset_using_Gemma4/runs/embeddinggemma300m_high_medium_triplet_v2_bs8)

Important artifacts from that run:

- [`training_recipe.json`](./08_training_dataset_using_Gemma4/runs/embeddinggemma300m_high_medium_triplet_v2_bs8/training_recipe.json)
- [`test_score.json`](./08_training_dataset_using_Gemma4/runs/embeddinggemma300m_high_medium_triplet_v2_bs8/test_score.json)

### 10. Retrieval-path evaluation of the fine-tuned model

Built under:

- [`10_finetuned_embeddinggemma300m_evaluation/evaluate_finetuned_embeddinggemma300m.py`](./10_finetuned_embeddinggemma300m_evaluation/evaluate_finetuned_embeddinggemma300m.py)

This stage:

- embeds content with the fine-tuned model
- embeds taxonomy rows
- runs FAISS
- compares fresh results against existing baselines

## Canonical Outputs

If someone is consuming this repo as a handoff package, the safest canonical outputs are:

- taxonomy:
  - [`taxonomy/Content_Taxonomy_3.1_6.tsv`](./taxonomy/Content_Taxonomy_3.1_6.tsv)
- pipeline docs:
  - [`content-fetching/readme.md`](./content-fetching/readme.md)
- GPT benchmark labels:
  - [`06_fetched_url_content_categories_by_GPT/new_urls__gpt-5.4.jsonl`](./06_fetched_url_content_categories_by_GPT/new_urls__gpt-5.4.jsonl)
- Gemma4 canonical silver labels:
  - current `google/gemma-4-E4B-it` outputs in [`07_fetched_url_content_categories_by_Gemma4`](./07_fetched_url_content_categories_by_Gemma4)
- consolidated Gemma training dataset:
  - [`08_training_dataset_using_Gemma4/gemma4_training_dataset.jsonl`](./08_training_dataset_using_Gemma4/gemma4_training_dataset.jsonl)
- split manifests and pair manifests:
  - [`08_training_dataset_using_Gemma4/gemma4_training_manifest.json`](./08_training_dataset_using_Gemma4/gemma4_training_manifest.json)
  - [`08_training_dataset_using_Gemma4/gemma4_training_splits_manifest.json`](./08_training_dataset_using_Gemma4/gemma4_training_splits_manifest.json)
  - [`08_training_dataset_using_Gemma4/gemma4_training_pairs_manifest.json`](./08_training_dataset_using_Gemma4/gemma4_training_pairs_manifest.json)
- successful fine-tuned model run:
  - [`08_training_dataset_using_Gemma4/runs/embeddinggemma300m_high_medium_triplet_v2_bs8`](./08_training_dataset_using_Gemma4/runs/embeddinggemma300m_high_medium_triplet_v2_bs8)
- final evaluation outputs:
  - [`10_finetuned_embeddinggemma300m_evaluation/new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md`](./10_finetuned_embeddinggemma300m_evaluation/new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md)
  - [`10_finetuned_embeddinggemma300m_evaluation/adserver_1000_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md`](./10_finetuned_embeddinggemma300m_evaluation/adserver_1000_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md)

## Non-Canonical / Safe-To-Ignore Outputs

The repo also contains experiments, backups, and older proof-of-concept branches that are useful for history but should not be treated as the current system of record.

Examples:

- failed / backup / alternate run files inside [`07_fetched_url_content_categories_by_Gemma4`](./07_fetched_url_content_categories_by_Gemma4)
- old output scratch folders:
  - [`fetched_url_content_embedding_files`](./fetched_url_content_embedding_files)
  - [`outputs`](./outputs)
  - [`outputs_2`](./outputs_2)
- Python cache:
  - [`__pycache__`](./__pycache__)

These are not the recommended entry points for future work.

## Taxonomy Enrichment Work

Taxonomy work is documented in:

- [`taxonomy/readme.md`](./taxonomy/readme.md)

The taxonomy evolved through several versions:

- [`Content_Taxonomy_3.1.tsv`](./taxonomy/Content_Taxonomy_3.1.tsv)
- [`Content_Taxonomy_3.1_2.tsv`](./taxonomy/Content_Taxonomy_3.1_2.tsv)
- [`Content_Taxonomy_3.1_3.tsv`](./taxonomy/Content_Taxonomy_3.1_3.tsv)
- [`Content_Taxonomy_3.1_4.tsv`](./taxonomy/Content_Taxonomy_3.1_4.tsv)
- [`Content_Taxonomy_3.1_5.tsv`](./taxonomy/Content_Taxonomy_3.1_5.tsv)
- [`Content_Taxonomy_3.1_6.tsv`](./taxonomy/Content_Taxonomy_3.1_6.tsv)

The final recommended retrieval taxonomy is:

- [`Content_Taxonomy_3.1_6.tsv`](./taxonomy/Content_Taxonomy_3.1_6.tsv)

Main taxonomy improvements that were made:

- added `Path`
- added richer `Description`
- added `Keywords`
- improved sparse rows
- removed generic parent-level keyword noise
- separated `Common Confusers` from `Description`
- kept `Negative Keywords` and `Common Confusers` as metadata, not embedding text

Important taxonomy conclusion:

- for dense retrieval, embed only:
  - `Path`
  - `Description`
  - `Keywords`
- do not embed:
  - `Common Confusers`
  - `Negative Keywords`

## How To Consume The Main Folders

### [`01_plain_urls_files`](./01_plain_urls_files)

Raw URL inputs.

### [`02_fetched_url_content_files`](./02_fetched_url_content_files)

Fetched page content.

Use this folder when you need:

- source text for embeddings
- source text for GPT/Gemma labeling
- source text for training-data assembly

### [`03_fetched_url_content_embedding_files`](./03_fetched_url_content_embedding_files)

Stage-03 embeddings.

Use this folder when you need:

- precomputed embedding records
- query text used for retrieval
- source embeddings for FAISS and reranking experiments

### [`04_fetched_url_content_embedding_categories_files`](./04_fetched_url_content_embedding_categories_files)

Stage-04 FAISS outputs.

Use this folder when you need:

- top-N taxonomy candidates from embedding retrieval
- direct ANN behavior comparisons

Important note:

- [`new_urls_findings.md`](./04_fetched_url_content_embedding_categories_files/new_urls_findings.md) contains the early comparison of `bge-m3` vs `embeddinggemma-300m`

### [`05_fetched_url_content_embedding_categories_reranked_files`](./05_fetched_url_content_embedding_categories_reranked_files)

Stage-05 reranked outputs.

Use this folder when you need:

- stronger candidate ordering than FAISS alone
- weak-supervision teacher signals

### [`06_fetched_url_content_categories_by_GPT`](./06_fetched_url_content_categories_by_GPT)

GPT batch baseline outputs.

Use this folder when you need:

- highest-quality benchmark labels currently available in this repo
- OpenAI Batch request/tracker artifacts

Important limitation:

- this folder is small because GPT labeling was too expensive to scale broadly

### [`07_fetched_url_content_categories_by_Gemma4`](./07_fetched_url_content_categories_by_Gemma4)

Gemma4 local teacher outputs.

Use this folder when you need:

- scalable silver-label categorization outputs
- local-LLM teacher labels for training

Important note:

- use canonical `google/gemma-4-E4B-it` outputs
- ignore backup and failed variants

### [`08_training_dataset_using_Gemma4`](./08_training_dataset_using_Gemma4)

Training-dataset preparation and fine-tuning artifacts.

Use this folder when you need:

- consolidated Gemma-labeled dataset
- train/valid/test splits
- training pairs
- fine-tuning script
- model run artifacts

Important files:

- [`gemma4_training_dataset.jsonl`](./08_training_dataset_using_Gemma4/gemma4_training_dataset.jsonl)
- [`gemma4_training_manifest.json`](./08_training_dataset_using_Gemma4/gemma4_training_manifest.json)
- [`gemma4_training_splits_manifest.json`](./08_training_dataset_using_Gemma4/gemma4_training_splits_manifest.json)
- [`gemma4_training_pairs_manifest.json`](./08_training_dataset_using_Gemma4/gemma4_training_pairs_manifest.json)

### [`09_fetched_url_content_categories_by_Claude`](./09_fetched_url_content_categories_by_Claude)

Claude benchmark outputs.

Use this folder when you need:

- side-by-side model comparison data
- an additional hosted-model reference point

Practical note:

- this folder is not part of the main training pipeline
- it is useful as supporting evidence, not as the primary training source

### [`10_finetuned_embeddinggemma300m_evaluation`](./10_finetuned_embeddinggemma300m_evaluation)

Evaluation of the fine-tuned embedding model.

Use this folder when you need:

- FAISS outputs from the fine-tuned model
- comparison JSON and markdown summaries

## Legacy / Experimental Folders

These folders are part of the research history of the repo and should be read as supporting experiments rather than the final pipeline.

### [`benchmarking`](./benchmarking)

Contains performance/concurrency benchmarking for:

- embeddings
- rerankers
- ColBERT-style pooling experiments

Most relevant findings files:

- [`benchmarking/vllm_embedding_concurrency_test.md`](./benchmarking/vllm_embedding_concurrency_test.md)
- [`benchmarking/vllm_reranker_concurrency_test.md`](./benchmarking/vllm_reranker_concurrency_test.md)

### [`categorize_on_hosted_models`](./categorize_on_hosted_models)

Hosted-model categorization experiments outside the numbered pipeline.

Related findings live in:

- [`categorize_on_hosted_models_findings`](./categorize_on_hosted_models_findings)

### [`categorize_on_hosted_models_findings`](./categorize_on_hosted_models_findings)

Findings and benchmark notes for hosted-model categorization experiments.

### [`gpt_faiss_reranker_enriched`](./gpt_faiss_reranker_enriched)

Earlier GPT + FAISS + reranker prototype scripts and JSON experiment outputs.

This is useful for research history, but not the current canonical implementation.

### [`tiered_approach`](./tiered_approach)

Older tiered categorization experiments.

### [`adserver_old_approach`](./adserver_old_approach)

Older adserver-specific categorization experiments and outputs.

### [`ModelFiles`](./ModelFiles)

Contains model prompt / modelfile assets such as:

- [`Modelfile.content.taxonomy`](./ModelFiles/Modelfile.content.taxonomy)
- [`Modelfile.content_2.taxonomy`](./ModelFiles/Modelfile.content_2.taxonomy)

These are part of experimentation history, not the main numbered pipeline.

### [`research`](./research)

Reference notes used during exploration, including:

- Triton
- NVIDIA Dynamo
- two-tower retrieval concepts

## Where To Find The Main Findings

### Taxonomy findings

- [`taxonomy/readme.md`](./taxonomy/readme.md)

### Embedding-model retrieval findings

- [`04_fetched_url_content_embedding_categories_files/new_urls_findings.md`](./04_fetched_url_content_embedding_categories_files/new_urls_findings.md)

Key takeaway:

- `google/embeddinggemma-300m` had clearer FAISS separation than `bge-m3` on `new_urls`

### GPT cost findings

- [`content-fetching/gpt-cost.md`](./content-fetching/gpt-cost.md)

Key takeaway:

- basis run estimate: about `$6.16` for `63` URLs
- projected cost for `50K` URLs: about `$4.9K`

### Gemma serving findings

- [`content-fetching/gemma.md`](./content-fetching/gemma.md)

Key takeaway:

- `google/gemma-4-31B-it` was not practical for this workload
- `google/gemma-4-26B-A4B-it` was not practically usable here
- `google/gemma-4-E4B-it` was the workable local model

### Gemma vs GPT findings

- [`07_fetched_url_content_categories_by_Gemma4/new_urls_gemma4e4b_vs_gpt54.md`](./07_fetched_url_content_categories_by_Gemma4/new_urls_gemma4e4b_vs_gpt54.md)

Key takeaway:

- best Gemma result on `new_urls`: **`55.0 / 100`** vs GPT-5.4 baseline
- usable as a local teacher
- not equivalent to GPT

### Fine-tuned embedding model findings

- [`10_finetuned_embeddinggemma300m_evaluation/new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md`](./10_finetuned_embeddinggemma300m_evaluation/new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md)
- [`10_finetuned_embeddinggemma300m_evaluation/adserver_1000_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md`](./10_finetuned_embeddinggemma300m_evaluation/adserver_1000_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md)

Key takeaway:

- the first fine-tuned checkpoint did **not** beat the original `embeddinggemma-300m` baseline

### Benchmarking findings

- [`benchmarking/vllm_embedding_concurrency_test.md`](./benchmarking/vllm_embedding_concurrency_test.md)
- [`benchmarking/vllm_reranker_concurrency_test.md`](./benchmarking/vllm_reranker_concurrency_test.md)

Key takeaway:

- model and concurrency choices in the main pipeline were informed by direct vLLM throughput and stability benchmarking, not by guesswork

## Main Quantitative Conclusions

### 1. GPT was too expensive for broad teacher-label generation

From [`content-fetching/gpt-cost.md`](./content-fetching/gpt-cost.md):

- estimated `gpt-5.4` batch cost for `50K` URLs: about **`$4.9K`**

This made GPT suitable for:

- benchmark sets
- reference baselines

But not suitable for:

- large-scale teacher labeling in this project

### 2. Gemma 4 E4B was the only practical local teacher model

From [`content-fetching/gemma.md`](./content-fetching/gemma.md):

- larger Gemma 4 variants were either too memory-heavy or too slow
- `google/gemma-4-E4B-it` could run locally and complete categorization jobs

### 3. Gemma 4 E4B was usable, but not GPT-equivalent

From [`new_urls_gemma4e4b_vs_gpt54.md`](./07_fetched_url_content_categories_by_Gemma4/new_urls_gemma4e4b_vs_gpt54.md):

- best direct Gemma result on `new_urls`: **`55.0 / 100`** vs GPT baseline

Interpretation:

- good enough as a silver-label teacher
- not good enough to replace GPT as the reference benchmark

### 3a. Claude remained a side benchmark, not the main teacher source

Claude outputs were generated and preserved in:

- [`09_fetched_url_content_categories_by_Claude`](./09_fetched_url_content_categories_by_Claude)

But they were not used as the main teacher-label source for dataset creation or model training.

### 4. The first fine-tuned `embeddinggemma-300m` run did not improve retrieval

On `new_urls`:

- fine-tuned model score vs GPT: **`35.62 / 100`**
- original `embeddinggemma-300m` baseline vs GPT: **`38.13 / 100`**

On `adserver_1000_urls` against Gemma4:

- fine-tuned model score vs Gemma4: **`30.17 / 100`**
- original `embeddinggemma-300m` baseline vs Gemma4: **`31.59 / 100`**

Interpretation:

- the first fine-tuned checkpoint underperformed the original baseline on both evaluation slices

### 5. The project should stop here rather than continue without a better plan

The main research question was effectively answered:

- a straightforward Gemma-labeled triplet-loss fine-tuning pass on `embeddinggemma-300m` did **not** produce a better retrieval model in this setup

That is a valid result.

## POC Code Index

If someone needs to continue from the proof-of-concept code, these are the most important entry points.

### End-to-end pipeline scripts

- [`content-fetching/01_save_url_content.py`](./content-fetching/01_save_url_content.py)
- [`content-fetching/02_generate_url_embeddings.py`](./content-fetching/02_generate_url_embeddings.py)
- [`content-fetching/03_generate_faiss_taxonomy_categories.py`](./content-fetching/03_generate_faiss_taxonomy_categories.py)
- [`content-fetching/04_generate_reranked_taxonomy_categories.py`](./content-fetching/04_generate_reranked_taxonomy_categories.py)
- [`content-fetching/05_generate_categories_by_GPT.py`](./content-fetching/05_generate_categories_by_GPT.py)
- [`content-fetching/06_generate_categories_by_Gemma4.py`](./content-fetching/06_generate_categories_by_Gemma4.py)

### Training / evaluation scripts

- [`08_training_dataset_using_Gemma4/build_gemma4_training_dataset.py`](./08_training_dataset_using_Gemma4/build_gemma4_training_dataset.py)
- [`08_training_dataset_using_Gemma4/build_gemma4_training_splits.py`](./08_training_dataset_using_Gemma4/build_gemma4_training_splits.py)
- [`08_training_dataset_using_Gemma4/build_gemma4_training_pairs.py`](./08_training_dataset_using_Gemma4/build_gemma4_training_pairs.py)
- [`08_training_dataset_using_Gemma4/train_embeddinggemma300m.py`](./08_training_dataset_using_Gemma4/train_embeddinggemma300m.py)
- [`10_finetuned_embeddinggemma300m_evaluation/evaluate_finetuned_embeddinggemma300m.py`](./10_finetuned_embeddinggemma300m_evaluation/evaluate_finetuned_embeddinggemma300m.py)

## Where The Project Was Closed

This project was closed after:

1. building the full data pipeline
2. building the taxonomy-enriched retrieval setup
3. producing Gemma-based teacher labels at scale
4. preparing a large training dataset
5. training one full fine-tuned `embeddinggemma-300m` model
6. evaluating the fine-tuned model on actual retrieval slices
7. observing that it did not beat the existing embedding baseline

The project was **not** closed because the code was incomplete.

It was closed because the current research direction had already produced a clear first answer:

- the first fine-tuning recipe was not good enough to justify immediate continuation

## How To Resume This Research

If this work is resumed later, the following path is recommended.

### 1. Keep the current taxonomy

Do not restart taxonomy work from scratch.

Resume from:

- [`taxonomy/Content_Taxonomy_3.1_6.tsv`](./taxonomy/Content_Taxonomy_3.1_6.tsv)

### 2. Keep GPT only as a benchmark slice

Do not use GPT for large-scale teacher labeling unless cost is explicitly approved.

Use GPT only for:

- benchmark slices
- agreement checks
- sanity-check evaluation sets

### 3. Use stronger silver-label construction

If training is resumed, do not rely only on raw Gemma teacher labels.

Better options:

- use consensus between:
  - stage-05 reranker outputs
  - stage-07 Gemma outputs
- filter more aggressively by teacher confidence
- build higher-precision subsets first

### 4. Improve the training objective

The first attempt used a simple triplet-loss setup.

If resumed, try:

- multiple-negatives ranking loss
- in-batch negatives
- better positive/negative mining
- stronger use of stage-05 hard negatives
- pair weighting by teacher confidence

### 5. Evaluate deeper in the real pipeline

Future evaluation should not stop at raw FAISS-only comparison.

Resume with:

- fine-tuned embeddings
- stage-04 FAISS
- stage-05 reranking
- compare against GPT/Gemma benchmark slices

### 6. Make evaluation GPU-explicit from the start

One practical lesson from step 10:

- local evaluation can silently fall back to CPU unless device handling is explicit

The evaluator was updated to support:

- `--device cuda`

That should be kept for future runs.

## Practical Resume Checklist

If someone returns to this project later, the shortest sensible restart path is:

1. Read:
   - [`taxonomy/readme.md`](./taxonomy/readme.md)
   - [`content-fetching/readme.md`](./content-fetching/readme.md)
   - [`08_training_dataset_using_Gemma4/readme.md`](./08_training_dataset_using_Gemma4/readme.md)
   - [`10_finetuned_embeddinggemma300m_evaluation/readme.md`](./10_finetuned_embeddinggemma300m_evaluation/readme.md)
2. Use:
   - [`taxonomy/Content_Taxonomy_3.1_6.tsv`](./taxonomy/Content_Taxonomy_3.1_6.tsv)
3. Start from:
   - [`08_training_dataset_using_Gemma4/gemma4_training_dataset.jsonl`](./08_training_dataset_using_Gemma4/gemma4_training_dataset.jsonl)
   - [`08_training_dataset_using_Gemma4/high_medium_*`](./08_training_dataset_using_Gemma4)
4. Compare any new model against:
   - [`10_finetuned_embeddinggemma300m_evaluation/new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md`](./10_finetuned_embeddinggemma300m_evaluation/new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md)
   - [`10_finetuned_embeddinggemma300m_evaluation/adserver_1000_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md`](./10_finetuned_embeddinggemma300m_evaluation/adserver_1000_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md)

## Final Conclusion

This project successfully produced:

- a full categorization pipeline
- a cleaned ANN-friendly taxonomy
- large-scale local silver labels with Gemma 4 E4B
- a training dataset and a first fine-tuned embedding model

But the first fine-tuning outcome was negative:

- the trained `embeddinggemma-300m` checkpoint did **not** outperform the original `embeddinggemma-300m` baseline

That means the current project should be treated as:

- a completed proof of concept
- a finished first research cycle
- a solid base for future work, but not something that should continue unchanged
