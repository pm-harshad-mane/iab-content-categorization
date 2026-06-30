# Content Fetching Workflow

This project uses a 7-stage pipeline for URL processing.

Each record written in the fetched-content and embedding stages keeps a stable `url_hash` field so the same URL can be tracked reliably across files.

## Long-Running Jobs

For long-running jobs, use `nohup` so the process continues even if the terminal is closed.

Recommended pattern:

```bash
nohup python3 <script> <args...> > <log_file> 2>&1 & echo $!
```

Example for content fetching:

```bash
nohup python3 content-fetching/01_save_url_content.py \
  --urls-file 01_plain_urls_files/HarshardData_New_1M_06.csv \
  --output 02_fetched_url_content_files/HarshardData_New_1M_06.jsonl \
  --workers 124 \
  > 02_fetched_url_content_files/HarshardData_New_1M_06.nohup.log 2>&1 & echo $!
```

Going forward, long tasks in this workflow should be launched with `nohup`.

## Folder Convention

### `01_plain_urls_files`

- Input files live here.
- Each file should contain plain URLs, one per line.
- Example:
  - [`01_plain_urls_files/HarshardData_New_1M.csv`](/home/harshad.mane/Harshad_Categorization/01_plain_urls_files/HarshardData_New_1M.csv)

### `02_fetched_url_content_files`

- Output of the content-fetching stage.
- Each input URL file becomes a JSONL file with the same base name.
- Example:
  - input: `01_plain_urls_files/new_urls.txt`
  - output: `02_fetched_url_content_files/new_urls.jsonl`

### `03_fetched_url_content_embedding_files`

- Output of the embedding-generation stage.
- Each fetched-content JSONL file becomes an embedding JSONL file with the same base name.
- Example:
  - input: `02_fetched_url_content_files/new_urls.jsonl`
  - output: `03_fetched_url_content_embedding_files/new_urls__BAAI_bge-m3.jsonl`

### `04_fetched_url_content_embedding_categories_files`

- Output of the FAISS taxonomy-categorization stage.
- Each embedding JSONL file becomes a category JSONL file with a related name.
- Example:
  - input: `03_fetched_url_content_embedding_files/new_urls__BAAI_bge-m3.jsonl`
- output: `04_fetched_url_content_embedding_categories_files/new_urls__BAAI_bge-m3__faiss.jsonl`

### `05_fetched_url_content_embedding_categories_reranked_files`

- Output of the reranking stage.
- Each FAISS category JSONL file becomes a reranked JSONL file with the same base name plus the reranker model name.
- Example:
  - input: `04_fetched_url_content_embedding_categories_files/new_urls__google_embeddinggemma-300m__faiss.jsonl`
  - output: `05_fetched_url_content_embedding_categories_reranked_files/new_urls__google_embeddinggemma-300m__faiss_reranked_mixedbread-ai_mxbai-rerank-base-v2.jsonl`

### `06_fetched_url_content_categories_by_GPT`

- Output of the OpenAI Batch GPT baseline-categorization stage.
- Each fetched-content JSONL file becomes a GPT-category JSONL file with the same base name plus the cloud model name.
- This folder also stores:
  - the batch tracker file
  - batch request JSONL payloads
  - downloaded OpenAI batch output/error files
- Example:
  - input: `02_fetched_url_content_files/new_urls.jsonl`
  - output: `06_fetched_url_content_categories_by_GPT/new_urls__gpt-5.4.jsonl`

### `07_fetched_url_content_categories_by_Gemma4`

- Output of the local Gemma 4 categorization stage.
- Each fetched-content JSONL file becomes a local-LLM category JSONL file with the same base name plus the hosted local model name.
- Example:
  - input: `02_fetched_url_content_files/new_urls.jsonl`
  - output: `07_fetched_url_content_categories_by_Gemma4/new_urls__google_gemma-4-E4B-it.jsonl`

## Step 1: Prepare Plain URL Files

Place the raw URL files in:

```text
01_plain_urls_files/
```

Format:

- one URL per line
- no JSON
- no extra columns required

## Step 2: Fetch Page Content

Use [`01_save_url_content.py`](/home/harshad.mane/Harshad_Categorization/content-fetching/01_save_url_content.py) to fetch page content and write JSONL output.

General pattern:

```bash
python3 content-fetching/01_save_url_content.py \
  --urls-file 01_plain_urls_files/<input_file> \
  --output 02_fetched_url_content_files/<same_base_name>.jsonl
```

Example:

```bash
python3 content-fetching/01_save_url_content.py \
  --urls-file 01_plain_urls_files/new_urls.txt \
  --output 02_fetched_url_content_files/new_urls.jsonl
```

Optional worker override:

```bash
python3 content-fetching/01_save_url_content.py \
  --urls-file 01_plain_urls_files/urls_may_14_26.txt \
  --output 02_fetched_url_content_files/urls_may_14_26.jsonl \
  --workers 64
```

Notes:

- The script is resumable.
- Output records include `url_hash`.
- Both successful and failed fetches are written to JSONL.

## Step 3: Generate Embeddings

Use [`02_generate_url_embeddings.py`](/home/harshad.mane/Harshad_Categorization/content-fetching/02_generate_url_embeddings.py) to create embeddings from the fetched-content JSONL files.

Before running this step, start a local vLLM embedding server.

Example with `BAAI/bge-m3` on port `8000`:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-m3 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

Example with `google/embeddinggemma-300m` on port `8000`:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model google/embeddinggemma-300m \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

General pattern:

```bash
python3 content-fetching/02_generate_url_embeddings.py \
  --model <embedding_model> \
  --input-files 02_fetched_url_content_files/<same_base_name>.jsonl \
  --output-dir 03_fetched_url_content_embedding_files
```

Example with `BAAI/bge-m3`:

```bash
python3 content-fetching/02_generate_url_embeddings.py \
  --model BAAI/bge-m3 \
  --input-files 02_fetched_url_content_files/new_urls.jsonl \
  --output-dir 03_fetched_url_content_embedding_files
```

High-concurrency example:

```bash
python3 content-fetching/02_generate_url_embeddings.py \
  --model BAAI/bge-m3 \
  --input-files 02_fetched_url_content_files/urls_may_14_26.jsonl \
  --output-dir 03_fetched_url_content_embedding_files \
  --workers 1000
```

Example with `google/embeddinggemma-300m`:

```bash
python3 content-fetching/02_generate_url_embeddings.py \
  --model google/embeddinggemma-300m \
  --input-files 02_fetched_url_content_files/new_urls.jsonl \
  --output-dir 03_fetched_url_content_embedding_files \
  --workers 1000
```

Output naming:

- input:
  - `02_fetched_url_content_files/new_urls.jsonl`
- output:
  - `03_fetched_url_content_embedding_files/new_urls__BAAI_bge-m3.jsonl`
  - `03_fetched_url_content_embedding_files/new_urls__google_embeddinggemma-300m.jsonl`

Notes:

- Each embedding record also keeps the same `url_hash` from the fetched-content stage.
- This makes it easy to join:
  - plain URL source
  - fetched content
  - embeddings
  for the same URL.

## Step 4: Generate Top-N Taxonomy Categories With FAISS

Use [`03_generate_faiss_taxonomy_categories.py`](/home/harshad.mane/Harshad_Categorization/content-fetching/03_generate_faiss_taxonomy_categories.py) to:

- load the taxonomy from [`taxonomy/Content_Taxonomy_3.1_6.tsv`](/home/harshad.mane/Harshad_Categorization/taxonomy/Content_Taxonomy_3.1_6.tsv)
- build a taxonomy embedding index with FAISS
- retrieve top taxonomy categories for each embedding record

This stage uses taxonomy text built from:

- `Path`
- `Description`
- `Keywords`

It explicitly does **not** use:

- `Common Confusers`
- `Negative Keywords`

while building the taxonomy embedding index.

Before running this step, start a local vLLM embedding server with the same embedding model used in the input embedding file.

Example with `BAAI/bge-m3` on port `8000`:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-m3 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

Example with `google/embeddinggemma-300m` on port `8000`:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model google/embeddinggemma-300m \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

General pattern:

```bash
python3 content-fetching/03_generate_faiss_taxonomy_categories.py \
  --input-files 03_fetched_url_content_embedding_files/<embedding_file>.jsonl \
  --output-dir 04_fetched_url_content_embedding_categories_files \
  --model <embedding_model> \
  --concurrent-records 500 \
  --top-k 10
```

Example:

```bash
python3 content-fetching/03_generate_faiss_taxonomy_categories.py \
  --input-files 03_fetched_url_content_embedding_files/new_urls__BAAI_bge-m3.jsonl \
  --output-dir 04_fetched_url_content_embedding_categories_files \
  --model BAAI/bge-m3 \
  --concurrent-records 500 \
  --top-k 10
```

Example with multiple input embedding files in one run:

```bash
python3 content-fetching/03_generate_faiss_taxonomy_categories.py \
  --input-files \
    03_fetched_url_content_embedding_files/adserver_1000_urls__BAAI_bge-m3.jsonl \
    03_fetched_url_content_embedding_files/urls_may_14_26__BAAI_bge-m3.jsonl \
  --output-dir 04_fetched_url_content_embedding_categories_files \
  --model BAAI/bge-m3 \
  --concurrent-records 500 \
  --top-k 10
```

`--input-files` accepts one or more embedding JSONL files. The script will generate one matching FAISS output file per input file.

Output naming:

- input:
  - `03_fetched_url_content_embedding_files/new_urls__BAAI_bge-m3.jsonl`
- output:
  - `04_fetched_url_content_embedding_categories_files/new_urls__BAAI_bge-m3__faiss.jsonl`

Notes:

- Each output record keeps the same `url_hash`.
- Each output record also keeps:
  - `embedding_model`
  - `model_details`
  - `faiss_top_k`
  - `top_categories`
  - `timing_ms`
- Per-record timing includes:
  - `faiss_search`
  - `total`

## Step 5: Rerank FAISS Categories With A Hosted Reranker

Use [`04_generate_reranked_taxonomy_categories.py`](/home/harshad.mane/Harshad_Categorization/content-fetching/04_generate_reranked_taxonomy_categories.py) to:

- read FAISS category files from stage 04
- join each FAISS record back to the matching embedding record from stage 03 using `url_hash`
- use the stage-03 `query_text` as the reranker query
- use FAISS top categories as reranker documents
- keep the top `5` reranked categories in the output

Reranker document shape:

- `path: <category path>`
- `description: <category description>`

Before running this step, start a local vLLM reranker server.

Example with `mixedbread-ai/mxbai-rerank-base-v2` on port `8000`:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model mixedbread-ai/mxbai-rerank-base-v2 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

General pattern:

```bash
python3 content-fetching/04_generate_reranked_taxonomy_categories.py \
  --input-files 04_fetched_url_content_embedding_categories_files/<faiss_file>.jsonl \
  --embedding-input-dir 03_fetched_url_content_embedding_files \
  --output-dir 05_fetched_url_content_embedding_categories_reranked_files \
  --reranker-model <reranker_model> \
  --api-base http://127.0.0.1:8000 \
  --concurrent-records 500 \
  --top-k 5
```

Example:

```bash
python3 content-fetching/04_generate_reranked_taxonomy_categories.py \
  --input-files 04_fetched_url_content_embedding_categories_files/new_urls__google_embeddinggemma-300m__faiss.jsonl \
  --embedding-input-dir 03_fetched_url_content_embedding_files \
  --output-dir 05_fetched_url_content_embedding_categories_reranked_files \
  --reranker-model mixedbread-ai/mxbai-rerank-base-v2 \
  --api-base http://127.0.0.1:8000 \
  --concurrent-records 500 \
  --top-k 5
```

Output naming:

- input:
  - `04_fetched_url_content_embedding_categories_files/new_urls__google_embeddinggemma-300m__faiss.jsonl`
- output:
  - `05_fetched_url_content_embedding_categories_reranked_files/new_urls__google_embeddinggemma-300m__faiss_reranked_mixedbread-ai_mxbai-rerank-base-v2.jsonl`

Notes:

- Each output record keeps the same `url_hash`.
- Each output record also keeps:
  - `embedding_model`
  - `reranker_model`
  - `faiss_top_categories`
  - `reranked_top_categories`
  - `model_details`
  - `timing_ms`
- This step uses:
  - page `query_text` from stage 03
  - FAISS top categories from stage 04
- Recommended concurrency for this step is `500`.

## Step 6: Generate GPT Baseline Categories With OpenAI Batch

Use [`05_generate_categories_by_GPT.py`](/home/harshad.mane/Harshad_Categorization/content-fetching/05_generate_categories_by_GPT.py) to:

- read fetched-content JSONL files directly from stage 02
- use [`taxonomy/Content_Taxonomy_3.1_6.tsv`](/home/harshad.mane/Harshad_Categorization/taxonomy/Content_Taxonomy_3.1_6.tsv)
- submit one OpenAI Batch job per input file
- keep track of:
  - uploaded OpenAI input file id
  - batch id
  - batch status
  - downloaded output/error file ids
  - final local output path
- finalize completed batch jobs into enriched JSONL output records

This stage is asynchronous. The typical flow is:

1. `submit`
2. `sync`
3. `finalize`

Required environment variable:

```bash
export OPENAI_API_KEY=...
```

Recommended cloud model for this stage:

```text
gpt-5.4
```

Why:

- `gpt-5.4` is the lower-cost baseline teacher model we currently prefer for this stage.
- The script also supports `gpt-5.5` if you want a stronger but more expensive baseline.
- The selected cloud model name is saved in each final output record as `cloud_model`.
- Additional batch/model metadata is saved under `model_details`.

The script keeps model output minimal:

- top `5` categories
- only `unique_id` and `score` are requested from the model
- category details are added locally from the taxonomy TSV during finalization

Submit a batch job:

```bash
python3 content-fetching/05_generate_categories_by_GPT.py submit \
  --input-files 02_fetched_url_content_files/new_urls.jsonl \
  --output-dir 06_fetched_url_content_categories_by_GPT \
  --model gpt-5.4 \
  --top-k 5
```

What `submit` does:

- builds one OpenAI Batch request file for each input content JSONL file
- uploads that request file to OpenAI with `purpose=batch`
- creates the batch job against `/v1/responses`
- stores tracker information locally in:
  - `06_fetched_url_content_categories_by_GPT/batch_tracker.jsonl`

Sync tracker status from OpenAI:

```bash
python3 content-fetching/05_generate_categories_by_GPT.py sync \
  --output-dir 06_fetched_url_content_categories_by_GPT
```

What `sync` does:

- reads the local tracker file
- fetches latest batch status from OpenAI for each tracked job
- updates:
  - `status`
  - `batch_status`
  - `openai_output_file_id`
  - `openai_error_file_id`
  - request counts and timestamps when available

Finalize completed jobs and write the enriched output JSONL:

```bash
python3 content-fetching/05_generate_categories_by_GPT.py finalize \
  --output-dir 06_fetched_url_content_categories_by_GPT
```

What `finalize` does:

- downloads the completed OpenAI batch output file
- downloads the OpenAI batch error file when present
- parses the model output
- enriches returned `unique_id` values with taxonomy details from:
  - [`taxonomy/Content_Taxonomy_3.1_6.tsv`](/home/harshad.mane/Harshad_Categorization/taxonomy/Content_Taxonomy_3.1_6.tsv)
- writes the final enriched JSONL output file for each completed batch

Output naming:

- input:
  - `02_fetched_url_content_files/new_urls.jsonl`
- output:
  - `06_fetched_url_content_categories_by_GPT/new_urls__gpt-5.4.jsonl`

Tracker and batch artifacts:

- tracker:
  - `06_fetched_url_content_categories_by_GPT/batch_tracker.jsonl`
- request payloads:
  - `06_fetched_url_content_categories_by_GPT/requests/*.jsonl`
- downloaded batch outputs:
  - `06_fetched_url_content_categories_by_GPT/downloads/*.jsonl`

Notes:

- Each final output record keeps the same `url_hash`.
- Source rows that were already `status="error"` in stage 02 are preserved as error rows in stage 06.
- Final successful rows keep:
  - `cloud_model`
  - `gpt_top_categories`
  - `usage`
  - `model_details`
- `model_details` includes batch-related identifiers and status information, such as:
  - `batch_id`
  - `input_file_id`
  - `output_file_id`
  - `error_file_id`
  - `completion_window`
- The taxonomy prompt uses only:
  - `Unique ID`
  - `Path`
  - `Description`
  - `Keywords`
- The prompt is structured with:
  - taxonomy first
  - page content after it
- The request body also includes a stable `prompt_cache_key` so repeated taxonomy prefixes are better aligned for prompt caching.

## Step 7: Generate Local Gemma 4 Baseline Categories With vLLM

Use [`06_generate_categories_by_Gemma4.py`](/home/harshad.mane/Harshad_Categorization/content-fetching/06_generate_categories_by_Gemma4.py) to:

- read fetched-content JSONL files directly from stage 02
- use [`taxonomy/Content_Taxonomy_3.1_6.tsv`](/home/harshad.mane/Harshad_Categorization/taxonomy/Content_Taxonomy_3.1_6.tsv)
- send taxonomy-first prompts to a locally hosted `google/gemma-4-E4B-it` vLLM server
- write one output JSONL file per input file
- resume safely by `url_hash` if the job is restarted

This stage is synchronous, but it is still effectively batch-style processing because one run processes an entire content file with configurable concurrent requests against vLLM.

Recommended local model for this stage:

```text
google/gemma-4-E4B-it
```

Recommended vLLM launch command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-E4B-it \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90
```

Recommended detached launch for long runs:

```bash
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python3 -u -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-E4B-it \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  > 07_fetched_url_content_categories_by_Gemma4/gemma4_e4b_vllm.nohup.log 2>&1 & echo $!
```

Readiness check:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Wait until the server responds before starting [`06_generate_categories_by_Gemma4.py`](/home/harshad.mane/Harshad_Categorization/content-fetching/06_generate_categories_by_Gemma4.py).

Recommended run pattern for long jobs:

```bash
nohup python3 content-fetching/06_generate_categories_by_Gemma4.py \
  --input-files 02_fetched_url_content_files/new_urls.jsonl \
  --output-dir 07_fetched_url_content_categories_by_Gemma4 \
  --model google/gemma-4-E4B-it \
  --workers 32 \
  --top-k 5 \
  > 07_fetched_url_content_categories_by_Gemma4/new_urls__google_gemma-4-E4B-it.nohup.log 2>&1 & echo $!
```

Direct run example:

```bash
python3 content-fetching/06_generate_categories_by_Gemma4.py \
  --input-files 02_fetched_url_content_files/new_urls.jsonl \
  --output-dir 07_fetched_url_content_categories_by_Gemma4 \
  --model google/gemma-4-E4B-it \
  --workers 32 \
  --top-k 5
```

Output naming:

- input:
  - `02_fetched_url_content_files/new_urls.jsonl`
- output:
  - `07_fetched_url_content_categories_by_Gemma4/new_urls__google_gemma-4-E4B-it.jsonl`

Notes:

- Each final output record keeps the same `url_hash`.
- Source rows that were already `status="error"` in stage 02 are preserved as error rows here too.
- Final successful rows keep:
  - `local_model`
  - `llm_top_categories`
  - `usage`
  - `model_details`
  - `timing_ms`
- The taxonomy prompt uses only:
  - `Unique ID`
  - `Path`
  - `Description`
  - `Keywords`
- The prompt is structured with:
  - taxonomy first
  - page content after it
- This layout keeps the repeated taxonomy prefix stable so vLLM prefix caching can help when it is enabled on the server.

## End-to-End Example

Fetch content:

```bash
python3 content-fetching/01_save_url_content.py \
  --urls-file 01_plain_urls_files/new_urls.txt \
  --output 02_fetched_url_content_files/new_urls.jsonl
```

Generate embeddings:

```bash
python3 content-fetching/02_generate_url_embeddings.py \
  --model BAAI/bge-m3 \
  --input-files 02_fetched_url_content_files/new_urls.jsonl \
  --output-dir 03_fetched_url_content_embedding_files \
  --workers 1000
```

Generate top categories with FAISS:

```bash
python3 content-fetching/03_generate_faiss_taxonomy_categories.py \
  --input-files 03_fetched_url_content_embedding_files/new_urls__BAAI_bge-m3.jsonl \
  --output-dir 04_fetched_url_content_embedding_categories_files \
  --model BAAI/bge-m3 \
  --concurrent-records 500 \
  --top-k 10
```

Rerank top categories:

```bash
python3 content-fetching/04_generate_reranked_taxonomy_categories.py \
  --input-files 04_fetched_url_content_embedding_categories_files/new_urls__BAAI_bge-m3__faiss.jsonl \
  --embedding-input-dir 03_fetched_url_content_embedding_files \
  --output-dir 05_fetched_url_content_embedding_categories_reranked_files \
  --reranker-model mixedbread-ai/mxbai-rerank-base-v2 \
  --api-base http://127.0.0.1:8000 \
  --concurrent-records 500 \
  --top-k 5
```

Submit GPT baseline batch:

```bash
python3 content-fetching/05_generate_categories_by_GPT.py submit \
  --input-files 02_fetched_url_content_files/new_urls.jsonl \
  --output-dir 06_fetched_url_content_categories_by_GPT \
  --model gpt-5.4 \
  --top-k 5
```

Sync GPT baseline batch:

```bash
python3 content-fetching/05_generate_categories_by_GPT.py sync \
  --output-dir 06_fetched_url_content_categories_by_GPT
```

Finalize GPT baseline batch after completion:

```bash
python3 content-fetching/05_generate_categories_by_GPT.py finalize \
  --output-dir 06_fetched_url_content_categories_by_GPT
```

Run the local Gemma 4 baseline:

```bash
python3 content-fetching/06_generate_categories_by_Gemma4.py \
  --input-files 02_fetched_url_content_files/new_urls.jsonl \
  --output-dir 07_fetched_url_content_categories_by_Gemma4 \
  --model google/gemma-4-E4B-it \
  --workers 32 \
  --top-k 5
```

## Recommended Naming Rule

Keep the same base filename across stages:

- `01_plain_urls_files/<name>.txt` or `<name>.csv`
- `02_fetched_url_content_files/<name>.jsonl`
- `03_fetched_url_content_embedding_files/<name>__<model>.jsonl`
- `04_fetched_url_content_embedding_categories_files/<name>__<model>__faiss.jsonl`
- `05_fetched_url_content_embedding_categories_reranked_files/<name>__<model>__faiss_reranked_<reranker-model>.jsonl`
- `06_fetched_url_content_categories_by_GPT/<name>__<cloud-model>.jsonl`
- `07_fetched_url_content_categories_by_Gemma4/<name>__<local-model>.jsonl`

This makes it easy to trace:

- raw URLs
- fetched content
- generated embeddings
- retrieved taxonomy categories
- reranked taxonomy categories
- GPT baseline taxonomy categories
- local Gemma 4 baseline taxonomy categories

for the same dataset.

Use `url_hash` as the stable identifier across stages.
