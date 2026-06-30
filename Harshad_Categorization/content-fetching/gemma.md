# Gemma Run Findings

This note captures the practical findings from trying to run Gemma 4 models locally on this machine for taxonomy categorization via vLLM.

## Machine Context

- GPUs available: `2 x NVIDIA L40S`
- Typical usable VRAM per GPU seen during these runs: about `44.4 GiB`

## Goal

Use a Gemma 4 instruction-tuned model as a local replacement candidate for the GPT batch baseline used in stage 6.

Target serving shape:

- vLLM OpenAI-compatible server on `http://127.0.0.1:8000`
- stage-7 script:
  - [`06_generate_categories_by_Gemma4.py`](/home/harshad.mane/Harshad_Categorization/content-fetching/06_generate_categories_by_Gemma4.py)
- taxonomy-first prompt
- page content after taxonomy
- structured JSON output with top categories

## Models Tried

### `google/gemma-4-31B-it`

#### Single GPU attempt

Command pattern used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.95
```

Result:

- failed before serving
- first failure mode: startup free-memory check failed
- then, even after moving to GPU 1, model load failed with CUDA OOM

Observed failure:

- free memory on one L40S was slightly below requested memory target at `0.95`
- later failure during model load:
  - `CUDA out of memory`
  - attempted allocation around `442 MiB`

Conclusion:

- `google/gemma-4-31B-it` is **not viable on one L40S** in this environment

#### Two-GPU attempt

Command pattern used:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -u -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90
```

Result:

- got much further
- weights loaded
- but failed during KV-cache sizing because default `max_model_len=262144` was too large

Observed failure:

- available KV cache memory: about `8.52 GiB`
- required for max model len `262144`: about `11.18 GiB`
- vLLM estimated max feasible model length: about `192576`

Retried with:

```bash
--max-model-len 131072
```

Result:

- this corrected configuration **did work**
- vLLM eventually reached:
  - `Starting vLLM server on http://0.0.0.0:8000`
- confirmed listener on port `8000`

Working server shape:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -u -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072
```

But categorization result quality/latency was not usable:

- stage-7 run against `new_urls.jsonl` with `workers=32`, `timeout=300`
- all `63` valid content rows timed out
- only the single source-error row completed immediately

Observed output shape:

- `64` total rows
- `0 ok`
- `64 error`
- `63` rows:
  - `error_code = llm_request_error`
  - `error_type = ReadTimeout`
  - request time about `300000 ms`

Retry with:

- `workers=1`
- `timeout=900`
- same full body text

Result:

- process stayed alive
- output file stayed at `0` rows for a long time
- looked blocked on the first request

Conclusion:

- even though `31B` can be made to serve on two GPUs, it is **not practical for this prompt shape** in the current setup

## `google/gemma-4-26B-A4B-it`

### Single GPU attempt

Command pattern used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-26B-A4B-it \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90
```

Result:

- failed during model load with CUDA OOM

Observed failure:

- OOM while allocating around `484 MiB`
- single GPU still not enough for this model in current environment

Conclusion:

- `google/gemma-4-26B-A4B-it` is **not viable on one L40S** here

### Two-GPU attempt

Command pattern used:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -u -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-26B-A4B-it \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90
```

Result:

- got through distributed startup and entered model-load path
- no immediate OOM with the 2-GPU setup
- but did not reach serving state during our checks
- process tree stayed alive for a long time while `8000` never opened

Observed state:

- API server alive
- `VLLM::EngineCore` alive
- `VLLM::Worker_TP0` and `VLLM::Worker_TP1` alive
- no `lsof` listener on `8000`
- no final traceback captured in those long-running checks

Interpretation:

- likely hanging or stuck somewhere in initialization
- not a usable serving configuration yet

Conclusion:

- `google/gemma-4-26B-A4B-it` on two GPUs is **more promising than one GPU**, but still **not yet usable** in the current configuration

## `google/gemma-4-E4B-it`

### Single GPU attempt

Command pattern used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-E4B-it \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90
```

Observed so far:

- startup begins cleanly
- architecture resolves correctly
- no immediate OOM seen during the early checks
- not enough observation was completed here to record a final serving outcome in this note

This remains the **most practical next Gemma candidate** to continue with.

## Stage-7 Script Findings

Script:

- [`06_generate_categories_by_Gemma4.py`](/home/harshad.mane/Harshad_Categorization/content-fetching/06_generate_categories_by_Gemma4.py)

Prompt shape used:

- taxonomy first
- page content after taxonomy
- full taxonomy fields:
  - `Unique ID`
  - `Path`
  - `Description`
  - `Keywords`

Key finding:

- very large Gemma models combined with the full taxonomy-first prompt created extremely slow request behavior
- lowering concurrency from `32` to `1` was not enough to make `31B` practical

## Practical Recommendation

Recommended order going forward:

1. Try `google/gemma-4-E4B-it`
2. Keep low concurrency first:
   - `workers=1` or `2`
3. Reuse the stage-7 script and output folder:
   - [`07_fetched_url_content_categories_by_Gemma4`](/home/harshad.mane/Harshad_Categorization/07_fetched_url_content_categories_by_Gemma4)
4. Treat `31B` and `26B-A4B-it` as experimental only, not the default local path

## Bottom Line

- `31B`:
  - can be forced to serve on two GPUs with `--max-model-len 131072`
  - but request latency was not usable for our taxonomy categorization run
- `26B-A4B-it`:
  - does not fit on one GPU
  - on two GPUs it entered model loading but did not reach serving state in our checks
- `E4B-it`:
  - currently the best candidate to continue with
