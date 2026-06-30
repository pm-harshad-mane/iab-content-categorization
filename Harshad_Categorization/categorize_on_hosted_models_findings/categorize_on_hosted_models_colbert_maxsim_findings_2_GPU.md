# categorize_on_hosted_models_colbert_maxsim Findings (2 GPUs)

## Scope

Benchmarked `categorize_on_hosted_models_colbert_maxsim.py` on the default `994`-row input against the current 2-GPU hosted deployment.

Model and endpoint configuration used:

- Embedding model on port `8000`: `BAAI/bge-m3` via `/v1/embeddings`
- ColBERT model on port `8001`: `colbert-ir/colbertv2.0` via `/pooling`
- Worker counts tested: `4`, `8`, `16`, `24`, `32`, `48`, `64`, `80`, `96`, `128`

Placement checks before the benchmark showed one `vllm` engine core per GPU:

- GPU 0: `BAAI/bge-m3` engine, about `3275 MB` to `3321 MB` resident
- GPU 1: `colbert-ir/colbertv2.0` engine, about `1819 MB` resident

All runs produced the same outcome count:

- Total input rows: `994`
- Successful rows: `831`
- Failed rows: `163`

## Throughput Sweep

Primary goal: maximize throughput.

| Workers | Total Rows/s | Successful Rows/s | Script Runtime (s) |
| --- | ---: | ---: | ---: |
| 4 | 23.35 | 19.52 | 42.563 |
| 8 | 32.76 | 27.39 | 30.344 |
| 16 | 36.10 | 30.18 | 27.534 |
| 24 | 36.48 | 30.50 | 27.249 |
| 32 | 35.79 | 29.92 | 27.776 |
| 48 | 37.96 | 31.74 | 26.185 |
| 64 | 39.00 | 32.60 | 25.489 |
| 80 | 36.01 | 30.11 | 27.603 |
| 96 | 36.33 | 30.37 | 27.363 |
| 128 | 36.85 | 30.81 | 26.972 |

### Recommendation

Use `--concurrent-records 64` for this current ColBERT-MaxSim setup if the objective is pure throughput on the 2-GPU deployment.

Notes:

- `64` workers was the best measured point for both total rows/s and successful rows/s.
- The curve continues improving past `32` and peaks at `64`, then falls back at `80` and above.
- There is a broad tail from `48` through `128` where throughput stays near the peak, but latency worsens sharply as concurrency rises.
- This pipeline still saturates earlier than the earlier reranker-based approach, but not as early as the first truncated sweep suggested.

## Per-Step Timing By Worker Load

Mean and p95 latency below are per successful record.

| Workers | Total Mean | Total p95 | Content Embedding Mean | FAISS Mean | ColBERT Embed Mean | MaxSim CPU Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 179.09 ms | 310 ms | 32.62 ms | 2.10 ms | 132.63 ms | 6.95 ms |
| 8 | 245.51 ms | 408 ms | 42.14 ms | 3.33 ms | 176.45 ms | 15.75 ms |
| 16 | 430.65 ms | 740 ms | 49.38 ms | 5.11 ms | 329.30 ms | 36.87 ms |
| 24 | 644.77 ms | 1160 ms | 58.64 ms | 6.85 ms | 501.73 ms | 65.39 ms |
| 32 | 874.07 ms | 1540 ms | 72.78 ms | 8.52 ms | 680.22 ms | 98.54 ms |
| 48 | 1195.85 ms | 2199 ms | 89.35 ms | 10.74 ms | 933.83 ms | 142.96 ms |
| 64 | 1555.75 ms | 2853 ms | 116.43 ms | 12.13 ms | 1210.75 ms | 196.37 ms |
| 80 | 2079.18 ms | 3822 ms | 138.37 ms | 18.42 ms | 1631.01 ms | 263.85 ms |
| 96 | 2457.21 ms | 4458 ms | 149.04 ms | 17.53 ms | 1928.04 ms | 315.06 ms |
| 128 | 3097.29 ms | 5712 ms | 187.59 ms | 21.39 ms | 2470.78 ms | 383.68 ms |

Key takeaways:

- `colbert_embed` is the dominant cost at every worker count.
- `maxsim_cpu` stays small at low concurrency, but it grows substantially as the candidate reranking queue backs up.
- Throughput peaks at `64` while latency is already well above `1.5s` mean per successful record, which indicates queueing pressure before either GPU is heavily utilized.

## GPU Monitoring At Best Worker Count

GPU usage and memory were spot-sampled during a rerun at `64` workers using direct `nvidia-smi` calls. A continuous shell-loop sampler was not reliable in this environment, so the GPU summary below is based on `5` direct samples during that rerun.

Monitored rerun summaries:

- Monitored `64`-worker rerun: `35.07` total rows/s, `29.32` successful rows/s, script runtime `28.342s`

GPU summary from those direct samples:

- GPU 0 (`BAAI/bge-m3`): mean util `5.2%`, peak util `16%`, mean memory `3445 MB`, peak memory `3445 MB`
- GPU 1 (`colbert-ir/colbertv2.0`): mean util `1.2%`, peak util `3%`, mean memory `1819 MB`, peak memory `1819 MB`

Interpretation:

- GPU memory residency is stable and matches the one-engine-per-GPU deployment.
- The sampled GPU utilization is still low at the best-throughput point, which suggests this workload is not GPU-saturated at the current request shape.
- The throughput ceiling appears to come more from request/queue overhead and the expensive ColBERT stage than from raw GPU compute saturation.

## Summary

For the current `BAAI/bge-m3` + `colbert-ir/colbertv2.0` ColBERT-MaxSim pipeline on 2 GPUs, the best measured throughput was at `64` workers with `39.00` total rows/s and `32.60` successful rows/s. The ColBERT embedding call dominates end-to-end latency, and concurrency above `64` keeps latency climbing steeply while throughput falls back.

## Single-Instance Tuning Attempt

I tested a first single-instance tuning change on the vLLM servers:

- BGE on port `8000`: `--gpu-memory-utilization 0.92 --performance-mode throughput`
- ColBERT on port `8001`: `--gpu-memory-utilization 0.92 --performance-mode throughput`

Important execution detail:

- The first restart was invalid because both vLLM engine cores landed on GPU 0.
- Those measurements were discarded.
- I then reran the test with explicit GPU pinning:
  - `CUDA_VISIBLE_DEVICES=0` for `BAAI/bge-m3`
  - `CUDA_VISIBLE_DEVICES=1` for `colbert-ir/colbertv2.0`

Pinned launch commands used for the tuned test:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-m3 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.92 \
  --performance-mode throughput
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
  --model colbert-ir/colbertv2.0 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.92 \
  --performance-mode throughput
```

Comparison at `64` workers:

| Config | Total Rows/s | Successful Rows/s | Script Runtime (s) |
| --- | ---: | ---: | ---: |
| Baseline pinned config | 39.00 | 32.60 | 25.489 |
| Tuned `performance-mode throughput` config | 33.55 | 28.05 | 29.631 |

Conclusion from this tuning attempt:

- `--performance-mode throughput` plus the higher memory cap was a regression for this workload.
- The regression persisted even after fixing GPU placement.
- The next knobs to test should be explicit batching controls such as `--max-num-seqs` and `--max-num-batched-tokens`, not this throughput-mode preset.

## Single-Instance Batching Tuning

After the throughput-mode regression, I tested explicit batching-token limits instead.

Fresh baseline context:

- vLLM version: `0.19.0`
- Default OpenAI API scheduler values on this hardware are effectively:
  - `max_num_batched_tokens = 2048`
  - `max_num_seqs = 256`
- Fresh request-shape metrics after a pinned baseline run showed:
  - BGE mean prompt size: about `295` tokens (`444308 / 1504`)
  - ColBERT mean prompt size: about `186` tokens (`280141 / 1504`)

That means the default `2048` token cap is only enough for roughly:

- `6` to `7` average BGE requests per batch
- `11` average ColBERT requests per batch

### ColBERT `max-num-batched-tokens`

Pinned launch command for the improving ColBERT setting:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
  --model colbert-ir/colbertv2.0 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8001 \
  --max-num-batched-tokens 8192
```

`64`-worker results:

| Config | Total Rows/s | Successful Rows/s | Script Runtime (s) |
| --- | ---: | ---: | ---: |
| Baseline pinned rerun | 33.16 | 27.72 | 29.975 |
| ColBERT `4096` only | 34.57 | 28.90 | 28.750 |
| ColBERT `8192` only | 35.00 | 29.26 | 28.399 |

Takeaway:

- Raising only the ColBERT batching-token cap improved throughput.
- `8192` was better than `4096` on this workload.

### BGE `max-num-batched-tokens`

For `BAAI/bge-m3`, vLLM rejects `--max-num-batched-tokens 4096` because the model has `max_model_len = 8192`, and vLLM requires the batching-token limit to be at least the model length.

Tested BGE command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-m3 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 8192
```

Combined `64`-worker result with `BGE 8192` and `ColBERT 8192`:

| Config | Total Rows/s | Successful Rows/s | Script Runtime (s) |
| --- | ---: | ---: | ---: |
| ColBERT `8192` only | 35.00 | 29.26 | 28.399 |
| BGE `8192` + ColBERT `8192` | 34.22 | 28.61 | 29.051 |

Takeaway:

- Increasing the BGE batching-token cap to `8192` did not help this pipeline.
- The best configuration found so far is:
  - baseline BGE
  - ColBERT with `--max-num-batched-tokens 8192`

### Best Current Launch Commands

Best configuration found so far:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-m3 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
  --model colbert-ir/colbertv2.0 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8001 \
  --max-num-batched-tokens 8192
```

### GPU Monitoring For Best Current Config

I reran the best current config at `64` workers and took `6` direct `nvidia-smi` spot samples during the run.

Monitored run summary:

- Wall-clock result from the script: `29.053s`
- Throughput: `34.21` total rows/s, `28.60` successful rows/s

GPU summary from those spot samples:

- GPU 0 (`BAAI/bge-m3`): mean util `4.67%`, peak util `19%`, mean memory `3325 MB`, peak memory `3351 MB`
- GPU 1 (`colbert-ir/colbertv2.0`, `max-num-batched-tokens 8192`): mean util `1.00%`, peak util `3%`, mean memory `1987 MB`, peak memory `1987 MB`

Important caveat:

- These are sparse spot samples, not continuous profiling.
- They are good enough to confirm that the current best config still does not appear GPU-saturated, but they can miss short utilization spikes.

## Artifacts

- `/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models_colbert_maxsim_1000.jsonl`
- `/tmp/colbert_maxsim_w8.jsonl`
- `/tmp/colbert_maxsim_w16.jsonl`
- `/tmp/colbert_maxsim_w24.jsonl`
- `/tmp/colbert_maxsim_w32.jsonl`
- `/tmp/colbert_maxsim_w48.jsonl`
- `/tmp/colbert_maxsim_w64.jsonl`
- `/tmp/colbert_maxsim_w80_clean.jsonl`
- `/tmp/colbert_maxsim_w96.jsonl`
- `/tmp/colbert_maxsim_w128_clean.jsonl`
- `/tmp/colbert_maxsim_w64_monitored.jsonl`
- `/tmp/colbert_maxsim_tuned_pinned_w64.jsonl`
- `/tmp/colbert_maxsim_baseline_pinned_w64_again.jsonl`
- `/tmp/colbert_maxsim_colbert4096_w64.jsonl`
- `/tmp/colbert_maxsim_colbert8192_w64.jsonl`
- `/tmp/colbert_maxsim_bge8192_colbert8192_w64.jsonl`
- `/tmp/colbert_maxsim_best_monitored_w64.jsonl`
