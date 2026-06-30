# categorize_on_hosted_models Findings (2 GPUs)

## Scope

Benchmarked `categorize_on_hosted_models.py` again using the default input file (`994` rows) against the hosted embedding service on port `8000` and hosted reranker service on port `8001`, after moving the two models onto separate GPUs.

Model configuration used:

- Embedding model on port `8000`: `BAAI/bge-base-en-v1.5`
- Reranker model on port `8001`: `BAAI/bge-reranker-v2-m3`

Placement checks before the benchmark confirmed one `vllm` engine core on each GPU:

- GPU 0: one `VLLM::EngineCore` using about `932 MB` at idle
- GPU 1: one `VLLM::EngineCore` using about `1934 MB` at idle

All runs produced the same outcome count:

- Total input rows: `994`
- Successful rows: `831`
- Failed rows: `163`

## Throughput Sweep

Primary goal: maximize throughput.

| Workers | Total Rows/s | Successful Rows/s | Wall Time (s) |
| --- | ---: | ---: | ---: |
| 4 | 43.48 | 36.35 | 22.86 |
| 8 | 65.48 | 54.74 | 15.18 |
| 16 | 70.95 | 59.31 | 14.01 |
| 32 | 71.31 | 59.61 | 13.94 |
| 48 | 71.41 | 59.70 | 13.92 |
| 64 | 71.25 | 59.57 | 13.95 |
| 80 | 70.65 | 59.06 | 14.07 |
| 96 | 71.56 | 59.83 | 13.89 |
| 112 | 71.61 | 59.87 | 13.88 |
| 128 | 70.15 | 58.65 | 14.17 |
| 160 | 69.61 | 58.19 | 14.28 |
| 192 | 70.00 | 58.52 | 14.20 |
| 256 | 71.15 | 59.48 | 13.97 |
| 384 | 69.80 | 58.36 | 14.24 |
| 500 | 71.51 | 59.78 | 13.90 |

### Recommendation

Use `--concurrent-records 112` if the objective is pure throughput on this 2-GPU deployment.

Notes:

- `112` workers was the best measured wall-clock point, but the entire plateau from roughly `32` through `256` is very tight.
- Throughput gains above `32` workers are marginal, while per-record latency keeps rising.
- Pushing past `128` workers did not improve throughput and generally made latency materially worse.
- `500` workers stayed within the same wall-clock plateau, but with much worse per-record latency than `112`.

## Per-Step Timing By Worker Load

The tables below show mean and p95 latency per successful record for every worker count tested.

### 4 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 101.29 | 156 |
| Embedding | 27.10 | 50 |
| Embedding API only | 19.63 | 27 |
| FAISS search | 1.34 | 3 |
| Rerank | 72.41 | 102 |
| Rerank API only | 72.38 | 102 |

### 8 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 131.75 | 202 |
| Embedding | 31.04 | 65 |
| Embedding API only | 21.66 | 33 |
| FAISS search | 1.43 | 3 |
| Rerank | 98.92 | 147 |
| Rerank API only | 98.89 | 147 |

### 16 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 242.19 | 386 |
| Embedding | 36.93 | 216 |
| Embedding API only | 24.53 | 50 |
| FAISS search | 1.65 | 4 |
| Rerank | 203.15 | 279 |
| Rerank API only | 203.11 | 279 |

### 32 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 478.80 | 652 |
| Embedding | 41.76 | 233 |
| Embedding API only | 28.90 | 77 |
| FAISS search | 1.62 | 3 |
| Rerank | 435.03 | 527 |
| Rerank API only | 435.00 | 527 |

### 48 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 704.50 | 922 |
| Embedding | 48.77 | 236 |
| Embedding API only | 34.11 | 212 |
| FAISS search | 1.97 | 4 |
| Rerank | 653.34 | 782 |
| Rerank API only | 653.31 | 782 |

### 64 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 947.69 | 1188 |
| Embedding | 60.32 | 261 |
| Embedding API only | 39.87 | 219 |
| FAISS search | 2.12 | 5 |
| Rerank | 884.85 | 1046 |
| Rerank API only | 884.81 | 1045 |

### 80 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 1165.98 | 1401 |
| Embedding | 61.69 | 258 |
| Embedding API only | 43.43 | 222 |
| FAISS search | 1.91 | 5 |
| Rerank | 1102.03 | 1298 |
| Rerank API only | 1101.99 | 1298 |

### 96 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 1383.05 | 1642 |
| Embedding | 54.69 | 249 |
| Embedding API only | 38.31 | 212 |
| FAISS search | 2.06 | 5 |
| Rerank | 1325.94 | 1542 |
| Rerank API only | 1325.91 | 1542 |

### 112 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 1568.17 | 1899 |
| Embedding | 64.22 | 250 |
| Embedding API only | 43.01 | 173 |
| FAISS search | 3.13 | 11 |
| Rerank | 1500.47 | 1813 |
| Rerank API only | 1500.43 | 1813 |

### 128 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 1801.02 | 2323 |
| Embedding | 91.73 | 475 |
| Embedding API only | 61.17 | 301 |
| FAISS search | 3.11 | 11 |
| Rerank | 1705.80 | 2275 |
| Rerank API only | 1705.75 | 2275 |

### 160 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 2204.87 | 2654 |
| Embedding | 82.92 | 333 |
| Embedding API only | 57.72 | 256 |
| FAISS search | 3.56 | 16 |
| Rerank | 2118.04 | 2532 |
| Rerank API only | 2118.00 | 2532 |

### 192 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 2554.55 | 3405 |
| Embedding | 131.18 | 616 |
| Embedding API only | 88.87 | 372 |
| FAISS search | 6.45 | 30 |
| Rerank | 2416.56 | 3314 |
| Rerank API only | 2416.52 | 3313 |

### 256 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 3113.45 | 4164 |
| Embedding | 140.53 | 520 |
| Embedding API only | 96.39 | 338 |
| FAISS search | 6.81 | 30 |
| Rerank | 2965.73 | 4039 |
| Rerank API only | 2965.70 | 4039 |

### 384 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 4103.61 | 6716 |
| Embedding | 378.77 | 1388 |
| Embedding API only | 248.98 | 897 |
| FAISS search | 14.38 | 62 |
| Rerank | 3710.15 | 6675 |
| Rerank API only | 3710.12 | 6675 |

### 500 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 4193.99 | 7347 |
| Embedding | 237.94 | 687 |
| Embedding API only | 162.55 | 464 |
| FAISS search | 12.93 | 44 |
| Rerank | 3942.84 | 7170 |
| Rerank API only | 3942.79 | 7170 |

Key takeaways:

- The reranker remains the dominant cost in the pipeline.
- Moving the models onto separate GPUs did not change the shape of the curve: throughput plateaus early, then latency expands rapidly as worker count rises.
- `112` beats `256` on throughput while cutting mean per-record total latency roughly in half.
- `500` workers is still on the same throughput plateau, but its latency profile is substantially worse than both `112` and `256`.

## Monitored Run At 112 Workers

A monitored rerun at `112` workers was used only to inspect GPU activity. The sampler itself adds some overhead, so its throughput numbers should not replace the sweep table above.

Monitored run summary:

- Wall time: `14.32s`
- Script-reported runtime: `13.697s`
- Throughput: `72.57` total rows/s, `60.67` successful rows/s

Latency per successful record:

- Total: mean `1614.36 ms`, p95 `1966 ms`
- Embedding: mean `55.88 ms`, p95 `258 ms`
- Embedding API only: mean `37.80 ms`, p95 `182 ms`
- FAISS search: mean `2.82 ms`, p95 `11 ms`
- Rerank: mean `1555.28 ms`, p95 `1903 ms`
- Rerank API only: mean `1555.23 ms`, p95 `1903 ms`

GPU summary from the monitor log:

- GPU 0: mean util `0.8%`, peak `6%`, memory `2135 MB`
- GPU 1: mean util `12.0%`, peak `100%`, memory `3005 MB`

Interpretation:

- Both models were deployed on separate GPUs, but the embedding side remained very light relative to reranking.
- GPU 1 is still the meaningful bottleneck under load.
- The second GPU removes co-location, but it does not create a large end-to-end throughput gain for this workload because the reranker continues to dominate.

## Monitored Run At 500 Workers

A monitored rerun at `500` workers was added to compare the very high-concurrency case directly against the earlier plateau winner.

Monitored run summary:

- Wall time: `14.15s`
- Script-reported runtime: `13.506s`
- Throughput: `73.60` total rows/s, `61.53` successful rows/s

Latency per successful record:

- Total: mean `4325.92 ms`, p95 `7429 ms`
- Embedding: mean `247.95 ms`, p95 `770 ms`
- Embedding API only: mean `166.71 ms`, p95 `486 ms`
- FAISS search: mean `15.30 ms`, p95 `56 ms`
- Rerank: mean `4062.38 ms`, p95 `7395 ms`
- Rerank API only: mean `4062.34 ms`, p95 `7395 ms`

GPU summary from the monitor log:

- GPU 0: mean util `1.2%`, peak `20%`, memory `2135 MB`
- GPU 1: mean util `10.92%`, peak `100%`, memory `3005 MB`

Interpretation:

- Even at `500` workers, the extra concurrency did not materially shift load onto the embedding GPU.
- The reranker GPU remains the limiting stage, and the queueing cost shows up as a large latency increase rather than a throughput gain.
- The monitored run shows why `500` is not a better operating point despite similar wall-clock throughput.

## Comparison To The Prior Single-GPU Findings

Compared with the earlier findings in `categorize_on_hosted_models_findings.md`:

- Prior best measured throughput: `73.09` total rows/s and `61.10` successful rows/s at `256` workers
- Current best measured throughput: `71.61` total rows/s and `59.87` successful rows/s at `112` workers
- Delta versus the prior best: `-2.02%` total throughput and `-2.01%` successful throughput
- Additional `500`-worker run: `71.51` total rows/s and `59.78` successful rows/s, which is still below the prior single-GPU best

Interpretation:

- The 2-GPU deployment did not improve measured wall-clock throughput in this rerun.
- The practical benefit here is a lower best-concurrency point (`112` instead of `256`) rather than a step-change in throughput.
- The dominant limiter is still the reranker stage, not GPU sharing between the two hosted models.

## Artifacts

Relevant outputs and logs:

- `/tmp/categorize_worker_sweep_2gpus_20260420_raw.tsv`
- `/tmp/categorize_worker_tiebreak_2gpus_20260420_raw.tsv`
- `/tmp/categorized_2gpus_w112_20260420.jsonl`
- `/tmp/categorized_2gpus_w112_20260420.stdout.log`
- `/tmp/categorized_2gpus_w112_20260420.time.txt`
- `/tmp/categorized_2gpus_w112_monitored_20260420.jsonl`
- `/tmp/categorized_2gpus_w112_monitored_20260420.stdout.log`
- `/tmp/categorized_2gpus_w112_monitored_20260420.time.txt`
- `/tmp/categorized_2gpus_w112_monitored_20260420.monitor.log`
- `/tmp/categorized_2gpus_w500_20260420.jsonl`
- `/tmp/categorized_2gpus_w500_20260420.stdout.log`
- `/tmp/categorized_2gpus_w500_20260420.time.txt`
- `/tmp/categorized_2gpus_w500_monitored_20260420.jsonl`
- `/tmp/categorized_2gpus_w500_monitored_20260420.stdout.log`
- `/tmp/categorized_2gpus_w500_monitored_20260420.time.txt`
- `/tmp/categorized_2gpus_w500_monitored_20260420.monitor.log`
