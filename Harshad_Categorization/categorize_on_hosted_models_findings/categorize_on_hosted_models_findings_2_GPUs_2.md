# categorize_on_hosted_models Findings (2 GPUs, BAAI/bge-m3)

## Scope

Benchmarked `categorize_on_hosted_models.py` again using the default input file (`994` rows) against the hosted embedding service on port `8000` and hosted reranker service on port `8001`, after switching the embedding model to `BAAI/bge-m3` on the 2-GPU deployment.

Model configuration used:

- Embedding model on port `8000`: `BAAI/bge-m3`
- Reranker model on port `8001`: `BAAI/bge-reranker-v2-m3`

Placement checks before the benchmark confirmed one `vllm` engine core on each GPU:

- GPU 0: one `VLLM::EngineCore` using about `1932 MB` at idle
- GPU 1: one `VLLM::EngineCore` using about `2132 MB` at idle

All runs produced the same outcome count:

- Total input rows: `994`
- Successful rows: `831`
- Failed rows: `163`

## Throughput Sweep

Primary goal: maximize throughput.

| Workers | Total Rows/s | Successful Rows/s | Wall Time (s) |
| --- | ---: | ---: | ---: |
| 4 | 44.34 | 37.07 | 22.42 |
| 8 | 63.80 | 53.34 | 15.58 |
| 16 | 69.56 | 58.15 | 14.29 |
| 32 | 70.30 | 58.77 | 14.14 |
| 48 | 69.27 | 57.91 | 14.35 |
| 64 | 65.14 | 54.46 | 15.26 |
| 80 | 69.66 | 58.23 | 14.27 |
| 96 | 69.75 | 58.32 | 14.25 |
| 112 | 68.79 | 57.51 | 14.45 |
| 128 | 68.93 | 57.63 | 14.42 |
| 160 | 67.85 | 56.72 | 14.65 |
| 192 | 67.85 | 56.72 | 14.65 |
| 256 | 68.36 | 57.15 | 14.54 |
| 384 | 69.08 | 57.75 | 14.39 |
| 500 | 68.36 | 57.15 | 14.54 |

### Recommendation

Use `--concurrent-records 32` if the objective is pure throughput on this 2-GPU `BAAI/bge-m3` deployment.

Notes:

- `32` workers was the best measured wall-clock point for both total rows/sec and successful rows/sec.
- The curve peaks much earlier than it did with `BAAI/bge-base-en-v1.5`; throughput is already near its maximum by `32` workers.
- Increasing concurrency above `32` mostly increases per-record latency rather than improving throughput.
- `500` workers remains on the same broad throughput plateau, but with dramatically worse latency than `32`.

## Per-Step Timing By Worker Load

The tables below show mean and p95 latency per successful record for every worker count tested.

### 4 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 98.23 | 150 |
| Embedding | 25.46 | 45 |
| Embedding API only | 25.08 | 44 |
| FAISS search | 1.66 | 3 |
| Rerank | 70.69 | 108 |
| Rerank API only | 70.66 | 108 |

### 8 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 136.63 | 193 |
| Embedding | 29.44 | 52 |
| Embedding API only | 29.04 | 52 |
| FAISS search | 1.90 | 4 |
| Rerank | 104.81 | 155 |
| Rerank API only | 104.78 | 155 |

### 16 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 247.04 | 319 |
| Embedding | 34.39 | 62 |
| Embedding API only | 33.89 | 61 |
| FAISS search | 2.26 | 5 |
| Rerank | 209.99 | 280 |
| Rerank API only | 209.95 | 280 |

### 32 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 487.60 | 574 |
| Embedding | 37.83 | 70 |
| Embedding API only | 37.31 | 69 |
| FAISS search | 2.40 | 6 |
| Rerank | 446.97 | 543 |
| Rerank API only | 446.93 | 543 |

### 48 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 727.26 | 859 |
| Embedding | 36.82 | 76 |
| Embedding API only | 36.29 | 74 |
| FAISS search | 2.37 | 6 |
| Rerank | 687.67 | 822 |
| Rerank API only | 687.64 | 822 |

### 64 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 1029.16 | 1537 |
| Embedding | 132.29 | 1054 |
| Embedding API only | 131.44 | 1054 |
| FAISS search | 3.58 | 12 |
| Rerank | 892.92 | 1058 |
| Rerank API only | 892.88 | 1058 |

### 80 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 1185.14 | 1359 |
| Embedding | 40.93 | 89 |
| Embedding API only | 40.42 | 88 |
| FAISS search | 2.26 | 5 |
| Rerank | 1141.56 | 1323 |
| Rerank API only | 1141.52 | 1323 |

### 96 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 1428.01 | 1624 |
| Embedding | 52.90 | 116 |
| Embedding API only | 52.33 | 116 |
| FAISS search | 2.34 | 6 |
| Rerank | 1372.40 | 1580 |
| Rerank API only | 1372.37 | 1580 |

### 112 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 1630.77 | 1855 |
| Embedding | 49.96 | 182 |
| Embedding API only | 48.89 | 167 |
| FAISS search | 3.83 | 15 |
| Rerank | 1576.61 | 1830 |
| Rerank API only | 1576.57 | 1830 |

### 128 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 1845.54 | 2120 |
| Embedding | 63.59 | 284 |
| Embedding API only | 62.58 | 282 |
| FAISS search | 4.03 | 15 |
| Rerank | 1777.54 | 2075 |
| Rerank API only | 1777.51 | 2075 |

### 160 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 2316.19 | 2699 |
| Embedding | 94.03 | 500 |
| Embedding API only | 93.28 | 500 |
| FAISS search | 3.78 | 14 |
| Rerank | 2217.97 | 2589 |
| Rerank API only | 2217.93 | 2589 |

### 192 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 2675.55 | 3132 |
| Embedding | 142.80 | 779 |
| Embedding API only | 140.42 | 763 |
| FAISS search | 8.03 | 40 |
| Rerank | 2524.38 | 3053 |
| Rerank API only | 2524.34 | 3053 |

### 256 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 3368.97 | 4368 |
| Embedding | 237.65 | 1221 |
| Embedding API only | 236.10 | 1219 |
| FAISS search | 6.76 | 31 |
| Rerank | 3124.24 | 4084 |
| Rerank API only | 3124.19 | 4084 |

### 384 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 4212.38 | 6101 |
| Embedding | 169.13 | 563 |
| Embedding API only | 165.77 | 552 |
| FAISS search | 10.48 | 40 |
| Rerank | 4032.43 | 6060 |
| Rerank API only | 4032.38 | 6060 |

### 500 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 4829.87 | 7843 |
| Embedding | 227.12 | 616 |
| Embedding API only | 223.74 | 610 |
| FAISS search | 17.09 | 71 |
| Rerank | 4585.38 | 7765 |
| Rerank API only | 4585.35 | 7765 |

Key takeaways:

- The reranker is still the dominant cost in the pipeline.
- With `BAAI/bge-m3`, the embedding stage is heavier than before, but reranking still dominates end-to-end latency.
- The best operating point shifted down to `32` workers; beyond that point, queueing growth outpaces any throughput gain.
- The high-concurrency tail (`256` to `500`) keeps throughput roughly flat while pushing total per-record latency into the multi-second range.

## Monitored Run At 32 Workers

A monitored rerun at `32` workers was used to inspect GPU activity at the best measured operating point. The sampler added noticeable overhead to this run, so its throughput should not replace the sweep table above.

Monitored run summary:

- Wall time: `27.21s`
- Script-reported runtime: `26.611s`
- Throughput: `37.35` total rows/s, `31.23` successful rows/s

Latency per successful record:

- Total: mean `983.73 ms`, p95 `4030 ms`
- Embedding: mean `39.46 ms`, p95 `91 ms`
- Embedding API only: mean `39.00 ms`, p95 `91 ms`
- FAISS search: mean `2.20 ms`, p95 `5 ms`
- Rerank: mean `941.64 ms`, p95 `3967 ms`
- Rerank API only: mean `941.60 ms`, p95 `3967 ms`

GPU summary from the monitor log:

- GPU 0: mean util `4.68%`, peak `52.0%`, memory `3467 MB`
- GPU 1: mean util `29.28%`, peak `100.0%`, memory `3005 MB`

Interpretation:

- `BAAI/bge-m3` makes the embedding GPU more active than the earlier `BAAI/bge-base-en-v1.5` run, but the reranker GPU is still busier overall.
- The best-throughput operating point is now lower, which suggests the heavier embedding stage changes the balance of the pipeline enough that extra client-side concurrency stops helping earlier.

## Monitored Run At 500 Workers

A monitored rerun at `500` workers was added to compare the very high-concurrency case directly against the new best point.

Monitored run summary:

- Wall time: `16.18s`
- Script-reported runtime: `15.604s`
- Throughput: `63.70` total rows/s, `53.26` successful rows/s

Latency per successful record:

- Total: mean `5753.93 ms`, p95 `8284 ms`
- Embedding: mean `238.58 ms`, p95 `639 ms`
- Embedding API only: mean `234.92 ms`, p95 `621 ms`
- FAISS search: mean `13.75 ms`, p95 `51 ms`
- Rerank: mean `5501.32 ms`, p95 `8253 ms`
- Rerank API only: mean `5501.27 ms`, p95 `8253 ms`

GPU summary from the monitor log:

- GPU 0: mean util `6.92%`, peak `100.0%`, memory `3467 MB`
- GPU 1: mean util `28.0%`, peak `100.0%`, memory `3005 MB`

Interpretation:

- At `500` workers, both GPUs can spike to `100%`, but the extra concurrency mostly turns into queueing delay rather than better sustained throughput.
- The reranker stage is still the long pole, and the total latency increase from `32` to `500` is severe.

## Comparison To Earlier Findings

Compared with the earlier findings:

- Prior 2-GPU `BAAI/bge-base-en-v1.5` best: `71.61` total rows/s and `59.87` successful rows/s at `112` workers
- Current 2-GPU `BAAI/bge-m3` best: `70.30` total rows/s and `58.77` successful rows/s at `32` workers
- Delta versus the prior 2-GPU best: `-1.83%` total throughput and `-1.84%` successful throughput
- Prior single-GPU best from `categorize_on_hosted_models_findings.md`: `73.09` total rows/s and `61.10` successful rows/s at `256` workers
- Delta versus the prior single-GPU best: `-3.82%` total throughput and `-3.81%` successful throughput

Interpretation:

- Switching to `BAAI/bge-m3` did not improve peak wall-clock throughput for this workload.
- The practical change is that the best concurrency shifted down from `112` workers to `32` workers.
- `BAAI/bge-m3` appears to make the embedding side more expensive, but not enough to displace the reranker as the primary bottleneck.

## Artifacts

Relevant outputs and logs:

- `/tmp/categorize_worker_sweep_2gpus_bgem3_20260420_raw.tsv`
- `/tmp/categorized_2gpus_bgem3_w32_20260420.jsonl`
- `/tmp/categorized_2gpus_bgem3_w32_20260420.stdout.log`
- `/tmp/categorized_2gpus_bgem3_w32_20260420.time.txt`
- `/tmp/categorized_2gpus_bgem3_w500_20260420.jsonl`
- `/tmp/categorized_2gpus_bgem3_w500_20260420.stdout.log`
- `/tmp/categorized_2gpus_bgem3_w500_20260420.time.txt`
- `/tmp/categorized_2gpus_bgem3_w32_monitored_20260420.jsonl`
- `/tmp/categorized_2gpus_bgem3_w32_monitored_20260420.stdout.log`
- `/tmp/categorized_2gpus_bgem3_w32_monitored_20260420.time.txt`
- `/tmp/categorized_2gpus_bgem3_w32_monitored_20260420.monitor.log`
- `/tmp/categorized_2gpus_bgem3_w500_monitored_20260420.jsonl`
- `/tmp/categorized_2gpus_bgem3_w500_monitored_20260420.stdout.log`
- `/tmp/categorized_2gpus_bgem3_w500_monitored_20260420.time.txt`
- `/tmp/categorized_2gpus_bgem3_w500_monitored_20260420.monitor.log`
