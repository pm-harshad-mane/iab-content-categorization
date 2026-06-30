# categorize_on_hosted_models Findings

## Scope

Benchmarked `categorize_on_hosted_models.py` using the default input file (`994` rows) against the hosted embedding service on port `8000` and hosted reranker service on port `8001`.

Model configuration used:

- Embedding model: `BAAI/bge-base-en-v1.5`
- Reranker model: `BAAI/bge-reranker-v2-m3`

All runs produced the same outcome count:

- Total input rows: `994`
- Successful rows: `831`
- Failed rows: `163`

## Throughput Sweep

Primary goal: maximize throughput.

| Workers | Total Rows/s | Successful Rows/s | Wall Time (s) |
| --- | ---: | ---: | ---: |
| 4 | 44.47 | 37.18 | 22.35 |
| 8 | 63.72 | 53.27 | 15.60 |
| 16 | 69.51 | 58.11 | 14.30 |
| 32 | 70.45 | 58.89 | 14.11 |
| 48 | 69.95 | 58.48 | 14.21 |
| 64 | 70.65 | 59.06 | 14.07 |
| 96 | 70.50 | 58.94 | 14.10 |
| 128 | 70.25 | 58.73 | 14.15 |
| 256 | 73.09 | 61.10 | 13.60 |

### Recommendation

Use `--concurrent-records 256` if the objective is pure throughput on the current setup.

Notes:

- `256` workers is now the best measured point for both total rows/sec and successful rows/sec.
- The system largely plateaus from roughly `32` to `128` workers, then improves slightly again at `256`.
- The throughput gain from `64` to `256` is still modest relative to the latency increase.

## Per-Step Timing By Worker Load

The tables below show mean and p95 latency per successful record for each worker count tested.

### 4 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 100.88 | 278 |
| Embedding | 32.72 | 221 |
| Embedding API only | 23.91 | 30 |
| FAISS search | 1.31 | 3 |
| Rerank | 66.45 | 101 |
| Rerank API only | 66.40 | 101 |

### 8 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 135.69 | 211 |
| Embedding | 34.70 | 71 |
| Embedding API only | 25.78 | 37 |
| FAISS search | 1.40 | 3 |
| Rerank | 99.13 | 140 |
| Rerank API only | 99.10 | 140 |

### 16 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 246.30 | 434 |
| Embedding | 49.86 | 250 |
| Embedding API only | 35.97 | 218 |
| FAISS search | 1.55 | 4 |
| Rerank | 194.33 | 264 |
| Rerank API only | 194.29 | 264 |

### 32 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 486.81 | 632 |
| Embedding | 44.89 | 236 |
| Embedding API only | 33.35 | 77 |
| FAISS search | 1.39 | 3 |
| Rerank | 440.14 | 526 |
| Rerank API only | 440.11 | 526 |

### 48 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 719.89 | 921 |
| Embedding | 53.32 | 244 |
| Embedding API only | 39.39 | 218 |
| FAISS search | 1.61 | 4 |
| Rerank | 664.54 | 817 |
| Rerank API only | 664.50 | 817 |

### 64 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 956.48 | 1128 |
| Embedding | 55.26 | 255 |
| Embedding API only | 40.90 | 224 |
| FAISS search | 1.88 | 5 |
| Rerank | 898.88 | 1050 |
| Rerank API only | 898.84 | 1050 |

### 96 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 1418.01 | 1791 |
| Embedding | 86.39 | 412 |
| Embedding API only | 61.99 | 248 |
| FAISS search | 1.95 | 5 |
| Rerank | 1329.29 | 1581 |
| Rerank API only | 1329.26 | 1581 |

### 128 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 1829.18 | 2269 |
| Embedding | 105.77 | 551 |
| Embedding API only | 72.78 | 322 |
| FAISS search | 2.98 | 10 |
| Rerank | 1719.98 | 2160 |
| Rerank API only | 1719.95 | 2160 |

### 256 Workers

| Step | Mean (ms) | p95 (ms) |
| --- | ---: | ---: |
| Total | 3042.80 | 4087 |
| Embedding | 199.37 | 664 |
| Embedding API only | 139.40 | 491 |
| FAISS search | 9.35 | 40 |
| Rerank | 2833.69 | 3982 |
| Rerank API only | 2833.64 | 3982 |

## Monitored Runs

### 32 Workers

- Wall time: `13.95s`
- Script-reported runtime: `13.331s`
- Throughput: `71.25` total rows/s, `59.57` successful rows/s

Latency per successful record:

- Total: mean `477.55 ms`, p95 `671 ms`
- Embedding: mean `49.97 ms`, p95 `245 ms`
- Embedding API only: mean `36.20 ms`, p95 `78 ms`
- FAISS search: mean `1.52 ms`, p95 `4 ms`
- Rerank: mean `425.65 ms`, p95 `512 ms`
- Rerank API only: mean `425.60 ms`, p95 `512 ms`

Resource summary:

- Embed server CPU: mean `11.43%`, peak `12.0%`
- Rerank server CPU: mean `18.07%`, peak `20.1%`
- GPU 0: mean util `66.67%`, peak `100%`, memory `4241 MB`
- GPU 1: util `0%`, memory `868 MB`

### 64 Workers

- Wall time: `13.67s`
- Script-reported runtime: `13.119s`
- Throughput: `72.71` total rows/s, `60.79` successful rows/s

Latency per successful record:

- Total: mean `929.87 ms`, p95 `1124 ms`
- Embedding: mean `57.05 ms`, p95 `257 ms`
- Embedding API only: mean `40.87 ms`, p95 `217 ms`
- FAISS search: mean `2.21 ms`, p95 `6 ms`
- Rerank: mean `870.16 ms`, p95 `1052 ms`
- Rerank API only: mean `870.12 ms`, p95 `1052 ms`

Resource summary:

- Embed server CPU: mean `7.5%`, peak `7.8%`
- Rerank server CPU: mean `11.9%`, peak `12.9%`
- GPU 0: mean util `100%`, peak `100%`, memory `4241 MB`
- GPU 1: util `0%`, memory `868 MB`

## Interpretation

- The reranker is the dominant cost in the pipeline.
- FAISS search is negligible.
- Increasing workers from `32` to `64` improves throughput slightly, but nearly doubles per-record latency.
- Increasing workers to `256` yields the best measured throughput, but pushes mean per-record total latency above `3s` and rerank latency above `2.8s`.
- GPU 0 is the active bottleneck. GPU 1 is effectively unused in these runs.
- If the goal is throughput, `256` is the best measured setting.
- If the goal is latency-efficiency rather than throughput, `32` is the better operating point.

## Artifacts

Relevant benchmark outputs and logs:

- `/tmp/categorize_worker_sweep_20260420_raw.tsv`
- `/tmp/categorize_worker_tiebreak_20260420_raw.tsv`
- `/tmp/categorized_w256_20260420.jsonl`
- `/tmp/categorized_w256_20260420.stdout.log`
- `/tmp/categorized_w256_20260420.time.txt`
- `/tmp/categorized_on_hosted_models_w32_monitored_20260420_escalated.jsonl`
- `/tmp/categorized_on_hosted_models_w32_monitored_20260420_escalated.monitor.log`
- `/tmp/categorized_on_hosted_models_w64_monitored_20260420.jsonl`
- `/tmp/categorized_on_hosted_models_w64_monitored_20260420.monitor.log`
