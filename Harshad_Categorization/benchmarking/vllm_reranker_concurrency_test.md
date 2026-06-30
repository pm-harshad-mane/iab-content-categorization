# vLLM Reranker Concurrency Test

## Goal

Measure how much concurrency local vLLM reranker servers can handle for the same kind of rerank traffic used by [`categorize_on_hosted_models.py`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models.py).

The benchmark uses:

- source categorization output: [`categorize_on_hosted_models_1000.jsonl`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models_1000.jsonl)
- original page-content input: [`save_url_content_1000.jsonl`](/home/harshad.mane/Harshad_Categorization/save_url_content_1000.jsonl)
- benchmark script: [`benchmark_vllm_reranker_concurrency.py`](/home/harshad.mane/Harshad_Categorization/benchmark_vllm_reranker_concurrency.py)

## Latency Terms

The tables in this file use two different p95 latency metrics:

- `Service p95 ms`
  - measured from the moment a worker thread actually starts the HTTP request to vLLM until the response comes back
  - includes the request/response itself plus whatever queueing and model execution happens after the request reaches vLLM
  - does **not** include time spent waiting in the benchmark client's executor before the HTTP call starts

- `End-to-end p95 ms`
  - measured from task submission in the benchmark client until the response comes back
  - includes:
    - client-side queue wait before the HTTP call begins
    - the HTTP request/response time
    - server-side queueing
    - model execution time

So:

- `End-to-end p95 ms` is the user-facing latency seen by the benchmark client
- `Service p95 ms` is the narrower latency once the HTTP request has actually begun

At high concurrency, `End-to-end p95 ms` is usually the more important metric because it captures queue buildup outside the HTTP call as well.

## Benchmark Shape

The benchmark is intentionally close to the production script:

- query text is rebuilt from the original page-content record using the same logic as [`categorize_on_hosted_models.py`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models.py)
- `max_body_chars = 4000`
- `rerank_query_max_chars = 1800`
- rerank documents are built exactly like production:
  `path: {candidate.path}\ndescription: {candidate.description}`
- candidate docs come from each record's saved `faiss_candidates`
- prefix caching is disabled in vLLM
- request combinations are duplicated with `repeat_factor` so higher concurrency levels are real in-flight load levels

This produces `831` base rerank requests, one per successful categorization row.

## Performance Scorecard

Meaning of “request-shape adjustment” in this file:

- the model does **not** work cleanly with the current production-like rerank request shape as-is
- the current production-like shape uses:
  - query text built like [`categorize_on_hosted_models.py`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models.py)
  - `rerank_query_max_chars = 1800`
  - rerank docs shaped as `path: ...\ndescription: ...`
- for some models, good throughput was only possible after changing the request shape
- in these tests, that adjustment usually meant reducing `rerank_query_max_chars` from `1800` to `512`

So:

- `clean drop-in` means the model worked with the current production-like request shape unchanged
- `best if request-shape adjustment is allowed` means the model only became a good option after we changed the request shape first

Best clean drop-in performer by `req/s` plus `service p95`:

1. `mixedbread-ai/mxbai-rerank-base-v2` — multilingual
2. `Alibaba-NLP/gte-reranker-modernbert-base` — English-only
3. `Alibaba-NLP/gte-multilingual-reranker-base` — multilingual
4. `mixedbread-ai/mxbai-rerank-large-v2` — multilingual
5. `BAAI/bge-reranker-v2-m3` — multilingual

Best performers if request-shape adjustment is allowed:

1. `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — multilingual, with `rerank_query_max_chars = 512`
2. `BAAI/bge-reranker-base` — Chinese and English, with `rerank_query_max_chars = 512`

Not viable under the current production-like request shape:

- `jinaai/jina-reranker-v2-base-multilingual`
- `cross-encoder/ms-marco-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L4-v2`
- `cross-encoder/ms-marco-MiniLM-L2-v2`

Special-case architecture:

- `colbert-ir/colbertv2.0` via `/pooling` is included for completeness, but it is not directly comparable to standard `/v1/rerank` models

## GTE Reranker ModernBERT Base

Language support:

- English-only

Server command used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model Alibaba-NLP/gte-reranker-modernbert-base \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

### Main Sweep

- `repeat_factor = 2`
- total requests = `1662`
- concurrencies:
  `32 64 128 256 512 1024`
- `2` trials per point

Median results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU peak % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32` | `124.246` | `100.000` | `321.656` | `12748.086` | `63.434` | `85.500` | `2399` |
| `64` | `116.569` | `100.000` | `652.683` | `13549.646` | `66.200` | `90.000` | `2399` |
| `128` | `116.562` | `100.000` | `1251.884` | `13477.376` | `67.666` | `89.500` | `2399` |
| `256` | `109.768` | `100.000` | `2794.482` | `14251.900` | `77.600` | `92.500` | `2399` |
| `512` | `115.690` | `100.000` | `4603.470` | `13122.630` | `77.975` | `100.000` | `2399` |
| `1024` | `109.780` | `100.000` | `8874.746` | `13328.190` | `73.250` | `100.000` | `2399` |

### Higher-Concurrency Extension

Additional one-trial runs:

| Concurrency | Repeat Factor | Req/s | Success % | Service p95 ms | End-to-end p95 ms | Mean GPU util % | Peak GPU util % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1536` | `10` | `103.426` | `100.000` | `15535.199` | `74858.980` | `65.577` | `100.000` |
| `2048` | `10` | `104.377` | `100.000` | `20594.074` | `73349.231` | `78.111` | `100.000` |
| `3072` | `10` | `107.452` | `100.000` | `29324.242` | `69302.013` | `69.958` | `100.000` |
| `4096` | `10` | `108.911` | `100.000` | `37626.137` | `66104.506` | `68.750` | `100.000` |
| `8192` | `10` | `110.318` | `100.000` | `56771.339` | `56773.359` | `76.231` | `100.000` |
| `16384` | `20` | `105.684` | `100.000` | `111195.433` | `111201.254` | `71.898` | `100.000` |

### Findings So Far

- `Alibaba-NLP/gte-reranker-modernbert-base` handled every tested concurrency level through `16384` with `100%` success.
- Throughput stayed roughly flat in the `104` to `124 req/s` range.
- GPU peak utilization hit `100%` repeatedly, but mean utilization stayed well below full saturation.
- GPU memory stayed stable at about `2399 MB`.
- Increasing concurrency above roughly `32` to `128` did not improve throughput meaningfully.
- Higher concurrency mainly increased queueing and service latency. By `16384`, p95 latency was about `111s`.

Current practical interpretation:

- highest observed successful concurrency: at least `16384`
- useful throughput ceiling: roughly `110` to `125 req/s`
- above that, extra concurrency mostly buys worse latency, not more throughput

## Next

Run the same benchmark for:

- `BAAI/bge-reranker-v2-m3`

## BGE Reranker v2 m3

Language support:

- multilingual

Server command used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-reranker-v2-m3 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

### Main Sweep

- `repeat_factor = 2`
- total requests = `1662`
- concurrencies:
  `32 64 128 256 512 1024`
- `2` trials per point

Median results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU peak % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32` | `68.963` | `100.000` | `536.798` | `22975.449` | `100.000` | `100.000` | `3321` |
| `64` | `68.173` | `100.000` | `1034.726` | `23181.477` | `99.945` | `100.000` | `3321` |
| `128` | `67.255` | `100.000` | `2006.013` | `23455.419` | `94.445` | `100.000` | `3321` |
| `256` | `67.260` | `100.000` | `3888.225` | `23254.338` | `94.166` | `100.000` | `3321` |
| `512` | `67.225` | `100.000` | `7657.641` | `22966.845` | `94.445` | `100.000` | `3321` |
| `1024` | `65.931` | `100.000` | `15201.923` | `22761.565` | `100.000` | `100.000` | `3321` |

### Higher-Concurrency Extension

Additional one-trial runs:

| Concurrency | Repeat Factor | Req/s | Success % | Service p95 ms | End-to-end p95 ms | Mean GPU util % | Peak GPU util % | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `2048` | `10` | `66.335` | `100.000` | `30721.538` | `115926.221` | `100.000` | `100.000` | stable |
| `4096` | `10` | `64.738` | `100.000` | `61405.680` | `115302.782` | `99.179` | `100.000` | stable |
| `8192` | `10` | `62.421` | `100.000` | `112008.865` | `112011.409` | `94.643` | `100.000` | stable but very slow |
| `16384` | `20` | `74.595` | `80.457` | `172220.688` | `172225.078` | `89.303` | `100.000` | `3248` read timeouts |

### Findings

- `BAAI/bge-reranker-v2-m3` is GPU-bound almost immediately.
- GPU 0 is effectively saturated from the lowest tested concurrency band.
- Throughput is flat at about `66` to `69 req/s` through `1024`, and remains in the same rough band even at `2048` to `8192`.
- Increasing concurrency mainly increases latency, not throughput.
- `8192` completed with `100%` success, but p95 latency was already around `112s`.
- `16384` is the first tested level that broke down, with success dropping to `80.457%` due to `3248` HTTP read timeouts.

Current practical interpretation:

- highest observed successful concurrency: `8192`
- first observed failure point: `16384`
- useful throughput ceiling: about `66` to `69 req/s`
- unlike GTE, this model does not benefit from high concurrency because the GPU is already full very early

## Attempted Multilingual GTE Reranker

I also attempted to benchmark:

- `Alibaba-NLP/gte-multilingual-reranker-base`

Language support:

- multilingual

Server command attempted:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model Alibaba-NLP/gte-multilingual-reranker-base \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

Result:

- the model did **not** start successfully under the current vLLM environment
- no concurrency benchmark was run

Observed blocker:

- vLLM `0.19.0` rejected the model architecture
- startup failed with:
  - `Model architectures ['NewForSequenceClassification'] are not supported for now`

Interpretation:

- this is a model-compatibility issue, not a concurrency limit
- there are no valid throughput or latency results for `Alibaba-NLP/gte-multilingual-reranker-base` in the current environment

### Corrected Launch Command

After investigating newer vLLM model-support documentation, I retried the same model with the required architecture override:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model Alibaba-NLP/gte-multilingual-reranker-base \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --hf-overrides '{"architectures": ["GteNewForSequenceClassification"]}'
```

Why this is needed:

- the model config reports the generic architecture name `NewForSequenceClassification`
- vLLM expects this model family to be loaded as `GteNewForSequenceClassification`
- without the override, startup fails

Observed after applying the override:

- vLLM resolved the architecture as `GteNewForSequenceClassification`
- the model proceeded to load instead of failing immediately on architecture validation

So the blocker was not “multilingual GTE reranker is unsupported forever,” but rather:

- the model needs the documented `--hf-overrides` fix when launched under vLLM

### Multilingual GTE Reranker Results

After launching with the required override, I benchmarked:

- `Alibaba-NLP/gte-multilingual-reranker-base`

using the same production-like reranker benchmark as the other models.

#### Main Sweep

- `repeat_factor = 2`
- total requests = `1662`
- concurrencies:
  `32 64 128 256 512 1024`
- `2` trials per point

Median results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU peak % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32` | `104.341` | `100.000` | `343.863` | `15221.619` | `52.100` | `71.000` | `5135` |
| `64` | `109.830` | `100.000` | `661.648` | `14391.218` | `57.800` | `68.000` | `5135` |
| `128` | `104.800` | `100.000` | `1534.231` | `14998.004` | `61.250` | `71.000` | `5135` |
| `256` | `109.904` | `100.000` | `2592.746` | `14147.690` | `63.200` | `84.000` | `5135` |
| `512` | `107.644` | `100.000` | `5007.552` | `14189.790` | `63.100` | `75.500` | `5135` |
| `1024` | `104.719` | `100.000` | `9218.321` | `13917.856` | `51.417` | `78.000` | `5135` |

#### Higher-Concurrency Extension

Additional one-trial runs:

| Concurrency | Repeat Factor | Req/s | Success % | Service p95 ms | End-to-end p95 ms | Mean GPU util % | Peak GPU util % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2048` | `10` | `99.227` | `100.000` | `21584.058` | `77337.828` | `52.269` | `84.000` |
| `4096` | `10` | `101.324` | `100.000` | `40520.784` | `71408.620` | `58.583` | `84.000` |
| `8192` | `10` | `105.444` | `100.000` | `60005.934` | `60008.581` | `60.080` | `88.000` |
| `16384` | `10` | `101.646` | `100.000` | `62587.423` | `62589.628` | `59.840` | `97.000` |

#### Interpretation

- `Alibaba-NLP/gte-multilingual-reranker-base` handled every tested concurrency level through `16384` with `100%` success.
- Throughput stayed in the `99` to `110 req/s` band.
- Service p95 increased steadily with concurrency, but the model remained stable.
- GPU memory usage was much higher than `Alibaba-NLP/gte-reranker-modernbert-base`:
  - about `5135 MB` vs about `2399 MB`
- Throughput was slightly lower than the English `gte-reranker-modernbert-base`, but still much stronger than `BAAI/bge-reranker-v2-m3`.

## Overall Takeaways

- `mixedbread-ai/mxbai-rerank-base-v2` is the strongest clean drop-in result in this file on the combined `req/s` plus `service p95` criteria.
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` and `BAAI/bge-reranker-base` are strong throughput options only if the rerank query cap is reduced to `512`.
- `Alibaba-NLP/gte-reranker-modernbert-base` remains the strongest clean lower-memory drop-in option.
- `mixedbread-ai/mxbai-rerank-large-v2` is slower than `mixedbread-ai/mxbai-rerank-base-v2` while using essentially the same very high GPU memory.
- `BAAI/bge-reranker-v2-m3` is the clearest example of an early GPU-bound reranker in this set.
- `jinaai/jina-reranker-v2-base-multilingual` and the `ms-marco-MiniLM` family are not clean drop-ins under the current production-like request shape because they return persistent `400` errors.

## Expansion Candidates Summary

Extra speed-oriented candidates tested after the initial reranker set:

- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
  - multilingual
  - very strong throughput, but only after reducing `rerank_query_max_chars` to `512`

- `mixedbread-ai/mxbai-rerank-base-v2`
  - multilingual
  - best clean drop-in throughput result overall

- `mixedbread-ai/mxbai-rerank-large-v2`
  - multilingual
  - clean but slower than the base model at essentially the same memory footprint

- `BAAI/bge-reranker-base`
  - Chinese and English
  - strong throughput with `rerank_query_max_chars = 512`

- `cross-encoder/ms-marco-MiniLM-L6-v2`
  - English-only
  - not clean under the current production-like request shape

- `cross-encoder/ms-marco-MiniLM-L4-v2`
  - English-only
  - not clean under the current production-like request shape

- `cross-encoder/ms-marco-MiniLM-L2-v2`
  - English-only
  - not clean under the current production-like request shape

## Jina Reranker v2 Base Multilingual

Language support:

- multilingual

Server command used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model jinaai/jina-reranker-v2-base-multilingual \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

Benchmark command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model jinaai/jina-reranker-v2-base-multilingual \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 2 \
  --concurrencies 32 64 128 256 512 1024 \
  --trials 2 \
  --output-json /tmp/jina_reranker_v2_base_multilingual_concurrency.json
```

### Results

Median results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU peak % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32` | `106.899` | `98.676` | `390.337` | `14735.740` | `42.816` | `54.000` | `2415` |
| `64` | `106.870` | `98.706` | `882.029` | `14812.505` | `39.834` | `54.000` | `2415` |
| `128` | `99.368` | `98.646` | `1813.087` | `15880.667` | `38.250` | `51.500` | `2415` |
| `256` | `101.847` | `98.646` | `3183.145` | `15391.701` | `47.250` | `56.000` | `2415` |
| `512` | `106.235` | `98.767` | `5098.129` | `14382.731` | `42.200` | `55.000` | `2415` |
| `1024` | `101.915` | `98.706` | `9875.659` | `14336.845` | `41.166` | `55.500` | `2415` |

Observed errors across the sweep:

- `240` HTTP `400` responses
- `18` connection resets
- `3` remote disconnects

### Interpretation

- Throughput was in the `99` to `107 req/s` band, which is reasonably strong.
- But this model was **not clean** under the current production-like request shape.
- A fixed block of requests returned HTTP `400` at every tested concurrency level, which means the issue is request compatibility rather than pure concurrency saturation.
- Because the baseline sweep already had persistent `400`s, I did **not** extend this model above `1024`.

Practical conclusion:

- `jinaai/jina-reranker-v2-base-multilingual` is not a clean drop-in replacement for the current request shape used in these tests
- before any higher-concurrency benchmarking, it would need input-shape investigation or model-specific request adaptation

## mMARCO mMiniLMv2 L12 H384 v1

Language support:

- multilingual

The original repo ID in the earlier notes was wrong. This model resolved and loaded successfully as:

- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

Server command used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

### Production-Like Compatibility Check

I first tried the same benchmark shape used for the other rerankers, including:

- `rerank_query_max_chars = 1800`

Command attempted:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 2 \
  --concurrencies 32 64 128 256 512 1024 \
  --trials 2 \
  --output-json /tmp/mmarco_mMiniLMv2_L12_H384_v1_concurrency.json
```

That failed at warmup with:

- HTTP `400` on the first production-like rerank request

This matters because [`categorize_on_hosted_models.py`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models.py) does **not** retry reranker requests on `400`. So with the current production-like `1800`-char rerank query cap, this model is not a clean drop-in.

### Compatibility-Adjusted Benchmark

To still measure its throughput characteristics, I reran the benchmark with a smaller rerank query cap:

- `rerank_query_max_chars = 512`

Main sweep command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 2 \
  --rerank-query-max-chars 512 \
  --concurrencies 32 64 128 256 512 1024 \
  --trials 2 \
  --output-json /tmp/mmarco_mMiniLMv2_L12_H384_v1_q512_concurrency.json
```

Higher-concurrency extension command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 10 \
  --rerank-query-max-chars 512 \
  --concurrencies 2048 4096 8192 16384 \
  --trials 1 \
  --output-json /tmp/mmarco_mMiniLMv2_L12_H384_v1_q512_above_1024.json
```

### Results With `rerank_query_max_chars = 512`

Median main-sweep results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU peak % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32` | `151.362` | `100.000` | `329.208` | `10494.981` | `16.125` | `18.500` | `2245` |
| `64` | `156.341` | `100.000` | `586.904` | `10311.956` | `16.166` | `20.000` | `2245` |
| `128` | `165.459` | `100.000` | `1235.277` | `9487.882` | `13.750` | `18.500` | `2245` |
| `256` | `142.894` | `100.000` | `2488.717` | `10869.279` | `14.400` | `20.500` | `2245` |
| `512` | `151.713` | `100.000` | `4984.231` | `10046.240` | `16.125` | `21.500` | `2245` |
| `1024` | `144.725` | `100.000` | `7846.495` | `9729.330` | `12.075` | `20.000` | `2245` |

Higher-concurrency extension:

| Concurrency | Repeat Factor | Req/s | Success % | Service p95 ms | End-to-end p95 ms | Mean GPU util % | Peak GPU util % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2048` | `10` | `127.387` | `100.000` | `18095.254` | `59975.631` | `14.227` | `20.000` | `2245` |
| `4096` | `10` | `138.659` | `100.000` | `31563.975` | `50798.133` | `13.850` | `22.000` | `2245` |
| `8192` | `10` | `145.411` | `100.000` | `40192.946` | `40194.660` | `13.600` | `20.000` | `2245` |
| `16384` | `10` | `148.317` | `100.000` | `39309.037` | `39310.275` | `16.579` | `23.000` | `2245` |

### Interpretation

- With a `512`-char rerank query cap, this model was clean and stable through `16384` concurrency with `100%` success.
- It was the fastest reranker measured so far on raw throughput, reaching roughly `145` to `165 req/s` in the main band.
- Service p95 was also strong for that throughput tier:
  - about `329 ms` at `32`
  - about `587 ms` at `64`
  - about `1235 ms` at `128`
- GPU utilization stayed surprisingly low, mostly in the `12%` to `16%` mean range, with low-20s peaks.
- GPU memory stayed low and flat at about `2245 MB`.
- Increasing concurrency beyond about `128` did not buy much more throughput; it mainly increased queueing latency.

Practical conclusion:

- fastest measured reranker so far by `req/s`: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- highest observed successful concurrency: at least `16384`
- but it is **not** a clean drop-in for the current production-like rerank query shape unless the rerank query cap is reduced from `1800` to something like `512`

## Mixedbread Rerank Base v2

Language support:

- multilingual, with 100+ language support in the model card

Server command used:

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

This model loaded successfully under vLLM and exposed `/v1/rerank` without any request-shape adjustment.

Main sweep command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model mixedbread-ai/mxbai-rerank-base-v2 \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 2 \
  --concurrencies 32 64 128 256 512 1024 \
  --trials 2 \
  --output-json /tmp/mxbai_rerank_base_v2_concurrency.json
```

Higher-concurrency extension command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model mixedbread-ai/mxbai-rerank-base-v2 \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 10 \
  --concurrencies 2048 4096 8192 16384 \
  --trials 1 \
  --output-json /tmp/mxbai_rerank_base_v2_above_1024.json
```

### Results

Median main-sweep results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU peak % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32` | `209.307` | `100.000` | `205.075` | `7546.023` | `44.166` | `73.000` | `42715` |
| `64` | `223.212` | `100.000` | `405.487` | `7111.577` | `57.500` | `83.000` | `42715` |
| `128` | `181.757` | `100.000` | `794.710` | `8973.885` | `36.625` | `54.000` | `42715` |
| `256` | `232.821` | `100.000` | `2144.909` | `6529.850` | `60.916` | `76.000` | `42715` |
| `512` | `218.276` | `100.000` | `3237.005` | `6550.984` | `64.666` | `100.000` | `42715` |
| `1024` | `207.870` | `100.000` | `6260.155` | `6261.236` | `39.250` | `60.500` | `42715` |

Higher-concurrency extension:

| Concurrency | Repeat Factor | Req/s | Success % | Service p95 ms | End-to-end p95 ms | Mean GPU util % | Peak GPU util % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2048` | `10` | `184.000` | `100.000` | `12667.764` | `39942.777` | `35.733` | `100.000` | `42715` |
| `4096` | `10` | `185.929` | `100.000` | `27870.234` | `35468.717` | `48.933` | `100.000` | `42715` |
| `8192` | `10` | `182.161` | `100.000` | `33507.639` | `33509.553` | `56.438` | `100.000` | `42715` |
| `16384` | `10` | `183.763` | `100.000` | `35326.634` | `35327.484` | `48.000` | `91.000` | `42715` |

### Interpretation

- `mixedbread-ai/mxbai-rerank-base-v2` is the fastest clean drop-in reranker measured so far.
- It handled every tested concurrency level through `16384` with `100%` success.
- Throughput stayed in a much higher band than the other clean drop-in rerankers:
  - roughly `208` to `233 req/s` in the main sweep
  - roughly `182` to `186 req/s` above `1024`
- Service p95 was especially strong at practical load:
  - about `205 ms` at `32`
  - about `405 ms` at `64`
- GPU utilization was higher than the smaller multilingual rerankers, but still not pinned constantly.
- The tradeoff is memory footprint:
  - about `42.7 GB` on the L40S

Practical conclusion:

- fastest clean drop-in reranker measured so far: `mixedbread-ai/mxbai-rerank-base-v2`
- highest observed successful concurrency: at least `16384`
- major operational caveat: very high GPU memory residency
- if GPU memory is available, this is now the most attractive multilingual reranker in the test set from a pure throughput perspective

## BGE Reranker Base

Language support:

- Chinese and English

Server command used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-reranker-base \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

### Production-Like Compatibility Check

I first tried the same production-like reranker benchmark shape used for the other models:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model BAAI/bge-reranker-base \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 2 \
  --concurrencies 32 64 128 256 512 1024 \
  --trials 2 \
  --output-json /tmp/bge_reranker_base_concurrency.json
```

That failed at warmup with:

- HTTP `400` on the first production-like rerank request

So, like `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, this model is not a clean drop-in with the current `1800`-char rerank query cap used by [`categorize_on_hosted_models.py`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models.py).

### Compatibility-Adjusted Benchmark

To still measure its throughput profile, I reran it with:

- `rerank_query_max_chars = 512`

Main sweep command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model BAAI/bge-reranker-base \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 2 \
  --rerank-query-max-chars 512 \
  --concurrencies 32 64 128 256 512 1024 \
  --trials 2 \
  --output-json /tmp/bge_reranker_base_q512_concurrency.json
```

Higher-concurrency extension command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model BAAI/bge-reranker-base \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 10 \
  --rerank-query-max-chars 512 \
  --concurrencies 2048 4096 8192 16384 \
  --trials 1 \
  --output-json /tmp/bge_reranker_base_q512_above_1024.json
```

### Results With `rerank_query_max_chars = 512`

Median main-sweep results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU peak % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32` | `156.439` | `100.000` | `323.760` | `10156.726` | `39.000` | `43.000` | `2765` |
| `64` | `134.770` | `100.000` | `699.450` | `11638.589` | `32.000` | `47.500` | `2765` |
| `128` | `135.139` | `100.000` | `1424.246` | `11726.949` | `29.500` | `46.500` | `2765` |
| `256` | `134.445` | `100.000` | `2442.066` | `11553.711` | `29.875` | `39.500` | `2765` |
| `512` | `143.419` | `100.000` | `3867.726` | `10488.428` | `27.250` | `44.000` | `2765` |
| `1024` | `152.656` | `100.000` | `7349.070` | `9119.391` | `35.125` | `46.500` | `2765` |

Higher-concurrency extension:

| Concurrency | Repeat Factor | Req/s | Success % | Service p95 ms | End-to-end p95 ms | Mean GPU util % | Peak GPU util % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2048` | `10` | `135.319` | `100.000` | `16667.593` | `55845.081` | `27.957` | `46.000` | `2765` |
| `4096` | `10` | `148.742` | `100.000` | `28554.256` | `46683.880` | `33.100` | `48.000` | `2765` |
| `8192` | `10` | `147.937` | `100.000` | `39639.099` | `39640.837` | `33.474` | `49.000` | `2765` |
| `16384` | `10` | `144.630` | `100.000` | `40080.028` | `40082.877` | `27.952` | `53.000` | `2765` |

### Interpretation

- `BAAI/bge-reranker-base` is materially faster than `BAAI/bge-reranker-v2-m3`.
- With a `512`-char rerank query cap, it stayed clean through `16384` concurrency with `100%` success.
- Its throughput band was strong:
  - about `134` to `156 req/s` in the main sweep
  - about `135` to `149 req/s` above `1024`
- GPU utilization stayed modest, mostly around the low-30s, and never looked close to saturation.
- GPU memory stayed low at about `2765 MB`.

Practical conclusion:

- `BAAI/bge-reranker-base` is a much better throughput option than `BAAI/bge-reranker-v2-m3`
- but, like `mMARCO`, it is not a clean drop-in unless the rerank query cap is reduced from `1800` to something like `512`

## MS MARCO MiniLM L6 v2

Language support:

- English-only

Server command used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

Benchmark command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model cross-encoder/ms-marco-MiniLM-L6-v2 \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 2 \
  --concurrencies 32 64 128 256 512 1024 \
  --trials 2 \
  --output-json /tmp/ms_marco_MiniLM_L6_v2_concurrency.json
```

### Results

Median results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU peak % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32` | `94.719` | `83.363` | `436.403` | `16705.018` | `7.285` | `8.500` | `1849` |
| `64` | `94.370` | `83.815` | `773.181` | `16775.603` | `6.857` | `8.000` | `1849` |
| `128` | `95.755` | `83.423` | `1491.434` | `16499.078` | `6.869` | `8.500` | `1849` |
| `256` | `95.456` | `83.845` | `2875.260` | `16330.682` | `6.000` | `7.500` | `1849` |
| `512` | `93.912` | `84.055` | `5491.280` | `16243.065` | `7.117` | `7.500` | `1849` |
| `1024` | `89.809` | `84.085` | `11443.569` | `16327.647` | `5.214` | `7.500` | `1849` |

Observed errors across the sweep:

- `3043` HTTP `400` responses
- `123` connection resets
- `72` remote disconnects

### Interpretation

- This model is not a clean drop-in reranker for the current request shape.
- It returned a fixed block of HTTP `400` responses at every tested concurrency level.
- GPU utilization stayed extremely low, so the problem is request compatibility, not GPU saturation.
- Because the baseline sweep already had persistent `400`s, I did **not** extend it above `1024`.

Practical conclusion:

- `cross-encoder/ms-marco-MiniLM-L6-v2` is not viable under the current production-like rerank request shape

## MS MARCO MiniLM L4 v2

Language support:

- English-only

Server command used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model cross-encoder/ms-marco-MiniLM-L4-v2 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

Benchmark command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model cross-encoder/ms-marco-MiniLM-L4-v2 \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 2 \
  --concurrencies 32 64 128 256 512 1024 \
  --trials 2 \
  --output-json /tmp/ms_marco_MiniLM_L4_v2_concurrency.json
```

### Results

Median results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU peak % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32` | `96.555` | `83.635` | `401.990` | `16354.836` | `5.476` | `6.000` | `1827` |
| `64` | `94.776` | `83.785` | `768.431` | `16646.909` | `4.595` | `6.000` | `1827` |
| `128` | `94.736` | `83.513` | `1499.135` | `16643.856` | `4.714` | `6.500` | `1827` |
| `256` | `96.474` | `83.905` | `2853.322` | `16173.269` | `5.000` | `7.000` | `1827` |
| `512` | `93.213` | `84.146` | `5747.325` | `16412.893` | `3.643` | `6.000` | `1827` |
| `1024` | `91.469` | `84.176` | `10956.870` | `16191.468` | `4.125` | `6.000` | `1827` |

Observed errors across the sweep:

- `3056` HTTP `400` responses
- `122` connection resets
- `41` remote disconnects

### Interpretation

- This model is also not a clean drop-in reranker for the current request shape.
- It showed the same fixed-`400` failure pattern as `cross-encoder/ms-marco-MiniLM-L6-v2`.
- GPU utilization stayed extremely low, so the blocker is not compute saturation.
- Because the baseline sweep already had persistent `400`s, I did **not** extend it above `1024`.

Practical conclusion:

- `cross-encoder/ms-marco-MiniLM-L4-v2` is not viable under the current production-like rerank request shape

## MS MARCO MiniLM L2 v2

Language support:

- English-only

Server command used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model cross-encoder/ms-marco-MiniLM-L2-v2 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

Benchmark command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model cross-encoder/ms-marco-MiniLM-L2-v2 \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 2 \
  --concurrencies 32 64 128 256 512 1024 \
  --trials 2 \
  --output-json /tmp/ms_marco_MiniLM_L2_v2_concurrency.json
```

### Results

Median results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU peak % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32` | `99.798` | `83.965` | `391.005` | `15856.370` | `3.000` | `4.000` | `1809` |
| `64` | `94.912` | `83.905` | `788.993` | `16674.406` | `3.333` | `4.000` | `1809` |
| `128` | `97.838` | `83.845` | `1458.734` | `16070.939` | `3.500` | `4.000` | `1809` |
| `256` | `93.272` | `83.815` | `2938.057` | `16712.210` | `2.965` | `4.000` | `1809` |
| `512` | `94.835` | `83.905` | `5694.259` | `16130.408` | `3.417` | `4.500` | `1809` |
| `1024` | `91.867` | `84.326` | `10994.553` | `15972.870` | `2.893` | `4.000` | `1809` |

Observed errors across the sweep:

- `3042` HTTP `400` responses
- `111` connection resets
- `46` remote disconnects

### Interpretation

- This model is also not a clean drop-in reranker for the current request shape.
- It showed the same fixed-`400` failure pattern as `cross-encoder/ms-marco-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L4-v2`.
- GPU utilization stayed extremely low, so the issue is request compatibility, not lack of GPU capacity.
- Because the baseline sweep already had persistent `400`s, I did **not** extend it above `1024`.

Practical conclusion:

- `cross-encoder/ms-marco-MiniLM-L2-v2` is not viable under the current production-like rerank request shape

## Mixedbread Rerank Large v2

Language support:

- multilingual, with 100+ language support in the model card

Server command used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model mixedbread-ai/mxbai-rerank-large-v2 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

Main sweep command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model mixedbread-ai/mxbai-rerank-large-v2 \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 2 \
  --concurrencies 32 64 128 256 512 1024 \
  --trials 2 \
  --output-json /tmp/mxbai_rerank_large_v2_concurrency.json
```

Higher-concurrency extension command used:

```bash
python3 benchmark_vllm_reranker_concurrency.py \
  --model mixedbread-ai/mxbai-rerank-large-v2 \
  --api-base http://127.0.0.1:8000 \
  --repeat-factor 10 \
  --concurrencies 2048 4096 8192 16384 \
  --trials 1 \
  --output-json /tmp/mxbai_rerank_large_v2_above_1024.json
```

### Results

Median main-sweep results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU peak % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32` | `99.544` | `100.000` | `356.214` | `15895.311` | `90.000` | `100.000` | `42777` |
| `64` | `100.065` | `100.000` | `685.832` | `15733.788` | `90.000` | `100.000` | `42777` |
| `128` | `99.363` | `100.000` | `1348.800` | `15785.750` | `83.333` | `100.000` | `42777` |
| `256` | `98.594` | `100.000` | `2647.874` | `15745.416` | `99.800` | `100.000` | `42777` |
| `512` | `98.095` | `100.000` | `5282.983` | `15475.272` | `89.750` | `100.000` | `42777` |
| `1024` | `97.448` | `100.000` | `11465.455` | `14757.059` | `88.833` | `100.000` | `42777` |

Higher-concurrency extension:

| Concurrency | Repeat Factor | Req/s | Success % | Service p95 ms | End-to-end p95 ms | Mean GPU util % | Peak GPU util % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2048` | `10` | `97.786` | `100.000` | `21520.880` | `77567.137` | `96.154` | `100.000` | `42777` |
| `4096` | `10` | `95.901` | `100.000` | `45465.004` | `75009.789` | `93.700` | `100.000` | `42777` |
| `8192` | `10` | `94.183` | `100.000` | `69936.375` | `69938.847` | `91.786` | `100.000` | `42777` |
| `16384` | `10` | `94.722` | `100.000` | `69555.624` | `69563.012` | `97.731` | `100.000` | `42777` |

### Interpretation

- `mixedbread-ai/mxbai-rerank-large-v2` is clean and stable through `16384` concurrency with `100%` success.
- It is clearly GPU-bound almost immediately.
- Throughput is much lower than `mixedbread-ai/mxbai-rerank-base-v2`:
  - about `98` to `100 req/s` in the main sweep
  - about `94` to `98 req/s` above `1024`
- GPU memory is essentially the same heavy footprint as the base model:
  - about `42.8 GB`
- Service p95 grows steadily with concurrency and becomes very large above `1024`.

Practical conclusion:

- `mixedbread-ai/mxbai-rerank-large-v2` is slower than `mixedbread-ai/mxbai-rerank-base-v2` while using essentially the same very high GPU memory
- on throughput alone, it is not the better serving choice in this environment

## Winner

Using `req/s` and `service p95` as the main decision factors, there are now two different winners depending on whether you allow request-shape adjustment.

Fastest clean drop-in reranker:

- `mixedbread-ai/mxbai-rerank-base-v2` — multilingual

Why:

- it is the fastest reranker measured that worked with the current production-like request shape as-is
- it sustained about `208` to `233 req/s` in the main sweep
- its service p95 was excellent for that throughput tier
  - about `205 ms` at `32`
  - about `405 ms` at `64`
- it remained stable through `16384` concurrency with `100%` success

Main caveat:

- it uses about `42.7 GB` of GPU memory, which is much heavier than the other rerankers

Fastest measured reranker overall:

- `mixedbread-ai/mxbai-rerank-base-v2` — multilingual

Why:

- it delivered the highest throughput seen so far without requiring any request-shape changes
  - about `208` to `233 req/s` in the main sweep
- its service p95 was stronger than every other tested reranker in the same clean-drop-in category
  - about `205 ms` at `32`
  - about `405 ms` at `64`
- it stayed stable through `16384` concurrency with `100%` success

Fastest measured reranker among models that require request-shape adjustment:

- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — multilingual

Why:

- even after reducing `rerank_query_max_chars` to `512`, it still delivered very strong throughput
  - about `145` to `165 req/s` in the main sweep
- its `service p95` was still strong for that tier
  - about `587 ms` at `64`
  - about `1235 ms` at `128`
- it stayed stable through `16384` concurrency with `100%` success

Important caveat:

- this required reducing `rerank_query_max_chars` to `512`
- with the current production-like `1800`-char rerank query cap, it failed on warmup with HTTP `400`

Cleanest drop-in winner for the current request shape:

- `Alibaba-NLP/gte-reranker-modernbert-base` — English-only

Why:

- it is the strongest performer that worked cleanly under the current production-like rerank request shape
- it sustained materially higher throughput than `BAAI/bge-reranker-v2-m3`
  - about `110` to `125 req/s` vs about `66` to `69 req/s`
- it remained stable through much higher tested concurrency
- its service p95 at practical load levels was better than the BGE reranker for a meaningfully higher throughput tier

Reranker ranking from a performance point of view for viable models:

1. `mixedbread-ai/mxbai-rerank-base-v2`
2. `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — multilingual, with `rerank_query_max_chars = 512`
3. `Alibaba-NLP/gte-reranker-modernbert-base` — English-only
4. `Alibaba-NLP/gte-multilingual-reranker-base` — multilingual
5. `jinaai/jina-reranker-v2-base-multilingual` — multilingual
6. `BAAI/bge-reranker-base` — Chinese and English, with `rerank_query_max_chars = 512`
7. `mixedbread-ai/mxbai-rerank-large-v2` — multilingual
8. `BAAI/bge-reranker-v2-m3` — multilingual
9. `colbert-ir/colbertv2.0` via `/pooling` — English-focused

Excluded from the ranking because they are not clean under the current production-like request shape:

- `jinaai/jina-reranker-v2-base-multilingual`
- `cross-encoder/ms-marco-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L4-v2`
- `cross-encoder/ms-marco-MiniLM-L2-v2`

Why ColBERT ranks last here:

- direct `/pooling` throughput was only about `46` to `50 req/s`
- it started dropping requests above `64` concurrency
- this path also is not a normal hosted rerank endpoint; it is only one piece of the broader ColBERT-MaxSim architecture

## ColBERT MaxSim Reranking Path

Language support:

- English-focused / not a multilingual reranker in this test set

ColBERT in [`categorize_on_hosted_models_colbert_maxsim.py`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models_colbert_maxsim.py) is not used like the two hosted reranker models above.

The reranking path is:

1. build one page query text
2. use dense embeddings + FAISS to retrieve taxonomy candidates
3. send the page query once to the ColBERT vLLM `/pooling` endpoint to get per-token query vectors
4. score FAISS candidates locally with NumPy MaxSim against a precomputed taxonomy token-vector store

Important differences from `/v1/rerank`:

- no `(query, documents[])` rerank API call per record
- one ColBERT token-embedding request per page, not one rerank request with candidate docs
- candidate category token vectors are precomputed once at startup
- reranking work is split between:
  - ColBERT query token embedding on vLLM
  - local CPU MaxSim over the candidate set

That means ColBERT results are **not directly comparable** to the isolated `/v1/rerank` concurrency numbers above. The correct comparison is pipeline throughput from the full ColBERT-MaxSim categorization run.

From [`categorize_on_hosted_models_colbert_maxsim_findings_2_GPU.md`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models_colbert_maxsim_findings_2_GPU.md):

- embedding model on `8000`: `BAAI/bge-m3` via `/v1/embeddings`
- ColBERT model on `8001`: `colbert-ir/colbertv2.0` via `/pooling`
- best measured worker count: `64`
- best measured throughput: `39.00` total rows/s, `32.60` successful rows/s

Per-record timing at the best measured point:

- total mean: `1555.75 ms`
- total p95: `2853 ms`
- content embedding mean: `116.43 ms`
- ColBERT query embed mean: `1210.75 ms`
- MaxSim CPU mean: `196.37 ms`

Operational findings:

- the ColBERT query-embedding step dominates latency
- the local MaxSim CPU step is smaller, but still non-trivial at high concurrency
- sampled GPU utilization remained low even at the best throughput point
- the observed throughput ceiling looked driven more by request/queue overhead and the expensive ColBERT stage than by GPU saturation

Practical interpretation:

- ColBERT-MaxSim is a separate reranking design, not a drop-in hosted reranker endpoint
- it delivered much lower end-to-end page throughput than the hosted reranker models in these tests
- its main value is quality/architecture experimentation, not high standalone reranker request throughput

## ColBERT `/pooling` Server Concurrency

Language support:

- English-focused / not a multilingual reranker in this test set

I also benchmarked the ColBERT vLLM server directly as a token-embedding service, because that is the actual hosted request path used by [`categorize_on_hosted_models_colbert_maxsim.py`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models_colbert_maxsim.py).

Server command used:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model colbert-ir/colbertv2.0 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

Benchmark script used:

- [`benchmark_vllm_colbert_pooling_concurrency.py`](/home/harshad.mane/Harshad_Categorization/benchmark_vllm_colbert_pooling_concurrency.py)

Benchmark shape:

- source input: [`save_url_content_1000.jsonl`](/home/harshad.mane/Harshad_Categorization/save_url_content_1000.jsonl)
- same page-query construction as the production ColBERT script
- same token-aware truncation approach as the production ColBERT script
- `max_model_tokens = 512`
- `repeat_factor = 8`
- total requests = `6648`
- endpoint: `/pooling`

Results:

| Concurrency | Req/s | Success % | Service p95 ms | End-to-end p95 ms | Mean tokens/request | Mean GPU util % | Peak GPU util % | GPU mem MB | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `64` | `45.747` | `100.000` | `2568.335` | `137950.542` | `307.006` | `2.396` | `5.000` | `2087` | none |
| `128` | `48.268` | `99.970` | `4999.956` | `129792.775` | `307.003` | `2.477` | `8.000` | `2087` | `2` remote disconnects |
| `256` | `45.730` | `99.789` | `7376.459` | `142981.479` | `307.056` | `2.756` | `16.000` | `2087` | `14` dropped connections |
| `512` | `47.022` | `99.880` | `12888.947` | `138006.614` | `306.819` | `2.068` | `9.000` | `2087` | `8` dropped connections |
| `1024` | `45.788` | `99.985` | `22911.363` | `113432.181` | `307.045` | `2.023` | `9.000` | `2087` | `1` remote disconnect |
| `2048` | `49.998` | `99.744` | `30382.861` | `102247.519` | `306.854` | `2.135` | `9.000` | `2087` | `17` dropped connections |
| `4096` | `50.086` | `99.819` | `33491.556` | `98352.791` | `307.125` | `2.026` | `8.000` | `2087` | `12` dropped connections |

Where “dropped connections” means:

- `RemoteDisconnected`
- `Connection reset by peer`

### ColBERT `/pooling` Findings

- The ColBERT `/pooling` server did **not** appear GPU-bound in this test.
- GPU utilization stayed extremely low, around `2%` mean, with single-digit peaks in most samples.
- Throughput stayed roughly flat around `46` to `50 req/s`.
- The server started dropping requests as soon as concurrency rose above `64`.
- If the rule is “stop once vLLM starts dropping requests,” then the highest clean tested concurrency was:
  - `64`

Practical interpretation:

- highest tested concurrency with `100%` success: `64`
- above `64`, the request path starts showing connection resets / remote disconnects
- increasing concurrency beyond that did not produce meaningful throughput gains
- the bottleneck is not obvious GPU saturation; it looks more like request-path or server-side handling overhead for ColBERT multi-vector `/pooling` responses
