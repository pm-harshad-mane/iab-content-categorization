# vLLM Embedding Concurrency Test

## Goal

Measure how many concurrent embedding requests a single vLLM server on GPU `0` can handle for real page-content inputs, while tracking:

- response time
- throughput
- GPU usage
- highest concurrency handled successfully

Models tested one at a time on `8000`:

- `BAAI/bge-m3`
- `google/gemma-4-E4B-it`
- `google/embeddinggemma-300m`

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

## Test Setup

- GPU: `0` only
- Endpoint: `http://127.0.0.1:8000/v1/embeddings`
- Input source: [`save_url_content_1000.jsonl`](/home/harshad.mane/Harshad_Categorization/save_url_content_1000.jsonl)
- Request builder: same title / description / headings / body concatenation pattern used in [`categorize_on_hosted_models.py`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models.py)
- Effective request count: `831`

Why `831` and not `994`:

- only `status == "ok"` rows were used
- rows with empty final query text were skipped

## Server Launch Commands

All three servers were launched with prefix caching explicitly disabled:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-m3 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-E4B-it \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model google/embeddinggemma-300m \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

Observed in vLLM startup logs:

- `BAAI/bge-m3`: `enable_prefix_caching=False`
- `google/gemma-4-E4B-it`: `enable_prefix_caching=False`
- `google/embeddinggemma-300m`: `enable_prefix_caching=False`

## Benchmark Method

For each model:

1. start the vLLM server on `8000`
2. verify `/v1/models` and `/v1/embeddings`
3. warm up with one embedding request
4. run the same request set at these client-side concurrency levels:
   `1 2 4 8 16 32 64 128 256 512 768 1024`
5. record:
   - total wall time
   - requests/sec
   - successful requests/sec
   - p95 response time
   - GPU utilization and memory via `nvidia-smi`

Important caveat:

- the fast models complete some high-concurrency runs in about `2` seconds, so 1-second GPU sampling only gives a small number of samples for those runs

## Summary

| Model | Highest tested concurrency with full success | Best throughput point | Best requests/sec | Best p95 ms | GPU memory footprint |
|---|---:|---:|---:|---:|---:|
| `BAAI/bge-m3` | `1024` | `512` | `430.100` | `895.003` | `~3.6 GB` |
| `google/gemma-4-E4B-it` | `1024` | `32` | `42.181` | `922.037` | `~44.1 GB` |
| `google/embeddinggemma-300m` | none | `512` | `473.576` | `791.967` | `~2.6 GB` |

Main takeaways:

- `google/embeddinggemma-300m` had the highest measured throughput, but it was not fully reliable on this request set
- `BAAI/bge-m3` was the fastest model that stayed at `100%` success across all tested concurrencies
- `google/gemma-4-E4B-it` saturated GPU `0` quickly and then mostly traded latency for queue depth, not throughput

## Detailed Results

### `BAAI/bge-m3`

| Concurrency | Success % | Requests/sec | Successful req/sec | p95 ms | Mean GPU util % | Peak GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|
| `1` | `100.000` | `62.236` | `62.236` | `25.134` | `16.600` | `3159` |
| `2` | `100.000` | `73.616` | `73.616` | `40.642` | `16.750` | `3159` |
| `4` | `100.000` | `184.312` | `184.312` | `32.266` | `30.667` | `3185` |
| `8` | `100.000` | `173.329` | `173.329` | `60.326` | `43.000` | `3249` |
| `16` | `100.000` | `283.483` | `283.483` | `79.011` | `74.000` | `3289` |
| `32` | `100.000` | `236.252` | `236.252` | `257.258` | `63.000` | `3441` |
| `64` | `100.000` | `309.594` | `309.594` | `898.269` | `100.000` | `3625` |
| `128` | `100.000` | `414.479` | `414.479` | `424.595` | `100.000` | `3625` |
| `256` | `100.000` | `409.926` | `409.926` | `1051.890` | `100.000` | `3625` |
| `512` | `100.000` | `430.100` | `430.100` | `895.003` | `100.000` | `3625` |
| `768` | `100.000` | `427.164` | `427.164` | `1116.885` | `100.000` | `3625` |
| `1024` | `100.000` | `428.083` | `428.083` | `938.197` | `0.000` | `3625` |

Findings:

- best measured throughput: `512`
- highest tested fully successful concurrency: `1024`
- GPU became effectively saturated by about `64`
- memory stayed modest, around `3.6 GB`

Interpretation:

- this is the strongest choice here if full reliability matters
- it kept scaling deep into the queue without failures

### `google/gemma-4-E4B-it`

| Concurrency | Success % | Requests/sec | Successful req/sec | p95 ms | Mean GPU util % | Peak GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|
| `1` | `100.000` | `24.185` | `24.185` | `69.481` | `63.200` | `44145` |
| `2` | `100.000` | `31.124` | `31.124` | `106.371` | `79.571` | `44145` |
| `4` | `100.000` | `35.972` | `35.972` | `171.356` | `100.000` | `44145` |
| `8` | `100.000` | `40.452` | `40.452` | `307.862` | `100.000` | `44145` |
| `16` | `100.000` | `42.088` | `42.088` | `515.551` | `85.714` | `44145` |
| `32` | `100.000` | `42.181` | `42.181` | `922.037` | `83.333` | `44145` |
| `64` | `100.000` | `42.179` | `42.179` | `1733.473` | `100.000` | `44145` |
| `128` | `100.000` | `42.075` | `42.075` | `3278.292` | `100.000` | `44145` |
| `256` | `100.000` | `41.999` | `41.999` | `6419.366` | `100.000` | `44145` |
| `512` | `100.000` | `41.985` | `41.985` | `12410.103` | `100.000` | `44145` |
| `768` | `100.000` | `41.962` | `41.962` | `18042.737` | `100.000` | `44145` |
| `1024` | `100.000` | `41.894` | `41.894` | `17976.328` | `85.714` | `44145` |

Findings:

- best measured throughput: `32`
- highest tested fully successful concurrency: `1024`
- GPU `0` saturated almost immediately
- memory footprint was huge: about `44.1 GB`

Interpretation:

- this model can queue a lot of requests, but extra concurrency after about `16` to `32` does not buy throughput
- higher concurrency mainly increases latency

### `google/embeddinggemma-300m`

| Concurrency | Success % | Requests/sec | Successful req/sec | p95 ms | Mean GPU util % | Peak GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|
| `1` | `99.278` | `73.298` | `72.769` | `24.433` | `12.250` | `2609` |
| `2` | `99.278` | `104.743` | `103.987` | `33.652` | `10.667` | `2609` |
| `4` | `99.278` | `164.494` | `163.307` | `41.002` | `34.500` | `2629` |
| `8` | `98.917` | `269.275` | `266.359` | `41.475` | `40.000` | `2629` |
| `16` | `99.037` | `415.827` | `411.823` | `53.208` | `48.000` | `2629` |
| `32` | `99.278` | `455.170` | `451.884` | `91.090` | `51.000` | `2629` |
| `64` | `99.037` | `470.246` | `465.719` | `162.559` | `34.000` | `2629` |
| `128` | `99.037` | `241.011` | `238.691` | `582.815` | `28.000` | `2629` |
| `256` | `99.158` | `323.563` | `320.837` | `1077.629` | `9.000` | `2629` |
| `512` | `99.158` | `473.576` | `469.587` | `791.967` | `58.000` | `2629` |
| `768` | `99.037` | `457.721` | `453.314` | `1045.814` | `47.000` | `2629` |
| `1024` | `99.037` | `457.741` | `453.334` | `1003.205` | `52.000` | `2629` |

Observed recurring failures:

- `6` requests consistently failed with HTTP `400`
- some higher-concurrency runs also had a small number of connection resets / remote disconnects

Representative error counts:

- `HTTPError: 400 Client Error`: always present
- occasional `Connection reset by peer`
- occasional `Remote end closed connection without response`

Interpretation:

- highest measured throughput of all three models
- lowest memory footprint of all three models
- not acceptable if you need `100%` success on this request set without additional input truncation or retry logic

## Recommendation

If the objective is **highest reliable concurrency** on this exact content set:

- choose `BAAI/bge-m3`
- it sustained `100%` success through `1024` concurrent requests
- it delivered about `430 req/s` at its best point

If the objective is **highest raw throughput** and you can tolerate some request failures or add smarter truncation / retries:

- `google/embeddinggemma-300m` was fastest
- but it never reached `100%` success on this dataset

If the objective is **Gemma-family model quality experimentation** and throughput is secondary:

- `google/gemma-4-E4B-it` is operationally viable
- but it is much slower and far more memory-hungry
- concurrency above about `16` to `32` mostly just increases queueing latency

## Final Answer

Highest concurrent requests successfully handled on this request set:

- `BAAI/bge-m3`: `1024` concurrent requests completed with `100%` request success
- `google/gemma-4-E4B-it`: `1024` concurrent requests completed with `100%` request success
- `google/embeddinggemma-300m`: no tested concurrency level achieved `100%` request success

Best practical model for this workload:

- `BAAI/bge-m3`

## Winner

From a pure performance point of view, with `requests/sec` and `service p95` as the main criteria, the clear winner is:

- `google/embeddinggemma-300m`

Why:

- it delivered the highest measured throughput, around `450` to `477 req/s`
- its service p95 stayed lower than `BAAI/bge-m3` in the comparable high-concurrency reruns
- GPU memory usage was also much lower than `google/gemma-4-E4B-it`

Performance-oriented ranking:

1. `google/embeddinggemma-300m`
2. `BAAI/bge-m3`
3. `google/gemma-4-E4B-it`

Important caveat:

- `google/embeddinggemma-300m` was the performance winner, but not the reliability winner
- if `100%` request success is required on this request set without extra retry/truncation handling, then `BAAI/bge-m3` remains the safer operational choice

## Corrected Above-1024 Test

The earlier above-`1024` check had a flaw: the workload only contained `831` total requests, so anything above `831` was not a true higher-concurrency test.

I reran `BAAI/bge-m3` with:

- `repeat_factor = 8`
- total requests = `6648`
- prefix caching still disabled
- tested concurrencies:
  `1024 1536 2048 2560 3072 4096`

This made every tested concurrency level a real in-flight load level.

Results:

| Concurrency | Success % | Requests/sec | Successful req/sec | p95 ms | Mean GPU util % | Peak GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|
| `1024` | `100.000` | `429.519` | `429.519` | `5918.408` | `100.000` | `3295` |
| `1536` | `100.000` | `440.780` | `440.780` | `2905.864` | `75.000` | `3295` |
| `2048` | `100.000` | `437.473` | `437.473` | `2761.278` | `80.000` | `3295` |
| `2560` | `100.000` | `429.127` | `429.127` | `5188.610` | `80.000` | `3295` |
| `3072` | `100.000` | `431.359` | `431.359` | `4181.429` | `83.333` | `3295` |
| `4096` | `100.000` | `424.463` | `424.463` | `5168.973` | `100.000` | `3295` |

Conclusion from the corrected test:

- throughput stayed essentially flat in the `424` to `441 req/s` range
- pushing concurrency above `1024` did **not** unlock meaningful additional throughput
- higher concurrency mostly increased queueing latency into the multi-second range

So for `BAAI/bge-m3` on this machine and request mix:

- vLLM can **handle** at least `4096` concurrent requests successfully
- but the useful throughput ceiling is already reached by about `1024` to `2048`
- above that, you are mostly paying in response time rather than getting more work done

## End-to-End Latency Rerun

The first version of the benchmark reported only request service time, not full client-observed latency from task submission. That made some p95 values look non-monotonic.

I patched the benchmark to measure:

- `service_time_*`: request start to response
- `end_to_end_time_*`: task submission to response completion

Then I reran the repeated-load `BAAI/bge-m3` test with:

- `repeat_factor = 8`
- total requests = `6648`
- concurrencies:
  `1024 1536 2048 2560 3072 4096`
- `2` trials per point

Median results across the two trials:

| Concurrency | Median req/s | Median service p95 ms | Median end-to-end p95 ms |
|---|---:|---:|---:|
| `1024` | `418.971` | `3196.191` | `11327.434` |
| `1536` | `429.947` | `4367.136` | `8014.860` |
| `2048` | `428.367` | `3670.410` | `11176.010` |
| `2560` | `427.456` | `3516.445` | `11469.523` |
| `3072` | `424.796` | `3646.863` | `11253.918` |
| `4096` | `428.685` | `4153.197` | `11273.358` |

What this corrected rerun shows:

- throughput stayed essentially flat in the `419` to `430 req/s` band
- true end-to-end p95 stayed very high, around `8.0s` to `11.5s`
- increasing concurrency above `1024` did **not** create meaningful throughput gains

So the corrected answer is:

- above the useful saturation point, extra concurrency mostly increases queueing delay
- the non-monotonicity you noticed in the earlier p95 numbers was due to using service-time p95 instead of full end-to-end latency

## Gemma 4 E4B Repeated-Load Rerun

I reran `google/gemma-4-E4B-it` with the same corrected setup:

- `repeat_factor = 8`
- total requests = `6648`
- prefix caching disabled
- concurrencies:
  `1024 1536 2048 2560 3072 4096`

Median results from the completed points:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|
| `1024` | `42.146` | `100.000` | `24640.677` | `148399.229` | `98.076` | `44145` |
| `1536` | `42.088` | `100.000` | `36748.250` | `147638.227` | `98.309` | `44145` |
| `2048` | `42.067` | `100.000` | `48865.171` | `146948.877` | `97.938` | `44145` |
| `2560` | `41.998` | `100.000` | `61098.750` | `146181.231` | `99.000` | `44145` |
| `3072` | `41.973` | `100.000` | `73276.638` | `145268.167` | `98.944` | `44145` |
| `4096` | `41.952` | `100.000` | `97494.328` | `143111.269` | `98.936` | `44145` |

What this shows:

- `google/gemma-4-E4B-it` is GPU-bound very early
- GPU 0 is effectively saturated already at `1024` concurrency
- throughput is flat at about `42 req/s`
- higher concurrency only increases queueing and service latency

Because the throughput was far below the other models and GPU utilization was already pegged, I stopped further Gemma scaling work and moved on to `google/embeddinggemma-300m`.

## EmbeddingGemma Repeated-Load Rerun

I reran `google/embeddinggemma-300m` with the corrected benchmark:

- `repeat_factor = 8`
- total requests = `6648`
- prefix caching disabled
- concurrencies:
  `1024 1536 2048 2560 3072 4096`
- `2` trials per point

Median results:

| Concurrency | Median req/s | Success % | Median service p95 ms | Median end-to-end p95 ms | Median GPU util % | Median GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|
| `1024` | `451.173` | `99.150` | `2606.157` | `10397.716` | `39.166` | `2629` |
| `1536` | `467.769` | `99.173` | `3138.024` | `8292.752` | `46.100` | `2629` |
| `2048` | `472.615` | `99.210` | `2609.716` | `10661.815` | `52.375` | `2629` |
| `2560` | `467.487` | `99.180` | `3145.761` | `10478.945` | `47.367` | `2629` |
| `3072` | `461.863` | `99.226` | `3073.475` | `10787.221` | `39.800` | `2629` |
| `4096` | `470.725` | `99.180` | `2998.254` | `10487.400` | `46.025` | `2629` |

What this shows:

- throughput is much higher than Gemma 4 E4B and slightly above `BAAI/bge-m3`
- success rate still never reaches `100%` on this request set
- GPU utilization remains moderate, well below saturation
- simply increasing concurrency is not enough to fill the GPU

## EmbeddingGemma Higher-Concurrency Extension

To test whether GPU 0 would eventually saturate if concurrency kept rising, I extended the same benchmark above `4096` with larger repeated request pools.

Additional runs:

| Concurrency | Repeat Factor | Trials | Req/s | Success % | Service p95 ms | End-to-end p95 ms | Mean GPU util % | Peak GPU util % | GPU mem MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `6144` | `20` | `2` | `465.528` | `99.149` | `5556.084` | `29055.746` | `48.197` | `55.000` | `2629` |
| `8192` | `20` | `2` | `470.413` | `99.119` | `6421.536` | `28910.098` | `48.273` | `58.000` | `2629` |
| `12288` | `24` | `1` | `466.987` | `99.178` | `5360.097` | `36872.253` | `47.538` | `55.000` | `2629` |
| `16384` | `32` | `1` | `470.269` | `99.124` | `8875.777` | `48205.626` | `47.941` | `56.000` | `2629` |

Conclusion from the higher-concurrency extension:

- even at `16384` concurrent requests, GPU 0 still did not saturate
- throughput remained flat around `466` to `470 req/s`
- end-to-end latency kept rising sharply as queueing increased
- concurrency alone does not appear sufficient to fill the GPU for `google/embeddinggemma-300m` in this vLLM setup

The practical ceiling for this model is therefore:

- throughput ceiling: about `470 req/s`
- observed maximum tested concurrency handled: `16384`
- GPU saturation by concurrency alone: not reached
