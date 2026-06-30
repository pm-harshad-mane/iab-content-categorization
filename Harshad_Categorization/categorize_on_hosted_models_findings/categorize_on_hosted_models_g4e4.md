# Findings: `categorize_on_hosted_models.py` with Gemma 4 E4B + BGE Reranker

## Test Setup

- Script: [`categorize_on_hosted_models.py`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models.py)
- Input size: `994` records
- Output file: [`categorize_on_hosted_models_1000_g4e4.jsonl`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models_1000_g4e4.jsonl)
- Embedding model: `google/gemma-4-E4B-it`
- Reranker model: `BAAI/bge-reranker-v2-m3`
- Embedding server: GPU `0`, port `8000`
- Reranker server: GPU `1`, port `8001`

## vLLM Commands

Embedding server on GPU `0`:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-E4B-it \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

Reranker server on GPU `1`:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-reranker-v2-m3 \
  --runner pooling \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.9
```

Run command:

```bash
python3 categorize_on_hosted_models.py \
  --embed-api-base http://127.0.0.1:8000 \
  --embed-model google/gemma-4-E4B-it \
  --rerank-api-base http://127.0.0.1:8001 \
  --rerank-model BAAI/bge-reranker-v2-m3 \
  --output-jsonl categorize_on_hosted_models_1000_g4e4.jsonl
```

## Endpoint Validation

Confirmed before the run:

- `8000` answered `/v1/models` and `/v1/embeddings`
- `8001` answered `/v1/models` and `/v1/rerank`

So this model pairing was compatible with the existing hosted pipeline without code changes.

## Main Run Results

Output:

- [`categorize_on_hosted_models_1000_g4e4.jsonl`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models_1000_g4e4.jsonl)

Summary:

- total rows: `994`
- successful rows: `831`
- failed rows: `163`
- runtime: `34635 ms`
- throughput: `28.699` total rows/sec
- throughput: `23.993` successful rows/sec

Successful-row timing stats:

| Metric | Mean ms | P95 ms |
|---|---:|---:|
| `content_embedding` | `80.746` | `151` |
| `content_embedding_api` | `80.309` | `150` |
| `faiss_search` | `2.248` | `3` |
| `rerank` | `68.739` | `102` |
| `rerank_api` | `68.705` | `102` |
| `total` | `152.002` | `241` |

## Monitored Rerun

To capture GPU behavior, the same benchmark was rerun while sampling `nvidia-smi` once per second.

Monitored output:

- `/tmp/categorize_on_hosted_models_1000_g4e4_monitored.jsonl`

Summary:

- total rows: `994`
- successful rows: `831`
- failed rows: `163`
- runtime: `27282 ms`
- throughput: `36.434` total rows/sec
- throughput: `30.460` successful rows/sec

Successful-row timing stats:

| Metric | Mean ms | P95 ms |
|---|---:|---:|
| `content_embedding` | `47.935` | `57` |
| `content_embedding_api` | `47.510` | `57` |
| `faiss_search` | `2.219` | `3` |
| `rerank` | `71.970` | `109` |
| `rerank_api` | `71.941` | `108` |
| `total` | `122.390` | `160` |

## GPU Observations

Spot-sampled `nvidia-smi` during the monitored rerun consistently showed:

- GPU `0` (`Gemma 4 E4B`): `44093 MB` used, observed utilization `0%`
- GPU `1` (`BGE reranker`): `2955 MB` used, observed utilization `0%`

Interpretation:

- both models were resident in GPU memory during the run
- the 1-second spot samples did not catch sustained GPU utilization
- this suggests the pipeline is not obviously GPU-saturated, but the sampling method can miss short spikes

So these utilization numbers should be treated as directional, not as a full profile.

## What This Means

The Gemma 4 E4B + BGE reranker configuration worked end to end with the current hosted script and produced valid output.

The monitored rerun was materially faster than the first run:

- main run: `28.699` total rows/sec
- monitored rerun: `36.434` total rows/sec

That improvement came mostly from faster embedding latency:

- `content_embedding` mean dropped from `80.746 ms` to `47.935 ms`
- rerank latency stayed in roughly the same range

The observed GPU data does not show saturation on either GPU, so the next throughput gains are unlikely to come from simply adding more GPU memory. The more plausible next levers are request concurrency, batching behavior inside vLLM, or a different embedding model/runtime choice.

## Concurrency Sweep

To test how many more concurrent requests the current vLLM setup can absorb, I reran [`categorize_on_hosted_models.py`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models.py) with higher `--concurrent-records` values while keeping the same two live model servers:

- embedding: `google/gemma-4-E4B-it` on `8000`
- reranker: `BAAI/bge-reranker-v2-m3` on `8001`

Results:

| Concurrent records | `run_total_ms` | Total rows/sec | Successful rows/sec |
|---|---:|---:|---:|
| `4` | `24929` | `39.873` | `33.335` |
| `8` | `16802` | `59.160` | `49.458` |
| `16` | `15406` | `64.520` | `53.940` |
| `32` | `14612` | `68.026` | `56.871` |
| `64` | `14953` | `66.475` | `55.574` |
| `96` | `14856` | `66.909` | `55.937` |
| `128` | `14828` | `67.035` | `56.043` |
| `192` | `14697` | `67.633` | `56.542` |
| `256` | `14860` | `66.891` | `55.922` |
| `384` | `15073` | `65.946` | `55.132` |
| `512` | `15314` | `64.908` | `54.264` |
| `768` | `15637` | `63.567` | `53.143` |
| `1024` | `15133` | `65.684` | `54.913` |

### What This Shows

- throughput improved sharply from `4` to `32`
- after `32`, throughput mostly flattened
- the best measured unmonitored point was `192` concurrent records
- even `512`, `768`, and `1024` still completed successfully with the same `831/994` success count

So the system can handle far more in-flight work than the earlier low-concurrency runs suggested. The main constraint is not correctness or outright failure. It is diminishing returns after roughly the `32` to `192` range.

### Practical Recommendation

If the objective is maximum throughput, the best point measured here is:

- `192` concurrent records
- `67.633` total rows/sec
- `56.542` successful rows/sec

If the objective is simply to allow many more requests to queue concurrently while keeping throughput roughly flat, the setup still behaved acceptably up to:

- `1024` concurrent records

That means the current vLLM services appear capable of absorbing at least `1024` in-flight record requests from this client without collapsing.

## GPU Samples at 1024 Concurrency

I also reran the benchmark at `1024` concurrent records with `nvidia-smi` spot sampling.

Monitored run:

- output: `/tmp/categorize_g4e4_w1024_monitored.jsonl`
- runtime: `14941 ms`
- throughput: `66.528` total rows/sec
- throughput: `55.619` successful rows/sec

Observed GPU samples during that run consistently showed:

- GPU `0` (`Gemma 4 E4B`): `44093 MB` used, observed utilization `0%`
- GPU `1` (`BGE reranker`): `2955 MB` used, observed utilization `0%`

Interpretation:

- the services remained memory-resident and stable even at `1024` client-side concurrency
- the spot samples still did not show sustained GPU utilization
- the likely bottleneck remains request scheduling / batching behavior rather than raw GPU saturation

## FAISS-Only vs Reranker

I also checked whether this setup could skip reranking and rely on Gemma 4 E4B FAISS results alone.

### Top-1 Agreement

Across the `831` successful records:

- FAISS top-1 matched the reranked top-1 in `96/831` records: `11.6%`
- FAISS top-1 differed from the reranked top-1 in `735/831` records: `88.4%`

So FAISS top-1 is not a reliable replacement for the reranker in this run.

### Top-5 Candidate Quality

Considering FAISS top-5 as a candidate set instead of a final ranking:

- reranked top-1 was present in FAISS top-5 for `416/831` records: `50.1%`
- reranked top-1 was present in FAISS top-10 for `831/831` records: `100%`

This means:

- FAISS top-10 worked well as a retrieval stage
- FAISS top-5 missed the eventual reranked winner about half the time

### Top-5 Set Overlap

Comparing FAISS top-5 with reranked top-5:

- exact same top-5 set: `4/831` records: `0.5%`
- exact same top-5 order: `0/831` records
- mean overlap: `2.48` categories out of `5`

Overlap distribution:

- overlap `0`: `6`
- overlap `1`: `86`
- overlap `2`: `335`
- overlap `3`: `313`
- overlap `4`: `87`
- overlap `5`: `4`

So FAISS top-5 is not a good substitute for the final reranked top-5 output.

### FAISS Score Thresholds

I also checked whether a high FAISS top-1 score could be used as a confidence gate.

Agreement with reranked top-1:

- FAISS top-1 score `>= 0.60`: `22/80` = `27.5%`
- FAISS top-1 score `>= 0.65`: `16/24` = `66.7%`
- FAISS top-1 score `>= 0.68`: `11/13` = `84.6%`

The only threshold that reached at least `90%` agreement was approximately `0.680117`, but it covered only `12` records. That is too little coverage to be useful as a general skip-rerank rule.

Top1-top2 FAISS margin was also not useful as a confidence signal in this run.

### Conclusion

For this Gemma 4 E4B embedding setup:

- do not skip the reranker if you want output behavior close to the current pipeline
- FAISS top-10 is strong enough as a candidate generator
- FAISS top-5 is not strong enough as a final answer set

Important caveat:

- this analysis measures agreement with the reranker, not true taxonomy accuracy
- without labeled ground truth, it does not prove the reranker is correct, only that it changes results substantially
