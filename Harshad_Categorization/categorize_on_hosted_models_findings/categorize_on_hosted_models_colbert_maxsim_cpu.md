# CPU Findings: `categorize_on_hosted_models_colbert_maxsim_cpu.py`

## Approach

CPU-only local categorization pipeline:

- Script: [`categorize_on_hosted_models_colbert_maxsim_cpu.py`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models_colbert_maxsim_cpu.py)
- Input: [`save_url_content_1000.jsonl`](/home/harshad.mane/Harshad_Categorization/save_url_content_1000.jsonl)
- Taxonomy: [`taxonomy/Content_Taxonomy_3.1_2.tsv`](/home/harshad.mane/Harshad_Categorization/taxonomy/Content_Taxonomy_3.1_2.tsv)
- Retrieval model: `BAAI/bge-m3`
- Reranker: `colbert-ir/colbertv2.0`
- No vLLM
- No batching
- One process handles one input record at a time
- Each worker process builds and owns its own local taxonomy FAISS index and ColBERT token store

## Machine

- CPU: `AMD EPYC 9634 84-Core Processor`
- Physical cores: `84`
- Logical CPUs: `168`
- RAM: `503 GiB`
- Available RAM at inspection time: about `452 GiB`
- Swap: `0`

## Output Files

- `10` workers:
  [`categorize_on_hosted_models_colbert_maxsim_cpu_wc_10.jsonl`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models_colbert_maxsim_cpu_wc_10.jsonl)
- `84` workers:
  [`categorize_on_hosted_models_colbert_maxsim_cpu_wc_84.jsonl`](/home/harshad.mane/Harshad_Categorization/categorize_on_hosted_models_colbert_maxsim_cpu_wc_84.jsonl)

## Measured Results

### `10` workers

- Record count: `994`
- Successful records: `831`
- Failed records: `163`
- Total runtime: `653376 ms`
- End-to-end total throughput: `994 / 653.376 = 1.52 records/sec`
- End-to-end successful throughput: `831 / 653.376 = 1.27 successful records/sec`

### `84` workers

- Record count: `994`
- Successful records: `831`
- Failed records: `163`
- Total runtime: `778623 ms`
- End-to-end total throughput: `994 / 778.623 = 1.28 records/sec`
- End-to-end successful throughput: `831 / 778.623 = 1.07 successful records/sec`

## Key Observation

`84` workers was slower than `10` workers for this CPU-only design.

- `10` workers: `653376 ms`
- `84` workers: `778623 ms`

That means aggressive worker replication hurt end-to-end throughput instead of improving it.

## Why `84` Workers Was Slower

This CPU design replicates expensive state per worker:

- local `bge-m3` model
- local ColBERT model
- worker-local FAISS taxonomy index
- worker-local ColBERT token store

At `84` workers, startup overhead became very large.

Observed during the `84`-worker run:

- all worker children were CPU-saturated during bootstrap
- memory usage rose to about `160 GiB used`
- available memory was still healthy at about `338 GiB`
- so the main issue was not OOM, it was bootstrap / replication cost

## Bootstrap vs Processing

Important caveat:

The first `84`-worker run was launched before explicit split timing was added to the script, so this run does **not** contain an exact measured `processing_only_ms`.

What was observed manually:

- no processed-record output yet at `2026-05-08 18:44:30 PDT`
- processed-record output definitely present by `2026-05-08 18:47:49 PDT`
- by `2026-05-08 18:48:49 PDT`, the run had already reached at least `[906/994]`
- the run finished around `18:49 PDT`

Conclusion:

- bootstrap dominated the `84`-worker run
- record processing after bootstrap was only a small fraction of total wall-clock time
- best current estimate for processing-only time on that `84`-worker run is roughly `2` to `5` minutes
- a rough central estimate used in discussion was `3` minutes

If we assume `3` minutes of processing-only time:

- total processing throughput: `994 / 180 = 5.52 records/sec`
- successful processing throughput: `831 / 180 = 4.62 successful records/sec`

This is an estimate, not an exact measured number.

## Instrumentation Added

The CPU script was later patched so future runs will report exact split timings:

- `worker_bootstrap_total_ms`
- `processing_only_ms`
- `avg_worker_init_ms`
- `max_worker_init_ms`

This means the next rerun of the CPU script can provide an exact post-bootstrap processing time instead of an estimate.

## Current Conclusion

For this machine and this implementation:

- CPU-only local inference is functional
- `10` workers outperformed `84` workers end to end
- the replicated per-worker bootstrap cost is the dominant problem at high worker counts
- higher worker count does not automatically translate to better throughput for this design

## Recommended Next Step

Run one more controlled CPU benchmark with the patched script so the output contains:

- exact bootstrap time
- exact processing-only time
- exact per-worker initialization statistics

Suggested rerun target:

- rerun `10` workers with patched timing
- rerun `84` workers with patched timing
- compare `worker_bootstrap_total_ms` vs `processing_only_ms`
