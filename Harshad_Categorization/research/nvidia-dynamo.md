# NVIDIA Dynamo For This Pipeline

## Context

Current pipeline:

- `BAAI/bge-m3` on vLLM for dense retrieval embeddings
- `colbert-ir/colbertv2.0` on vLLM for token-level pooling embeddings used by local MaxSim reranking
- Current serving shape is two separate inference stages, both on single-GPU single-instance vLLM servers

The question is whether NVIDIA Dynamo can increase throughput for this workload.

## What Dynamo Is Good At

NVIDIA Dynamo is a distributed inference framework for generative AI and reasoning workloads. Its main value proposition is:

- disaggregated prefill/decode serving
- KV-aware routing
- dynamic scheduling and routing across workers
- a frontend that can expose OpenAI-compatible APIs, including `/v1/embeddings`
- backend integration with vLLM while preserving native vLLM engine args

Practical reading of that:

- Dynamo is strongest when you have multiple workers, multiple GPUs, or multi-node deployments
- Dynamo is especially optimized for generative LLM serving where prefill and decode are separate phases

## Fit For Our Current Problem

### What fits

Dynamo can plausibly help with:

- putting a routing/frontend layer in front of multiple vLLM replicas
- scaling out BGE embedding workers behind one OpenAI-compatible embeddings endpoint
- adding centralized routing, discovery, and observability if this system grows beyond two manually managed servers
- passing vLLM tuning flags through Dynamo-managed workers, since Dynamo vLLM uses vLLM’s native arg parser

### What does not fit well

The main Dynamo feature, disaggregated prefill/decode serving, is not a natural fit for this pipeline.

Why:

- our workload is embedding/pooling inference, not text generation
- there is no long-running decode phase
- there is no token generation loop where KV cache transfer and prefill/decode specialization would help
- Dynamo’s AIConfigurator is built around generative LLM metrics like `TTFT`, `TPOT`, input sequence length, and output sequence length

That means the marquee Dynamo optimization path is solving a different problem than ours.

## Most Important Conclusion

For this exact pipeline, Dynamo is unlikely to unlock a big throughput jump by itself on only 2 GPUs.

The reason is simple:

- Dynamo’s biggest wins come from routing and scaling generative LLM workers
- our current bottlenecked stages are embedding/pooling requests on two small models
- we already observed low GPU utilization, which suggests request shape, batching, and client/server scheduling are the bigger issues

So the short answer is:

- **Dynamo is probably not the primary fix for current throughput**
- **it becomes more attractive if we scale to more replicas / more GPUs / more nodes**

## Best-Case Dynamo Use For Us

If we choose to use Dynamo anyway, the most realistic path is:

1. Keep the current architecture aggregated, not disaggregated.
2. Use Dynamo as a frontend + worker runtime for vLLM backends.
3. Replicate the BGE service first if dense embedding QPS becomes the limiting stage.
4. Replicate ColBERT workers only if we create a clean serving path for token-level pooling outputs.

Expected benefit from that setup:

- better replica management
- request routing to multiple workers
- cleaner operational model than hand-managed ports and processes
- room to grow to more GPUs later

Expected non-benefit:

- little or no magic gain from prefill/decode separation, because that is not our workload shape

## Biggest Integration Risk

The Dynamo frontend officially advertises OpenAI-compatible endpoints including `/v1/embeddings`, but I did not find official frontend documentation for a `/pooling` endpoint.

That matters because:

- BGE fits naturally behind `/v1/embeddings`
- our ColBERT setup currently depends on token-level pooling output from `/pooling`

So if we adopt Dynamo:

- BGE is straightforward
- ColBERT may require either:
  - direct access to the backend worker instead of the Dynamo frontend, or
  - a custom adapter / custom frontend path if token-level pooling must remain exposed

This is the main technical reason not to rush into Dynamo for the current system.

## Recommended Decision

### Recommendation now

Do **not** make NVIDIA Dynamo the next optimization project for this exact 2-GPU setup.

Instead, prioritize:

- deeper batching/scheduler tuning on direct vLLM serving
- request-shape optimization
- reducing client-side overhead
- testing more than one vLLM replica only after single-instance tuning is exhausted

### When Dynamo becomes worth it

Revisit Dynamo if one or more of these become true:

- we move beyond 2 GPUs
- we want one managed routing layer for multiple replicas
- we need multi-node scaling
- we convert part of the system to generative LLM serving
- we want centralized discovery, frontend routing, and operational tooling rather than hand-managed vLLM processes

## Practical Recommendation If We Pilot Dynamo

If we still want to test it, keep the pilot narrow:

1. Start with **BGE only** behind Dynamo’s embeddings-compatible frontend.
2. Keep ColBERT on direct vLLM `/pooling` until we prove a good token-pooling integration path.
3. Use **aggregated serving**, not disaggregated serving.
4. Treat the pilot as an ops/scaling experiment, not as a guaranteed throughput win.

Success criteria for that pilot:

- simpler routing and scaling
- no regression in rows/sec
- no regression in ColBERT integration

## Bottom Line

NVIDIA Dynamo is a strong serving platform, but for this specific pipeline it is mostly an **infrastructure/routing option**, not an obvious **throughput breakthrough**.

For the current problem:

- **likely low upside right now:** disaggregated Dynamo architecture
- **possible medium-term upside:** Dynamo as a frontend/runtime once we have more replicas or more GPUs
- **highest near-term upside:** continue optimizing direct vLLM batching and request flow

## Sources

- NVIDIA Dynamo architecture overview
- NVIDIA Dynamo frontend docs
- NVIDIA Dynamo vLLM backend docs
- NVIDIA Dynamo disaggregated serving docs
- NVIDIA AIConfigurator / disaggregated serving guide
