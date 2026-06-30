# FAISS Confidence Analysis for `categorize_on_hosted_models_1000.jsonl`

## Scope

This report analyzes the output in `categorize_on_hosted_models_1000.jsonl` to answer one question:

When does the FAISS retrieval step look confident enough that we might skip the reranker?

The file contains 994 total rows, of which 831 have `ok=true` and include:

- top-10 FAISS candidates with `faiss_score`
- final top-5 reranked candidates with `rerank_score`
- per-record timings

There is no ground-truth label file here, so this report does **not** measure accuracy directly. Instead, it uses **agreement with the current reranked top-1** as a proxy:

- if FAISS top-1 equals reranked top-1, FAISS looked stable enough for that page
- if reranker promoted a different candidate, FAISS looked less certain for that page

That proxy is useful for runtime strategy decisions, but it is not the same as correctness.

## Summary

The main findings are:

- FAISS itself is cheap. Mean `faiss_search` time is `1.32 ms`.
- The reranker is the dominant expensive step. Mean `rerank` time is `66.17 ms`.
- Mean total per-record time is `100.52 ms`.
- FAISS top-1 matches the current reranked top-1 on only `32.37%` of successful rows.
- The most useful confidence signal in this dataset is **the gap between FAISS rank 1 and rank 2**, not the absolute FAISS top-1 score.
- A second good signal is **how spread out the top-5 FAISS scores are**.

In short:

- **large FAISS score gap -> more confidence**
- **tight cluster of top scores -> low confidence**

## Timing Observation

Average timings across the 831 successful rows:

| Step | Mean ms |
| --- | ---: |
| content embedding | 32.70 |
| FAISS search | 1.32 |
| rerank | 66.17 |
| total | 100.52 |

Reranking is about two thirds of the total average record cost. If we can safely skip reranking for even 20% to 30% of pages, the end-to-end runtime impact is noticeable.

## What Was Measured

For each successful row, I computed these FAISS confidence features from the top-10 candidate list:

- `top1`: FAISS score of rank 1
- `margin12`: `faiss_score(rank1) - faiss_score(rank2)`
- `margin15`: `faiss_score(rank1) - faiss_score(rank5)`
- `std5`: population standard deviation of the FAISS scores for ranks 1 through 5

Interpretation:

- `top1` asks whether the best candidate looks strong in absolute terms
- `margin12` asks whether the winner clearly beats the runner-up
- `margin15` asks whether the winner clearly separates from the rest of the top-5
- `std5` asks whether the top-5 are spread out or tightly clustered

## Overall FAISS vs Reranker Behavior

### Top-1 agreement

FAISS top-1 and reranked top-1 are identical for:

- `269 / 831` rows
- `32.37%` of successful rows

So reranking is changing the winner on:

- `562 / 831` rows
- `67.63%` of successful rows

That is much too often to justify a naive FAISS-only strategy for the full dataset.

### Where the reranker pulls the final winner from

The final reranked top-1 came from these original FAISS ranks:

| Original FAISS rank | Count | Share |
| --- | ---: | ---: |
| 1 | 269 | 32.37% |
| 2 | 87 | 10.47% |
| 3 | 108 | 13.00% |
| 4 | 53 | 6.38% |
| 5 | 78 | 9.39% |
| 6 | 55 | 6.62% |
| 7 | 43 | 5.17% |
| 8 | 47 | 5.66% |
| 9 | 50 | 6.02% |
| 10 | 41 | 4.93% |

Notably:

- reranker picks a candidate from FAISS ranks `2-5` in `39.24%` of rows
- reranker picks a candidate from FAISS ranks `6-10` in `28.40%` of rows

That means the reranker is not just doing light cleanup. It is often selecting a candidate well below FAISS rank 1.

## Score Distribution Observations

### Absolute top-1 score is not a strong confidence signal

FAISS top-1 score quantiles:

| Quantile | Score |
| --- | ---: |
| min | 0.470406 |
| p10 | 0.560291 |
| p25 | 0.593339 |
| p50 | 0.620941 |
| p75 | 0.650106 |
| p90 | 0.676367 |
| p95 | 0.692166 |
| p99 | 0.716276 |
| max | 0.770615 |

Observation:

- high `top1` score alone does **not** guarantee reranker agreement
- the top-1 agreement rate is weakest in the highest `top1` quintile as well as the lowest one

Likely reason:

- some pages strongly match a **broad parent category**, while the reranker prefers a **more specific child category**
- this makes absolute similarity less reliable than relative separation

### Rank-1 vs rank-2 gap is strongly informative

`margin12` quantiles:

| Quantile | Score |
| --- | ---: |
| min | 0.000050 |
| p10 | 0.001802 |
| p25 | 0.004275 |
| p50 | 0.011700 |
| p75 | 0.031630 |
| p90 | 0.085117 |
| p95 | 0.094408 |
| p99 | 0.110421 |
| max | 0.122742 |

Agreement with reranked top-1 rises quickly as `margin12` grows:

| Rule for skipping rerank | Rows skipped | Skip share | In skipped rows, FAISS top-1 matches reranked top-1 |
| --- | ---: | ---: | ---: |
| `margin12 >= 0.02` | 291 | 35.02% | 59.79% |
| `margin12 >= 0.03` | 221 | 26.59% | 69.68% |
| `margin12 >= 0.05` | 152 | 18.29% | 80.92% |
| `margin12 >= 0.08` | 100 | 12.03% | 85.00% |

Interpretation:

- when rank 1 only barely beats rank 2, FAISS is often uncertain
- when rank 1 clearly separates from rank 2, FAISS is much more stable

### Top-5 spread is also useful

`std5` quantiles:

| Quantile | Score |
| --- | ---: |
| min | 0.000562 |
| p10 | 0.004703 |
| p25 | 0.007262 |
| p50 | 0.011852 |
| p75 | 0.020996 |
| p90 | 0.038322 |
| p95 | 0.041776 |
| p99 | 0.047583 |
| max | 0.053272 |

Agreement with reranked top-1 improves when the top-5 are more spread out:

| Rule for skipping rerank | Rows skipped | Skip share | In skipped rows, FAISS top-1 matches reranked top-1 |
| --- | ---: | ---: | ---: |
| `std5 >= 0.015` | 319 | 38.39% | 57.37% |
| `std5 >= 0.02` | 225 | 27.08% | 68.00% |
| `std5 >= 0.03` | 147 | 17.69% | 82.31% |

This tells the same story as `margin12`:

- tight clusters are ambiguous
- wide spread is a better sign of confidence

## Concrete Examples

### High-confidence FAISS case

Examples such as weather.com forecast pages were stable:

- FAISS top-1 path: `Science > Weather`
- `margin12` around `0.095` to `0.102`
- reranker kept the same top-1

These are pages where the top category is very distinct and beats the runner-up decisively.

### Low-confidence FAISS case

Examples with tiny `margin12` often changed after reranking:

- lyrics page: FAISS top-1 `Entertainment > Music > Songwriters/Folk`, reranker top-1 `Entertainment > Music > Hip Hop Music`, `margin12 = 0.002494`
- CarGurus sedan listing: FAISS top-1 `Automotive > Auto Type > Budget Cars`, reranker top-1 `Automotive > Auto Body Styles > Sedan`, `margin12 = 0.002836`
- restaurant/travel page: FAISS top-1 `Travel`, reranker top-1 `Attractions > Bars & Restaurants`, `margin12 = 0.001715`

These are exactly the cases where the embedding stage retrieves a small cluster of semantically related categories, and the reranker resolves the more precise one.

### Important caveat: even moderate margins can still be wrong

There are also rows with moderate `margin12` where reranker still overturned FAISS:

- Yahoo Finance pages
- real-estate listing pages
- gaming subgenre pages

This happens when:

- FAISS prefers a broad or neighboring category
- the reranker prefers a more specific subtype after reading the text pair directly

So `margin12` is useful, but not sufficient by itself.

## Candidate Strategies

Below, "hybrid matches current" means:

- rerank only the pages flagged as uncertain
- keep FAISS top-1 for the rest
- compare that hybrid result against the current always-rerank output

Again, this is **agreement with the current reranker**, not true accuracy.

### Strategy A: Conservative

Use reranker unless FAISS is very clearly separated.

Rule:

- skip rerank only when `margin12 >= 0.08`

Observed tradeoff:

- rerank `87.97%` of successful rows
- skip `12.03%`
- hybrid matches current reranked top-1 on `98.19%` of rows
- estimated average runtime saved: about `7.96 ms` per successful row

This is the safest threshold in this analysis.

### Strategy B: Balanced

Use reranker for ambiguous cases, skip for clearly separated winners.

Rule:

- skip rerank when `margin12 >= 0.05`

Observed tradeoff:

- rerank `81.71%`
- skip `18.29%`
- hybrid matches current reranked top-1 on `96.51%`
- estimated average runtime saved: about `12.10 ms` per successful row

This looks like the most reasonable first experiment if preserving current behavior matters.

### Strategy C: Aggressive

Push more traffic through FAISS only.

Rule:

- skip rerank when `margin12 >= 0.03`

Observed tradeoff:

- rerank `73.41%`
- skip `26.59%`
- hybrid matches current reranked top-1 on `91.94%`
- estimated average runtime saved: about `17.59 ms` per successful row

This saves more time, but now roughly 1 in 12 pages diverges from the current reranked behavior.

### Strategy D: Two-signal rule

Use either the rank-1/rank-2 gap or the top-5 spread.

Rule:

- rerank when `margin12 < 0.03` **or** `std5 < 0.02`
- skip rerank otherwise

Observed tradeoff:

- rerank `77.26%`
- skip `22.74%`
- hybrid matches current reranked top-1 on `94.22%`
- estimated average runtime saved: about `15.05 ms` per successful row

This is a reasonable compromise if you want a slightly richer confidence rule than a single threshold.

## Recommended Decision Framework

### If the goal is minimum behavior change

Use a conservative skip rule:

- skip rerank only when `margin12 >= 0.08`

Reason:

- this keeps almost all current reranker behavior intact
- it only bypasses reranking on strongly separated FAISS winners

### If the goal is meaningful runtime reduction with moderate risk

Start with:

- skip rerank when `margin12 >= 0.05`

Reason:

- this is still selective
- it captures the clearest confident subset
- it saves around 18% of reranker calls with limited divergence from current outputs

### If the goal is maximum speed

Use a more aggressive rule such as:

- skip rerank when `margin12 >= 0.03`

Reason:

- this cuts reranker usage more substantially
- but it also changes output more often relative to the current pipeline

I would not recommend going more aggressive than this without evaluating against labeled data.

## Practical Guidance for "FAISS Looks Uncertain"

Based on this dataset, FAISS looks uncertain when one or more of these are true:

- `margin12` is very small, especially below about `0.01`
- `margin15` is small, meaning rank 1 does not separate from the rest of the top-5
- `std5` is small, meaning the top-5 are tightly clustered
- the page type is one where broad parent categories and specific child categories are both plausible

Examples of likely uncertain content:

- finance quote/statistics pages
- real-estate listing pages
- entertainment/music pages with many adjacent subgenres
- marketplace or directory pages where several sibling taxonomy nodes are semantically close

Examples of likely confident content:

- strongly single-topic pages such as weather forecasts
- highly distinctive sports or fantasy sports pages
- pages whose wording directly matches a narrow taxonomy description

## Limitations

- There is no labeled truth set here, so agreement with the reranker is only a proxy.
- The reranker is not guaranteed to be correct either.
- These thresholds are specific to:
  - this taxonomy
  - this embedding model
  - this reranker
  - this page-content mix
- If the model or taxonomy changes, the thresholds should be recalibrated.

## Recommendation

For this exact dataset and pipeline, the best first operational strategy is:

1. Treat `margin12` as the primary FAISS confidence signal.
2. Start with a conservative threshold such as `margin12 >= 0.05` for skipping rerank.
3. If you want slightly more coverage, add a second spread feature like `std5`.
4. Validate the resulting hybrid policy against a labeled sample before adopting a more aggressive threshold.

If you want a single sentence version:

Use the reranker when the top FAISS candidates are clustered together; consider skipping it only when FAISS rank 1 clearly separates from rank 2 by a large margin.
