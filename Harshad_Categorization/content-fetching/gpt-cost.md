# GPT Batch Cost Notes

## Basis Run

Source run:

- input file:
  - [`02_fetched_url_content_files/new_urls.jsonl`](/home/harshad.mane/Harshad_Categorization/02_fetched_url_content_files/new_urls.jsonl)
- output file:
  - [`06_fetched_url_content_categories_by_GPT/new_urls__gpt-5.4.jsonl`](/home/harshad.mane/Harshad_Categorization/06_fetched_url_content_categories_by_GPT/new_urls__gpt-5.4.jsonl)
- model:
  - `gpt-5.4`

Batch size:

- total input records: `64`
- submitted to OpenAI Batch: `63`
- source-error rows skipped from submission: `1`

Observed usage across the `63` successful GPT requests:

- input tokens: `4,909,598`
- output tokens: `3,688`
- total tokens: `4,913,286`

Average per submitted URL:

- input tokens per URL: about `77,930`
- output tokens per URL: about `58.5`

## Pricing Basis

OpenAI pricing reference for `gpt-5.4`:

- standard input: `$2.50 / 1M`
- cached input: `$0.25 / 1M`
- standard output: `$15.00 / 1M`
- Batch API: `50%` off input and output rates

For a simple upper-bound batch estimate, use:

- batch input: `$1.25 / 1M`
- batch output: `$7.50 / 1M`

## Estimated Cost For The Basis Run

Estimated batch cost using the upper-bound batch rates:

- input cost:
  - `4.909598M × $1.25 ≈ $6.14`
- output cost:
  - `0.003688M × $7.50 ≈ $0.03`
- estimated total:
  - **about `$6.16`**

Important caveat:

- this is an estimate, not the exact OpenAI invoice amount
- the local output files do not currently include a cached-vs-non-cached input token breakdown
- if prompt caching reduced billed input tokens materially, the real cost could be lower than this estimate

## 50K URL Projection

Using the basis-run estimate:

- estimated cost per submitted URL:
  - `$6.16 / 63 ≈ $0.0978`

Projected cost for `50,000` URLs:

- `50,000 × $0.0978 ≈ $4,890`

Rounded practical estimate:

- **about `$4.9K` for 50K URLs**

## Interpretation

The cost is dominated by input tokens, not output tokens.

That is expected because:

- the taxonomy prompt is large
- we include:
  - `Unique ID`
  - `Path`
  - `Description`
  - `Keywords`
  for all taxonomy rows in each request
- model output is intentionally tiny:
  - top `5` categories
  - only `unique_id` and `score`

## Practical Takeaway

If we continue using the current full-taxonomy-per-request prompting strategy:

- GPT batch labeling is viable for smaller benchmark sets
- it becomes expensive for large-scale teacher-label generation

For large datasets like `50K` URLs, the current prompt design implies a cost in the **low thousands of dollars**, even with Batch pricing.
