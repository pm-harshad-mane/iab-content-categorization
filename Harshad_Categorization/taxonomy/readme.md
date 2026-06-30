# Taxonomy Notes

## Purpose

This folder contains multiple taxonomy files because we started from a raw content taxonomy, then created progressively cleaner and more retrieval-friendly variants for the categorization pipeline.

The main categorization project uses dense embedding models such as:

- `BAAI/bge-m3`
- `google/embeddinggemma-300m`

Those models work best when the taxonomy text is:

- specific
- semantically positive
- consistent across rows
- free from noisy generic filler

They do **not** work best when positive and negative concepts are mixed into the same embedding text.

## File Lineage

### `Content_Taxonomy_3.1.tsv`

- Raw source taxonomy.
- `707` rows, `8` columns.
- Header is not immediately model-friendly:
  - `Relational ID System`, blank columns, `Content Taxonomy v3.1 Tiered Categories`, `Extension`
- This looks like the original source export rather than a file ready for programmatic loading.

### `Content_Taxonomy_3.1_2.tsv`

- Cleaned working source for the categorization project.
- `706` rows, `7` columns.
- Header:
  - `Unique ID`
  - `Parent`
  - `Tier 1`
  - `Tier 2`
  - `Tier 3`
  - `Tier 4`
  - `Description`
- This is the most reliable structured source version of the content taxonomy.
- Current categorization loaders were built around this shape.

### `Content_Taxonomy_3.1_enriched.tsv`

- Flattened enrichment experiment.
- `705` rows, `5` columns.
- Header:
  - `unique_id`
  - `parent_id`
  - `path`
  - `description`
  - `keywords`
- Useful because it already has:
  - flattened `path`
  - explicit `keywords`
- Not ideal as the main taxonomy source because:
  - descriptions are mostly template-generated
  - many rows are generic
  - some keyword artifacts exist, such as malformed stems or overly broad parent terms

### `Content_Taxonomy_3.1_3.tsv`

- First retrieval-oriented rewrite derived from `Content_Taxonomy_3.1_2.tsv`.
- `705` rows, `10` columns.
- Added:
  - `Path`
  - richer `Description`
  - `Keywords`
  - `Negative Keywords`
- This version was useful as a first pass but still had major issues:
  - too many broad parent keywords inside leaf rows
  - too many generic negative terms
  - some noisy tokenization artifacts

### `Content_Taxonomy_3.1_4.tsv`

- Cleaner refinement of `3.1_3`.
- Same `10` columns as `3.1_3`.
- Improvements:
  - reduced generic parent terms inside leaf keywords
  - removed many broad negative artifacts
  - removed noisy possessive artifacts such as `women s` / `men s`
  - made negatives more sibling/confuser oriented
- This is the better version if we want a retrieval-friendly taxonomy with explicit positive and negative metadata.

### `Content_Taxonomy_3.1_5.tsv`

- Targeted follow-up to `3.1_4`.
- Same `10` columns.
- Purpose:
  - enrich the sparse rows that still had weak keyword coverage
- Improvements:
  - raised minimum keyword coverage across rows
  - added stronger domain-specific keywords to weak categories such as:
    - religion subcategories
    - sparse sports nodes
    - technology leaves
    - travel-type leaves
    - video game genre leaves

### `Content_Taxonomy_3.1_6.tsv`

- ANN-friendly refinement of `3.1_5`.
- `705` rows including header, `11` columns.
- Same positive retrieval structure as `3.1_5`, plus:
  - `Common Confusers`
- Main improvement:
  - removed `Common confusers: ...` text from the embedded `Description`
  - moved those terms into a separate metadata column
- This is the strongest current variant for dense retrieval because the main embedded text is more positive-only and less likely to mix target concepts with sibling concepts.

### `Ad_Product_Taxonomy_2.0.tsv`

- Separate taxonomy.
- Not the content taxonomy.
- `584` rows, `6` columns.
- Header:
  - `Unique ID`
  - `Parent ID`
  - `Name`
  - `Tier 1`
  - `Tier 2`
  - `Tier 3`
- This should be treated as a different taxonomy family, likely for ad/product classification rather than general page-content categorization.

## Recommended Use

### Best source of truth for hierarchy

Use `Content_Taxonomy_3.1_2.tsv` as the canonical hierarchical source because it is the cleanest structured version of the original content taxonomy.

### Best retrieval-oriented derived file

Use `Content_Taxonomy_3.1_6.tsv` if the categorization pipeline is updated to read:

- `Path`
- `Description`
- `Keywords`
- optionally:
  - `Common Confusers` as metadata only
  - `Negative Keywords` as metadata only

### Which derived file is stronger

- `3.1_4` is cleaner and safer than `3.1_3`
- `3.1_5` is the same general structure as `3.1_4`, but with better coverage on previously sparse rows
- `3.1_6` keeps the stronger keyword coverage from `3.1_5` while removing `Common confusers` from the embedded description text

So if we want the strongest enriched taxonomy variant right now for ANN / FAISS retrieval, `Content_Taxonomy_3.1_6.tsv` is the best candidate.

## Important Note About Negative Keywords And Common Confusers

### Short answer

Do **not** concatenate either of these into the same text that is sent to dense embedding models like:

- `BAAI/bge-m3`
- `google/embeddinggemma-300m`

The risky fields are:

- `Negative Keywords`
- `Common Confusers`

### Why

Dense embedding models produce **one vector for the whole input text**.

If taxonomy text contains both:

- positive terms for the target category
- negative terms for sibling or confuser categories

the resulting vector can move toward **all** of those concepts, not just the intended category.

Example:

- Category: `Religion & Spirituality > Islam`
- Positive keywords:
  - `islam, muslim, quran, ramadan`
- Negative keywords:
  - `christianity, buddhism, hinduism`

If all of that is embedded together, the vector may become a “comparative religion” vector rather than a specifically “Islam” vector.

The same problem happens when `Description` includes phrases like:

- `Common confusers: augmented reality, robotics, virtual reality`

Those terms are often treated by the embedding model as semantically related concepts, not as “avoid these” instructions.

### Recommended practice

For dense retrieval / FAISS:

- embed:
  - `Path`
  - `Description`
  - `Keywords`
- do **not** embed:
  - `Common Confusers`
  - `Negative Keywords`

Keep `Common Confusers` and `Negative Keywords` separate for:

- diagnostics
- rule-based penalties
- reranker-side features
- post-retrieval filtering

## Practical Guidance For The Categorization Project

### If the existing code is unchanged

The current loaders were built around `Content_Taxonomy_3.1_2.tsv`.

That means `3.1_4`, `3.1_5`, and `3.1_6` are **not** drop-in replacements unless the pipeline is updated to read the new columns.

### If the pipeline is updated

Preferred taxonomy text for embedding:

```text
path: <Path>
description: <Description>
keywords: <Keywords>
```

Do not append `Common Confusers` or `Negative Keywords` to that same embedding text.

### Suggested default

If the pipeline is upgraded to support enriched taxonomy rows, use:

- source file:
  - `Content_Taxonomy_3.1_6.tsv`
- embedded text:
  - `Path + Description + Keywords`
- non-embedded metadata:
  - `Common Confusers`
  - `Negative Keywords`

## Examples

### Example category present in the taxonomy

`Religion & Spirituality > Islam`

This exists as:

- `Unique ID = 461`

In the enriched variants, this row has:

- specific positive keywords such as:
  - `islam`
  - `muslim`
  - `quran`
  - `ramadan`
  - `eid`
  - `hadith`
- sibling-style negative keywords such as:
  - `christianity`
  - `judaism`
  - `hinduism`
  - `buddhism`

In `3.1_6`, sibling confuser terms can also live in `Common Confusers` instead of being mixed into the main description.

That structure is useful, but only if confusers and negatives are kept separate from the dense embedding input.

## Current Recommendation

If we need one file to continue improving the categorization project:

- keep `Content_Taxonomy_3.1_2.tsv` as the canonical raw hierarchy
- use `Content_Taxonomy_3.1_6.tsv` as the best current enriched retrieval variant for ANN / FAISS
- treat `Common Confusers` as metadata, not embedding text
- treat `Negative Keywords` as metadata, not embedding text
