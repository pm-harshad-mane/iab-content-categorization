# Step 10: Fine-Tuned embeddinggemma-300m Evaluation

This folder contains the first retrieval-path evaluation of the fine-tuned `embeddinggemma-300m` model trained in [`08_training_dataset_using_Gemma4`](/home/harshad.mane/Harshad_Categorization/08_training_dataset_using_Gemma4).

## Files

- [`evaluate_finetuned_embeddinggemma300m.py`](/home/harshad.mane/Harshad_Categorization/10_finetuned_embeddinggemma300m_evaluation/evaluate_finetuned_embeddinggemma300m.py)
  - Loads the fine-tuned local SentenceTransformer model.
  - Rebuilds page `query_text` from stage-02 content using the same format as stage 03.
  - Embeds taxonomy rows from [`Content_Taxonomy_3.1_6.tsv`](/home/harshad.mane/Harshad_Categorization/taxonomy/Content_Taxonomy_3.1_6.tsv).
  - Runs local FAISS retrieval for the target input file.
  - Compares the fresh FAISS results against:
    - existing stage-04 `google/embeddinggemma-300m` output
    - GPT baseline output

- [`new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__faiss.jsonl`](/home/harshad.mane/Harshad_Categorization/10_finetuned_embeddinggemma300m_evaluation/new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__faiss.jsonl)
  - Fresh FAISS retrieval output from the fine-tuned model on `new_urls`.

- [`new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.json`](/home/harshad.mane/Harshad_Categorization/10_finetuned_embeddinggemma300m_evaluation/new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.json)
  - Machine-readable comparison summary vs GPT and vs the existing stage-04 `embeddinggemma-300m` output.

- [`new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md`](/home/harshad.mane/Harshad_Categorization/10_finetuned_embeddinggemma300m_evaluation/new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__comparison.md)
  - Human-readable summary of the same comparison.

## Command

```bash
python3 10_finetuned_embeddinggemma300m_evaluation/evaluate_finetuned_embeddinggemma300m.py
```

## Current Result on `new_urls`

- Fine-tuned model vs GPT:
  - top-1 match rate: `0.3492`
  - avg overlap count: `1.2381`
  - NDCG@5: `0.4136`
  - heuristic score / 100: `35.62`

- Existing stage-04 `embeddinggemma-300m` vs GPT:
  - top-1 match rate: `0.3651`
  - avg overlap count: `1.4286`
  - NDCG@5: `0.4322`
  - heuristic score / 100: `38.13`

So on `new_urls`, this first fine-tuned checkpoint did **not** beat the current stage-04 `embeddinggemma-300m` baseline yet.

## Next Step

The next useful step is to evaluate the model deeper in the full pipeline:

1. run stage 04 on a larger evaluation slice
2. optionally run stage 05 reranking on top of it
3. compare again against GPT and the existing embedding baselines
