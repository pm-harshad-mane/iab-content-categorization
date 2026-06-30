# Fine-Tuned embeddinggemma-300m Evaluation

- Input file: `02_fetched_url_content_files/new_urls.jsonl`
- Trained model: `08_training_dataset_using_Gemma4/runs/embeddinggemma300m_high_medium_triplet_v2_bs8`
- Output file: `10_finetuned_embeddinggemma300m_evaluation/new_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__faiss.jsonl`
- Taxonomy source: `taxonomy/Content_Taxonomy_3.1_6.tsv`
- Taxonomy rows: `704`
- Taxonomy setup ms: `13175.806`

## Trained vs GPT-5.4

- Common rows: `63`
- Top-1 match rate: `0.3492`
- GPT top-1 in trained top-5: `0.4762`
- Avg overlap count: `1.2381`
- Avg F1: `0.2845`
- NDCG@5: `0.4136`
- Heuristic score / 100: `35.62`

## Existing embeddinggemma-300m vs GPT-5.4

- Common rows: `63`
- Top-1 match rate: `0.3651`
- GPT top-1 in existing top-5: `0.5556`
- Avg overlap count: `1.4286`
- Avg F1: `0.3192`
- NDCG@5: `0.4322`
- Heuristic score / 100: `38.13`
