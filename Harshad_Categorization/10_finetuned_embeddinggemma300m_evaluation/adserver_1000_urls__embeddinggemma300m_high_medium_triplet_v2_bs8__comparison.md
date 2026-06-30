# Fine-Tuned embeddinggemma-300m Evaluation

- Input file: `02_fetched_url_content_files/adserver_1000_urls.jsonl`
- Trained model: `08_training_dataset_using_Gemma4/runs/embeddinggemma300m_high_medium_triplet_v2_bs8`
- Output file: `10_finetuned_embeddinggemma300m_evaluation/adserver_1000_urls__embeddinggemma300m_high_medium_triplet_v2_bs8__faiss.jsonl`
- Taxonomy source: `taxonomy/Content_Taxonomy_3.1_6.tsv`
- Taxonomy rows: `704`
- Taxonomy setup ms: `7745.538`
- Device: `cuda`
- Existing stage-04 baseline: `04_fetched_url_content_embedding_categories_files/adserver_1000_urls__google_embeddinggemma-300m__faiss.jsonl`
- Comparison baseline: `07_fetched_url_content_categories_by_Gemma4/adserver_1000_urls__google_gemma-4-E4B-it.jsonl` (`gemma4`, field `llm_top_categories`)

## Trained vs gemma4

- Common rows: `831`
- Top-1 match rate: `0.3995`
- gemma4 top-1 in trained top-5: `0.5872`
- Avg overlap count: `0.8496`
- Avg F1: `0.1925`
- NDCG@5: `0.3677`
- Heuristic score / 100: `30.17`

## Existing embeddinggemma-300m vs gemma4

- Common rows: `825`
- Top-1 match rate: `0.2327`
- gemma4 top-1 in existing top-5: `0.4109`
- Avg overlap count: `1.1273`
- Avg F1: `0.2480`
- NDCG@5: `0.3624`
- Heuristic score / 100: `31.59`
