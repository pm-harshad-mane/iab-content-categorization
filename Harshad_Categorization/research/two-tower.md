# Two-Tower Architecture In Ad / Recommendation Stacks

## Short Answer

In large-scale ad-tech and recommendation systems, the two-tower model is usually used at the **retrieval / candidate generation / matching** stage, not at the final ranking stage.

Typical stage split:

1. **Retrieval / matching**
   - very large candidate pool
   - low-latency requirement
   - factorized models such as **two-tower / dual-encoder**
   - output is usually a few thousand candidates
2. **Ranking**
   - much smaller candidate set
   - heavier cross-feature / sequence / foundation-style models
   - output is the final ordering or ad selection

## What "Two-Tower" Usually Means

The standard industrial meaning is:

- one tower builds a representation for the **query / user / request context**
- one tower builds a representation for the **item / ad / document**
- retrieval is done with a cheap similarity function such as dot product

This makes two-tower attractive for:

- ANN retrieval
- precomputed item / ad embeddings
- large corpus search over millions to tens of millions of candidates

## Meta

### Where Two-Tower Fits

From Meta's public ads-retrieval writing, retrieval is the **first stage** of a multi-stage ads recommendation system. That stage reduces **tens of millions of ads** to **a few thousand relevant candidates**, and later stages use **larger and more sophisticated ranking models** to decide the final ads shown. Public Meta writing also says this retrieval stage historically used more conventional approaches, while newer systems are moving beyond common two-tower baselines. Source: Meta Andromeda engineering post, 2024.

### Current Public Meta Story

The clearest public Meta picture is:

- **retrieval stage**
  - historically comparable to common two-tower / ANN retrieval approaches
  - now being replaced or exceeded by newer retrieval systems
- **ranking stage**
  - larger recommendation / ranking models
  - historically DLRM-style ads recommendation models
  - more recently sequence-learning models and newer large foundation-style ranking models

### Publicly Named Meta Models / Systems By Stage

**Retrieval / matching**

- **Two-tower neural networks**
  - described by Meta as a common retrieval baseline in recommendation systems such as Instagram Explore
  - useful because caching and precomputation allow scalable retrieval
- **Andromeda**
  - Meta's newer ads retrieval engine
  - explicitly presented as better than "commonly used two-tower neural networks or approximate nearest neighbor search"
  - uses a jointly trained hierarchical retrieval/indexing design instead of a simple classic two-tower setup

**Ranking**

- **DLRM-based ads recommendation models**
  - public Meta post says ads recommendation was powered by DLRMs with many engineered features
- **Sequence-learning ads recommendation models**
  - Meta says it is replacing older DLRM limitations with event-based and sequence-learning architectures
- **GEM**
  - foundation model used to improve downstream ads recommendation models across the stack
- **Adaptive Ranking Model**
  - LLM-scale ads ranking runtime model

### Meta Takeaway

If you ask "what does two-tower do in Meta-style ads systems?", the public answer is:

- **two-tower belongs to retrieval / candidate generation**
- **heavier DLRM / sequence / foundation / ranking models belong after retrieval**
- Meta's newest public ads retrieval work suggests it is moving **beyond** standard two-tower retrieval in production ads systems

## Google

### Where Two-Tower Fits

Google's public production papers are consistent with the same split:

- **candidate generation / retrieval** first
- **ranking** second

The best public Google examples are recommendation systems rather than Google Ads-specific writeups, but the architecture pattern is the same.

### Publicly Named Google Models / Systems By Stage

**Retrieval / matching**

- **Two-tower neural network retrieval**
  - Google Research's Mixed Negative Sampling paper explicitly says the two-tower framework was used to improve a large-scale production app recommendation retrieval system
  - this is the clearest public Google statement of two-tower in production retrieval
- **Candidate generation model in YouTube recommendations**
  - Google public YouTube paper describes a deep candidate-generation stage followed by a separate ranking stage
  - even though the paper is not framed as "Google Ads", it shows the same large-scale industrial stage split

**Ranking**

- **Deep ranking model**
  - YouTube recommendations paper describes a separate ranking network after candidate generation
- **Two-tower for ULTR**
  - Google also has work using two-tower ideas in unbiased learning-to-rank
  - this is a different use from retrieval; here two-tower is used as a debiasing factorization inside ranking research, not the classic user/item retrieval tower split

### Google Takeaway

For Google-style industrial systems, the public pattern is:

- **two-tower is mainly a retrieval / matching model**
- **ranking is done by a separate, heavier model**
- Google also has a second meaning of "two-tower" in ranking research for debiasing, but that is not the usual ad-retrieval meaning

## Explicit Model / System Names Mentioned In The Sources

This section is narrower than the architecture summary above. It lists the actual model or system names that the cited public sources mention directly, plus a few nearby public names that help map the stage split more concretely.

### Meta

**Retrieval / matching**

- **Two Towers neural network**
  - explicitly named in Meta's Instagram Explore engineering writeup as a retrieval model and also as a cache-friendly first-stage ranker
- **Word2Vec**
  - Meta says its earlier ML retrieval approach used Word2Vec-style user and media/author embeddings before extending that setup with Two Towers
- **Andromeda**
  - explicitly named as Meta's newer personalized ads retrieval engine

**Ranking / post-retrieval modeling**

- **MTML**
  - the Instagram Explore writeup explicitly names the heavy second-stage ranker as a **multi-task multi label (MTML) neural network**
- **Value Model (VM)**
  - the same Explore article explicitly names the final weighted scoring formula used to combine predicted engagement probabilities
- **DLRM**
  - Meta's ads recommendation stack is explicitly described as being powered by deep learning recommendation models (DLRMs) before the newer sequence-learning shift
- **Custom transformer architecture**
  - the sequence-learning article does not give a separate productized brand name, but it does explicitly say Meta built a custom transformer architecture for the new ads recommendation system
- **GEM**
  - explicitly named as the **Generative Ads Model**
- **Adaptive Ranking Model**
  - explicitly named as a later large-scale ranking model in Meta's ads stack
- **Wukong**
  - the GEM article explicitly says GEM enhances the **Wukong** architecture for non-sequence feature interaction modeling

### Google

**Retrieval / matching**

- **Deep candidate generation model**
  - explicitly named in the YouTube recommendations paper
- **Two-tower neural network**
  - explicitly named in the Mixed Negative Sampling paper as the production retrieval framework
- **Dual encoder**
  - the same Google paper says two-tower is also known as a dual encoder in the NLP community
- **Mixed Negative Sampling (MNS)**
  - explicitly named as the negative-sampling method used to improve the production two-tower retrieval model

**Ranking**

- **Deep ranking model**
  - explicitly named in the YouTube recommendations paper
- **Additive two-tower models**
  - explicitly named in the Google ULTR paper as the common debiasing form of two-tower ranking models

### Practical Reading Of The Names

- When public sources say **Two Towers**, **two-tower neural network**, or **dual encoder**, they are usually talking about the fast retrieval stage.
- When public sources mention **MTML**, **DLRM**, **deep ranking model**, **custom transformer architecture**, **GEM**, or **Adaptive Ranking Model**, they are talking about later ranking stages.
- Some public sources do not expose a polished internal product name. In those cases, the best available public identifier is the model family name used in the article itself, such as **deep candidate generation model** or **custom transformer architecture**.

## Open-Sourced Models / Frameworks Relevant To Two-Tower

There are public open-source releases from Meta and Google that are clearly relevant to two-tower or dual-encoder architectures. The important caveat is that these are usually **frameworks, reference implementations, or research retrieval models**, not the exact internal ads-retrieval models described in the engineering articles above.

### Meta

- **[TorchRec](https://github.com/meta-pytorch/torchrec)**
  - Meta open-sourced TorchRec as its PyTorch recommendation stack, and the launch post explicitly says it initially targets **two-tower** architectures.
  - This is best thought of as the open-source building block for training and serving large-scale two-tower recommendation models, rather than a released pretrained ads retrieval checkpoint.
- **[DPR (Dense Passage Retriever)](https://github.com/facebookresearch/DPR)**
  - Meta / Facebook Research open-sourced DPR, a **bi-encoder** dense retriever for question-answer retrieval.
  - DPR is not an ads model, but architecturally it is a textbook two-tower / dual-encoder retrieval system.
- **[FAISS](https://github.com/facebookresearch/faiss)**
  - FAISS is not the tower model itself, but it is a major Meta open-source building block commonly paired with two-tower retrieval systems for ANN lookup over precomputed item embeddings.

### Google

- **[TensorFlow Recommenders (TFRS) retrieval models](https://www.tensorflow.org/recommenders)**
  - Google open-sourced TensorFlow Recommenders, and its retrieval task documentation explicitly describes a **two-tower, factorized structure** with separate query and candidate towers.
  - The public examples include a basic user-item retrieval model and a sequential retrieval variant where the query tower uses a **GRU**.
- **[T5X Retrieval](https://github.com/google-research/t5x_retrieval)**
  - Google Research open-sourced T5X Retrieval as a framework for neural retrieval and ranking.
  - The repo explicitly includes **DualEncoderModel** / **DualEncoderDecoderModel** style retrieval code.
- **[GTR](https://github.com/google-research/t5x_retrieval)**
  - In T5X Retrieval, Google released **Generalizable T5 Retrieval (GTR)** checkpoints.
  - These are explicitly described as **dual encoders** that embed two pieces of text into dense vectors for retrieval.
- **[SentenceT5](https://github.com/google-research/t5x_retrieval)**
  - Also released through T5X Retrieval.
  - SentenceT5 is primarily a sentence encoder family, not a classic ad-tech two-tower recommender, but it sits in the same dual-encoder retrieval ecosystem and can be used in tower-style retrieval setups.
- **[ScaNN](https://github.com/google-research/google-research/tree/master/scann)**
  - Like FAISS on the Meta side, ScaNN is not the tower model itself, but it is an open-source Google ANN retrieval component designed to serve dense retrieval systems efficiently.

### Practical Takeaway On Open Source Availability

- **Yes**, both Meta and Google have open-source assets relevant to two-tower architectures.
- **No**, the exact production ads retrieval models from the public Meta ads articles are generally **not** released as downloadable pretrained tower checkpoints.
- What is available publicly is mostly:
  - large-scale recommender **frameworks**
  - retrieval **reference implementations**
  - dense retrieval / dual-encoder **research models**
  - ANN serving libraries used alongside tower models

### Open-Source Assets At A Glance

| Company | Open-source asset | Type | Real pretrained model? | Closer to ads retrieval or generic retrieval? |
|---|---|---|---|---|
| Meta | [TorchRec](https://github.com/meta-pytorch/torchrec) | Recommender framework / training-serving stack | No, mainly framework | Closer to ads / recommender infrastructure |
| Meta | [DPR](https://github.com/facebookresearch/DPR) | Bi-encoder retrieval model + code | Yes | Closer to generic text retrieval |
| Meta | [FAISS](https://github.com/facebookresearch/faiss) | ANN indexing / vector search library | No, infrastructure only | Shared infra component used by retrieval systems |
| Google | [TensorFlow Recommenders (TFRS)](https://www.tensorflow.org/recommenders) | Recommender framework with two-tower retrieval examples | Mostly no, framework/examples | Closer to recommender / ads-style retrieval patterns |
| Google | [T5X Retrieval](https://github.com/google-research/t5x_retrieval) | Retrieval framework | Framework plus released checkpoints | Closer to generic text retrieval |
| Google | [GTR](https://github.com/google-research/t5x_retrieval) | Dual-encoder pretrained retrieval models | Yes | Closer to generic text retrieval |
| Google | [SentenceT5](https://github.com/google-research/t5x_retrieval) | Sentence encoder checkpoints | Yes | Generic retrieval / representation learning |
| Google | [ScaNN](https://github.com/google-research/google-research/tree/master/scann) | ANN indexing / vector search library | No, infrastructure only | Shared infra component used by retrieval systems |

## Practical Mapping For Ad Tech

If you are designing an ad-tech system and want to map public Meta/Google patterns onto implementation stages:

### Stage 1: Retrieval / matching

Typical model family:

- two-tower / dual-encoder
- ANN retrieval
- precomputed ad embeddings

Goal:

- reduce millions of ads/items to thousands

### Stage 2: First-stage ranking

Typical model family:

- lightweight neural ranker
- richer feature crosses than retrieval

Goal:

- reduce thousands to hundreds

### Stage 3: Heavy ranking / final ranking

Typical model family:

- DLRM-like models
- sequence models / transformer-like ranking models
- foundation-style ranking models

Goal:

- final ad selection and ordering

## Production Rerankers Used After Two-Tower Retrieval

In large production systems, the model used after two-tower retrieval is usually **not** a generic off-the-shelf reranker checkpoint. Public writing from Meta, Google, and Pinterest points much more strongly to **custom ranking models** trained on proprietary engagement, click, conversion, and business-value labels.

### What Companies Publicly Say They Use

### Meta

- **[MTML](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)**
  - Meta's Instagram Explore system explicitly names the heavy second-stage ranker as a **multi-task multi-label (MTML) neural network**
  - this is the clearest public example of a production reranker after two-tower retrieval
- **[Value Model (VM)](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)**
  - Explore then combines the predicted event probabilities into a final scoring formula called **VM**
- **[DLRM](https://engineering.fb.com/2024/11/19/data-infrastructure/sequence-learning-personalized-ads-recommendations/)**
  - Meta's older ads recommendation stack is publicly described as being powered by **DLRM**-based models
- **[Custom transformer architecture](https://engineering.fb.com/2024/11/19/data-infrastructure/sequence-learning-personalized-ads-recommendations/)**
  - Meta's newer ads recommendation stack is publicly described as using a **custom transformer architecture** for sequence learning
- **[GEM](https://engineering.fb.com/2025/05/29/ml-applications/meta-generative-ads-model-gem/)**
  - publicly named as the **Generative Ads Model**, used to improve downstream ads recommendation models
- **[Adaptive Ranking Model](https://engineering.fb.com/2026/03/12/ml-applications/meta-adaptive-ranking-model-llm-scale-ads/)**
  - publicly named as a large-scale later-stage ranking model in Meta's ads stack

### Google

- **[Deep ranking model](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)**
  - the YouTube recommendations paper explicitly names a separate **deep ranking model** after candidate generation
  - Google does not expose a branded internal product name there, but the ranker is clearly a heavier post-retrieval model
- **[Additive two-tower models for ULTR](https://research.google/pubs/revisiting-two-tower-models-for-unbiased-learning-to-rank/)**
  - in a different ranking context, Google also discusses **additive two-tower models** for unbiased learning to rank
  - this is ranking research rather than the common retrieval-plus-rerank production pattern

### Pinterest

- **[XGBoost GBDT lightweight ranker](https://medium.com/pinterest-engineering/improving-the-quality-of-recommended-pins-with-lightweight-ranking-8ff5477b20e3)**
  - Pinterest publicly says an earlier production lightweight ranking stage used an **XGBoost GBDT** model
- **[MMoE](https://medium.com/pinterest-engineering/multi-gate-mixture-of-experts-mmoe-model-architecture-and-knowledge-distillation-in-ads-08ec7f4aa857)**
  - Pinterest publicly describes **Multi-gate Mixture-of-Experts (MMoE)** as a production ads engagement modeling architecture
- **[DCNv2](https://medium.com/pinterest-engineering/multi-gate-mixture-of-experts-mmoe-model-architecture-and-knowledge-distillation-in-ads-08ec7f4aa857)**
  - Pinterest publicly describes **DCNv2** as the production baseline architecture in ads engagement modeling
- **[Transformer-based sequence models](https://medium.com/pinterest-engineering/user-action-sequence-modeling-for-pinterest-ads-engagement-modeling-21139cab8f4e)**
  - Pinterest publicly describes long user-sequence transformer components in its ads engagement stack
- **[MTMD](https://www.adkdd.org/papers/mtmd%3A-a-multi-task-multi-domain-framework-for-unified-ad-lightweight-ranking-at-pinterest/2025)**
  - Pinterest publicly says **MTMD (Multi-Task Multi-Domain)** was deployed in production for ad recommendation and replaced 9 production models

### Practical Conclusion

- In ad-tech and large recommender systems, the reranker is usually a **custom multi-task ranking network**
- Common publicly disclosed model families are:
  - **MTML / multi-task neural rankers**
  - **DLRM / DCNv2-style web-scale rankers**
  - **MMoE / MTMD mixture-of-experts rankers**
  - **Transformer / sequence-learning rankers**
- So the usual pattern is:
  - **two-tower / dual-encoder for retrieval**
  - **custom neural ranker for reranking**

## Bottom Line

For companies like Meta and Google, the public evidence points to this rule:

- **Two-tower = retrieval / candidate generation / matching**
- **Ranking uses different, heavier models**

And for Meta specifically, the latest public ads-retrieval material suggests:

- classic two-tower retrieval is now more of a **common baseline / older pattern**
- newer production retrieval is shifting to **more expressive retrieval architectures** such as Andromeda

## Public Sources Referenced

- [Meta Engineering, **Meta Andromeda: Supercharging Advantage+ automation with the next-gen personalized ads retrieval engine** (2024)](https://engineering.fb.com/2024/12/12/ml-applications/meta-andromeda-supercharging-advantage-automation-with-the-next-gen-personalized-ads-retrieval-engine/)
- [Meta Engineering, **Sequence learning: A paradigm shift for personalized ads recommendations** (2024)](https://engineering.fb.com/2024/11/19/data-infrastructure/sequence-learning-personalized-ads-recommendations/)
- [Meta Engineering, **Meta's Generative Ads Model (GEM): The Central Brain Accelerating Ads Recommendation AI Innovation** (2025)](https://engineering.fb.com/2025/05/29/ml-applications/meta-generative-ads-model-gem/)
- [Meta Engineering, **Meta Adaptive Ranking Model: Bending the Inference Scaling Curve to Serve LLM-Scale Models for Ads** (2026)](https://engineering.fb.com/2026/03/12/ml-applications/meta-adaptive-ranking-model-llm-scale-ads/)
- [Meta Engineering, **Scaling the Instagram Explore recommendations system** (2023)](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)
- [Google Research, **Deep Neural Networks for YouTube Recommendations** (2016)](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)
- [Google Research, **Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations** (2020)](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)
- [Google Research, **Revisiting two tower models for unbiased learning to rank** (2022)](https://research.google/pubs/revisiting-two-tower-models-for-unbiased-learning-to-rank/)
- [Pinterest Engineering, **Improving the Quality of Recommended Pins with Lightweight Ranking** (2020)](https://medium.com/pinterest-engineering/improving-the-quality-of-recommended-pins-with-lightweight-ranking-8ff5477b20e3)
- [Pinterest Engineering, **Multi-gate-Mixture-of-Experts (MMoE) model architecture and knowledge distillation in Ads Engagement modeling development** (2025)](https://medium.com/pinterest-engineering/multi-gate-mixture-of-experts-mmoe-model-architecture-and-knowledge-distillation-in-ads-08ec7f4aa857)
- [Pinterest Engineering, **User Action Sequence Modeling for Pinterest Ads Engagement Modeling** (2024)](https://medium.com/pinterest-engineering/user-action-sequence-modeling-for-pinterest-ads-engagement-modeling-21139cab8f4e)
- [Pinterest Engineering, **Unifying Ads Engagement Modeling Across Pinterest Surfaces** (2026)](https://medium.com/pinterest-engineering/unifying-ads-engagement-modeling-across-pinterest-surfaces-4b5cd3d99e67)
- [AdKDD 2025, **MTMD: A Multi-Task Multi-Domain Framework for Unified Ad Lightweight Ranking at Pinterest** (2025)](https://www.adkdd.org/papers/mtmd%3A-a-multi-task-multi-domain-framework-for-unified-ad-lightweight-ranking-at-pinterest/2025)
