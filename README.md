# 📊 RAG Evaluation Report  
**Chunking, Retrieval, and Stability Analysis**

---

## 1. Project Overview

This project evaluates a **Retrieval-Augmented Generation (RAG)** pipeline applied to **long parliamentary debates**.  
The main objective is to analyze how different design choices affect:

- Retrieval quality
- Answer stability
- Faithfulness of generated answers

The evaluation focuses on three core dimensions:

1. Chunking strategy  
2. Retrieval representation (BM25 vs embeddings)  
3. Sensitivity to the number of retrieved chunks (K)

---

## 2. Chunking Methodology (Assigned Method)

### Father–Son (Parent–Child) Chunking

The main chunking strategy used in this project is the **Father–Son (Parent–Child)** method, which was assigned for this evaluation.

Each document is first segmented into **parent chunks**, which group consecutive sentences to preserve broader thematic context.  
Each parent chunk is then subdivided into **child chunks**, which are smaller, sentence-based units with minimal overlap.

This design provides two complementary benefits:

- **Improved recall**: parent chunks ensure that relevant regions of the document are not missed.
- **Improved precision**: child chunks allow retrieval to focus on tightly scoped semantic content.

This hierarchical approach is particularly well suited for long and structured documents such as parliamentary debates.

---

## 3. Large Language Model (LLM)

We used **Qwen 2.5 (3B)** via **Ollama** as the language model in our RAG pipeline.

This choice was motivated by:

- Local and offline execution (full reproducibility)
- Low computational cost and fast inference
- Adequate generation quality when combined with high-quality retrieved context

Since this is a RAG system, overall answer quality depends more on **retrieval quality** than on model size.

---

## 4. Retrieval Representations

Two retrieval approaches were evaluated:

### 4.1 BM25 (Lexical Retrieval)

BM25 relies on exact term matching and is especially effective for:

- Factual questions
- Named entities
- Dates, numbers, and organizations

### 4.2 Dense Embeddings (Semantic Retrieval)

Dense embeddings were generated using **MPNet (`all-mpnet-base-v2`)**.  
This method captures semantic similarity and enables retrieval even when the query is a paraphrase of the source text.
### Practical Note on Embedding Computation

The answer embeddings used for stability and similarity analysis were computed using **Google Colab with GPU acceleration**.

This was necessary because generating dense embeddings with `all-mpnet-base-v2` over hundreds of generated answers is computationally expensive on CPU.  
Using a GPU significantly reduced embedding computation time while ensuring identical results.

Once computed, the embeddings were saved and reused locally for all subsequent analyses and plotting steps.


---

## 5. Choice of K (Number of Retrieved Chunks)

We evaluated **K = 3, 5, and 8**, representing different precision–recall regimes:

- **K = 3**: high precision, limited recall  
- **K = 5**: balanced precision and recall  
- **K = 8**: higher recall, increased risk of noise  

To measure sensitivity to K, we computed **answer stability across K values**, using cosine similarity between generated answers.

---

## 6. Plot A — Stability by Configuration

![Plot A – Stability by configuration](evaluation/plots/plotA_stability_by_config.png)

This plot shows the **mean cosine similarity of answers across different K values**, grouped by configuration.

### Observations

- **Child chunking (Father–Son)** yields the highest stability.
- Fixed-size chunking is significantly less stable, especially with BM25.
- Retrieval representation has less impact than chunking strategy.

### Conclusion

Chunking strategy is the dominant factor influencing answer stability.  
Hierarchical chunking clearly outperforms fixed-size chunking.

---

## 7. Plot C — Stability by Query Type

![Plot C – Stability by query type](evaluation/plots/plotC_stability_by_query_type.png)

This plot compares answer stability between **factual** and **conceptual** queries.

### Observations

- Factual queries exhibit slightly higher stability.
- Conceptual queries are more sensitive to retrieval variation.

### Interpretation

Conceptual questions require higher-level reasoning and synthesis, making them more vulnerable to changes in retrieved context.

---

## 8. Plot B — BM25 vs Embeddings Answer Similarity

![Plot B – BM25 vs Embeddings similarity](evaluation/plots/plotB_hist_bm25_vs_emb.png)

This histogram shows cosine similarity between answers generated using BM25 retrieval and embedding-based retrieval, for the **same query, chunking strategy, and K**.

### Key Insights

- The distribution is **bimodal**:
  - One peak near 1.0 → nearly identical answers
  - One peak between 0.1 and 0.3 → fundamentally different answers
- This indicates that BM25 and embeddings often retrieve **different contextual evidence**.

---

## 9. Chunk-Level Retrieval Analysis (Precision and Recall)

We manually analyzed retrieved chunks to evaluate:

- **Recall**: whether all relevant chunks were retrieved
- **Precision**: whether retrieved chunks were relevant

### Findings

- Fixed-size chunking often required large K to achieve acceptable recall, introducing noise.
- Father–Son chunking retrieved more relevant chunks with fewer irrelevant ones.
- Precision and recall were both higher with hierarchical chunking.

---

## 10. Answer Faithfulness and Grounding

We evaluated whether generated answers were faithful to the retrieved chunks.

### Observations

- Stable configurations produced answers closely grounded in retrieved text.
- Noisy retrieval led to topic drift and over-generalization.
- Answer quality correlated strongly with **context relevance**, not model size.

---

## 11. Tested Queries and Qualitative Answer Analysis

In order to complement the quantitative evaluation, we tested the RAG pipeline on **real user queries** covering both **factual** and **conceptual** information needs.

These queries were used to generate the answers analyzed in the stability and similarity plots.

---

### 11.1 Example Factual Query

**Tested question**

> What was the main argument regarding the immigration bill that was presented?

This question targets a **specific parliamentary argument** and requires retrieving precise political positions expressed during debates.

#### Observations across configurations

- With **fixed-size chunking** and high K, answers tended to:
  - Include unrelated political topics
  - Drift toward broader immigration discussions
- With **Father–Son (child) chunking**, answers:
  - Focused consistently on the devolution of immigration powers
  - Remained stable as K increased
  - Closely matched the retrieved chunks

This behavior directly explains the higher stability scores observed for hierarchical chunking in **Plot A**.

---

### 11.2 Example Conceptual Query

**Tested question**

> What is the central tension that emerges from the speeches between the need for national security and the protection of citizens’ privacy in the digital age?

This question is **conceptual** and requires synthesizing arguments across multiple speeches rather than extracting a single fact.

#### Observations across configurations

- **BM25-based retrieval** tended to emphasize:
  - Encryption and law-enforcement access
  - Keyword-matching discussions
- **Embedding-based retrieval** enabled:
  - Higher-level synthesis
  - Integration of multiple viewpoints on privacy and security

However, conceptual questions were more sensitive to retrieval variation, which explains their slightly lower stability scores in **Plot C**.

---

### 11.3 Relation to Quantitative Results

These tested queries confirm the quantitative findings:

- Stable configurations correspond to answers grounded in relevant chunks
- Unstable configurations often introduce noise or topic drift
- Hierarchical chunking improves both **answer stability** and **faithfulness**

The qualitative behavior observed for these queries directly supports the trends shown in **Plots A, B, and C**.
tive findings.

---

## 12. Final Takeaway

This evaluation demonstrates that **hierarchical chunking is the most critical factor** in building a stable and reliable RAG system.

When combined with appropriate retrieval strategies, it enables robust performance even with a lightweight local LLM.
