# 📊 RAG Evaluation Report  
**Chunking, Retrieval, and Stability Analysis**

---

## 1. Chunking Methodology (Assigned Method)

### Father–Son (Parent–Child) Chunking

The main chunking strategy used in this project is the **Father–Son (Parent–Child)** method, which was the method assigned to us.  
This hierarchical chunking approach is designed to balance **context preservation** and **retrieval precision** in long parliamentary documents.

Each document is first segmented into **parent chunks**, created by grouping consecutive sentences up to a relatively large word limit. These parent chunks preserve thematic continuity across extended parts of the text.  
Each parent chunk is then subdivided into **child chunks**, which are smaller, sentence-based segments with minimal overlap.

This design has two main advantages:

- **Improved recall**: parent chunks ensure that relevant regions of the document are not missed.
- **Improved precision**: child chunks reduce noise by focusing retrieval on tightly scoped semantic units.

This approach is particularly well suited for long and structured texts such as parliamentary debates.

---

## 2. Large Language Model (LLM)

We used **Qwen 2.5 (3B)** via **Ollama** as the LLM in our RAG pipeline.

The reasons for this choice are:

- Local and offline execution, ensuring reproducibility  
- Low computational cost and fast inference  
- Adequate generation quality when combined with high-quality retrieved context  

Since this is a Retrieval-Augmented Generation system, answer quality depends primarily on retrieval quality rather than model size.

---

## 3. Vector Representation Methods

Two retrieval representations were evaluated:

### BM25 (Lexical Retrieval)

BM25 relies on exact word matching and is especially effective for:

- Factual queries  
- Named entities  
- Dates, numbers, and organizations  

### Dense Embeddings (Semantic Retrieval)

We used **MPNet (`all-mpnet-base-v2`)** to generate dense embeddings.  
This approach captures semantic similarity and enables retrieval even when the query is a paraphrase of the source text.

---

## 4. Choice of K (Number of Retrieved Chunks)

We evaluated **K = 3, 5, and 8**, representing different trade-offs:

- **K = 3**: high precision, lower recall  
- **K = 5**: balanced precision and recall  
- **K = 8**: higher recall but increased noise  

To analyze sensitivity to K, we measured **answer stability across K values** using cosine similarity between generated answers.

---

## 5. Stability by Configuration (Plot A)

![Plot A – Stability by configuration](evaluation/plots/plotA_stability_by_config.png)

**Figure 1 – Stability by configuration**

This plot shows the mean cosine similarity of generated answers across different K values for each configuration.

**Observations:**

- Configurations using **child chunking** show significantly higher stability.
- Fixed-size chunking, especially with BM25, is substantially less stable.

**Conclusion:**  
The Father–Son chunking method clearly outperforms fixed-size chunking in terms of robustness and consistency.

---

## 6. Stability by Query Type (Plot C)

![Plot C – Stability by query type](evaluation/plots/plotC_stability_by_query_type.png)

**Figure 2 – Stability by query type**

Factual queries exhibit slightly higher stability than conceptual queries.  
This is expected, as conceptual questions require higher-level reasoning and are more sensitive to retrieval variation.

---

## 7. BM25 vs Embeddings Answer Similarity (Plot B)

![Plot B – BM25 vs Embeddings similarity](evaluation/plots/plotB_hist_bm25_vs_emb.png)

**Figure 3 – Histogram of cosine similarity between BM25-based and embedding-based answers**

This histogram compares answers produced using BM25 retrieval versus embedding-based retrieval.

**Key insight:**

- The distribution is bimodal.  
- Some answers are nearly identical (cosine ≈ 1.0).  
- Others differ significantly (cosine ≈ 0.1–0.3).  

This shows that BM25 and embeddings often retrieve fundamentally different contexts.

---

## 8. Retrieval Quality Analysis

- **Relevant chunks were retrieved**, especially with child chunking.  
- **Recall improved with larger K**, but at the cost of increased noise.  
- **Child chunks reduced irrelevant retrieval**, even for larger K.  
- Generated answers in stable configurations closely matched the retrieved chunks.

---

## 9. Key Insights and Conclusions

1. Chunking strategy has the strongest impact on RAG stability.  
2. Father–Son chunking produces more consistent and reliable answers.  
3. BM25 performs well on factual queries, while embeddings are better suited for conceptual queries.  
4. **K = 5** provides the best trade-off between precision and recall.  
5. Answer stability is an effective metric for evaluating RAG pipelines.

---

## 10. Example Queries with Answer Excerpts

This section presents **real user questions** together with **selected excerpts from generated answers**, in order to qualitatively demonstrate the effects of chunking strategy, retrieval method, and **K** on answer quality, focus, and stability.

---

### Example 1 – Factual / Argument-Centered Question

**User question**

> What was the main argument regarding the immigration bill that was presented?

---

### Fixed Chunking + BM25

#### K = 3

**Answer excerpt**
> “The main arguments regarding the immigration bill presented mainly revolved around concerns about devolving immigration powers to the Scottish Government. Critics argued that this would not solve existing problems but create new ones, such as difficulties in border security and consistency with UK immigration policy.”

**Observation**  
The answer is relevant but relatively narrow and misses additional arguments raised later in the debate.

---

#### K = 5

**Answer excerpt**
> “The main arguments presented regarding the immigration bill included concerns about inefficiency, border security between England and Scotland, and whether the Scottish Government could manage immigration powers effectively.”

**Observation**  
Recall improves, but the answer becomes more verbose and slightly repetitive.

---

#### K = 8

**Answer excerpt**
> “The main arguments regarding the immigration bill seemed to focus on strengthening the process and making getting a British passport more difficult... There was also discussion about fraud within the government and criticism of the Prime Minister’s handling of the Budget.”

**Observation**  
The answer suffers from **topic drift**, introducing unrelated political issues due to noisy retrieval.

---

### Father–Son (Child) Chunking + BM25

#### K = 5

**Answer excerpt**
> “The main argument regarding the immigration bill was that it fails to offer the security, clarity or accountability needed to deliver outcomes. The Bill proposes devolving immigration powers to the Scottish Government, which raises concerns about border management and policy consistency.”

**Observation**  
The answer is **focused, concise, and directly grounded in the relevant debate**.

---

#### K = 8

**Answer excerpt**
> “The Bill fails to offer the security, clarity, or accountability needed. There are unanswered questions regarding immigration management between England and Scotland, including border security and legal safeguards.”

**Observation**  
Despite a larger K, the answer remains stable and on-topic, demonstrating the robustness of hierarchical chunking.

---

### Father–Son (Child) Chunking + Embeddings

#### K = 8

**Answer excerpt**
> “The main arguments against the immigration bill included concerns about border security, vulnerable populations being put at risk, and disagreements over the effectiveness of current immigration policies.”

**Observation**  
Although semantically relevant, the answer partially shifts to **US immigration debates**, highlighting a limitation of dense retrieval without lexical grounding.

---

## Example 2 – Conceptual / Analytical Question

**User question**

> What is the central tension that emerges from the speeches between the need for national security and the protection of citizens’ privacy in the digital age?

---

### Fixed Chunking + BM25 (K = 3)

**Answer excerpt**
> “The central tension revolves around encryption and law enforcement access to data. Speakers warned that any ‘way in’ for security services could be exploited by criminals or hostile states, undermining citizens’ privacy.”

**Observation**  
Accurate and grounded, but highly dependent on keyword overlap.

---

### Father–Son (Child) Chunking + Embeddings (K = 3)

**Answer excerpt**
> “The tension lies in balancing national security with privacy rights, particularly regarding end-to-end encryption, digital ID systems, and lawful access. Speakers emphasized the risk of weakening encryption while acknowledging security threats.”

**Observation**  
The answer provides **higher-level synthesis** and abstracts across multiple debates, making it well suited for conceptual questions.

---

## 11. Final Takeaway

This evaluation demonstrates that **hierarchical chunking has a stronger impact on answer quality and stability than retrieval method or K alone**.  
Combined with semantic retrieval, it enables a **robust and stable RAG system**, even when using a lightweight local LLM.
