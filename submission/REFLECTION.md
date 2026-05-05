# Reflection — Lab 19

**Name:** Nguyen Minh Hieu
**Student ID:** 2A202600180
**Cohort:** A20
**Execution Path:** lite

---

## 1. Evaluation (≤ 200 words)

> On the golden set of 50 queries, which mode won for each query type (`exact` / `paraphrase` / `mixed`), and why? When would you **not** use hybrid (i.e., when is pure BM25 or pure vector the right choice)?

**Answer:**
Based on the benchmark results using OpenAI `text-embedding-3-small` and the `pyvi` tokenizer:
- **`exact` & `mixed` queries**: Both Keyword (BM25) and Hybrid achieved 100.0% precision. Lexical search is naturally superior here as it matches specific terms directly present in the source text.
- **`paraphrase` queries**: Hybrid (56.0%) and Semantic (52.7%) outperformed pure Keyword (46.7%). Vector embeddings excel at capturing semantic intent even when the user uses synonyms or different phrasing.

**When NOT to use Hybrid:**
- **Pure BM25** is preferred for searching exact technical identifiers (SKUs, UUIDs, specific error codes) where semantic "near-matches" are unwanted noise. It is also significantly cheaper and faster for high-volume, low-resource environments.
- **Pure Vector** is the better choice for conceptual or cross-lingual tasks where users rarely share the same vocabulary as the author, and where maintaining an inverted index adds unnecessary infrastructure complexity.

---

## 2. Most Surprising Finding

The most surprising discovery was the "synergy effect" between NLP tokenization and Vector Search. Simply using OpenAI's powerful SOTA embeddings wasn't enough to guarantee a Hybrid win because a naive whitespace-based Keyword search became the bottleneck. By integrating the **`pyvi` Vietnamese tokenizer**, I boosted the Keyword component from 77.8% to 84.0%. This uplift allowed the Hybrid RRF model to effectively combine the best of both worlds, reaching a final precision of **86.8%**, surpassing both individual components.

---

## 3. Bonus Challenge

- [x] I have completed the bonus challenge: **"AI Assistant with Hybrid Memory"**.
- [x] Implementation: Built a `HybridMemoryAgent` combining **Episodic Memory** (Qdrant Vector Store) and **Stable User Profiles** (Feast Feature Store).
- [x] Optimization: Integrated `pyvi` tokenizer for robust Vietnamese retrieval and implemented LRU caching for OpenAI latency optimization.

