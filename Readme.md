# PHASE 1 — Core Product (MVP, 2 weeks max)

*Goal: Professional, fast, clean site.*

1. Define positioning

   * Clear 2-line identity statement
   * Short technical bio
   * Links (GitHub, LinkedIn)

2. Frontend foundation

   * stack SvelteKit
   * Minimal design system
   * Responsive layout
   * Lighthouse performance > 90

3. Projects page

   * 3–5 solid projects
   * Architecture diagrams (simple, clean)
   * What you learned (critical reflection)

---

# PHASE 2 — Knowledge Representation Layer

Goal: Demonstrate structured thinking.

4. Create structured knowledge dataset

   * Skills
   * Projects
   * Concepts
   * Tools
   * Relationships

Represent it as:

* JSON graph schema

5. Build interactive graph visualization

   * D3.js (or bevy ?)
   * Clickable nodes
   * Clean physics, no chaos
   * Tooltip with explanation

---

# PHASE 3 — RAG Chatbot (Engineering Signal)

Goal: Demonstrate ML pipeline competence.

6. Prepare content corpus

   * CV
   * Project READMEs
   * Blog-style explanations
   * Technical reflections

7. Embedding pipeline

   * Chunking strategy
   * Embeddings generation
   * Store in vector DB (Qdrant is perfect if you want Rust alignment)

8. Retrieval API

   * Query embedding
   * Top-k retrieval
   * Context assembly

9. LLM inference

   * Structured prompt
   * Grounded answers only

10. UI integration

* Minimal chat interface
* Interactively display steps of the retrieval

Transparency = credibility.

---

Possible services:

1. Frontend (static site)
2. Graph API (if dynamic)
3. RAG service
4. Embedding ingestion service

If you split:

* rag-service
* embedding-worker
* graph-api
* frontend

---

# PHASE 4 — Engineering Polish

11. Add observability

* Logging
* Structured logs
* Basic metrics

12. Add caching to retrieval layer

13. Write architecture documentation

14. Write a short technical blog post:
    “How I Built My Personal RAG System in Rust”

---
