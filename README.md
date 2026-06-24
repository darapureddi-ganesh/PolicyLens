# PolicyLens

**Rule-based NLP knowledge graph for insurance policy querying — symbolic AI, zero LLM dependency.**

PolicyLens parses an Indian motor insurance policy PDF into a structured knowledge graph and
answers natural-language queries through deterministic graph traversal. No neural network, no
embeddings, no vector database, no fine-tuning — every answer traces back to an exact policy
clause, making it fully auditable.

This is the symbolic-AI counterpart to a RAG + LLM chatbot (Insurebot-AI), built to demonstrate
the **flexibility vs. explainability** tradeoff in human-centric AI design.

## Why

LLM-based RAG chatbots can answer insurance queries conversationally but cannot explain *why*
they gave an answer. Regulators, auditors, and researchers need traceability — every answer must
point to a specific clause. PolicyLens solves this with rule-based entity extraction and graph
traversal: every inference is fully inspectable.

## Architecture

Five sequential modules:

| # | Module | File | Responsibility |
|---|--------|------|-----------------|
| 1 | PDF Parser | `src/parser.py` | PyMuPDF reads the policy PDF and splits it into sections and clauses by heading/numbering pattern |
| 2 | Entity Extractor | `src/entity_extractor.py` | spaCy + regex extract coverage types, exclusions, monetary limits, deductibles, add-ons against a fixed domain ontology |
| 3 | Graph Builder | `src/graph_builder.py` | NetworkX `DiGraph`: clause nodes + entity nodes, edges = `HAS_COVERAGE`, `EXCLUDES`, `HAS_DEDUCTIBLE`, `PROVIDES_ADDON`, `HAS_CONDITION`, `WAIVED_BY` |
| 4 | Query Engine | `src/query_engine.py` | Query → spaCy lemmatisation + keyword/phrase match → graph traversal → ranked clause + path |
| 5 | Streamlit UI | `app.py` | Query box, answer panel, source clause highlight, optional PyVis graph render |

## Sample Queries

| Query | Response |
|---|---|
| Is theft covered? | YES — Section 3, Clause 2: Theft of vehicle is covered up to IDV. |
| What is excluded? | Lists every exclusion node with its source clause (drunk driving, no valid licence, consequential loss, war, racing, engine flood damage). |
| What is the deductible? | Compulsory deductible: Rs. 1,000 for private cars (Section 4, Clause 1). Voluntary deductible (Section 4, Clause 2). |
| Is engine flood damage covered? | PARTIAL — excluded under Section 5, Clause 6, unless the Engine Protect add-on is purchased. |

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| PDF Parsing | PyMuPDF (`fitz`) | Text extraction with layout awareness |
| NLP | spaCy (`en_core_web_sm`) | Tokenisation, lemmatisation to assist phrase matching |
| Knowledge Graph | NetworkX | DiGraph construction, traversal |
| Graph Visualisation | PyVis | Interactive HTML graph render inside Streamlit |
| UI | Streamlit | Query interface, deploy target |
| Regex | `re` (stdlib) | Monetary values, percentages |

## Project Structure

```
policylens/
├── data/
│   └── sample_motor_policy.pdf   # bundled synthetic policy
├── src/
│   ├── parser.py                 # Module 1
│   ├── entity_extractor.py       # Module 2
│   ├── graph_builder.py          # Module 3
│   └── query_engine.py           # Module 4
├── app.py                        # Streamlit UI (Module 5)
├── requirements.txt
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads a bundled synthetic sample policy by default; you can also upload your own
text-layer motor policy PDF from the sidebar.

## Non-Goals (v1)

- No LLM, no embeddings, no vector database
- Domain-locked to motor insurance, not a general-purpose chatbot
- No model training or fine-tuning of any kind
- Text-layer PDFs only (no OCR / scanned documents)

## Research Relevance

Built as a practical instantiation of human-centric reasoning and decision support: a system
that answers complex policy questions while keeping its reasoning transparent and auditable,
in direct contrast to opaque RAG/LLM pipelines.

---
PolicyLens — Darapureddi Ganesh — June 2026
