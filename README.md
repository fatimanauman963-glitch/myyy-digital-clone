---
title: Fatima's Digital Clone
emoji: 🎨
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# Fatima's Digital Clone (Hybrid RAG)

> **This is an AI clone, not the real person.**

A Streamlit chatbot that answers questions as me, using only documents I wrote about myself.
Pipeline: chunk -> BM25 index + dense embeddings -> retrieve (BM25 / Dense / Hybrid) -> RRF fusion -> Groq LLM with a first-person persona prompt.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (question box, mode selector, answer, retrieved chunks panel, AI-clone banner) |
| `rag.py` | Chunking, BM25, dense search, Reciprocal Rank Fusion |
| `llm.py` | Persona prompt + Groq API call (key read from secrets) |
| `build_embeddings.py` | Precomputes `embeddings.npy` and `chunks.json` |
| `report_table.py` | Prints the 5-question comparison table for this report |
| `data/` | My knowledge base (7 markdown files) |

## Run locally
```bash
pip install -r requirements.txt
python build_embeddings.py
# put GROQ_API_KEY in .streamlit/secrets.toml (see secrets.toml.example)
streamlit run app.py
```

## Report

### 1. Knowledge base and why
I used 7 documents: a bio and a CV (facts), a hobbies and personality note (opinions), a how-I-work note, a long project write-up, my RAG study notes (long-form) and a 25-question first-person FAQ. I chose them because I can judge immediately whether the answers are right. Rare keywords like "QR Attendance System" and "Sialkot" give BM25 something to win on, and descriptive sentences give dense search something to match paraphrases.

### 2. Test questions (BM25 vs Dense vs Hybrid)
Run `python report_table.py`, paste the table here and fill the last column.

| Question | BM25 result | Dense result | Hybrid result | Best mode and why |
|---|---|---|---|---|
| Q1 (exact rare keyword) | | | | |
| Q2 (paraphrase, no shared words) | | | | |
| Q3 | | | | |
| Q4 | | | | |
| Q5 | | | | |

Fusion method: Reciprocal Rank Fusion, score = sum of 1 / (60 + rank). The other common method is a weighted score `alpha * dense + (1 - alpha) * bm25` after min-max normalising each.

### 3. Failure case
TODO: write one question where the clone gave a wrong or weak answer and explain why (for example: the answer was split across two chunks, or the question used a word that appears in many chunks).

### 4. One improvement
TODO: for example a cross-encoder reranker over the hybrid results, or a small hand-labelled evaluation set with hit@3 per mode.
