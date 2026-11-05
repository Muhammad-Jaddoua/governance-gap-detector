# Governance Gap Detector

A local-first **RAG (Retrieval-Augmented Generation)** pipeline that answers compliance
questions against a data-governance framework — and, crucially, tells you when the
framework *doesn't* answer them.

Built end to end from raw PDF to HTTP API: extraction → chunking → embedding → vector
search → grounded generation. No hosted APIs, no API keys. Qdrant runs in Docker and the
LLM runs through Ollama, so the whole thing works offline.

The reference corpus is the Saudi **NDMO Data Management and Personal Data Protection
Standards** (v1.5, 173 pages) — but the pipeline is document-agnostic; point it at any PDF.

---

## Why "gap detector" and not "chatbot"

In a compliance setting, a confident wrong answer is worse than no answer. A regulator
doesn't want prose — they want to know *which clause* covers a requirement, and whether
one exists at all.

So the system is built to fail loudly in three independent ways:

| Guard | Mechanism | Why it matters |
|---|---|---|
| **Grounding** | Prompt forbids outside knowledge; must reply `Not found in retrieved sections.` when the context is silent | Stops the LLM from filling gaps with plausible-sounding invention |
| **Explicit gap flag** | Answers must prefix `GAP IDENTIFIED:` when coverage is partial; parsed into a boolean `gap_identified` | Makes "the standard is incomplete here" a machine-readable result, not a nuance buried in text |
| **Retrieval confidence** | If *every* top-k chunk scores below `0.3` cosine, the response carries a `warning` | Separates "the model was wrong" from "you asked something this document never covers" |

Every answer ships with the `retrieved_clauses` that produced it — chunk IDs, similarity
scores, and source text — so a human can audit the reasoning instead of trusting it.

---

## Architecture

```
          ndmo_framework.pdf  (173 pages)
                   │
   01_extract.py   │  pdfplumber, page-boundary markers preserved
                   ▼
            extracted.txt  (~186k chars)
                   │
   02_chunk.py     │  sliding window, 512 chars / 50 overlap
                   ▼
             chunks.json  (402 chunks)
                   │
 03_embed_store.py │  all-MiniLM-L6-v2 → 384-dim vectors
                   ▼
       ┌───────────────────────┐
       │  Qdrant (Docker)      │   collection: ndmo_chunks
       │  cosine distance      │   payload: text, chunk_id, source, start_char
       └───────────────────────┘
                   │
                   │  top-k search + source filter
                   ▼
   04_query.py  /  server.py   →  Ollama (tinyllama / mistral)
                                   grounded prompt → assessed answer
```

### Design decisions worth defending

**Chunk size: 512 characters, 50 overlap.** `02_chunk.py` runs 256/512/1024 side by side
and prints samples from each. 256 severed individual requirements mid-sentence; 1024
pulled two unrelated clauses into one vector and blurred its meaning. 512 kept one
requirement per chunk, and the 50-char overlap stops a clause that straddles a boundary
from vanishing from both neighbours.

**Cosine over Euclidean.** Sentence-transformer embeddings carry meaning in direction, not
magnitude. Two passages on the same obligation should match even when one is three times
longer.

**Metadata payload, not just vectors.** Each point stores `source`, `chunk_id`, and
`start_char`. `source` powers a hard pre-filter at query time — in a multi-document
deployment, an answer about one regulation must never quote another. `start_char` lets any
retrieved claim be traced back to its exact offset in the original text.

**Model loaded once, in the FastAPI lifespan.** `all-MiniLM-L6-v2` takes seconds to
initialise. Loading it per request would dominate latency; the lifespan hook pays that cost
once at startup and shares the handles.

**`top_k` bounded to 1–20.** Tuning showed `top_k=1` missed requirements split across
chunks, while `top_k=10`+ buried the relevant clause in noise a small LLM couldn't filter.
5 is the default.

---

## Running it

**Prerequisites:** Python 3.13, Docker, [Ollama](https://ollama.com).

```bash
# 1. Dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Vector store
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant

# 3. Local LLM
ollama pull tinyllama      # or: ollama pull mistral (better answers, slower)

# 4. Corpus — drop any PDF into data/ as ndmo_framework.pdf
#    The NDMO standards are published at https://sdaia.gov.sa

# 5. Ingest
python 01_extract.py       # PDF  → data/extracted.txt
python 02_chunk.py         # text → data/chunks.json
python 03_embed_store.py   # embed + upsert into Qdrant

# 6. Query — CLI
python 04_query.py

# 7. Query — API
uvicorn server:app --reload
```

### API

`POST /assess`

```bash
curl -X POST localhost:8000/assess -H 'Content-Type: application/json' -d '{
  "query": "What are the responsibilities of a data owner?",
  "top_k": 5
}'
```

```json
{
  "query": "What are the responsibilities of a data owner?",
  "answer": "The context assigns the Data Owner accountability for ...",
  "gap_identified": false,
  "retrieved_clauses": [
    { "chunk_id": 147, "score": 0.6213, "text": "...", "source": "ndmo_framework.pdf" }
  ],
  "model_used": "tinyllama",
  "warning": null
}
```

Also exposed: `GET /health` (Qdrant reachable, collection present, models in use) and
`GET /collection/info` (point count, vector size, distance metric) — both intended as
post-ingestion sanity checks and container readiness probes.

---

## Repository layout

| File | Role |
|---|---|
| `01_extract.py` | PDF → text via pdfplumber, page markers retained |
| `02_chunk.py` | Sliding-window chunker + the size/overlap comparison used to pick 512/50 |
| `03_embed_store.py` | Embeds chunks, (re)creates the Qdrant collection, upserts with payload |
| `04_query.py` | CLI retrieve-and-generate loop; prints scores alongside answers |
| `server.py` | FastAPI service — `/assess`, `/health`, `/collection/info` |

Generated artifacts (`data/extracted.txt`, `data/chunks.json`), the source PDF, and
`qdrant_storage/` are gitignored; steps 4–5 recreate them.

---

## Known limitations

Stated plainly, because they're the next things worth building:

- **Character-based chunking is structure-blind.** It splits on offsets, not on headings or
  clause numbers. A section-aware splitter would produce cleaner retrieval units.
- **`tinyllama` is the weak link.** Retrieval is solid; the small model is what strains on
  multi-clause synthesis. `mistral` is a drop-in swap via `LLM_MODEL` in `server.py`.
- **Dense retrieval only.** Regulatory text is full of exact identifiers ("Article 4.2")
  that lexical search handles better — hybrid BM25 + dense would likely beat either alone.
- **The 0.3 warning threshold is empirical**, tuned by hand on this corpus rather than
  calibrated against a labelled set.
- **No evaluation harness.** Retrieval quality was judged by reading scores and answers.
  A fixed question set with graded relevance is the honest next step.
