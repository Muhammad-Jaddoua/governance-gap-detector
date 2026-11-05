from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import COLLECTION, SOURCE_DOC

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "tinyllama"   # swap to "mistral" once you have it pulled
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333

# ── Lifespan: load heavy objects once at startup ──────────────────────────────
# Avoids reloading the embedding model on every request.
resources: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    resources["model"]  = SentenceTransformer(EMBEDDING_MODEL)
    print("Connecting to Qdrant...")
    resources["qdrant"] = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    print("Ready.")
    yield
    resources.clear()

app = FastAPI(title="Governance Gap Detector", lifespan=lifespan)

# ── Schemas ───────────────────────────────────────────────────────────────────
class AssessRequest(BaseModel):
    query: str
    top_k: int = 5
    source_filter: str = SOURCE_DOC  # filter by document — critical for compliance traceability

class RetrievedClause(BaseModel):
    chunk_id: int
    score: float
    text: str
    source: str

class AssessResponse(BaseModel):
    query: str
    answer: str
    gap_identified: bool          # did the LLM find a gap or not
    retrieved_clauses: list[RetrievedClause]
    model_used: str
    warning: str | None = None    # populated if top scores are low (weak retrieval)

# ── Core logic ────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int, source: str) -> list[RetrievedClause]:
    model  = resources["model"]
    client = resources["qdrant"]

    query_vector = model.encode([query])[0].tolist()

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
        limit=top_k,
        with_payload=True,
    ).points

    return [
        RetrievedClause(
            chunk_id=r.payload["chunk_id"],
            score=round(r.score, 4),
            text=r.payload["text"],
            source=r.payload["source"],
        )
        for r in results
    ]


def generate(query: str, clauses: list[RetrievedClause]) -> tuple[str, bool]:
    """Returns (answer_text, gap_identified)."""
    context = "\n\n".join(
        [f"[Chunk {c.chunk_id} | score {c.score}]\n{c.text}" for c in clauses]
    )

    prompt = f"""You are a data governance compliance assistant.
Your job is to assess whether the provided context from a governance framework
addresses the user's query, and to identify any compliance gaps.

Rules:
- Answer ONLY using the provided context. Do not use outside knowledge.
- If the context fully addresses the query, explain how.
- If the context partially addresses or misses the query, explicitly state: "GAP IDENTIFIED:" followed by what is missing.
- If the answer is entirely absent, respond with exactly: "Not found in retrieved sections."

Context:
{context}

Query: {query}

Assessment:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]
    gap_identified = "GAP IDENTIFIED:" in answer or "Not found in retrieved sections" in answer
    return answer, gap_identified


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/assess", response_model=AssessResponse)
def assess(request: AssessRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if request.top_k < 1 or request.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20.")

    # Retrieve
    clauses = retrieve(request.query, request.top_k, request.source_filter)

    if not clauses:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found in collection '{COLLECTION}' "
                   f"for source '{request.source_filter}'. "
                   "Check that ingestion ran successfully."
        )

    # Warn if retrieval confidence is low — all top scores below 0.3
    # means the query is likely outside the document's coverage
    warning = None
    if all(c.score < 0.3 for c in clauses):
        warning = (
            "All retrieved chunks have low similarity scores (< 0.3). "
            "The query may be outside the document's scope, "
            "or the embedding model is not suited to this vocabulary."
        )

    # Generate
    answer, gap_identified = generate(request.query, clauses)

    return AssessResponse(
        query=request.query,
        answer=answer,
        gap_identified=gap_identified,
        retrieved_clauses=clauses,
        model_used=LLM_MODEL,
        warning=warning,
    )


@app.get("/health")
def health():
    client = resources.get("qdrant")
    if not client:
        raise HTTPException(status_code=503, detail="Qdrant client not initialised.")

    collections = [c.name for c in client.get_collections().collections]
    collection_ready = COLLECTION in collections

    return {
        "status": "ok" if collection_ready else "degraded",
        "collection": COLLECTION,
        "collection_ready": collection_ready,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
    }


@app.get("/collection/info")
def collection_info():
    """How many chunks are stored. Useful sanity check after ingestion."""
    client = resources.get("qdrant")
    if not client:
        raise HTTPException(status_code=503, detail="Qdrant client not initialised.")
    info = client.get_collection(COLLECTION)
    return {
        "collection": COLLECTION,
        "total_points": info.points_count,
        "vector_size": info.config.params.vectors.size,
        "distance": str(info.config.params.vectors.distance),
    }