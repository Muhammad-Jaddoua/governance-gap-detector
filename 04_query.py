import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, QueryRequest
import ollama

from config import COLLECTION, SOURCE_DOC

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient("localhost", port=6333)

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    query_embedding = model.encode([query])[0].tolist()
    
    results = client.query_points(  
        collection_name=COLLECTION,
        query=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="source", 
                                 match=MatchValue(value=SOURCE_DOC))]
        ),
        limit=top_k,
        with_payload=True
    ).points

    return [{"text": r.payload["text"], "score": r.score, 
             "chunk_id": r.payload["chunk_id"]} for r in results]

def generate(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join([c["text"] for c in chunks])
    prompt = f"""You are a data governance compliance assistant.
Answer based ONLY on the context below from the NDMO framework.
If the answer is not present, say exactly: "Not found in retrieved sections."

Context:
{context}

Question: {query}
Answer:"""
    
    response = ollama.chat(model="tinyllama",
                           messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def run(query: str, top_k: int = 5):
    print(f"\nQuery: {query}")
    print("-" * 60)
    chunks = retrieve(query, top_k)
    for c in chunks:
        print(f"  Score: {c['score']:.4f} | Chunk {c['chunk_id']}: "
              f"{c['text'][:100]}...")
    answer = generate(query, chunks)
    print(f"\nAnswer:\n{answer}\n")
    return answer

if __name__ == "__main__":
    run("What are the responsibilities of a data owner?")

    # Other probes used while tuning retrieval:
    # run("How should organizations handle data classification?")
    # run("What maturity level requires automated data lineage tracking?")
    # run("tell me about Compliance and Enforcement?")
    # run("What are data quality requirements?", top_k=1)    # recall too narrow
    # run("What are data quality requirements?", top_k=10)   # context gets noisy
    #
    # Deliberately out-of-corpus — checks that the system degrades honestly
    # instead of hallucinating. Scores collapse below 0.3 and the prompt
    # forces "Not found in retrieved sections."
    # run("What is the best Toyota mileage based in international?")