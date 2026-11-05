import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DATA = Path(__file__).parent / "data"

def embed_and_store(chunks_path: str, collection_name: str = "ndmo_chunks"):
    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Embed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"Embedding shape: {embeddings.shape}")  # (n_chunks, 384)

    # Connect to Qdrant (must be running via Docker)
    client = QdrantClient("localhost", port=6333)

    # Create collection — delete first if re-running
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # Store points with metadata payload
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "text": chunks[i]["text"],
                "chunk_id": chunks[i]["id"],
                "source": "ndmo_framework.pdf",
                "start_char": chunks[i]["start_char"],
            }
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=collection_name, points=points)
    print(f"Stored {len(points)} points in Qdrant collection '{collection_name}'")

if __name__ == "__main__":
    # Start Qdrant first: docker run -p 6333:6333 qdrant/qdrant
    embed_and_store(str(DATA / "chunks.json"))