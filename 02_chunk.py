import json
from pathlib import Path

DATA = Path(__file__).parent / "data"

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict]:
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append({
            "id": chunk_id,
            "text": text[start:end],
            "start_char": start,
            "end_char": min(end, len(text)),
        })
        chunk_id += 1
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    with open(DATA / "extracted.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Run all three and compare
    for size, overlap in [(256, 25), (512, 50), (1024, 100)]:
        chunks = chunk_text(text, size, overlap)
        print(f"\nSize={size}, Overlap={overlap} → {len(chunks)} chunks")
        print(f"Sample chunk 10:\n'{chunks[10]['text'][:300]}'")
    
    chosen_chunks = chunk_text(text, chunk_size=512, overlap=50)
    with open(DATA / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chosen_chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(chosen_chunks)} chunks")