from pathlib import Path

import pdfplumber

from config import SOURCE_DOC

DATA = Path(__file__).parent / "data"

def extract_text(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n--- Page {i+1} ---\n{text}"
    return full_text

if __name__ == "__main__":
    text = extract_text(str(DATA / SOURCE_DOC))
    print(f"Total characters extracted: {len(text)}")
    print("\nFirst 800 characters:")
    print(text[:800])
    
    with open(DATA / "extracted.txt", "w", encoding="utf-8") as f:
        f.write(text)