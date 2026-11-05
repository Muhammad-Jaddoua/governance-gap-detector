"""Corpus configuration — the only file you edit to swap documents.

The pipeline is not tied to the NDMO standards. Drop any PDF into `data/`,
point SOURCE_DOC at its filename, give it its own COLLECTION, and re-run
stages 01 → 03. Nothing else needs to change.
"""

# Filename of the PDF inside data/ — also stored on every vector as the
# `source` payload field, which query-time filtering uses to keep answers
# from one document out of answers about another.
SOURCE_DOC = "ndmo_framework.pdf"

# Qdrant collection holding this corpus. Use one collection per document
# so corpora can be rebuilt independently.
COLLECTION = "ndmo_chunks"
