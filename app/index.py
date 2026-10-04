"""
Build and load the FAISS index.

IndexFlatIP + L2-normalised vectors == exact cosine similarity search.
Flat (brute force) is the right call here: a few hundred papers is tens of
thousands of vectors, which searches in single-digit milliseconds. An
approximate index (IVF, HNSW) would add tuning parameters and recall loss to
solve a speed problem this corpus does not have. Say that in an interview --
"I chose exact search because the corpus is small" is a better answer than
naming a fancier index you didn't need.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk, chunk_pdf
from .config import (
    CHUNKS_PATH,
    EMBED_BATCH,
    EMBED_DEVICE,
    EMBED_MODEL,
    INDEX_PATH,
    PDF_DIR,
)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load once per process. Loading MiniLM takes ~2s; doing it per request
    is the single most common performance bug in a RAG demo."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    vecs = get_model().encode(
        texts,
        batch_size=EMBED_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,   # required for IP == cosine
        show_progress_bar=len(texts) > 256,
    )
    return vecs.astype("float32")


def build(pdf_dir: Path = PDF_DIR) -> dict:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}. Put some papers there first.")

    all_chunks: list[Chunk] = []
    skipped: list[str] = []
    for p in pdfs:
        cs = chunk_pdf(p)
        if not cs:
            skipped.append(p.name)  # almost always a scanned PDF with no text layer
        all_chunks.extend(cs)

    if not all_chunks:
        raise SystemExit("Every PDF produced zero chunks. Are they scanned images?")

    vecs = embed([c.text for c in all_chunks])

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    return {
        "pdfs": len(pdfs),
        "chunks": len(all_chunks),
        "dim": int(vecs.shape[1]),
        "skipped": skipped,
    }


def load() -> tuple["faiss.Index", list[dict]]:
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Index not built. Run:  python -m app.index"
        )
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"Index has {index.ntotal} vectors but chunks file has {len(chunks)} "
            "rows. They are out of sync -- rebuild."
        )
    return index, chunks


if __name__ == "__main__":
    stats = build()
    print(f"Indexed {stats['chunks']} chunks from {stats['pdfs']} PDFs "
          f"(dim={stats['dim']}).")
    if stats["skipped"]:
        print("No text layer, skipped:", ", ".join(stats["skipped"]))
