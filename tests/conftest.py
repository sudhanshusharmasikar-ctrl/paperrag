"""
Shared fixtures for the test suite.

The tests never load the real embedding model. `fake_embed` turns text into a
bag-of-words vector (each content word adds 1 to a slot chosen by a stable
hash), so texts that share words get a high cosine score and unrelated texts
score near zero. That is enough to test chunking, the index files, the
abstention guard and the API offline, in seconds. How good the *real* model's
retrieval is gets measured by eval/run_eval.py, not by unit tests.
"""
from __future__ import annotations

import os
import tempfile

# app.config reads these when it is first imported, so set them before any
# app import. Storage goes to a throwaway folder, so a test run can never
# touch your real index, and variables exported in your shell can't change
# the results.
_TMP = tempfile.mkdtemp(prefix="paperrag-tests-")
os.environ.update(
    {
        "PAPERRAG_STORAGE": os.path.join(_TMP, "storage"),
        "PAPERRAG_PDF_DIR": os.path.join(_TMP, "pdfs"),
        "PAPERRAG_GEN_MODE": "extractive",
        "PAPERRAG_SIM_THRESHOLD": "0.35",
        "PAPERRAG_TOP_K": "5",
        "PAPERRAG_CHUNK_CHARS": "900",
        "PAPERRAG_CHUNK_OVERLAP": "150",
    }
)

import re
import zlib

import numpy as np
import pymupdf as fitz
import pytest

from app import index as index_mod
from app import retrieve as retrieve_mod

DIM = 384  # same width as MiniLM, so the FAISS index has its real shape
STOPWORDS = set(
    "a an and are as at be by did do does for from how in is it of on or that "
    "the their they this to was we were what when which who why with".split()
)


def fake_embed(texts: list[str]) -> np.ndarray:
    vecs = np.zeros((len(texts), DIM), dtype="float32")
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word not in STOPWORDS:
                # crc32, not hash(): Python's hash() of a string changes every run
                vecs[row, zlib.crc32(word.encode()) % DIM] += 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)  # unit length, like the real embed()


@pytest.fixture(autouse=True)
def offline_embedder(monkeypatch):
    """Use fake_embed everywhere, and fail loudly if anything loads MiniLM."""
    monkeypatch.setattr(index_mod, "embed", fake_embed)
    monkeypatch.setattr(retrieve_mod, "embed", fake_embed)

    def refuse():
        raise AssertionError("a test tried to load the real embedding model")

    monkeypatch.setattr(index_mod, "get_model", refuse)


def write_pdf(path, pages):
    """One list of paragraphs per page; each paragraph becomes one text block."""
    doc = fitz.open()
    for paragraphs in pages:
        page = doc.new_page(width=612, height=792)
        y = 60
        for text in paragraphs:
            room = page.insert_textbox(fitz.Rect(60, y, 552, y + 150), text, fontsize=10)
            assert room >= 0, "paragraph too long for its text box"
            y += 160
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def make_pdf(tmp_path):
    return lambda name, pages: write_pdf(tmp_path / name, pages)


# Two tiny "papers". Each page fits in one chunk, so every question below has
# exactly one right page.
PAPERS = {
    "transformers.pdf": [
        [
            "The transformer replaces recurrence with self attention. Every token "
            "attends to every other token, so long range dependencies are learned in "
            "a single layer and training runs in parallel across the sequence.",
            "Positional encodings add word order information, because attention by "
            "itself ignores the order of the tokens in the input sequence.",
        ],
        [
            "We train with the Adam optimizer, a learning rate warmup of four thousand "
            "steps, label smoothing of zero point one and dropout of zero point one on "
            "every sublayer.",
        ],
    ],
    "retrieval.pdf": [
        [
            "Dense retrieval encodes questions and passages into vectors and ranks "
            "passages by cosine similarity, which finds paraphrases that keyword "
            "search misses.",
            "BM25 keyword search remains a strong baseline for rare terms such as "
            "product codes, names and exact error messages.",
        ],
    ],
}

@pytest.fixture
def built_index(tmp_path, monkeypatch):
    """Chunk and index PAPERS into this test's own folder; returns build() stats."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    for name, pages in PAPERS.items():
        write_pdf(pdf_dir / name, pages)
    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setattr(index_mod, "INDEX_PATH", storage / "faiss.index")
    monkeypatch.setattr(index_mod, "CHUNKS_PATH", storage / "chunks.jsonl")
    return index_mod.build(pdf_dir)


@pytest.fixture
def missing_index(tmp_path, monkeypatch):
    """Point the app at index files that don't exist."""
    monkeypatch.setattr(index_mod, "INDEX_PATH", tmp_path / "missing.index")
    monkeypatch.setattr(index_mod, "CHUNKS_PATH", tmp_path / "missing.jsonl")
