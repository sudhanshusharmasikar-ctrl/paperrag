"""
Central configuration.

Everything an interviewer might ask "why that number?" lives here, in one place,
with a comment explaining the choice. Change values via environment variables so
you can sweep them from the eval script without editing code.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------- paths ----------
PDF_DIR = Path(os.getenv("PAPERRAG_PDF_DIR", ROOT / "data" / "pdfs"))
STORAGE_DIR = Path(os.getenv("PAPERRAG_STORAGE", ROOT / "storage"))
INDEX_PATH = STORAGE_DIR / "faiss.index"
CHUNKS_PATH = STORAGE_DIR / "chunks.jsonl"

# ---------- chunking ----------
# Chunk size is in characters, not tokens, on purpose: it keeps ingestion
# independent of the tokenizer, and MiniLM truncates at 256 word-pieces anyway
# (~1000 chars of English), so anything above ~1200 is silently discarded.
CHUNK_CHARS = int(os.getenv("PAPERRAG_CHUNK_CHARS", 900))
CHUNK_OVERLAP = int(os.getenv("PAPERRAG_CHUNK_OVERLAP", 150))
MIN_CHUNK_CHARS = 120  # drop headers, page numbers, stray figure labels

# Two-column pages: a text block counts as sitting in one column if it reaches
# no more than this many points (1/72 inch) past the middle of the page. Column
# text stops short of the middle; full-width text (title, abstract, a wide
# caption) crosses it by hundreds of points, so 20 is a generous margin.
COLUMN_TOLERANCE = 20

# ---------- embedding ----------
# all-MiniLM-L6-v2: 384-dim, ~80MB, runs on CPU in milliseconds.
# Chosen over larger models because this whole thing must deploy on a free tier.
EMBED_MODEL = os.getenv("PAPERRAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384
EMBED_BATCH = 64

# ---------- retrieval ----------
TOP_K = int(os.getenv("PAPERRAG_TOP_K", 5))

# The abstention threshold. Vectors are L2-normalised and the index is inner
# product, so scores are cosine similarity in [-1, 1].
# DO NOT trust this default. Sweep it with eval/run_eval.py on your own
# questions and set it from the data. The right value depends on your corpus.
SIM_THRESHOLD = float(os.getenv("PAPERRAG_SIM_THRESHOLD", 0.35))

# ---------- generation ----------
# "extractive"  -> no LLM, returns the retrieved spans verbatim with citations.
#                  Always available, zero keys, zero cost. Good default.
# "mistral"     -> calls the Mistral API. Needs MISTRAL_API_KEY.
GEN_MODE = os.getenv("PAPERRAG_GEN_MODE", "extractive")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MAX_ANSWER_TOKENS = 400

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)
