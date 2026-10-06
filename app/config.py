"""
Central configuration.

Everything an interviewer might ask "why that number?" lives here, in one place,
with a comment explaining the choice. Change values via environment variables so
you can sweep them from the eval script without editing code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Read settings from a .env file in the project folder, if there is one
# (cp .env.example .env). A variable already set in your shell wins over the
# file, so you can still change a setting for a single command.
load_dotenv(ROOT / ".env")

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
# Where the model runs. Unset, the library picks: an NVIDIA GPU ("cuda"), the
# GPU of an Apple-silicon Mac ("mps"), or else the CPU. Don't force "cpu" on a
# Mac: on an M-series MacBook Air that crashed with a segmentation fault, most
# likely because faiss and PyTorch each load their own copy of the OpenMP
# threading library, and that only bites when both run on the CPU.
EMBED_DEVICE = os.getenv("PAPERRAG_DEVICE") or None

# ---------- retrieval ----------
TOP_K = int(os.getenv("PAPERRAG_TOP_K", 5))

# Which passages to cite. "dense": the embedding model alone. "hybrid": the
# embedding model and BM25 keyword search (app/bm25.py) together, merged by
# reciprocal rank fusion, so a passage with the exact rare term ("WordPiece",
# "30,000") can make the list. The guard below always uses the embedding
# score, so both modes refuse the same questions; only the cited pages differ.
# eval/run_eval.py measures the two side by side: on the 51-question set,
# hybrid cited the right page for 24 of the 30 answered questions, dense for
# 19 (README, Evaluation), so hybrid is the default.
RETRIEVAL_MODE = os.getenv("PAPERRAG_RETRIEVAL", "hybrid")
# Cite each page at most once. A page holds several chunks, and the best ones
# often come from the same page: in 5 of the 6 misses left after hybrid
# search, one page took two or three of the five citations. Keeping only each
# page's best chunk frees those slots for other pages. It never drops a page
# the plain ranking cited and leaves the first citation as it was; the
# evaluation measures what it gains.
DISTINCT_PAGES = os.getenv("PAPERRAG_DISTINCT_PAGES", "0") != "0"
# Reciprocal rank fusion gives a passage 1 / (RRF_K + rank) from each ranking.
# 60 comes from the paper that introduced it (Cormack et al., 2009) and is the
# usual default: it keeps one list's first place from outweighing everything.
RRF_K = 60

# The abstention threshold. Vectors are L2-normalised and the index is inner
# product, so scores are cosine similarity in [-1, 1].
# 0.50 comes from sweeping eval/run_eval.py over the 51-question evaluation set
# (README, Evaluation): it refused 14 of the 17 unanswerable questions and 4 of
# the 34 answerable ones. The right value depends on the papers indexed, so
# re-run the sweep when you change them.
SIM_THRESHOLD = float(os.getenv("PAPERRAG_SIM_THRESHOLD", 0.50))

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
