# PaperRAG

Citation-grounded retrieval over research papers. Every answer cites the source
file and page it came from, and the system refuses to answer when retrieval is
too weak to support one.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11 or newer
pip install -r requirements.txt   # pinned, tested versions

cp .env.example .env          # defaults need no API key
mkdir -p data/pdfs && cp /path/to/your/papers/*.pdf data/pdfs/
python -m app.index           # build the FAISS index

uvicorn app.api:app --reload  # terminal 1
streamlit run ui/streamlit_app.py   # terminal 2
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests run offline in a few seconds: a small bag-of-words embedder stands
in for MiniLM, and the tests generate their own PDFs. They cover text cleaning
and chunking, the index files, the abstention guard, answer formatting, the
API and the eval arithmetic. Retrieval *quality* is measured by
`eval/run_eval.py`, not by these tests.

Two tests are marked `xfail` (expected to fail): they describe the known
chunking bugs listed under [Known limitations](#known-limitations).

## How it works

```
PDF ─► PyMuPDF blocks ─► page-bounded chunks ─► MiniLM embeddings
                                                      │
                                                      ▼
question ─► embed ─► FAISS (exact cosine) ─► top-k ─► guard ─► answer + citations
                                                       │
                                                       └─► abstain
```

Chunks never cross a page boundary, which is what makes page-level citation
possible at all.

## Design decisions

| Choice | Why | What was rejected |
|---|---|---|
| `IndexFlatIP` (exact) | Tens of thousands of vectors search in single-digit ms | IVF/HNSW — recall loss and tuning knobs to solve a speed problem this corpus doesn't have |
| MiniLM-L6-v2 (384d) | ~80MB, CPU-fast, deploys on a free tier | Larger embedders — better recall, but the demo stops being deployable |
| Chunk at block level | Preserves page number and reading order | Fixed-size split over the whole document — destroys citability |
| Threshold on top-1 | Mean-of-top-k drifts with k, making the threshold silently k-dependent | Mean or median scoring |
| `extractive` default | Zero keys, zero cost, and a 100%-faithful control condition | LLM-only — no baseline to compare faithfulness against |

Two independent guards: retrieval-side (similarity below threshold) and
generation-side (model emits `INSUFFICIENT_CONTEXT`). They catch different
failures — the first catches off-topic questions, the second catches questions
where the passages are on-topic but don't contain the answer.

## Known limitations

- Two-column pages: blocks are sorted top-to-bottom, then left-to-right, which
  interleaves the two columns when their paragraphs start at different heights.
- The overlap carried into a new chunk isn't counted against `CHUNK_CHARS`, so
  a chunk that starts with a long paragraph can run up to ~150 characters over.
- Scanned PDFs with no text layer produce zero chunks and are skipped. OCR is
  not implemented.
- Tables and figures are dropped by the noise filter, so numeric questions that
  only appear in a table will fail.
- No reranker. A cross-encoder over the top-20 would likely improve precision;
  it was not needed to hit acceptable citation hit-rate on this corpus.

## Evaluation

Numbers in the results table below must come from `eval/run_eval.py` on a
question set you write yourself. See the docstring in that file.

```bash
python -m eval.run_eval --sweep              # pick a threshold
python -m eval.run_eval --threshold 0.42     # final table
```

| Metric | Value |
|---|---|
| Corpus | _N_ PDFs, _N_ chunks |
| Eval set | _N_ questions (_N_ answerable, _N_ unanswerable) |
| Unsupported answers, no guard | _N_% |
| Unsupported answers, with guard | _N_% |
| False refusal rate | _N_% |
| Citation hit rate | _N_% |
| p95 latency (CPU) | _N_ ms |

Fill these in from your own run. Do not put this project on a resume until
this table has real numbers in it.
