# PaperRAG

[![tests](https://github.com/sudhanshusharmasikar-ctrl/paperrag/actions/workflows/tests.yml/badge.svg)](https://github.com/sudhanshusharmasikar-ctrl/paperrag/actions/workflows/tests.yml)

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
API, the eval arithmetic, the question checker and reading settings from
`.env`. Retrieval *quality* is measured by `eval/run_eval.py`, not by these
tests.

GitHub Actions runs the same tests after every push, on Python 3.11 and 3.14
(see `.github/workflows/tests.yml`). The badge at the top shows the result
for `main`.

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
possible at all. Two-column pages are read column by column: a block that
crosses the middle of the page (title, abstract, a wide figure caption) splits
the page into bands, and inside each band the left column is read before the
right one.

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

- Pages with three or more columns aren't put in reading order.
- A single sentence longer than `CHUNK_CHARS` still becomes one oversized chunk.
- Scanned PDFs with no text layer produce zero chunks and are skipped. OCR is
  not implemented.
- Tables and figures are dropped by the noise filter, so numeric questions that
  only appear in a table will fail.
- No reranker. A cross-encoder over the top-20 would likely improve precision;
  it was not needed to hit acceptable citation hit-rate on this corpus.

## Evaluation

`eval/questions.jsonl` holds 51 questions about 8 papers: the Transformer,
BERT, BERTweet, ViT and CLIP papers, and three papers on multimodal stance
detection. 34 are answerable; each names the page that answers it and carries
a short evidence phrase copied from that page. 17 are unanswerable: on topic,
but about models that none of the 8 papers mention. The questions were drafted
from passages sampled across all 8 papers and checked by hand against the
PDFs. Because they start from the papers' own passages, the citation hit rate
may be higher than it would be with real users' questions.

The PDFs aren't in this repository. Put them in `data/pdfs/` (or point
`PAPERRAG_PDF_DIR` at their folder) under the names the questions use:
`attention.pdf`, `bert.pdf`, `bertweet.pdf`, `clip.pdf`, `multimodal.pdf`,
`multiturn.pdf`, `tmad.pdf` and `vit.pdf`. Then:

```bash
python -m app.index                          # index the papers
python -m eval.check_questions               # answers are on their pages; no paper answers an unanswerable one
python -m eval.run_eval --sweep              # pick a threshold
python -m eval.run_eval --threshold 0.42     # final table
```

A citation counts as correct when it points to the page listed for the
question, or to another page of the same paper that contains the same evidence
phrase (`check_questions` lists those pages). Latency is the time to embed a
question and search the index on CPU, measured after one warm-up question.

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
