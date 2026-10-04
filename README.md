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
API, the eval arithmetic, the question checker, reading settings from `.env`
and where the web page's threshold slider starts. Retrieval *quality* is
measured by `eval/run_eval.py`, not by these tests.

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
- A sentence that runs across a page break is split between two chunks, the
  price of chunks that never cross a page.
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
python -m eval.run_eval --threshold 0.50     # final numbers, and every miss
```

A citation counts as correct when it points to the page listed for the
question, or to another page of the same paper that contains the same evidence
phrase (`check_questions` lists those pages). Latency is the time to embed a
question and search the index on whatever device the model runs on (the
script prints it), measured after one untimed pass over all the questions.
With `--threshold`, the script also lists the questions that went wrong.

### Results

Measured on the 8 papers above (905 chunks), with the threshold at 0.50.
Latency is retrieval only, on an Apple-silicon MacBook Air, where the model
runs on the GPU (`mps`).

| Metric | Value |
|---|---|
| Corpus | 8 PDFs, 905 chunks |
| Eval set | 51 questions (34 answerable, 17 unanswerable) |
| Unsupported answers, no guard | 33.3% (all 17 unanswerable questions get an answer) |
| Unsupported answers, with guard | 5.9% (3 of 51) |
| False refusal rate | 11.8% (4 of 34 answerable questions) |
| Citation hit rate | 63.3% (19 of the 30 answered questions cite the right page) |
| Retrieval latency | median 9.3 ms, p95 10.4 ms |

In one line: the guard refused 14 of the 17 unanswerable questions (none
without it) and wrongly refused 4 of the 34 answerable ones.

### Choosing the threshold

Selected rows of `python -m eval.run_eval --sweep`:

| Threshold | Unsupported answers | False refusals | Citation hit rate |
|---|---|---|---|
| 0.34 | 19.6% (10 of 17 unanswerable answered) | 0% | 61.8% |
| 0.42 | 15.7% (8) | 0% | 61.8% |
| 0.48 | 9.8% (5) | 11.8% (4 of 34) | 63.3% |
| **0.50** | **5.9% (3)** | **11.8% (4)** | **63.3%** |
| 0.52 | 3.9% (2) | 14.7% (5) | 65.5% |
| 0.54 | 2.0% (1) | 23.5% (8) | 69.2% |
| 0.56 | 0% | 32.4% (11) | 69.6% |

Up to 0.42 no answerable question is refused, but 8 or more of the 17
unanswerable ones still get an answer. From 0.42 to 0.50, five more
unanswerable questions are refused for the price of four answerable ones.
Above 0.50, each further refusal of an unanswerable question costs one to
three answerable ones, and at 0.56 a third of the answerable questions are
refused. So the default is now 0.50; the old starting value, 0.35, let 10 of
the 17 unanswerable questions through. The citation hit rate rises with the
threshold only because the refused questions, often the hard ones, drop out
of the count.

### What went wrong at 0.50

`python -m eval.run_eval --threshold 0.50` lists every miss.

- **3 unanswerable questions got an answer**, all just above the threshold
  (0.507 to 0.549): the size of LAION-5B, T5's pre-training dataset and
  XLNet's objective. The papers never mention these models, but they discuss
  pre-training data and objectives, so a passage on the same topic scores
  high. A similarity threshold catches off-topic questions, not on-topic ones
  whose answer is missing; that is the job of the second guard
  (`INSUFFICIENT_CONTEXT` in mistral mode), which this evaluation doesn't use.
- **4 answerable questions were refused** (0.434 to 0.478): InfoNCE in CLIP,
  the fusion methods compared with TMPT, the tokenizer MLLM-SD uses and where
  ViT puts LayerNorm. Each asks for one specific detail, and such questions
  score lower than broad ones.
- **11 answers cited the wrong page**, usually the right paper but another
  page of it, because an overview passage outranks the detail. One answer is
  a sentence split across a page break (Transformer, pages 6 and 7), which
  page-bounded chunks can't keep together. A few cited pages may also hold the
  answer (ViT's appendix on position embeddings, for one), so the strict 63%
  may understate retrieval a little.

Next steps, each to be measured by re-running this evaluation: keyword search
(BM25) next to vector search, so exact terms like "WordPiece" or "30,000" are
found; a cross-encoder reranker over the top 20; and a stronger embedding
model.
