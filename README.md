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
and chunking, the index files, the abstention guard, BM25 and hybrid search,
one chunk per page, answer formatting, the API, the eval arithmetic, the question checker,
reading settings from `.env` and where the web page's threshold slider
starts. Retrieval *quality* is
measured by `eval/run_eval.py`, not by these tests.

GitHub Actions runs the same tests after every push, on Python 3.11 and 3.14
(see `.github/workflows/tests.yml`). The badge at the top shows the result
for `main`.

## How it works

```
PDF ─► PyMuPDF blocks ─► page-bounded chunks ─► MiniLM embeddings + BM25 word index
                                                      │
                                                      ▼
question ─┬─► embed ─► FAISS (exact cosine) ──┬─► merge the rankings (RRF) ─► top-k ─► answer + citations
          └─► BM25 keyword scores ────────────┘
          guard: best cosine below the threshold ─► abstain
```

Chunks never cross a page boundary, which is what makes page-level citation
possible at all. Two-column pages are read column by column: a block that
crosses the middle of the page (title, abstract, a wide figure caption) splits
the page into bands, and inside each band the left column is read before the
right one.

### Hybrid search

Retrieval combines the embeddings with keyword search. The
embedding model matches meaning, so it finds a passage that says the same
thing in other words, but it is weaker at exact terms such as "WordPiece",
"30,000" or "LayerNorm". BM25 (`app/bm25.py`) scores passages by the
question's words they contain, weighting rare words most. The two rankings
are merged by reciprocal rank fusion: each gives a passage 1 / (60 + rank),
and the shares add up, so no cosine score is ever compared with a BM25 score.

The guard still uses the best embedding score, so hybrid and embeddings alone
(`PAPERRAG_RETRIEVAL=dense`) refuse exactly the same questions; only the cited
pages differ. That makes the comparison in the evaluation clean: hybrid cited
the right page for 24 of the 30 answered questions, embeddings alone for 19,
so hybrid is the default.

### One chunk per page

A page is often cut into several chunks, and the best-ranked chunks tend to
come from the same page: in 5 of the 6 misses left after hybrid search, one
page took two or three of the five citations. `PAPERRAG_DISTINCT_PAGES=1`
keeps only each page's best chunk, so the five citations are five different
pages. It can't lose a page: the pages of the plain top five all stay, in the
same order, and pages further down fill the slots repeats used to take. So
the first citation doesn't change, and the right page can only be gained.
On the 51-question set it raised the answers citing the right page from 24
to 26 of 30 and lost none, so it is on by default;
`PAPERRAG_DISTINCT_PAGES=0` turns it off.

## Design decisions

| Choice | Why | What was rejected |
|---|---|---|
| `IndexFlatIP` (exact) | Tens of thousands of vectors search in single-digit ms | IVF/HNSW — recall loss and tuning knobs to solve a speed problem this corpus doesn't have |
| MiniLM-L6-v2 (384d) | ~80MB, CPU-fast, deploys on a free tier | Larger embedders — better recall, but the demo stops being deployable |
| Chunk at block level | Preserves page number and reading order | Fixed-size split over the whole document — destroys citability |
| Threshold on top-1 | Mean-of-top-k drifts with k, making the threshold silently k-dependent | Mean or median scoring |
| Reciprocal rank fusion for hybrid | Uses ranks only, so no tuning of how a cosine compares with a BM25 score | A weighted sum of normalised scores, one more knob to tune |
| One chunk per page in the citations | Five different pages cover more ground, and no page the plain ranking cited is ever dropped | Several chunks of the best page, which repeat what the first one shows |
| BM25 written in-house (40 lines) | No new dependency; every line is tested and explainable | `rank_bm25` — fine, but one more package for a short formula |
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
python -m eval.run_eval --threshold 0.50     # final numbers, dense vs hybrid, and every miss
```

A citation counts as correct when it points to the page listed for the
question, or to another page of the same paper that contains the same evidence
phrase (`check_questions` lists those pages). Latency is the time to embed a
question and search the index on whatever device the model runs on (the
script prints it), measured after one untimed pass over all the questions.
With `--threshold`, the script runs every question with three setups, each
adding one change to the one before: embeddings alone, hybrid search, and
hybrid search with one chunk per page. It shows how often each cites the right
page (among the 5 citations, and as the first one) and how many different
pages it cites, lists the questions each change gained or lost, and lists
every miss of the last setup. The guard is the same in all three, so they
refuse the same questions.

### Results

Measured on the 8 papers above (905 chunks), with the threshold at 0.50,
hybrid search and one chunk per page. Latency is retrieval only, on an
Apple-silicon MacBook Air, where the model runs on the GPU (`mps`).

| Metric | Value |
|---|---|
| Corpus | 8 PDFs, 905 chunks |
| Eval set | 51 questions (34 answerable, 17 unanswerable) |
| Unsupported answers, no guard | 33.3% (all 17 unanswerable questions get an answer) |
| Unsupported answers, with guard | 5.9% (3 of 51) |
| False refusal rate | 11.8% (4 of 34 answerable questions) |
| Citation hit rate | 86.7% (26 of the 30 answered questions cite the right page; 24 with hybrid search alone, 19 with embeddings alone) |
| Right page cited first | 40.0% (12 of 30; 13 with embeddings alone) |
| Retrieval latency | median 11.6 ms, p95 12.5 ms (10.4 and 11.8 ms with embeddings alone, in the same run) |

In one line: the guard refused 14 of the 17 unanswerable questions (none
without it) and wrongly refused 4 of the 34 answerable ones, and hybrid
search citing one chunk per page cited the right page for 26 of the 30
answered questions, up from 19 with embeddings alone.

### Three setups compared

`python -m eval.run_eval --threshold 0.50` runs every question with three
setups, each adding one change to the one before. Same guard, so the same
30 questions get an answer:

| Citations of the 30 answered questions | Embeddings alone | Hybrid | Hybrid, 1 chunk per page |
|---|---|---|---|
| Right page among the 5 citations | 19 (63.3%) | 24 (80.0%) | **26 (86.7%)** |
| Right page cited first | 13 (43.3%) | 12 (40.0%) | 12 (40.0%) |
| Different pages cited (average) | 4.0 | 4.0 | 5.0 |
| Retrieval time, median / p95 | 10.4 / 11.8 ms | 11.7 / 13.0 ms | 11.6 / 12.5 ms |

- **Hybrid search found the right page for 6 more questions**, mostly ones that hinge
  on an exact term: the Transformer paper's other name for self-attention
  ("intra-attention"), the size of BERT's WordPiece vocabulary, ViT's position
  embeddings and the pre-training the ViT authors leave for future work, what
  attention in end-to-end memory networks is based on, and the applications
  where zero-shot CLIP looks promising.
- **It lost 1**: for the encoders that give T-MAD its best results, pages 8,
  15 and 2 of the same paper pushed page 9 out of the five.
- **One chunk per page found 2 more and lost none**, as it can't. One is that
  T-MAD question, where three chunks of page 15 had taken three of the five
  slots. The other asks when a self-attention layer is cheaper than a
  recurrent one: page 6 of the Transformer paper had taken three slots, and
  page 7, which has the answer, now gets one.
- **The first citation got no better** (13, then 12 and 12). Keyword search
  and one chunk per page widen what makes the five; putting the best passage
  first is what a reranker is for.
- Retrieval times vary by about a millisecond from run to run (the previous
  run measured hybrid search at 10.3 ms), and every setup now ranks all the
  chunks, which one chunk per page needs.
- The questions were drafted from the papers' own passages, so they share
  words with the right page, which favours keyword search. Real users'
  questions may share fewer, and the gain may be smaller.

### Choosing the threshold

Selected rows of `python -m eval.run_eval --sweep`:

| Threshold | Unsupported answers | False refusals | Citation hit rate (embeddings alone) |
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
- **4 answers cite the wrong page** (6 with hybrid search alone, 11 with
  embeddings alone). Each now cites five different pages, mostly of the
  right paper, but the page with the answer ranks below them: the hardware
  BERT-Large was trained on (page 13 wanted; pages 14, 8, 3, 9 and 16
  cited), the name of the new multi-turn stance dataset, the model that
  writes MLLM-SD's image captions, and how ViT's classification head changes
  for fine-tuning.

Next steps, each to be measured by re-running this evaluation: a
cross-encoder reranker over the top 20, to put the best passage first (the
first citation is right for 12 of the 30) and to reach the 4 pages still
missed; and a stronger embedding model.
