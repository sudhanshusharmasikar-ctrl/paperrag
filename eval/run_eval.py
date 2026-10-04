"""
The eval that produces your resume numbers. Nobody can run this for you --
it needs YOUR questions over YOUR corpus.

Build eval/questions.jsonl with two kinds of question:

  answerable   -- the answer IS in your indexed papers
  unanswerable -- plausibly phrased, on-topic, but the answer is NOT in them
                  (e.g. ask about a paper you deliberately left out of data/pdfs)

You need both. A system that answers everything scores perfectly on answerable
questions and is still useless. The unanswerable set is what the guard is for.

Aim for 40-60 questions, at least a third unanswerable. Writing them takes
about two hours and is the part that makes the bullet point true.

Check the file against your index before trusting any numbers:
    python -m eval.check_questions

Then:
    python -m eval.run_eval --sweep
to pick a threshold, and
    python -m eval.run_eval --threshold 0.50
to produce the final table.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.config import SIM_THRESHOLD, TOP_K
from app.index import get_model
from app.retrieve import Retriever
from eval.check_questions import with_evidence_pages

QUESTIONS = Path(__file__).parent / "questions.jsonl"


def load_questions() -> list[dict]:
    if not QUESTIONS.exists():
        raise SystemExit(
            f"{QUESTIONS} not found. Copy questions.example.jsonl and write "
            "your own. This is the work; it cannot be skipped."
        )
    rows = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        assert "question" in r and "answerable" in r, f"bad row: {r}"
    return rows


def evaluate(retriever: Retriever, rows: list[dict], threshold: float, top_k: int) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]

    # Without the guard the system answers everything, so every unanswerable
    # question yields an unsupported answer by definition. That is the "before".
    unsupported_before = len(unans) / len(rows) * 100

    wrongly_answered = 0
    wrongly_refused = 0
    page_hits = 0
    page_checked = 0
    took_ms: list[float] = []
    # Which questions went wrong, so a reader can see why the numbers are what they are.
    misses: dict[str, list] = {"answered_unanswerable": [], "refused_answerable": [], "wrong_page": []}

    def ask(question: str):
        # Time embedding the question and searching the index: the retrieval
        # latency. In mistral mode the LLM call adds its own time on top.
        start = time.perf_counter()
        res = retriever(question, top_k=top_k, threshold=threshold)
        took_ms.append((time.perf_counter() - start) * 1000)
        return res

    for r in unans:
        res = ask(r["question"])
        if res.abstain is False:
            wrongly_answered += 1
            misses["answered_unanswerable"].append((r["question"], getattr(res, "top_score", None)))

    for r in ans:
        res = ask(r["question"])
        if res.abstain:
            wrongly_refused += 1
            misses["refused_answerable"].append((r["question"], getattr(res, "top_score", None)))
        elif r.get("expected_pages"):
            page_checked += 1
            got = [(h.source, h.page) for h in res.hits]
            want = {(p["source"], p["page"]) for p in r["expected_pages"]}
            if set(got) & want:
                page_hits += 1
            else:
                misses["wrong_page"].append((r["question"], sorted(want), got))

    unsupported_after = wrongly_answered / len(rows) * 100

    return {
        "threshold": round(threshold, 3),
        "top_k": top_k,
        "n_total": len(rows),
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "unsupported_before_pct": round(unsupported_before, 1),
        "unsupported_after_pct": round(unsupported_after, 1),
        "false_refusal_pct": round(wrongly_refused / max(len(ans), 1) * 100, 1),
        "citation_hit_rate_pct": (
            round(page_hits / page_checked * 100, 1) if page_checked else None
        ),
        "latency_ms_p50": round(float(np.percentile(took_ms, 50)), 1) if took_ms else None,
        "latency_ms_p95": round(float(np.percentile(took_ms, 95)), 1) if took_ms else None,
        "misses": misses,
    }


def print_misses(misses: dict[str, list]) -> None:
    def score(s):
        return f"{s:.3f}" if s is not None else "  -  "

    def pages(ps):
        return ", ".join(f"{s} p.{p}" for s, p in ps)

    print(f"\nUnanswerable questions that still got an answer ({len(misses['answered_unanswerable'])}):")
    for q, s in misses["answered_unanswerable"]:
        print(f"  {score(s)}  {q}")
    print(f"\nAnswerable questions that were refused ({len(misses['refused_answerable'])}):")
    for q, s in misses["refused_answerable"]:
        print(f"  {score(s)}  {q}")
    print(f"\nAnswered, but no citation pointed to the right page ({len(misses['wrong_page'])}):")
    for q, want, got in misses["wrong_page"]:
        print(f"  {q}\n      wanted {pages(want)}; cited {pages(got)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    rows = load_questions()
    retriever = Retriever()
    # Another page of the same paper that contains the question's evidence
    # phrase states the same fact, so citing it counts as correct too.
    rows = with_evidence_pages(rows, retriever.chunks)
    # One untimed pass first: it loads the model, and a fresh process does one-off
    # setup work the first time it sees inputs, which a running server has done.
    for r in rows:
        retriever(r["question"], top_k=args.top_k)
    device = get_model().device

    if args.sweep:
        print(f"{'thr':>6} {'unsup_after%':>13} {'false_refuse%':>14} {'cite_hit%':>10}")
        print("-" * 46)
        first = None
        for step in range(24):  # 0.20, 0.22, ... 0.66 (adding 0.02 in a loop drifts past 0.66)
            m = evaluate(retriever, rows, round(0.20 + 0.02 * step, 2), args.top_k)
            first = first or m
            print(f"{m['threshold']:>6} {m['unsupported_after_pct']:>13} "
                  f"{m['false_refusal_pct']:>14} {str(m['citation_hit_rate_pct']):>10}")
        print(f"\nRetrieval time per question on {device}: median {first['latency_ms_p50']} ms, "
              f"p95 {first['latency_ms_p95']} ms")
        print(
            "\nPick the threshold where unsupported answers drop sharply but "
            "false refusals are still acceptable. There is no single right "
            "answer -- the trade-off IS the finding. Put the curve in your README."
        )
        return

    thr = args.threshold if args.threshold is not None else SIM_THRESHOLD
    m = evaluate(retriever, rows, thr, args.top_k)
    misses = m.pop("misses")
    print(json.dumps(m, indent=2))
    print(f"(latency measured on {device})")
    print_misses(misses)
    # Counts are easier to defend than "33.3% -> x%": without a guard every
    # unanswerable question gets an answer, so the "before" is just their share.
    refused = m["n_unanswerable"] - len(misses["answered_unanswerable"])
    print(
        f"\nResume line:\n"
        f"  the guard refused {refused} of {m['n_unanswerable']} unanswerable questions "
        f"(none without it) and wrongly refused {len(misses['refused_answerable'])} of "
        f"{m['n_answerable']} answerable ones, on a hand-checked set of {m['n_total']} questions"
    )


if __name__ == "__main__":
    main()
