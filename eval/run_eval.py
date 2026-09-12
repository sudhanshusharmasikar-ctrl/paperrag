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

Then:
    python -m eval.run_eval --sweep
to pick a threshold, and
    python -m eval.run_eval --threshold 0.42
to produce the final table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import TOP_K
from app.retrieve import Retriever

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

    for r in unans:
        if retriever(r["question"], top_k=top_k, threshold=threshold).abstain is False:
            wrongly_answered += 1

    for r in ans:
        res = retriever(r["question"], top_k=top_k, threshold=threshold)
        if res.abstain:
            wrongly_refused += 1
        elif r.get("expected_pages"):
            page_checked += 1
            got = {(h.source, h.page) for h in res.hits}
            want = {(p["source"], p["page"]) for p in r["expected_pages"]}
            if got & want:
                page_hits += 1

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
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    rows = load_questions()
    retriever = Retriever()

    if args.sweep:
        print(f"{'thr':>6} {'unsup_after%':>13} {'false_refuse%':>14} {'cite_hit%':>10}")
        print("-" * 46)
        t = 0.20
        while t <= 0.66:
            m = evaluate(retriever, rows, t, args.top_k)
            print(f"{m['threshold']:>6} {m['unsupported_after_pct']:>13} "
                  f"{m['false_refusal_pct']:>14} {str(m['citation_hit_rate_pct']):>10}")
            t += 0.02
        print(
            "\nPick the threshold where unsupported answers drop sharply but "
            "false refusals are still acceptable. There is no single right "
            "answer -- the trade-off IS the finding. Put the curve in your README."
        )
        return

    thr = args.threshold if args.threshold is not None else 0.35
    m = evaluate(retriever, rows, thr, args.top_k)
    print(json.dumps(m, indent=2))
    print(
        f"\nResume line:\n"
        f"  reduced unsupported answers from {m['unsupported_before_pct']}% to "
        f"{m['unsupported_after_pct']}% on a hand-labelled set of "
        f"{m['n_total']} questions"
    )


if __name__ == "__main__":
    main()
