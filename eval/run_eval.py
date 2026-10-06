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
to pick a threshold,
    python -m eval.run_eval --threshold 0.50
to produce the final table, and
    python -m eval.run_eval --threshold 0.50 --mistral
to also run mistral mode on every question the score guard lets through.

--threshold runs every question with three retrieval setups, each adding one
change to the one before: dense (embeddings only), hybrid (embeddings plus
BM25 keyword search) and hybrid with one chunk per page. The guard is the
same in all three, so they refuse the same questions; the table shows how
often each one cites the right page.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from app.config import DISTINCT_PAGES, MISTRAL_API_KEY, MISTRAL_MODEL, RETRIEVAL_MODE, SIM_THRESHOLD, TOP_K
from app.generate import LLMAuthError, LLMError, answer
from app.index import get_model
from app.retrieve import Retriever
from eval.check_questions import with_evidence_pages

QUESTIONS = Path(__file__).parent / "questions.jsonl"
# The comparison --threshold prints: each setup adds one change to the one
# before it, so each column shows what that change did.
VARIANTS = [
    ("dense", {"mode": "dense", "distinct_pages": False}),
    ("hybrid", {"mode": "hybrid", "distinct_pages": False}),
    ("hybrid, 1 per page", {"mode": "hybrid", "distinct_pages": True}),
]


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


def evaluate(retriever: Retriever, rows: list[dict], threshold: float, top_k: int,
             **options) -> dict:
    """options (mode, distinct_pages) go to the retriever; left out, the
    settings in app/config.py apply."""
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]

    # Without the guard the system answers everything, so every unanswerable
    # question yields an unsupported answer by definition. That is the "before".
    unsupported_before = len(unans) / len(rows) * 100

    wrongly_answered = 0
    wrongly_refused = 0
    page_hits = 0
    first_hits = 0  # the right page is the first citation
    page_checked = 0
    pages_cited: list[int] = []  # how many different pages each answer cites
    took_ms: list[float] = []
    # Which questions went wrong, so a reader can see why the numbers are what they are.
    misses: dict[str, list] = {"answered_unanswerable": [], "refused_answerable": [], "wrong_page": []}

    def ask(question: str):
        # Time embedding the question and searching the index: the retrieval
        # latency. In mistral mode the LLM call adds its own time on top.
        start = time.perf_counter()
        res = retriever(question, top_k=top_k, threshold=threshold, **options)
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
            pages_cited.append(len(set(got)))
            want = {(p["source"], p["page"]) for p in r["expected_pages"]}
            if set(got) & want:
                page_hits += 1
                first_hits += got[0] in want
            else:
                misses["wrong_page"].append((r["question"], sorted(want), got))

    unsupported_after = wrongly_answered / len(rows) * 100

    return {
        "options": options,
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
        "first_hit_rate_pct": (
            round(first_hits / page_checked * 100, 1) if page_checked else None
        ),
        "answered_checked": page_checked,
        "cited_right_page": page_hits,
        "cited_right_page_first": first_hits,
        "pages_cited_avg": round(float(np.mean(pages_cited)), 1) if pages_cited else None,
        "latency_ms_p50": round(float(np.percentile(took_ms, 50)), 1) if took_ms else None,
        "latency_ms_p95": round(float(np.percentile(took_ms, 95)), 1) if took_ms else None,
        "misses": misses,
    }


def _pages(ps) -> str:
    return ", ".join(f"{s} p.{p}" for s, p in ps)


def print_misses(misses: dict[str, list]) -> None:
    def score(s):
        return f"{s:.3f}" if s is not None else "  -  "

    print(f"\nUnanswerable questions that still got an answer ({len(misses['answered_unanswerable'])}):")
    for q, s in misses["answered_unanswerable"]:
        print(f"  {score(s)}  {q}")
    print(f"\nAnswerable questions that were refused ({len(misses['refused_answerable'])}):")
    for q, s in misses["refused_answerable"]:
        print(f"  {score(s)}  {q}")
    print(f"\nAnswered, but no citation pointed to the right page ({len(misses['wrong_page'])}):")
    for q, want, got in misses["wrong_page"]:
        print(f"  {q}\n      wanted {_pages(want)}; cited {_pages(got)}")


def changed_pages(before: dict, after: dict) -> tuple[list[str], list[tuple]]:
    """The questions where `after` found a right page `before` had missed, and
    the misses `after` added. Both answer the same questions (same guard)."""
    missed_before = {q for q, _, _ in before["misses"]["wrong_page"]}
    missed_after = {q: (want, got) for q, want, got in after["misses"]["wrong_page"]}
    fixed = [q for q, _, _ in before["misses"]["wrong_page"] if q not in missed_after]
    lost = [(q, *missed_after[q]) for q, _, _ in after["misses"]["wrong_page"] if q not in missed_before]
    return fixed, lost


def print_comparison(results: list[tuple[str, dict]], device) -> None:
    """One column per setup; then, step by step, what each change fixed and lost."""
    first = results[0][1]
    n = first["answered_checked"]

    def share(m, key):
        return f"{m[key]} of {n} ({round(m[key] / n * 100, 1) if n else 0}%)"

    unanswered = len(first["misses"]["answered_unanswerable"])
    print(f"Threshold {first['threshold']}, top {first['top_k']} citations, {first['n_total']} questions "
          f"({first['n_answerable']} answerable, {first['n_unanswerable']} unanswerable).\n")
    print("The guard is the same in every column:")
    print(f"  unanswerable questions that got an answer   {unanswered} of {first['n_unanswerable']}"
          f"  (with no guard, all {first['n_unanswerable']})")
    print(f"  answerable questions refused                {len(first['misses']['refused_answerable'])} "
          f"of {first['n_answerable']}\n")
    lines = [
        ("Right page among the citations", lambda m: share(m, "cited_right_page")),
        ("Right page cited first", lambda m: share(m, "cited_right_page_first")),
        ("Different pages cited (average)", lambda m: str(m["pages_cited_avg"])),
        ("Retrieval time, median / p95 (ms)", lambda m: f"{m['latency_ms_p50']} / {m['latency_ms_p95']}"),
    ]
    print(f"{'Citations of the answered questions':38}" + "".join(f"{label:>20}" for label, _ in results))
    for name, fmt in lines:
        print(f"{name:38}" + "".join(f"{fmt(m):>20}" for _, m in results))
    print(f"(latency measured on {device})")

    for (a_label, a), (b_label, b) in zip(results, results[1:]):
        fixed, lost = changed_pages(a, b)
        print(f"\n{b_label}: found the right page where {a_label} didn't ({len(fixed)}):")
        for q in fixed:
            print(f"  {q}")
        print(f"{b_label}: lost a right page that {a_label} had found ({len(lost)}):")
        for q, want, got in lost:
            print(f"  {q}\n      wanted {_pages(want)}; cited {_pages(got)}")


# When Mistral gives no answer to this many questions in a row, something is
# wrong for every question (a used-up limit, a model the plan doesn't allow).
STOP_AFTER_NO_REPLY = 3


def cited_pages(text: str, hits) -> set[tuple[str, int]]:
    """The pages of the passages an answer cites as [1], [2] or [1, 3]."""
    numbers = {int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text)
               for n in group.split(",")}
    return {(hits[n - 1].source, hits[n - 1].page) for n in numbers if 1 <= n <= len(hits)}


def evaluate_mistral(retriever: Retriever, rows: list[dict], threshold: float, top_k: int,
                     ask=answer) -> dict:
    """Mistral mode on every question the score guard lets through. Does its
    second guard (INSUFFICIENT_CONTEXT) catch the unanswerable ones the score
    guard missed, how many good questions does it refuse, and do its answers
    cite the right page?"""
    records = []
    no_reply_in_a_row = 0
    for r in rows:
        res = retriever(r["question"], top_k=top_k, threshold=threshold)
        if res.abstain:
            continue  # the score guard refused it; mistral mode never sees it
        rec = {"question": r["question"], "answerable": r["answerable"]}
        try:
            out = ask(res, mode="mistral")
        except LLMAuthError:
            raise  # a missing or rejected key fails every question: stop now
        except LLMError as e:
            rec.update(outcome="error", error=str(e))
            records.append(rec)
            no_reply_in_a_row += 1
            if no_reply_in_a_row == STOP_AFTER_NO_REPLY:
                raise LLMError(f"Mistral gave no answer to {STOP_AFTER_NO_REPLY} questions in a row, "
                               f"so the results would mean nothing. The last error: {e}")
            continue
        no_reply_in_a_row = 0
        if out["abstained"]:
            rec["outcome"] = "refused"
        else:
            pages = cited_pages(out["answer"], res.hits)
            want = {(p["source"], p["page"]) for p in r.get("expected_pages", [])}
            rec.update(outcome="answered", answer=out["answer"], cited=sorted(pages),
                       right_page=bool(pages & want) if want else None)
        records.append(rec)

    unans = [x for x in records if not x["answerable"]]
    ans = [x for x in records if x["answerable"]]
    answered = [x for x in ans if x["outcome"] == "answered"]
    return {
        "passed_unanswerable": len(unans),
        "refused_unanswerable": sum(x["outcome"] == "refused" for x in unans),
        "passed_answerable": len(ans),
        "refused_answerable": sum(x["outcome"] == "refused" for x in ans),
        "answered": len(answered),
        "cite_a_passage": sum(bool(x["cited"]) for x in answered),
        "cite_right_page": sum(x["right_page"] is True for x in answered),
        "checked_pages": sum(x["right_page"] is not None for x in answered),
        "no_reply": sum(x["outcome"] == "error" for x in records),
        "records": records,
    }


def print_mistral(m: dict) -> None:
    total = m["passed_unanswerable"] + m["passed_answerable"]
    print(f"\nMistral mode ({MISTRAL_MODEL}) on the {total} questions the score guard let through:")
    print(f"  unanswerable ones it refused (second guard)   {m['refused_unanswerable']} of {m['passed_unanswerable']}")
    print(f"  answerable ones it refused                    {m['refused_answerable']} of {m['passed_answerable']}")
    print(f"  answers that cite a passage, like [1]         {m['cite_a_passage']} of {m['answered']}")
    print(f"  answers citing a passage from the right page  {m['cite_right_page']} of {m['checked_pages']}")
    print(f"  no answer from Mistral                        {m['no_reply']}")
    groups = [
        ("Unanswerable questions Mistral still answered",
         lambda x: not x["answerable"] and x["outcome"] == "answered"),
        ("Answerable questions Mistral refused", lambda x: x["answerable"] and x["outcome"] == "refused"),
        ("Answers that cite no passage from the right page",
         lambda x: x["answerable"] and x["outcome"] == "answered" and x["right_page"] is False),
        ("No answer from Mistral", lambda x: x["outcome"] == "error"),
    ]
    for title, keep in groups:
        hits = [x for x in m["records"] if keep(x)]
        print(f"\n{title} ({len(hits)}):")
        for x in hits:
            print(f"  {x['question']}")
            if x.get("answer"):
                print(f"      answer: {' '.join(x['answer'].split())[:240]}")
            if x.get("cited"):
                print(f"      cited: {_pages(x['cited'])}")
            if x.get("error"):
                print(f"      error: {x['error'][:200]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--mistral", action="store_true",
                    help="also run mistral mode on every question the score guard lets through")
    args = ap.parse_args()
    if args.mistral and not MISTRAL_API_KEY:
        raise SystemExit("MISTRAL_API_KEY is not set. Put it in .env (see README) and run again.")

    rows = load_questions()
    retriever = Retriever()
    # Another page of the same paper that contains the question's evidence
    # phrase states the same fact, so citing it counts as correct too.
    rows = with_evidence_pages(rows, retriever.chunks)
    # One untimed pass first: it loads the model, and a fresh process does one-off
    # setup work the first time it sees inputs, which a running server has done.
    for r in rows:
        for _, options in VARIANTS:
            retriever(r["question"], top_k=args.top_k, **options)
    device = get_model().device

    if args.sweep:
        print(f"Citations from {RETRIEVAL_MODE} search{', one chunk per page' if DISTINCT_PAGES else ''} "
              "(set PAPERRAG_RETRIEVAL and PAPERRAG_DISTINCT_PAGES to change it).")
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
    results = [(label, evaluate(retriever, rows, thr, args.top_k, **options)) for label, options in VARIANTS]
    print_comparison(results, device)
    (_, dense), (_, hybrid), (last_label, last) = results
    print(f"\nEvery miss with {last_label}:")
    print_misses(last["misses"])
    # Counts are easier to defend than "33.3% -> x%": without a guard every
    # unanswerable question gets an answer, so the "before" is just their share.
    refused = dense["n_unanswerable"] - len(dense["misses"]["answered_unanswerable"])
    print(
        f"\nResume line:\n"
        f"  the guard refused {refused} of {dense['n_unanswerable']} unanswerable questions "
        f"(none without it) and wrongly refused {len(dense['misses']['refused_answerable'])} of "
        f"{dense['n_answerable']} answerable ones, on a hand-checked set of {dense['n_total']} questions; "
        f"hybrid search citing one chunk per page cited the right page for {last['cited_right_page']} of "
        f"the {last['answered_checked']} answered ones ({hybrid['cited_right_page']} with hybrid search "
        f"alone, {dense['cited_right_page']} with embeddings alone)"
    )

    if args.mistral:
        try:
            print_mistral(evaluate_mistral(retriever, rows, thr, args.top_k))
        except LLMError as e:  # a rejected key, or Mistral failing every question
            raise SystemExit(f"\n{e}")


if __name__ == "__main__":
    main()
