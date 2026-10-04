"""
Check eval/questions.jsonl against the papers you indexed, before trusting the
numbers from run_eval.py. Run it again whenever the papers or the questions
change:

    python -m eval.check_questions

An answerable question carries "evidence": a short phrase copied from the page
that answers it. The phrase must appear on every page in expected_pages. If it
also appears on other pages of the same paper, those pages state the same fact,
so run_eval.py counts a citation of them as correct too; they are listed here
so you can see them.

An unanswerable question carries "absent": names (of models, datasets) that
must not appear anywhere in the indexed papers. If one does, a paper might
contain the answer after all, and the question has to be rewritten.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata

from app import index as index_mod


def normalise(text: str) -> str:
    """Compare text as a reader would: ignore case, spacing and font variants."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def page_texts(chunks: list[dict]) -> dict[tuple[str, int], str]:
    """The normalised text of every page, keyed by (file name, page number)."""
    pages: dict[tuple[str, int], list[str]] = {}
    for c in chunks:
        pages.setdefault((c["source"], int(c["page"])), []).append(c["text"])
    return {key: normalise(" ".join(parts)) for key, parts in pages.items()}


def evidence_pages(row: dict, pages: dict[tuple[str, int], str]) -> list[tuple[str, int]]:
    """Every page of the expected papers whose text contains the row's evidence."""
    evidence = normalise(row.get("evidence", ""))
    if not evidence:
        return []
    sources = {p["source"] for p in row.get("expected_pages", [])}
    return sorted(key for key, text in pages.items() if key[0] in sources and evidence in text)


def with_evidence_pages(rows: list[dict], chunks: list[dict]) -> list[dict]:
    """Copies of the rows whose expected_pages also include the evidence pages."""
    pages = page_texts(chunks)
    out = []
    for row in rows:
        row = dict(row)
        if row["answerable"]:
            listed = {(p["source"], p["page"]) for p in row.get("expected_pages", [])}
            found = set(evidence_pages(row, pages))
            row["expected_pages"] = [{"source": s, "page": p} for s, p in sorted(listed | found)]
        out.append(row)
    return out


def check(rows: list[dict], chunks: list[dict]) -> tuple[list[str], list[str]]:
    """Return (problems, notes). Problems make the numbers untrustworthy."""
    pages = page_texts(chunks)
    sources = {source for source, _ in pages}
    problems: list[str] = []
    notes: list[str] = []

    for n, row in enumerate(rows, start=1):
        q = row["question"]
        if row["answerable"]:
            expected = row.get("expected_pages") or []
            evidence = normalise(row.get("evidence", ""))
            if not expected or not evidence:
                problems.append(f"{n}. needs both expected_pages and evidence: {q}")
                continue
            for p in expected:
                if p["source"] not in sources:
                    problems.append(f"{n}. {p['source']} is not in the index: {q}")
                elif evidence not in pages.get((p["source"], p["page"]), ""):
                    problems.append(f"{n}. evidence not found on {p['source']} p.{p['page']}: {q}")
            listed = {(p["source"], p["page"]) for p in expected}
            more = [key for key in evidence_pages(row, pages) if key not in listed]
            if more:
                where = ", ".join(f"{s} p.{p}" for s, p in more)
                notes.append(f"{n}. same evidence also on {where}: {q}")
        else:
            terms = row.get("absent") or []
            if not terms:
                problems.append(f"{n}. an unanswerable question needs 'absent' names: {q}")
            for term in terms:
                pattern = re.compile(r"\b" + re.escape(normalise(term)) + r"\b")
                found = sorted(key for key, text in pages.items() if pattern.search(text))
                if found:
                    where = ", ".join(f"{s} p.{p}" for s, p in found[:3])
                    problems.append(f"{n}. '{term}' appears in {where}, so the answer may be there: {q}")

    unanswerable = sum(not r["answerable"] for r in rows)
    if unanswerable * 3 < len(rows):
        problems.append(f"only {unanswerable} of {len(rows)} questions are unanswerable; "
                        "make it at least a third")
    return problems, notes


def main() -> None:
    from eval.run_eval import load_questions  # imported here: run_eval imports this module

    rows = load_questions()
    if not index_mod.CHUNKS_PATH.exists():
        raise SystemExit("Index not built. Run:  python -m app.index")
    with open(index_mod.CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    problems, notes = check(rows, chunks)
    answerable = sum(r["answerable"] for r in rows)
    print(f"{len(rows)} questions: {answerable} answerable, {len(rows) - answerable} unanswerable, "
          f"checked against {len(chunks)} chunks.")
    if notes:
        print("\nPages that state the same fact (counted as correct citations):")
        for line in notes:
            print("  " + line)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print("  " + line)
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
