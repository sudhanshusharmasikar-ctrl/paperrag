"""
eval/check_questions.py decides which pages count as a correct citation and
catches broken questions before they turn into numbers. The last test checks
eval/questions.jsonl itself, which CI can do without the papers.
"""
import json

from eval.check_questions import check, normalise, with_evidence_pages
from eval.run_eval import QUESTIONS

CHUNKS = [
    {"source": "bert.pdf", "page": 3, "text": "BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M)."},
    {"source": "bert.pdf", "page": 8, "text": "BERTBASE contains 110M parameters and BERTLARGE contains 340M parameters."},
    {"source": "vit.pdf", "page": 3, "text": "We use standard learnable 1D position embeddings."},
]


def answerable(evidence, source="bert.pdf", page=8):
    return {"question": "How many parameters does BERT-Large have?", "answerable": True,
            "expected_pages": [{"source": source, "page": page}], "evidence": evidence}


def unanswerable(name):
    return {"question": f"How many parameters does {name} have?", "answerable": False,
            "absent": [name]}


def test_evidence_on_its_page_passes_and_other_pages_with_it_are_noted():
    problems, notes = check([answerable("340M"), unanswerable("T5")], CHUNKS)
    assert problems == []
    assert len(notes) == 1 and "bert.pdf p.3" in notes[0]


def test_citing_another_page_with_the_same_evidence_counts_as_correct():
    rows = [answerable("340M"), unanswerable("T5")]
    out = with_evidence_pages(rows, CHUNKS)
    assert out[0]["expected_pages"] == [{"source": "bert.pdf", "page": 3},
                                        {"source": "bert.pdf", "page": 8}]
    assert out[1] == unanswerable("T5")
    assert rows[0]["expected_pages"] == [{"source": "bert.pdf", "page": 8}]  # input untouched


def test_wrong_evidence_and_unknown_files_are_problems():
    rows = [answerable("350M"), answerable("340M", source="bart.pdf"),
            unanswerable("T5"), unanswerable("XLNet")]
    problems, _ = check(rows, CHUNKS)
    assert any("evidence not found on bert.pdf p.8" in p for p in problems)
    assert any("bart.pdf is not in the index" in p for p in problems)


def test_unanswerable_question_fails_if_a_paper_names_its_model():
    chunks = CHUNKS + [{"source": "vit.pdf", "page": 9, "text": "Unlike T5, ViT has no decoder."}]
    problems, _ = check([answerable("340M"), unanswerable("T5")], chunks)
    assert len(problems) == 1 and "'T5' appears in vit.pdf p.9" in problems[0]


def test_names_must_match_whole_words():
    chunks = CHUNKS + [{"source": "vit.pdf", "page": 9, "text": "Electrical engineers labelled it."}]
    problems, _ = check([answerable("340M"), unanswerable("ELECTRA")], chunks)
    assert problems == []


def test_too_few_unanswerable_questions_is_a_problem():
    rows = [answerable("340M")] * 3 + [unanswerable("T5")]
    problems, _ = check(rows, CHUNKS)
    assert any("at least a third" in p for p in problems)


def test_normalise_ignores_case_spacing_and_font_variants():
    assert normalise("Image  descriptions\nusing GPT4-Vision") == "image descriptions using gpt4-vision"
    # PDFs often store maths-style letters; NFKC turns them into plain ones
    assert normalise("as𝐶𝑎𝑝𝑡𝑖𝑜𝑛") == "ascaption"


def test_the_question_file_is_complete():
    rows = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [r["question"] for r in rows]
    assert len(questions) == len(set(questions)), "a question appears twice"
    for r in rows:
        if r["answerable"]:
            assert r["evidence"].strip() and r["expected_pages"], r["question"]
            for p in r["expected_pages"]:
                assert p["source"].endswith(".pdf") and p["page"] >= 1, r["question"]
        else:
            assert r["absent"], r["question"]
    assert sum(not r["answerable"] for r in rows) * 3 >= len(rows)
