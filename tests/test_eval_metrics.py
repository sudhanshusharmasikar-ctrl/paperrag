"""The numbers for the README table come from evaluate(), so check its arithmetic."""
from types import SimpleNamespace

import pytest

from app.generate import LLMAuthError, LLMError
from eval.run_eval import changed_pages, cited_pages, evaluate, evaluate_mistral


class ScriptedRetriever:
    """Answers each question with a fixed (abstain, pages) result."""

    def __init__(self, script):
        self.script, self.modes = script, []

    def __call__(self, question, top_k, threshold, **options):
        self.modes.append(options)
        abstain, pages = self.script[question]
        hits = [SimpleNamespace(source="paper.pdf", page=p) for p in pages]
        return SimpleNamespace(abstain=abstain, hits=hits)


def test_metrics_on_a_hand_checked_example():
    expected = [{"source": "paper.pdf", "page": 3}]
    rows = [{"question": f"a{i}", "answerable": True, "expected_pages": expected} for i in range(6)]
    rows += [{"question": f"u{i}", "answerable": False} for i in range(4)]
    script = {
        "a0": (True, []),                                    # answerable, but refused
        "a1": (False, [3, 7]), "a2": (False, [3]), "a3": (False, [3]), "a4": (False, [1, 3]),
        "a5": (False, [9]),                                  # answered from the wrong page
        "u0": (False, [2]),                                  # unanswerable, but answered
        "u1": (True, []), "u2": (True, []), "u3": (True, []),
    }
    m = evaluate(ScriptedRetriever(script), rows, threshold=0.4, top_k=5)
    assert m["unsupported_before_pct"] == 40.0  # with no guard, all 4 unanswerable get answers
    assert m["unsupported_after_pct"] == 10.0   # only u0 gets through: 1 of 10
    assert m["false_refusal_pct"] == 16.7       # a0: 1 of 6 answerable
    assert m["citation_hit_rate_pct"] == 80.0   # right page in 4 of the 5 answered
    assert m["first_hit_rate_pct"] == 60.0      # and first in 3 of them (a4 cites p.1 before p.3)
    assert (m["cited_right_page"], m["cited_right_page_first"], m["answered_checked"]) == (4, 3, 5)
    assert m["pages_cited_avg"] == 1.4          # 2 + 1 + 1 + 2 + 1 different pages, over 5 answers
    assert 0 <= m["latency_ms_p50"] <= m["latency_ms_p95"]  # timed, even if tiny here
    # and it says which questions went wrong
    assert [q for q, _ in m["misses"]["answered_unanswerable"]] == ["u0"]
    assert [q for q, _ in m["misses"]["refused_answerable"]] == ["a0"]
    assert m["misses"]["wrong_page"] == [("a5", [("paper.pdf", 3)], [("paper.pdf", 9)])]


def test_the_retrieval_options_reach_the_retriever():
    rows = [{"question": "a", "answerable": True, "expected_pages": [{"source": "paper.pdf", "page": 1}]}]
    retriever = ScriptedRetriever({"a": (False, [1])})
    options = {"mode": "hybrid", "distinct_pages": True}
    assert evaluate(retriever, rows, threshold=0.4, top_k=5, **options)["options"] == options
    assert retriever.modes == [options]


def test_changed_pages_lists_what_hybrid_fixed_and_what_it_lost():
    def miss(q, want, got):
        return (q, [("paper.pdf", want)], [("paper.pdf", got)])

    dense = {"misses": {"wrong_page": [miss("q1", 1, 2), miss("q2", 3, 4)]}}
    hybrid = {"misses": {"wrong_page": [miss("q2", 3, 5), miss("q3", 6, 7)]}}
    fixed, lost = changed_pages(dense, hybrid)
    assert fixed == ["q1"]                    # right page with hybrid only
    assert lost == [miss("q3", 6, 7)]         # right page with dense only


def test_cited_pages_reads_the_passage_numbers():
    hits = [SimpleNamespace(source="paper.pdf", page=p) for p in (3, 7, 9)]
    assert cited_pages("Adam [1], with warmup [2, 3]; see also [9].", hits) == \
        {("paper.pdf", 3), ("paper.pdf", 7), ("paper.pdf", 9)}
    assert cited_pages("No citation here.", hits) == set()


ROWS = [
    {"question": "a1", "answerable": True, "expected_pages": [{"source": "paper.pdf", "page": 3}]},
    {"question": "a2", "answerable": True, "expected_pages": [{"source": "paper.pdf", "page": 3}]},
    {"question": "a3", "answerable": True, "expected_pages": [{"source": "paper.pdf", "page": 3}]},
    {"question": "a4", "answerable": True, "expected_pages": [{"source": "paper.pdf", "page": 3}]},
    {"question": "u1", "answerable": False},
    {"question": "u2", "answerable": False},
]
GUARD = {"a1": (False, [3, 7]), "a2": (False, [7, 3]), "a3": (False, [3]), "a4": (True, []),
         "u1": (False, [5]), "u2": (False, [5])}


def scripted_answers(replies):
    """Mistral's reply for each question; an exception is raised as if the call failed."""
    def ask(res, mode):
        reply = replies[res.question]
        if isinstance(reply, Exception):
            raise reply
        return {"answer": reply, "abstained": reply == "INSUFFICIENT_CONTEXT"}
    return ask


class GuardRetriever(ScriptedRetriever):
    def __call__(self, question, top_k, threshold, **options):
        res = super().__call__(question, top_k, threshold, **options)
        res.question = question
        return res


def test_mistral_mode_is_scored_on_what_the_score_guard_let_through():
    replies = {"a1": "Adam [1].", "a2": "Adam [1].", "a3": "INSUFFICIENT_CONTEXT",
               "u1": "INSUFFICIENT_CONTEXT", "u2": "Something made up [1]."}
    m = evaluate_mistral(GuardRetriever(GUARD), ROWS, threshold=0.5, top_k=5, ask=scripted_answers(replies))
    assert (m["passed_unanswerable"], m["refused_unanswerable"]) == (2, 1)   # u2 slipped through both guards
    assert (m["passed_answerable"], m["refused_answerable"]) == (3, 1)       # a4 never reached Mistral
    assert (m["answered"], m["cite_a_passage"], m["cite_right_page"]) == (2, 2, 1)  # a2's [1] is p.7
    assert m["no_reply"] == 0


def test_mistral_failing_question_after_question_stops_the_run():
    busy = LLMError("Mistral answered 429 (Rate limit exceeded), still failing after 5 tries")
    replies = {q: busy for q in GUARD}
    with pytest.raises(LLMError, match="no answer to 3 questions in a row.*Rate limit exceeded"):
        evaluate_mistral(GuardRetriever(GUARD), ROWS, threshold=0.5, top_k=5, ask=scripted_answers(replies))


def test_a_rejected_key_stops_the_mistral_run_at_once():
    replies = {q: LLMAuthError("Mistral rejected the API key (401: Unauthorized).") for q in GUARD}
    with pytest.raises(LLMAuthError):
        evaluate_mistral(GuardRetriever(GUARD), ROWS, threshold=0.5, top_k=5, ask=scripted_answers(replies))
