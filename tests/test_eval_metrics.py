"""The numbers for the README table come from evaluate(), so check its arithmetic."""
from types import SimpleNamespace

from eval.run_eval import changed_pages, evaluate


class ScriptedRetriever:
    """Answers each question with a fixed (abstain, pages) result."""

    def __init__(self, script):
        self.script, self.modes = script, []

    def __call__(self, question, top_k, threshold, mode=None):
        self.modes.append(mode)
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
    assert 0 <= m["latency_ms_p50"] <= m["latency_ms_p95"]  # timed, even if tiny here
    # and it says which questions went wrong
    assert [q for q, _ in m["misses"]["answered_unanswerable"]] == ["u0"]
    assert [q for q, _ in m["misses"]["refused_answerable"]] == ["a0"]
    assert m["misses"]["wrong_page"] == [("a5", [("paper.pdf", 3)], [("paper.pdf", 9)])]


def test_the_retrieval_mode_reaches_the_retriever():
    rows = [{"question": "a", "answerable": True, "expected_pages": [{"source": "paper.pdf", "page": 1}]}]
    retriever = ScriptedRetriever({"a": (False, [1])})
    assert evaluate(retriever, rows, threshold=0.4, top_k=5, mode="hybrid")["mode"] == "hybrid"
    assert retriever.modes == ["hybrid"]


def test_changed_pages_lists_what_hybrid_fixed_and_what_it_lost():
    def miss(q, want, got):
        return (q, [("paper.pdf", want)], [("paper.pdf", got)])

    dense = {"misses": {"wrong_page": [miss("q1", 1, 2), miss("q2", 3, 4)]}}
    hybrid = {"misses": {"wrong_page": [miss("q2", 3, 5), miss("q3", 6, 7)]}}
    fixed, lost = changed_pages(dense, hybrid)
    assert fixed == ["q1"]                    # right page with hybrid only
    assert lost == [miss("q3", 6, 7)]         # right page with dense only
