"""The numbers for the README table come from evaluate(), so check its arithmetic."""
from types import SimpleNamespace

from eval.run_eval import evaluate


class ScriptedRetriever:
    """Answers each question with a fixed (abstain, pages) result."""

    def __init__(self, script):
        self.script = script

    def __call__(self, question, top_k, threshold):
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
