"""Turning a retrieval result into the answer dict shared by the API, UI and eval."""
import pytest

from app import generate
from app.retrieve import Hit, Retrieval

HIT = Hit(chunk_id="paper::p4::c0", source="paper.pdf", page=4,
          text="The model is trained with a learning rate of 3e-4.", score=0.71)


def retrieval(abstain=False):
    return Retrieval(query="What learning rate?", hits=[HIT], top_score=HIT.score,
                     abstain=abstain, threshold=0.35)


def test_refusal_has_no_citations_and_shows_both_scores():
    result = generate.answer(retrieval(abstain=True), mode="extractive")
    assert result["abstained"] is True
    assert result["citations"] == []
    assert "0.710" in result["answer"] and "0.350" in result["answer"]


def test_extractive_answer_quotes_the_passage_with_its_page():
    result = generate.answer(retrieval(), mode="extractive")
    assert result["abstained"] is False
    assert "[1] paper.pdf p.4" in result["answer"]
    assert HIT.text in result["answer"]
    assert result["citations"] == [
        {"n": 1, "source": "paper.pdf", "page": 4, "score": 0.71, "chunk_id": "paper::p4::c0"}
    ]


def test_model_saying_insufficient_context_becomes_a_refusal(monkeypatch):
    monkeypatch.setattr(generate, "_mistral", lambda r: "INSUFFICIENT_CONTEXT")
    result = generate.answer(retrieval(), mode="mistral")
    assert result["abstained"] is True
    assert result["citations"] == []


def test_mistral_mode_without_a_key_fails_before_any_request(monkeypatch):
    monkeypatch.setattr(generate, "MISTRAL_API_KEY", "")
    monkeypatch.setattr(generate.requests, "post",
                        lambda *args, **kwargs: pytest.fail("sent a request without a key"))
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        generate.answer(retrieval(), mode="mistral")
