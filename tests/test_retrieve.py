"""Retrieval and the abstention guard, on the two tiny papers in conftest.py."""
import pytest

from app import retrieve as retrieve_mod
from app.retrieve import Retriever

ON_TOPIC = "Which optimizer, learning rate warmup, label smoothing and dropout did they train with?"
OFF_TOPIC = "What is a good recipe for chocolate cake with butter?"


def test_best_matching_page_comes_first(built_index):
    r = Retriever()(ON_TOPIC)
    assert not r.abstain
    assert r.hits[0].citation() == "transformers.pdf p.2"
    scores = [h.score for h in r.hits]
    assert scores == sorted(scores, reverse=True)


def test_off_topic_question_is_refused(built_index):
    r = Retriever()(OFF_TOPIC)
    assert r.abstain
    assert r.top_score < r.threshold


def test_guard_compares_the_top_score_with_the_threshold(built_index):
    retriever = Retriever()
    top = retriever(ON_TOPIC).top_score
    assert retriever(ON_TOPIC, threshold=top + 0.01).abstain
    assert not retriever(ON_TOPIC, threshold=top - 0.01).abstain


def test_blank_question_is_refused_without_searching(built_index, monkeypatch):
    retriever = Retriever()
    monkeypatch.setattr(retrieve_mod, "embed", lambda texts: pytest.fail("embedded a blank question"))
    r = retriever("   ")
    assert r.abstain and r.hits == [] and r.top_score == 0.0


def test_top_k_larger_than_the_index_is_safe(built_index):
    assert len(Retriever()(ON_TOPIC, top_k=50).hits) == built_index["chunks"]
