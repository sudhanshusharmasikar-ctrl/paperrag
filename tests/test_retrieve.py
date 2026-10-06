"""Retrieval and the abstention guard, on the two tiny papers in conftest.py."""
import numpy as np
import pytest

from app import index as index_mod
from app import retrieve as retrieve_mod
from app.retrieve import Retriever, one_per_page, rrf
from conftest import write_pdf

ON_TOPIC = "Which optimizer, learning rate warmup, label smoothing and dropout did they train with?"
OFF_TOPIC = "What is a good recipe for chocolate cake with butter?"


def test_best_matching_page_comes_first(built_index):
    r = Retriever()(ON_TOPIC, mode="dense")  # dense ranks by similarity alone
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


def test_rrf_adds_up_reciprocal_ranks():
    fused = rrf([[7, 8, 9], [9, 7]], k=60)
    assert fused == pytest.approx({7: 1 / 61 + 1 / 62, 8: 1 / 62, 9: 1 / 63 + 1 / 61})
    assert sorted(fused, key=lambda i: -fused[i]) == [7, 9, 8]  # near the top of both lists wins


@pytest.mark.parametrize("question", [ON_TOPIC, OFF_TOPIC])
def test_hybrid_refuses_exactly_what_dense_refuses(built_index, question):
    retriever = Retriever()
    dense, hybrid = retriever(question, mode="dense"), retriever(question, mode="hybrid")
    assert hybrid.abstain == dense.abstain
    assert hybrid.top_score == pytest.approx(dense.top_score)


def test_hybrid_brings_in_a_keyword_match_that_dense_ranks_lower(built_index, monkeypatch):
    retriever = Retriever()
    dense = retriever(ON_TOPIC, top_k=1, mode="dense")
    assert dense.hits[0].citation() == "transformers.pdf p.2"
    # Pretend only the retrieval paper's page has the question's rare words.
    target = next(i for i, c in enumerate(retriever.chunks) if c["source"] == "retrieval.pdf")
    keyword = np.zeros(len(retriever.chunks))
    keyword[target] = 5.0
    monkeypatch.setattr(retriever.bm25, "scores", lambda question: keyword)
    hybrid = retriever(ON_TOPIC, top_k=1, mode="hybrid")
    assert hybrid.hits[0].citation() == "retrieval.pdf p.1"
    # The hit keeps its own embedding score; the guard still uses the best one.
    assert hybrid.hits[0].score < dense.hits[0].score == pytest.approx(hybrid.top_score)


def test_hybrid_returns_each_passage_once(built_index):
    hits = Retriever()(ON_TOPIC, top_k=50, mode="hybrid").hits
    assert len(hits) == built_index["chunks"] == len({h.chunk_id for h in hits})


def test_an_unknown_mode_is_an_error(built_index):
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        Retriever()(ON_TOPIC, mode="sparse")


def chunk(source, page):
    return {"source": source, "page": page}


def test_one_per_page_keeps_each_pages_best_chunk():
    chunks = [chunk("a.pdf", 1), chunk("a.pdf", 1), chunk("a.pdf", 2), chunk("b.pdf", 1), chunk("a.pdf", 2)]
    assert one_per_page([1, 0, 4, 3, 2], chunks) == [1, 4, 3]


def test_one_per_page_never_loses_a_page_the_plain_ranking_cited():
    # The top k pages of the plain ranking are always among the top k after it,
    # so the right page can be gained but never lost, and the first stays first.
    rng = np.random.default_rng(7)
    chunks = [chunk(f"p{rng.integers(3)}.pdf", int(rng.integers(1, 4))) for _ in range(40)]
    for _ in range(200):
        order = [int(i) for i in rng.permutation(len(chunks))]
        k = int(rng.integers(1, 8))
        pages = lambda ranking: [(chunks[i]["source"], chunks[i]["page"]) for i in ranking[:k]]
        after = one_per_page(order, chunks)
        assert set(pages(order)) <= set(pages(after))
        assert after[0] == order[0]


@pytest.fixture
def long_page_index(tmp_path, monkeypatch):
    """One paper whose page 1 is long enough for two chunks, both about the
    optimizer; page 2 says a little about it too."""
    para = ("The optimizer is Adam with a learning rate warmup, and the optimizer settings "
            "follow the warmup schedule closely. ") * 4
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    write_pdf(pdfs / "long.pdf", [[para, para.replace("Adam", "AdamW")],
                                  ["The batch size, the optimizer and the warmup are listed in a table of "
                                   "training settings, next to the dropout rate and the number of steps."]])
    monkeypatch.setattr(index_mod, "INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr(index_mod, "CHUNKS_PATH", tmp_path / "chunks.jsonl")
    return index_mod.build(pdfs)


def test_one_chunk_per_page_cites_another_page_instead_of_a_repeat(long_page_index):
    retriever = Retriever()
    pages = [c["page"] for c in retriever.chunks]
    assert pages.count(1) >= 2, "page 1 should have been cut into two chunks"
    question = "Which optimizer and learning rate warmup schedule did they use?"
    plain = retriever(question, top_k=2, mode="dense", distinct_pages=False)
    spread = retriever(question, top_k=2, mode="dense", distinct_pages=True)
    assert [h.page for h in plain.hits] == [1, 1]
    assert [h.page for h in spread.hits] == [1, 2]
    assert spread.hits[0].chunk_id == plain.hits[0].chunk_id  # the first citation doesn't change
