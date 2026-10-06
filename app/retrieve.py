"""
Retrieval and the abstention guard.

The guard is the part of this project worth talking about. A plain RAG system
always answers: if retrieval returns garbage, the LLM writes a fluent,
confident, wrong answer grounded in nothing. The guard makes "I don't know" a
first-class outcome by refusing when the best match is too weak.

Design decision worth defending: the threshold is applied to the *top-1* score,
not the mean of top-k. Mean scores drift down as k grows, which would make the
threshold silently k-dependent and impossible to sweep cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .config import DISTINCT_PAGES, RETRIEVAL_MODE, RRF_K, SIM_THRESHOLD, TOP_K
from .index import embed, load


@dataclass
class Hit:
    chunk_id: str
    source: str
    page: int
    text: str
    score: float

    def citation(self) -> str:
        return f"{self.source} p.{self.page}"


@dataclass
class Retrieval:
    query: str
    hits: list[Hit]
    top_score: float
    abstain: bool
    threshold: float


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion. Each ranking gives a passage 1 / (k + rank),
    counting ranks from 1, and the shares add up. Only ranks are used, so a
    cosine score never has to be compared with a BM25 score, and a passage
    near the top of both lists beats one at the top of just one."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
    return fused


def one_per_page(order: list[int], chunks: list[dict]) -> list[int]:
    """The ranking with only each page's best chunk. The pages of the plain
    top k all stay, in the same order, and pages further down move up into
    the slots that repeats of a page used to take."""
    seen: set[tuple[str, int]] = set()
    out = []
    for i in order:
        page = (chunks[i]["source"], int(chunks[i]["page"]))
        if page not in seen:
            seen.add(page)
            out.append(i)
    return out


class Retriever:
    def __init__(self) -> None:
        self.index, self.chunks = load()
        # Built in memory at startup: well under a second for thousands of chunks.
        self.bm25 = BM25([c["text"] for c in self.chunks])

    def _hit(self, i: int, score: float) -> Hit:
        c = self.chunks[i]
        return Hit(chunk_id=c["chunk_id"], source=c["source"], page=int(c["page"]),
                   text=c["text"], score=score)

    def __call__(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIM_THRESHOLD,
        mode: str | None = None,
        distinct_pages: bool | None = None,
    ) -> Retrieval:
        mode = mode or RETRIEVAL_MODE
        distinct_pages = DISTINCT_PAGES if distinct_pages is None else distinct_pages
        if mode not in ("dense", "hybrid"):
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if not query.strip():
            return Retrieval(query, [], 0.0, True, threshold)

        # Rank every passage. Exact search over all of them takes milliseconds
        # at this size, and gives each one its cosine score.
        qv = embed([query])
        scores, idxs = self.index.search(qv, self.index.ntotal)
        cosine = {int(i): float(s) for s, i in zip(scores[0], idxs[0]) if i >= 0}
        order = list(cosine)  # best first
        if mode == "hybrid":
            keyword = self.bm25.scores(query)
            keyword_rank = [int(i) for i in np.argsort(-keyword, kind="stable") if keyword[i] > 0]
            fused = rrf([order, keyword_rank])
            order = sorted(fused, key=lambda i: -fused[i])  # ties keep the dense order
        if distinct_pages:
            order = one_per_page(order, self.chunks)
        hits = [self._hit(i, cosine[i]) for i in order[:top_k]]
        # The guard looks at the best embedding score, whatever the mode.
        top = max(cosine.values(), default=0.0)

        return Retrieval(
            query=query,
            hits=hits,
            top_score=top,
            abstain=top < threshold,
            threshold=threshold,
        )

    def scores_only(self, query: str, top_k: int = TOP_K) -> np.ndarray:
        """Used by the threshold sweep so it doesn't rebuild Hit objects."""
        qv = embed([query])
        scores, _ = self.index.search(qv, min(top_k, self.index.ntotal))
        return scores[0]
