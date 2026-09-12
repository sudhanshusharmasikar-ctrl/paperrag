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

from .config import SIM_THRESHOLD, TOP_K
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


class Retriever:
    def __init__(self) -> None:
        self.index, self.chunks = load()

    def __call__(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIM_THRESHOLD,
    ) -> Retrieval:
        if not query.strip():
            return Retrieval(query, [], 0.0, True, threshold)

        qv = embed([query])
        scores, idxs = self.index.search(qv, min(top_k, self.index.ntotal))
        scores, idxs = scores[0], idxs[0]

        hits: list[Hit] = []
        for score, i in zip(scores, idxs):
            if i < 0:
                continue
            c = self.chunks[int(i)]
            hits.append(
                Hit(
                    chunk_id=c["chunk_id"],
                    source=c["source"],
                    page=int(c["page"]),
                    text=c["text"],
                    score=float(score),
                )
            )

        top = hits[0].score if hits else 0.0
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
