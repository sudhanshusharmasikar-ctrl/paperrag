"""
Answer generation.

Two modes, and the default needs no API key at all:

  extractive -- return the retrieved spans verbatim with their citations.
                Faithfulness is 100% by construction because nothing is
                generated. Useful as the honest baseline you compare against.

  mistral    -- send the retrieved context to an LLM with a prompt that forbids
                using outside knowledge and requires bracketed citations.

Keeping extractive as the default is deliberate: the deployed demo stays free
and always works, and you have a control condition for the faithfulness
benchmark instead of only measuring the LLM.
"""
from __future__ import annotations

import os
import textwrap

import requests

from .config import (
    GEN_MODE,
    MAX_ANSWER_TOKENS,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
)
from .retrieve import Retrieval

ABSTAIN_MESSAGE = (
    "I don't have enough supporting material in the indexed papers to answer "
    "that. The closest passage scored {score:.3f}, below the {threshold:.3f} "
    "threshold."
)

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You answer questions strictly from the numbered context passages provided.

    Rules:
    - Use only the passages. Never add facts from your own knowledge.
    - Cite the passage number in square brackets after each claim, like [2].
    - If the passages do not contain the answer, reply exactly:
      INSUFFICIENT_CONTEXT
    - Be concise. Three sentences unless the question demands more.
    """
)


def _format_context(r: Retrieval) -> str:
    return "\n\n".join(
        f"[{i}] ({h.citation()}) {h.text}" for i, h in enumerate(r.hits, start=1)
    )


def _extractive(r: Retrieval) -> str:
    parts = []
    for i, h in enumerate(r.hits, start=1):
        parts.append(f"[{i}] {h.citation()} (similarity {h.score:.3f})\n{h.text}")
    return (
        "Extractive mode -- passages returned verbatim, nothing generated:\n\n"
        + "\n\n".join(parts)
    )


def _mistral(r: Retrieval) -> str:
    if not MISTRAL_API_KEY:
        raise RuntimeError("PAPERRAG_GEN_MODE=mistral but MISTRAL_API_KEY is unset.")
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MISTRAL_MODEL,
            "max_tokens": MAX_ANSWER_TOKENS,
            "temperature": 0.0,  # determinism matters for the faithfulness eval
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context passages:\n\n{_format_context(r)}\n\n"
                        f"Question: {r.query}"
                    ),
                },
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def answer(r: Retrieval, mode: str | None = None) -> dict:
    """
    Returns a dict rather than a string so the API, the UI and the eval script
    all consume the same shape.
    """
    mode = mode or GEN_MODE

    if r.abstain:
        return {
            "answer": ABSTAIN_MESSAGE.format(
                score=r.top_score, threshold=r.threshold
            ),
            "abstained": True,
            "mode": mode,
            "top_score": r.top_score,
            "citations": [],
        }

    text = _extractive(r) if mode == "extractive" else _mistral(r)

    # The model is allowed to abstain even when retrieval cleared the bar --
    # the passages may be on-topic but not actually contain the answer. This is
    # the second, independent guard.
    if "INSUFFICIENT_CONTEXT" in text:
        return {
            "answer": "The retrieved passages are related but do not contain "
                      "the answer to that question.",
            "abstained": True,
            "mode": mode,
            "top_score": r.top_score,
            "citations": [],
        }

    return {
        "answer": text,
        "abstained": False,
        "mode": mode,
        "top_score": r.top_score,
        "citations": [
            {
                "n": i,
                "source": h.source,
                "page": h.page,
                "score": round(h.score, 4),
                "chunk_id": h.chunk_id,
            }
            for i, h in enumerate(r.hits, start=1)
        ],
    }
