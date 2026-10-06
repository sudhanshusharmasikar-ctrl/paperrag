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
import time

import requests

from .config import (
    GEN_MODE,
    LLM_MIN_INTERVAL,
    LLM_RETRIES,
    MAX_ANSWER_TOKENS,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
)
from .retrieve import Retrieval

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


class LLMError(RuntimeError):
    """The LLM couldn't be reached, or refused the request."""


class LLMAuthError(LLMError):
    """No API key, or a rejected one: retrying won't help."""


_last_call = 0.0  # time.monotonic() when the last request was sent

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


def _reason(resp) -> str:
    """Mistral's own explanation in an error reply. A 429 alone doesn't say
    whether you sent too fast, used up a limit, or the model is full."""
    try:
        body = resp.json()
        message = body.get("message") or body.get("detail") or body
    except (ValueError, AttributeError):
        message = resp.text
    return " ".join(str(message).split())[:200] or "no details"


def _post(payload: dict) -> dict:
    """POST to Mistral, keeping requests LLM_MIN_INTERVAL apart and retrying a
    rate limit (429), a server error (5xx) or a dropped connection with a
    growing wait (1, 2, 4, 8 s, or what the Retry-After header asks for)."""
    global _last_call
    if not MISTRAL_API_KEY:
        raise LLMAuthError("MISTRAL_API_KEY is not set. Put it in .env (see README).")
    problem = ""
    for attempt in range(LLM_RETRIES + 1):
        wait = _last_call + LLM_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        retry_after = ""
        try:
            resp = requests.post(
                MISTRAL_URL,
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
        except requests.RequestException as e:
            problem = f"could not reach Mistral ({e.__class__.__name__})"
        else:
            if resp.status_code in (401, 403):
                raise LLMAuthError(f"Mistral rejected the API key ({resp.status_code}: {_reason(resp)}). "
                                   "Check MISTRAL_API_KEY in .env.")
            if resp.status_code == 429 or resp.status_code >= 500:
                problem = f"Mistral answered {resp.status_code} ({_reason(resp)})"
                retry_after = resp.headers.get("Retry-After", "")
            elif resp.status_code >= 400:  # a mistake in the request: retrying won't help
                raise LLMError(f"Mistral refused the request ({resp.status_code}): {_reason(resp)}")
            else:
                return resp.json()
        if attempt < LLM_RETRIES:
            time.sleep(min(float(retry_after) if retry_after.isdigit() else 2 ** attempt, 30))
    raise LLMError(f"{problem}, still failing after {LLM_RETRIES + 1} tries")


def _mistral(r: Retrieval) -> str:
    reply = _post({
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
    })
    return reply["choices"][0]["message"]["content"].strip()


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
