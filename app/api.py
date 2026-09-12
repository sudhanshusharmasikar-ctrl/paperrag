"""
FastAPI backend.

Run:  uvicorn app.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import GEN_MODE, SIM_THRESHOLD, TOP_K
from .generate import answer
from .retrieve import Retriever

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and index once at startup, not per request.
    try:
        _state["retriever"] = Retriever()
    except FileNotFoundError as e:
        _state["error"] = str(e)
    yield
    _state.clear()


app = FastAPI(
    title="PaperRAG",
    description="Citation-grounded retrieval over research papers.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(TOP_K, ge=1, le=20)
    threshold: float = Field(SIM_THRESHOLD, ge=-1.0, le=1.0)
    mode: str | None = Field(None, pattern="^(extractive|mistral)$")


class Citation(BaseModel):
    n: int
    source: str
    page: int
    score: float
    chunk_id: str


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    top_score: float
    threshold: float
    mode: str
    citations: list[Citation]
    latency_ms: float


def _retriever() -> Retriever:
    if "retriever" not in _state:
        raise HTTPException(503, _state.get("error", "Index not loaded."))
    return _state["retriever"]


@app.get("/health")
def health() -> dict:
    ready = "retriever" in _state
    return {
        "status": "ok" if ready else "index_missing",
        "chunks": len(_state["retriever"].chunks) if ready else 0,
        "gen_mode": GEN_MODE,
        "threshold": SIM_THRESHOLD,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    r = _retriever()(req.question, top_k=req.top_k, threshold=req.threshold)
    result = answer(r, mode=req.mode)
    return AskResponse(
        question=req.question,
        answer=result["answer"],
        abstained=result["abstained"],
        top_score=round(r.top_score, 4),
        threshold=req.threshold,
        mode=result["mode"],
        citations=result["citations"],
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
