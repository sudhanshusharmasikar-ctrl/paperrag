"""The FastAPI app: the health check, /ask, and input validation."""
import pytest
from fastapi.testclient import TestClient

from app import api as api_mod
from app.api import app
from app.generate import LLMError

ON_TOPIC = "Which optimizer, learning rate warmup, label smoothing and dropout did they train with?"
OFF_TOPIC = "What is a good recipe for chocolate cake with butter?"


@pytest.fixture
def client(built_index):
    with TestClient(app) as c:  # "with" runs the startup code that loads the index
        yield c


def test_health_reports_the_loaded_index(client, built_index):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == built_index["chunks"]


def test_ask_answers_with_page_citations(client):
    resp = client.post("/ask", json={"question": ON_TOPIC})
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is False
    assert (body["citations"][0]["source"], body["citations"][0]["page"]) == ("transformers.pdf", 2)
    assert body["latency_ms"] >= 0


def test_ask_refuses_an_off_topic_question(client):
    body = client.post("/ask", json={"question": OFF_TOPIC}).json()
    assert body["abstained"] is True
    assert body["citations"] == []


@pytest.mark.parametrize("payload", [
    {"question": "hi"},                               # shorter than 3 characters
    {"question": "What optimizer?", "top_k": 0},      # top_k must be 1-20
    {"question": "What optimizer?", "mode": "gpt"},   # unknown generation mode
])
def test_ask_rejects_invalid_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_without_an_index_the_api_says_how_to_build_it(missing_index):
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "index_missing"
        resp = c.post("/ask", json={"question": "What optimizer?"})
        assert resp.status_code == 503
        assert "python -m app.index" in resp.json()["detail"]


def test_a_failed_mistral_call_becomes_a_502_that_says_why(client, monkeypatch):
    def broken(r, mode=None):
        raise LLMError("Mistral answered 429 (Rate limit exceeded), still failing after 5 tries")

    monkeypatch.setattr(api_mod, "answer", broken)
    resp = client.post("/ask", json={"question": ON_TOPIC, "mode": "mistral"})
    assert resp.status_code == 502
    assert "Rate limit exceeded" in resp.json()["detail"]
