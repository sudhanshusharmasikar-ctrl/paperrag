"""
The Mistral client spaces its requests out and retries failures that pass on
their own (a rate limit, a server error, a dropped connection), but stops at
once on a missing or rejected key, and every error says what Mistral said.
Time and the server are fakes here: no request leaves the machine and no test
waits.
"""
import json
import types

import pytest
import requests

from app import generate
from app.config import LLM_MIN_INTERVAL, LLM_RETRIES
from app.retrieve import Hit, Retrieval

HIT = Hit(chunk_id="paper::p4::c0", source="paper.pdf", page=4,
          text="The model is trained with a learning rate of 3e-4.", score=0.71)
QUESTION = Retrieval(query="What learning rate?", hits=[HIT], top_score=0.71, abstain=False, threshold=0.5)


class Reply:
    """The answer on success; otherwise Mistral's error, as JSON unless as_json=False."""

    def __init__(self, status=200, content=None, headers=None, as_json=True):
        self.status_code, self.headers, self.as_json = status, headers or {}, as_json
        if content is None:
            content = "The learning rate is 3e-4 [1]." if status == 200 else f"error {status}"
        if status == 200:
            self.body = {"choices": [{"message": {"content": content}}]}
        else:
            self.body = {"object": "error", "message": content, "type": "error", "code": str(status)}
        self.text = json.dumps(self.body) if as_json else content

    def json(self):
        if not self.as_json:
            raise ValueError("not JSON")
        return self.body


@pytest.fixture
def server(monkeypatch):
    """Scripted replies, fake time; records every request and every sleep."""
    state = types.SimpleNamespace(replies=[], calls=[], sleeps=[], now=1000.0)

    def post(url, headers, json, timeout):
        state.calls.append(json)
        reply = state.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def sleep(seconds):
        state.sleeps.append(seconds)
        state.now += seconds

    monkeypatch.setattr(generate, "requests", types.SimpleNamespace(
        post=post, RequestException=requests.RequestException))
    monkeypatch.setattr(generate, "time", types.SimpleNamespace(sleep=sleep, monotonic=lambda: state.now))
    monkeypatch.setattr(generate, "MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(generate, "_last_call", 0.0)
    return state


def test_an_answer_comes_back_with_its_citations(server):
    server.replies = [Reply()]
    result = generate.answer(QUESTION, mode="mistral")
    assert result["abstained"] is False
    assert result["answer"] == "The learning rate is 3e-4 [1]."
    assert result["citations"][0]["page"] == 4
    assert server.calls[0]["temperature"] == 0.0


def test_insufficient_context_becomes_a_refusal(server):
    server.replies = [Reply(content="INSUFFICIENT_CONTEXT")]
    assert generate.answer(QUESTION, mode="mistral")["abstained"] is True


def test_rate_limits_and_server_errors_are_retried_with_growing_waits(server):
    server.replies = [Reply(429), Reply(503), Reply()]
    generate.answer(QUESTION, mode="mistral")
    assert len(server.calls) == 3
    assert 1 in server.sleeps and 2 in server.sleeps  # 2**0, then 2**1 seconds


def test_the_retry_after_header_is_respected(server):
    server.replies = [Reply(429, headers={"Retry-After": "7"}), Reply()]
    generate.answer(QUESTION, mode="mistral")
    assert 7 in server.sleeps


def test_a_dropped_connection_is_retried(server):
    server.replies = [requests.ConnectionError("reset"), Reply()]
    assert generate.answer(QUESTION, mode="mistral")["abstained"] is False


def test_it_gives_up_after_the_last_try_and_says_what_mistral_said(server):
    server.replies = [Reply(429, content="Rate limit exceeded")] * (LLM_RETRIES + 1)
    with pytest.raises(generate.LLMError,
                       match=rf"429 \(Rate limit exceeded\), still failing after {LLM_RETRIES + 1} tries"):
        generate.answer(QUESTION, mode="mistral")
    assert len(server.calls) == LLM_RETRIES + 1


def test_an_error_page_that_is_not_json_still_gives_a_reason(server):
    server.replies = [Reply(502, content="<html> Bad gateway </html>", as_json=False)] * (LLM_RETRIES + 1)
    with pytest.raises(generate.LLMError, match=r"502 \(<html> Bad gateway </html>\)"):
        generate.answer(QUESTION, mode="mistral")


def test_a_rejected_key_stops_at_once(server):
    server.replies = [Reply(401, content="Unauthorized")]
    with pytest.raises(generate.LLMAuthError, match="rejected the API key"):
        generate.answer(QUESTION, mode="mistral")
    assert len(server.calls) == 1 and server.sleeps == []


def test_a_missing_key_stops_before_any_request(server, monkeypatch):
    monkeypatch.setattr(generate, "MISTRAL_API_KEY", "")
    with pytest.raises(generate.LLMAuthError, match="not set"):
        generate.answer(QUESTION, mode="mistral")
    assert server.calls == []


def test_another_client_error_is_reported_not_retried(server):
    server.replies = [Reply(400, content="Invalid model: mistral-tiny-9000")]
    with pytest.raises(generate.LLMError, match="Invalid model"):
        generate.answer(QUESTION, mode="mistral")
    assert len(server.calls) == 1


def test_back_to_back_requests_are_spaced_out(server):
    server.replies = [Reply(), Reply()]
    generate.answer(QUESTION, mode="mistral")
    generate.answer(QUESTION, mode="mistral")
    assert server.sleeps == [pytest.approx(LLM_MIN_INTERVAL)]
