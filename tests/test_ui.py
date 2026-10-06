"""The web page starts its threshold slider and its generation mode at the
server's settings, since it sends both with every question, and it shows why
a request failed."""
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

UI = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")


def sidebar_after_start(health_check):
    with mock.patch("requests.get", health_check):
        at = AppTest.from_file(UI, default_timeout=30).run()
    assert not at.exception
    return at.sidebar


def threshold_slider(sidebar):
    return next(s for s in sidebar.slider if s.label.startswith("Abstention threshold"))


def test_slider_starts_at_the_servers_threshold():
    reply = mock.Mock(json=lambda: {"status": "ok", "chunks": 905, "threshold": 0.42})
    sidebar = sidebar_after_start(mock.Mock(return_value=reply))
    assert threshold_slider(sidebar).value == 0.42
    assert "905 chunks" in sidebar.success[0].value


def test_slider_falls_back_to_the_default_when_the_api_is_down():
    sidebar = sidebar_after_start(mock.Mock(side_effect=ConnectionError("no server")))
    assert threshold_slider(sidebar).value == 0.50
    assert "unreachable" in sidebar.error[0].value


def test_the_mode_starts_at_the_servers_setting():
    reply = mock.Mock(json=lambda: {"status": "ok", "chunks": 905, "threshold": 0.5, "gen_mode": "mistral"})
    sidebar = sidebar_after_start(mock.Mock(return_value=reply))
    assert sidebar.radio[0].value == "mistral"


def test_a_failed_request_shows_the_reason():
    health = mock.Mock(json=lambda: {"status": "ok", "chunks": 905, "threshold": 0.5, "gen_mode": "mistral"})
    refused = mock.Mock(ok=False, status_code=502,
                        json=lambda: {"detail": "MISTRAL_API_KEY is not set. Put it in .env (see README)."})
    with mock.patch("requests.get", return_value=health), mock.patch("requests.post", return_value=refused):
        at = AppTest.from_file(UI, default_timeout=30).run()
        at.text_input[0].input("What learning rate did they use?")
        at.button[0].click().run()
    assert not at.exception
    assert at.error[0].value == "Request failed (502): MISTRAL_API_KEY is not set. Put it in .env (see README)."
