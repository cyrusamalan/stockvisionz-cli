from __future__ import annotations

import urllib.error

from cli.api_config import DEFAULT_API_URL
from cli import http_client
from cli.http_client import _base_url, get


def test_base_url_defaults_to_deployed_api(monkeypatch):
    monkeypatch.delenv("STOCKVISIONZ_API_URL", raising=False)
    assert _base_url() == DEFAULT_API_URL


def test_base_url_can_be_overridden(monkeypatch):
    monkeypatch.setenv("STOCKVISIONZ_API_URL", "https://example.com/")
    assert _base_url() == "https://example.com"


def test_network_error_returns_friendly_sentinel(monkeypatch):
    """A DNS/connection/timeout failure must not escape as a raw traceback."""

    def boom(req, timeout=None):  # noqa: ANN001
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    res = get("/v1/me", token="svz_x")
    assert res.status == http_client.NETWORK_ERROR_STATUS == 0
    assert isinstance(res.json, dict) and "error" in res.json


def test_request_sets_user_agent(monkeypatch):
    captured: dict[str, str | None] = {}

    class FakeResp:
        status = 200

        def read(self):
            return b'{"ok": true}'

        def readable(self):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *a):  # noqa: ANN002
            return False

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        captured["ua"] = req.headers.get("User-agent")
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    res = get("/v1/me", token="svz_x")
    assert res.status == 200
    assert (captured["ua"] or "").startswith("stockvisionz-cli/")

