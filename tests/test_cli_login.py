from __future__ import annotations

import argparse
import builtins
from types import SimpleNamespace

import pytest

from cli import main
from cli.http_client import ApiResponse


def test_cmd_login_opens_browser_and_saves_token(monkeypatch):
    started = ApiResponse(
        200,
        {
            "deviceCode": "dev",
            "userCode": "AB-CD",
            "verificationUrl": "https://x/cli/login?code=AB-CD",
            "interval": 0,
            "expiresIn": 600,
        },
    )
    monkeypatch.setattr(main, "post", lambda *a, **k: started)
    monkeypatch.setattr(main, "get", lambda *a, **k: ApiResponse(200, {"accessToken": "svz_new"}))

    opened: dict[str, str] = {}
    monkeypatch.setattr(main.webbrowser, "open", lambda url: opened.setdefault("url", url))
    saved: dict[str, str] = {}
    monkeypatch.setattr(main, "save_token", lambda tok: saved.setdefault("tok", tok))

    assert main.cmd_login(argparse.Namespace()) == 0
    assert saved["tok"] == "svz_new"
    assert opened["url"].endswith("code=AB-CD")


def test_cmd_login_times_out_instead_of_looping(monkeypatch):
    started = ApiResponse(
        200,
        {
            "deviceCode": "dev",
            "userCode": "AB-CD",
            "verificationUrl": "https://x",
            "interval": 0,
            "expiresIn": 0,  # deadline is immediate
        },
    )
    monkeypatch.setattr(main, "post", lambda *a, **k: started)
    monkeypatch.setattr(main, "get", lambda *a, **k: ApiResponse(202, {"status": "authorization_pending"}))
    monkeypatch.setattr(main.webbrowser, "open", lambda url: None)

    with pytest.raises(SystemExit) as exc:
        main.cmd_login(argparse.Namespace())
    assert "timed out" in str(exc.value).lower()


def test_cmd_logout_revokes_then_clears(monkeypatch):
    monkeypatch.setattr(main, "load_credentials", lambda: SimpleNamespace(token="svz_x"))
    calls: dict[str, object] = {}

    def fake_post(path, token=None, **k):  # noqa: ANN001
        calls["path"] = path
        calls["token"] = token
        return ApiResponse(200, {"revoked": True})

    monkeypatch.setattr(main, "post", fake_post)
    cleared: dict[str, bool] = {}
    monkeypatch.setattr(main, "clear_credentials", lambda: cleared.setdefault("done", True))

    assert main.cmd_auth_logout(argparse.Namespace()) == 0
    assert calls["path"] == "/v1/auth/revoke"
    assert calls["token"] == "svz_x"
    assert cleared["done"] is True


def test_cmd_logout_survives_revoke_failure(monkeypatch):
    monkeypatch.setattr(main, "load_credentials", lambda: SimpleNamespace(token="svz_x"))

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(main, "post", boom)
    cleared: dict[str, bool] = {}
    monkeypatch.setattr(main, "clear_credentials", lambda: cleared.setdefault("done", True))

    assert main.cmd_auth_logout(argparse.Namespace()) == 0
    assert cleared["done"] is True


def test_cmd_run_missing_local_deps_is_friendly(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "ml.lab_backtest":
            raise ImportError("No module named 'pandas'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as exc:
        main.cmd_run(argparse.Namespace(symbol="AAPL"))
    assert "stockvisionz-cli" in str(exc.value)
