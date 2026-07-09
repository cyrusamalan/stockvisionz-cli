"""Regression tests for CLI run save prompt and post-run behavior."""

from __future__ import annotations

import argparse
from datetime import date
from unittest.mock import MagicMock

import cli.main as main_mod
from cli.prompts import confirm_save, should_auto_save, should_prompt_save


def test_confirm_save_accepts_y_and_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert confirm_save() is True
    monkeypatch.setattr("builtins.input", lambda _: "YES")
    assert confirm_save() is True


def test_confirm_save_defaults_no(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert confirm_save() is False
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert confirm_save() is False


def test_should_prompt_save_respects_flags(monkeypatch):
    monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: True))
    assert should_prompt_save(auto_save=False, no_save=False, json_mode=False) is True
    assert should_prompt_save(auto_save=True, no_save=False, json_mode=False) is False
    assert should_prompt_save(auto_save=False, no_save=True, json_mode=False) is False
    assert should_prompt_save(auto_save=False, no_save=False, json_mode=True) is False


def test_should_auto_save_only_when_flag_set():
    assert should_auto_save(auto_save=True, no_save=False, json_mode=False) is True
    assert should_auto_save(auto_save=False, no_save=False, json_mode=False) is False
    assert should_auto_save(auto_save=True, no_save=False, json_mode=True) is False


def test_cmd_run_prints_summary_and_skips_save_with_no_save(monkeypatch, capsys):
    sample_result = {
        "symbol": "AAPL",
        "model_family": "logistic_regression",
        "validation_mode": "rolling",
        "window_start": "2023-01-27",
        "window_end": "2023-10-21",
        "sharpe_ratio": 1.0,
        "sortino_ratio": 1.0,
        "cagr": 0.1,
        "max_drawdown": 0.05,
        "win_rate": 0.5,
        "total_trades": 5,
        "fold_count": 1,
        "test_bars": 63,
        "fallback": False,
    }

    def fake_post(path, *, token, json_body=None):  # noqa: ANN001
        resp = MagicMock()
        if path == "/v1/market-ingest":
            resp.status = 201
            resp.json = {"jobId": 1, "status": "queued"}
        elif path.endswith("/complete"):
            resp.status = 200
            resp.json = {"jobId": 42, "status": "complete"}
        else:
            resp.status = 202
            resp.json = {"jobId": 42, "status": "queued"}
        return resp

    def fake_get(path, *, token, query=None):  # noqa: ANN001
        resp = MagicMock()
        if path.startswith("/v1/market-ingest/"):
            resp.status = 200
            resp.json = {"status": "already_available", "lastBarDate": "2023-10-21"}
        elif path == "/v1/lab/backtest/data":
            resp.status = 200
            resp.json = {
                "bars": [
                    {
                        "barDate": "2023-01-27",
                        "open": 1,
                        "high": 1,
                        "low": 1,
                        "close": 1,
                        "volume": 100,
                    }
                ]
            }
        else:
            resp.status = 200
            resp.json = {"status": "complete"}
        return resp

    monkeypatch.setattr(main_mod, "load_credentials", lambda: MagicMock(token="tok"))
    monkeypatch.setattr(main_mod, "post", fake_post)
    monkeypatch.setattr(main_mod, "get", fake_get)
    monkeypatch.setattr(main_mod, "_poll_job", lambda **k: {"status": "already_available"})
    monkeypatch.setattr(
        "ml.lab_backtest.run_lab_backtest",
        lambda *a, **k: sample_result,
    )

    args = argparse.Namespace(
        symbol="AAPL",
        start=date(2023, 1, 27),
        end=date(2023, 10, 21),
        model_family="logistic_regression",
        validation_mode="rolling",
        save=False,
        no_save=True,
        json=False,
        name=None,
        preset=None,
    )
    assert main_mod.cmd_run(args) == 0
    out = capsys.readouterr().out
    assert "Backtest complete — AAPL" in out
    assert "Save this run" not in out


def test_cmd_run_json_mode_emits_json(monkeypatch, capsys):
    sample_result = {
        "symbol": "AAPL",
        "model_family": "logistic_regression",
        "validation_mode": "rolling",
        "window_start": "2023-01-27",
        "window_end": "2023-10-21",
        "sharpe_ratio": 1.0,
        "sortino_ratio": 1.0,
        "cagr": 0.1,
        "max_drawdown": 0.05,
        "win_rate": 0.5,
        "total_trades": 5,
        "fold_count": 1,
        "test_bars": 63,
        "fallback": False,
    }

    def fake_post(path, *, token, json_body=None):  # noqa: ANN001
        resp = MagicMock()
        if path.endswith("/complete"):
            resp.status = 200
        elif path == "/v1/lab/backtest":
            resp.status = 202
        else:
            resp.status = 201
        resp.json = {"jobId": 42, "status": "queued"}
        return resp

    def fake_get(path, *, token, query=None):  # noqa: ANN001
        resp = MagicMock()
        if path.endswith("/data"):
            resp.status = 200
            resp.json = {
                "bars": [
                    {
                        "barDate": "2023-01-27",
                        "open": 1,
                        "high": 1,
                        "low": 1,
                        "close": 1,
                        "volume": 100,
                    }
                ]
            }
        else:
            resp.status = 200
            resp.json = {"status": "already_available"}
        return resp

    monkeypatch.setattr(main_mod, "load_credentials", lambda: MagicMock(token="tok"))
    monkeypatch.setattr(main_mod, "post", fake_post)
    monkeypatch.setattr(main_mod, "get", fake_get)
    monkeypatch.setattr(main_mod, "_poll_job", lambda **k: {"status": "already_available"})
    monkeypatch.setattr("ml.lab_backtest.run_lab_backtest", lambda *a, **k: sample_result)

    args = argparse.Namespace(
        symbol="AAPL",
        start=date(2023, 1, 27),
        end=date(2023, 10, 21),
        model_family="logistic_regression",
        validation_mode="rolling",
        save=False,
        no_save=False,
        json=True,
        name=None,
        preset=None,
    )
    assert main_mod.cmd_run(args) == 0
    out = capsys.readouterr().out
    assert '"jobId": 42' in out
    assert '"result"' in out
    assert "Backtest complete" not in out
