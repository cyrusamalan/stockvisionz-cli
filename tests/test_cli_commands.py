from __future__ import annotations

import argparse
from datetime import date
from unittest.mock import MagicMock

import cli.main as main_mod
from cli.dates import preset_range
from cli.main import build_parser, resolve_run_window


def test_preset_range_ytd():
    start, end = preset_range("ytd", today=date(2024, 6, 15))
    assert start == date(2024, 1, 1)
    assert end == date(2024, 6, 15)


def test_preset_range_max_floor():
    start, end = preset_range("max", today=date(2024, 6, 15))
    assert start == date(2014, 1, 1)
    assert end == date(2024, 6, 15)


def test_preset_range_6m():
    start, end = preset_range("6m", today=date(2024, 6, 15))
    assert start == date(2023, 12, 15)
    assert end == date(2024, 6, 15)


def test_resolve_run_window_uses_preset():
    args = argparse.Namespace(preset="1y", start=None, end=None)
    start, end = resolve_run_window(args)
    assert start < end


def test_resolve_run_window_requires_dates_without_preset():
    args = argparse.Namespace(preset=None, start=None, end=date(2024, 1, 1))
    try:
        resolve_run_window(args)
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_build_parser_includes_new_commands():
    parser = build_parser()
    subs = None
    for action in parser._actions:
        if getattr(action, "dest", None) == "cmd":
            subs = action.choices
            break
    assert subs is not None
    for name in ("login", "logout", "version", "config", "ingest", "jobs", "save", "run"):
        assert name in subs


def test_require_token_points_at_login(monkeypatch):
    monkeypatch.setattr(main_mod, "load_credentials", lambda: None)
    try:
        main_mod._require_token()
        raised = False
    except SystemExit as exc:
        raised = True
        assert "stockvisionz login" in str(exc)
    assert raised


def test_cmd_auth_status_not_logged_in(monkeypatch, capsys):
    monkeypatch.setattr(main_mod, "load_credentials", lambda: None)
    assert main_mod.cmd_auth_status(argparse.Namespace()) == 1
    out = capsys.readouterr().out
    assert "not logged in" in out.lower()


def test_cmd_auth_status_logged_in(monkeypatch, capsys):
    monkeypatch.setattr(main_mod, "load_credentials", lambda: MagicMock(token="tok"))
    assert main_mod.cmd_auth_status(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "logged in" in out.lower()


def test_cmd_ingest_polls_job(monkeypatch, capsys):
    monkeypatch.setattr(main_mod, "load_credentials", lambda: MagicMock(token="tok"))
    monkeypatch.setattr(
        main_mod,
        "post",
        lambda path, *, token, json_body=None: MagicMock(
            status=201, json={"jobId": 9, "status": "queued"}
        ),
    )
    monkeypatch.setattr(
        main_mod,
        "_poll_job",
        lambda **k: {"status": "already_available", "lastBarDate": "2024-01-02"},
    )
    args = argparse.Namespace(symbol="AAPL", json=False)
    assert main_mod.cmd_ingest(args) == 0
    out = capsys.readouterr().out
    assert "Ingest ready" in out


def test_cmd_jobs_show_prints_status(monkeypatch, capsys):
    monkeypatch.setattr(main_mod, "load_credentials", lambda: MagicMock(token="tok"))
    monkeypatch.setattr(
        main_mod,
        "get",
        lambda path, *, token, query=None: MagicMock(
            status=200,
            json={"jobId": 42, "status": "queued", "statusMessage": "Waiting"},
        ),
    )
    args = argparse.Namespace(job_id=42, json=False)
    assert main_mod.cmd_jobs_show(args) == 0
    out = capsys.readouterr().out
    assert "Job 42" in out
    assert "queued" in out
