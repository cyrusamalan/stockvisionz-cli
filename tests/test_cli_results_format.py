"""Regression tests for CLI backtest result formatting."""

from __future__ import annotations

from cli.results import format_backtest_summary, format_job_list


def _sample_payload(**overrides) -> dict:
    base = {
        "symbol": "AAPL",
        "model_family": "logistic_regression",
        "validation_mode": "rolling",
        "window_start": "2023-01-27",
        "window_end": "2023-10-21",
        "sharpe_ratio": 1.42,
        "sortino_ratio": 1.89,
        "cagr": 0.123,
        "max_drawdown": 0.084,
        "win_rate": 0.542,
        "total_trades": 38,
        "fold_count": 3,
        "test_bars": 189,
        "directional_accuracy": 0.581,
        "fallback": False,
    }
    base.update(overrides)
    return base


def test_format_backtest_summary_happy_path():
    text = format_backtest_summary(_sample_payload(), job_id=42)
    assert "AAPL" in text and "logistic_regression" in text and "rolling" in text
    assert "2023-01-27" in text and "2023-10-21" in text
    assert "Sharpe        1.42" in text
    assert "Sortino       1.89" in text
    assert "+12.3%" in text
    assert "Max Drawdown  8.4%" in text
    assert "Win Rate      54.2%" in text
    assert "Trades        38" in text
    assert "Folds: 3" in text and "Test bars: 189" in text
    assert "Directional accuracy: 58.1%" in text
    assert "Job #42" in text
    assert "dashboard/lab" in text


def test_format_backtest_summary_shows_fallback_warning():
    text = format_backtest_summary(_sample_payload(fallback=True), job_id=1)
    assert "Warning: one or more folds used fallback training" in text


def test_format_backtest_summary_handles_missing_optional_fields():
    text = format_backtest_summary(
        _sample_payload(
            win_rate=None,
            total_trades=0,
            directional_accuracy=None,
        ),
        job_id=7,
    )
    assert "Win Rate      —" in text
    assert "Trades        0" in text
    assert "Directional accuracy" not in text


def test_format_job_list_renders_rows():
    text = format_job_list(
        [
            {
                "jobId": 42,
                "symbol": "aapl",
                "modelFamily": "lightgbm",
                "status": "complete",
                "createdAt": "2026-07-01T12:00:00+00:00",
            }
        ]
    )
    assert "42" in text
    assert "AAPL" in text  # symbol upper-cased
    assert "lightgbm" in text
    assert "complete" in text
    assert "2026-07-01" in text  # ISO timestamp trimmed to date


def test_format_job_list_empty():
    text = format_job_list([])
    assert "No backtest jobs yet" in text
