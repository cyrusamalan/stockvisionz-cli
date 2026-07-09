from __future__ import annotations

import os
from typing import Any


def default_web_url() -> str:
    return (os.environ.get("STOCKVISIONZ_WEB_URL") or "https://stockvisionz.com").rstrip("/")


def _fmt_pct(value: float, *, signed: bool = False) -> str:
    pct = value * 100
    if signed:
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"
    return f"{pct:.1f}%"


def _short_date(value: Any) -> str:
    """Trim an ISO timestamp to YYYY-MM-DD for compact listing."""
    text = str(value or "")
    return text[:10] if text else "—"


def format_job_list(jobs: list[dict[str, Any]]) -> str:
    """Render a compact table of recent backtest jobs for terminal output."""
    if not jobs:
        return "No backtest jobs yet. Run: stockvisionz run --symbol AAPL --preset 1y"

    header = f"  {'JOB':<8}{'SYMBOL':<10}{'FAMILY':<22}{'STATUS':<14}{'CREATED':<12}"
    lines = ["", header, "  " + "-" * (len(header) - 2)]
    for job in jobs:
        job_id = str(job.get("jobId", "?"))
        symbol = str(job.get("symbol") or "?").upper()
        family = str(job.get("modelFamily") or "unknown")
        status = str(job.get("status") or "unknown")
        created = _short_date(job.get("createdAt"))
        lines.append(f"  {job_id:<8}{symbol:<10}{family:<22}{status:<14}{created:<12}")
    lines.append("")
    return "\n".join(lines)


def format_backtest_summary(payload: dict[str, Any], *, job_id: int) -> str:
    """Render a human-readable backtest summary for terminal output."""
    symbol = str(payload.get("symbol") or "?").upper()
    family = str(payload.get("model_family") or "unknown")
    mode = str(payload.get("validation_mode") or "rolling")
    window_start = str(payload.get("window_start") or "?")
    window_end = str(payload.get("window_end") or "?")

    sharpe = float(payload.get("sharpe_ratio") or 0.0)
    sortino = float(payload.get("sortino_ratio") or 0.0)
    cagr = float(payload.get("cagr") or 0.0)
    max_dd = float(payload.get("max_drawdown") or 0.0)
    win_rate = payload.get("win_rate")
    total_trades = int(payload.get("total_trades") or 0)
    fold_count = int(payload.get("fold_count") or 0)
    test_bars = int(payload.get("test_bars") or 0)
    fallback = bool(payload.get("fallback"))
    directional_accuracy = payload.get("directional_accuracy")

    lines = [
        "",
        f"Backtest complete — {symbol} · {family} · {mode}",
        f"Window: {window_start} → {window_end}",
        "",
        f"  {'Sharpe':<14}{sharpe:.2f}",
        f"  {'Sortino':<14}{sortino:.2f}",
        f"  {'CAGR':<14}{_fmt_pct(cagr, signed=True)}",
        f"  {'Max Drawdown':<14}{_fmt_pct(max_dd)}",
    ]

    if win_rate is not None and total_trades > 0:
        lines.append(f"  {'Win Rate':<14}{_fmt_pct(float(win_rate))}")
    else:
        lines.append(f"  {'Win Rate':<14}—")

    lines.append(f"  {'Trades':<14}{total_trades}")
    lines.append("")

    meta_parts = [f"Folds: {fold_count}", f"Test bars: {test_bars}"]
    if directional_accuracy is not None:
        meta_parts.append(f"Directional accuracy: {float(directional_accuracy) * 100:.1f}%")
    lines.append(f"  {' · '.join(meta_parts)}")

    if fallback:
        lines.append("")
        lines.append("  Warning: one or more folds used fallback training (insufficient labeled events).")

    web_url = default_web_url()
    lines.append("")
    lines.append(f"  Job #{job_id} · View in Lab: {web_url}/dashboard/lab")
    lines.append("")

    return "\n".join(lines)
