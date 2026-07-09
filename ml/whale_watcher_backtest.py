"""
Whale Watcher strategy simulation — stealth accumulation with break-of-structure entry.

Spec §2: candidate on volume/price divergence, directional bias from volume_bias,
confirmation on break of event-window high/low after the signal bar, exit via stop/TP/horizon.
Confirmation is a real filter: an opposite-side break invalidates the pending setup, and a
range that never breaks expires at the deadline without a trade.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from ml.backtest import (
    BacktestResult,
    TradeRecord,
    cagr,
    max_drawdown,
    sharpe,
    sortino,
    win_rate,
)
from ml.volume_profile_tracker import (
    DEFAULT_HORIZON,
    DEFAULT_SIGNAL_THRESHOLD,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    EVENT_WINDOW,
    VOLUME_Z_EVENT,
    is_stealth_event,
)

DEFAULT_INITIAL_CAPITAL: Final[float] = 10_000.0
# Bars to wait for a structural break before the pending setup expires unfilled.
CONFIRMATION_DEADLINE_BARS: Final[int] = 15


def _accumulation_bounds(
    high: np.ndarray, low: np.ndarray, signal_bar: int
) -> tuple[float, float]:
    """High/low of the completed accumulation window ending at `signal_bar`."""
    start = max(0, signal_bar - EVENT_WINDOW + 1)
    return float(np.max(high[start : signal_bar + 1])), float(np.min(low[start : signal_bar + 1]))


def run_whale_watcher_backtest(
    features: pd.DataFrame,
    proba: pd.Series,
    *,
    signal_threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
    horizon: int = DEFAULT_HORIZON,
    volume_z_entry: float = VOLUME_Z_EVENT,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> BacktestResult:
    """
    Simulate Whale Watcher over a feature frame and return performance metrics.

    `volume_z_entry` is the stealth-event volume-spike threshold (spec default 2.0).
    """
    frame = features.reset_index(drop=True)
    n = len(frame)
    if n == 0:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, [])

    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    volume_bias = frame["volume_bias"].to_numpy(dtype=float)
    stealth = is_stealth_event(frame, volume_z_event=volume_z_entry).to_numpy()
    proba_arr = proba.reset_index(drop=True).to_numpy(dtype=float)

    equity = initial_capital
    curve: list[float] = []
    trade_returns: list[float] = []
    trades: list[TradeRecord] = []

    position = 0
    entry_price = 0.0
    entry_bar = 0
    bars_held = 0

    pending = False
    pending_bias = 0
    pending_signal_bar = -1
    window_high = 0.0
    window_low = 0.0
    pending_deadline = -1

    for i in range(n):
        if position != 0 and i > 0 and close[i - 1] != 0:
            daily_ret = position * (close[i] - close[i - 1]) / close[i - 1]
            equity *= 1.0 + daily_ret
        curve.append(equity)

        if position != 0:
            bars_held += 1
            trade_ret = position * (close[i] - entry_price) / entry_price
            hit_tp = trade_ret >= take_profit_pct
            hit_sl = trade_ret <= -stop_loss_pct
            time_stop = bars_held >= horizon
            if hit_tp or hit_sl or time_stop:
                trade_returns.append(trade_ret)
                trades.append(TradeRecord(entry_bar=entry_bar, return_pct=float(trade_ret)))
                position = 0
                pending = False

        # Confirmation on bars after the stealth signal: enter only on a break in the
        # bias direction; an opposite-side break invalidates the setup, and an unbroken
        # range expires at the deadline. Entries on the final bar are skipped; they
        # would force-close at the same price and record a meaningless 0-return trade.
        if position == 0 and pending and i > pending_signal_bar:
            can_enter = i < n - 1
            if can_enter and pending_bias == 1 and close[i] >= window_high:
                position = 1
            elif can_enter and pending_bias == -1 and close[i] <= window_low:
                position = -1
            elif (pending_bias == 1 and close[i] <= window_low) or (
                pending_bias == -1 and close[i] >= window_high
            ):
                pending = False
            elif i >= pending_deadline:
                pending = False
            if position != 0:
                entry_price = close[i]
                entry_bar = i
                bars_held = 0
                pending = False

        if (
            position == 0
            and not pending
            and stealth[i]
            and proba_arr[i] >= signal_threshold
            and close[i] > 0
            and i < n - 1
        ):
            pending = True
            pending_bias = 1 if volume_bias[i] >= 0 else -1
            pending_signal_bar = i
            window_high, window_low = _accumulation_bounds(high, low, i)
            wait = min(horizon, CONFIRMATION_DEADLINE_BARS)
            pending_deadline = min(i + wait, n - 1)

    if position != 0:
        trade_ret = position * (close[-1] - entry_price) / entry_price
        trade_returns.append(trade_ret)
        trades.append(TradeRecord(entry_bar=entry_bar, return_pct=float(trade_ret)))

    curve_arr = np.asarray(curve, dtype=float)
    daily_returns = np.diff(curve_arr) / curve_arr[:-1] if curve_arr.size >= 2 else np.array([])

    return BacktestResult(
        sharpe_ratio=sharpe(daily_returns),
        sortino_ratio=sortino(daily_returns),
        cagr=cagr(curve_arr),
        max_drawdown=max_drawdown(curve_arr),
        win_rate=win_rate(trade_returns),
        total_trades=len(trade_returns),
        equity_curve=[float(x) for x in curve_arr],
        trades=trades,
    )
