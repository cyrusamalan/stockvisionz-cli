"""
Reusable backtest + performance metrics for all StockVisionz models.

`run_backtest` turns a feature frame + per-bar reversion probabilities into a daily
mark-to-market equity curve and the exact metric set stored in `performance_metric`
(sharpe_ratio, sortino_ratio, cagr, max_drawdown, win_rate, total_trades, equity_curve).
The metric helpers are standalone so the other 10 models can reuse them unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from ml.mean_reversion_drop import (
    DEFAULT_HORIZON,
    DEFAULT_SIGNAL_THRESHOLD,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    RSI_OVERSOLD,
    is_stretch_event,
)

TRADING_DAYS_PER_YEAR: Final[int] = 252
DEFAULT_VOLUME_Z_ENTRY: Final[float] = 1.5
DEFAULT_INITIAL_CAPITAL: Final[float] = 10_000.0

# Contrarian thesis invalidation (spec §1): if RSI keeps pushing deeper into the extreme
# that triggered entry for this many consecutive bars, the stretch is continuing rather
# than exhausting, so cut the trade early instead of waiting for the price stop. Set <= 0
# to disable the RSI-continuation exit.
DEFAULT_RSI_INVALIDATION_BARS: Final[int] = 2


@dataclass
class TradeRecord:
    """One closed round-trip trade, bar indices relative to the simulated window."""

    entry_bar: int
    return_pct: float
    entry_date: str | None = None


@dataclass
class BacktestResult:
    """One-to-one with the `performance_metric` table columns."""

    sharpe_ratio: float
    sortino_ratio: float
    cagr: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: list[float] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)

    def as_metrics_dict(self) -> dict[str, object]:
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "equity_curve": self.equity_curve,
        }


def sharpe(daily_returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe (risk-free 0). Returns 0.0 when there is no dispersion."""
    r = np.asarray(daily_returns, dtype=float)
    if r.size < 2:
        return 0.0
    std = r.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def sortino(daily_returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sortino: like Sharpe but penalizes only downside deviation."""
    r = np.asarray(daily_returns, dtype=float)
    if r.size < 2:
        return 0.0
    downside = r[r < 0]
    if downside.size == 0:
        return 0.0
    downside_std = np.sqrt(np.mean(np.square(downside)))
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(r.mean() / downside_std * np.sqrt(periods_per_year))


def cagr(equity_curve: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate implied by an equity curve."""
    curve = np.asarray(equity_curve, dtype=float)
    if curve.size < 2 or curve[0] <= 0 or curve[-1] <= 0:
        return 0.0
    years = (curve.size - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((curve[-1] / curve[0]) ** (1.0 / years) - 1.0)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Largest peak-to-trough decline as a negative fraction (0.0 if never underwater)."""
    curve = np.asarray(equity_curve, dtype=float)
    if curve.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(curve)
    drawdowns = curve / running_peak - 1.0
    return float(drawdowns.min())


def win_rate(trade_returns: list[float]) -> float:
    """Fraction of closed trades with a positive return."""
    if not trade_returns:
        return 0.0
    wins = sum(1 for ret in trade_returns if ret > 0)
    return wins / len(trade_returns)


def run_backtest(
    features: pd.DataFrame,
    proba: pd.Series,
    *,
    signal_threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
    volume_z_entry: float = DEFAULT_VOLUME_Z_ENTRY,
    horizon: int = DEFAULT_HORIZON,
    rsi_invalidation_bars: int = DEFAULT_RSI_INVALIDATION_BARS,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> BacktestResult:
    """
    Simulate the Contrarian strategy over a feature frame and return performance metrics.

    Entry (spec §1): on a stretch-event bar where P(reversion) > threshold and
    volume_z > confirmation; long if oversold, short if overbought. Exit at the sma_20
    reversion target, take-profit, stop-loss, a thesis invalidation (RSI pushing deeper
    into the entry extreme for `rsi_invalidation_bars` consecutive bars — set <= 0 to
    disable), or a `horizon`-bar time stop, whichever comes first. Equity is marked to
    market daily (full-capital position sizing) so the curve has one point per bar,
    matching `performance_metric.equity_curve`.
    """
    frame = features.reset_index(drop=True)
    n = len(frame)
    if n == 0:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, [])

    close = frame["close"].to_numpy(dtype=float)
    sma = frame["sma_20"].to_numpy(dtype=float)
    rsi = frame["rsi"].to_numpy(dtype=float)
    volume_z = frame["volume_z"].to_numpy(dtype=float)
    stretch = is_stretch_event(frame).to_numpy()
    proba_arr = proba.reset_index(drop=True).to_numpy(dtype=float)

    equity = initial_capital
    curve: list[float] = []
    trade_returns: list[float] = []
    trades: list[TradeRecord] = []

    position = 0  # 0 flat, +1 long, -1 short
    entry_price = 0.0
    entry_bar = 0
    target = 0.0
    bars_held = 0
    invalidation_streak = 0  # consecutive bars RSI pushed deeper into the entry extreme

    for i in range(n):
        if position != 0 and i > 0 and close[i - 1] != 0:
            daily_ret = position * (close[i] - close[i - 1]) / close[i - 1]
            equity *= 1.0 + daily_ret
        curve.append(equity)

        if position != 0:
            bars_held += 1
            # A long entered oversold is invalidated by RSI falling further; a short
            # entered overbought by RSI rising further. Count consecutive deeper bars.
            if i > 0:
                moved_deeper = (position == 1 and rsi[i] < rsi[i - 1]) or (
                    position == -1 and rsi[i] > rsi[i - 1]
                )
                invalidation_streak = invalidation_streak + 1 if moved_deeper else 0
            trade_ret = position * (close[i] - entry_price) / entry_price
            hit_target = (position == 1 and close[i] >= target) or (
                position == -1 and close[i] <= target
            )
            hit_tp = trade_ret >= take_profit_pct
            hit_sl = trade_ret <= -stop_loss_pct
            hit_invalidation = (
                rsi_invalidation_bars > 0 and invalidation_streak >= rsi_invalidation_bars
            )
            time_stop = bars_held >= horizon
            if hit_target or hit_tp or hit_sl or hit_invalidation or time_stop:
                trade_returns.append(trade_ret)
                trades.append(TradeRecord(entry_bar=entry_bar, return_pct=float(trade_ret)))
                position = 0

        # Skip entries on the final bar: they would force-close at the same price
        # and record a meaningless 0-return trade.
        if (
            position == 0
            and stretch[i]
            and proba_arr[i] >= signal_threshold
            and volume_z[i] > volume_z_entry
            and close[i] > 0
            and i < n - 1
        ):
            position = 1 if rsi[i] < RSI_OVERSOLD else -1
            entry_price = close[i]
            entry_bar = i
            target = sma[i]
            bars_held = 0
            invalidation_streak = 0

    # Force-close an open position at the final bar so win-rate counts it.
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
