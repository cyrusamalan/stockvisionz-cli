"""
Walk-forward validation for Lab backtests (Contrarian + Whale Watcher).

Mode rolling: fixed train window slides forward each step.
Mode anchored: train anchored at window start and expands each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import partial
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from ml.backtest import (
    BacktestResult,
    TradeRecord,
    cagr,
    max_drawdown,
    sharpe,
    sortino,
    run_backtest,
)
from ml.lab_shared import (
    LOOKBACK_DAYS,
    LAB_SIGNAL_THRESHOLD,
    LAB_VOLUME_Z_ENTRY,
    LAB_WHALE_SIGNAL_THRESHOLD,
    LAB_WHALE_VOLUME_Z_ENTRY,
    directional_accuracy,
    entry_funnel_counts,
    feature_date_range,
    normalize_curve,
    price_curve_from_close,
)
from ml.mean_reversion_drop import (
    DEFAULT_HORIZON as CONTRARIAN_HORIZON,
    DEFAULT_STOP_LOSS_PCT as CONTRARIAN_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT as CONTRARIAN_TAKE_PROFIT_PCT,
    build_labels as contrarian_build_labels,
    is_stretch_event,
    train_on_features as contrarian_train_on_features,
)
from ml.volume_profile_tracker import (
    DEFAULT_HORIZON as WHALE_HORIZON,
    DEFAULT_SIGNAL_THRESHOLD as WHALE_SIGNAL_THRESHOLD,
    DEFAULT_STOP_LOSS_PCT as WHALE_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT as WHALE_TAKE_PROFIT_PCT,
    build_labels as whale_build_labels,
    is_stealth_event,
    train_on_features as whale_train_on_features,
)
from ml.whale_watcher_backtest import run_whale_watcher_backtest

ValidationMode = Literal["rolling", "anchored"]

LAB_TRAIN_BARS: Final[int] = 504
LAB_TEST_BARS: Final[int] = 63
LAB_STEP_BARS: Final[int] = 63
LAB_PURGE_BARS: Final[int] = 5


@dataclass(frozen=True)
class FoldSlice:
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class FoldResult:
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_date_start: str | None
    train_date_end: str | None
    test_date_start: str | None
    test_date_end: str | None
    train_bars: int
    test_bars: int
    fallback: bool
    total_trades: int
    sharpe_ratio: float
    sortino_ratio: float
    cagr: float
    max_drawdown: float
    win_rate: float
    directional_accuracy: float | None
    stretch_accuracy_samples: int
    train_labeled_events: int
    funnel: dict[str, Any]
    equity_curve: list[float] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)


def _trade_entry_dates(features: pd.DataFrame, trades: list[TradeRecord]) -> list[TradeRecord]:
    """Attach ISO bar dates to trade records when `bar_date` is present."""
    if not trades or "bar_date" not in features.columns:
        return trades
    dates = pd.to_datetime(features["bar_date"], errors="coerce")
    out: list[TradeRecord] = []
    for trade in trades:
        entry_date = None
        if 0 <= trade.entry_bar < len(dates):
            ts = dates.iloc[trade.entry_bar]
            if pd.notna(ts):
                entry_date = ts.date().isoformat()
        out.append(
            TradeRecord(
                entry_bar=trade.entry_bar,
                return_pct=trade.return_pct,
                entry_date=entry_date,
            )
        )
    return out


def min_bars_for_one_fold(
    *,
    train_bars: int = LAB_TRAIN_BARS,
    test_bars: int = LAB_TEST_BARS,
    purge_bars: int = LAB_PURGE_BARS,
) -> int:
    return train_bars + purge_bars + test_bars


def generate_folds(
    n_bars: int,
    mode: ValidationMode,
    *,
    train_bars: int = LAB_TRAIN_BARS,
    test_bars: int = LAB_TEST_BARS,
    step_bars: int = LAB_STEP_BARS,
    purge_bars: int = LAB_PURGE_BARS,
) -> list[FoldSlice]:
    """Build chronological fold index ranges inside a feature frame of length n_bars."""
    if n_bars < min_bars_for_one_fold(train_bars=train_bars, test_bars=test_bars, purge_bars=purge_bars):
        raise ValueError(
            "Need at least ~2y + 3mo of data for one fold; widen dates or use a longer Max range."
        )

    folds: list[FoldSlice] = []
    fold_index = 0

    if mode == "rolling":
        offset = 0
        while True:
            train_start = offset
            train_end = offset + train_bars
            test_start = train_end + purge_bars
            test_end = test_start + test_bars
            if test_end > n_bars:
                break
            folds.append(
                FoldSlice(
                    fold_index=fold_index,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            fold_index += 1
            offset += step_bars
    elif mode == "anchored":
        k = 0
        while True:
            train_start = 0
            train_end = train_bars + k * step_bars
            test_start = train_end + purge_bars
            test_end = test_start + test_bars
            if test_end > n_bars or train_end >= test_start:
                break
            folds.append(
                FoldSlice(
                    fold_index=fold_index,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            fold_index += 1
            k += 1
    else:
        raise ValueError(f"unknown validation mode: {mode!r}")

    if not folds:
        raise ValueError(
            "Need at least ~2y + 3mo of data for one fold; widen dates or use a longer Max range."
        )
    return folds


def _flat_no_trade_result(n_bars: int) -> BacktestResult:
    """Fallback folds have no trained model, so simulate no trades over a flat curve."""
    return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, [100.0] * n_bars, [])


def _run_single_fold(
    window_features: pd.DataFrame,
    fold: FoldSlice,
    *,
    model_family: str,
    horizon: int,
    signal_threshold: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    volume_z_entry: float,
) -> FoldResult:
    train_features = window_features.iloc[fold.train_start : fold.train_end].reset_index(drop=True)
    test_features = window_features.iloc[fold.test_start : fold.test_end].reset_index(drop=True)

    # A fallback model has no fitted booster/pipeline; its constant base-rate proba is
    # not a trading signal, so fallback folds simulate no trades instead of flipping
    # between all-in and no-trades on whether the base rate clears the threshold.
    if model_family == "lightgbm":
        train_labels = whale_build_labels(train_features, horizon)
        model = whale_train_on_features(train_features, train_labels, horizon=horizon)
        proba = model.predict_proba(test_features)
        if model.is_fallback:
            result = _flat_no_trade_result(len(test_features))
        else:
            result = run_whale_watcher_backtest(
                test_features,
                proba,
                signal_threshold=signal_threshold,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                horizon=horizon,
                volume_z_entry=volume_z_entry,
            )
        event_fn = partial(is_stealth_event, volume_z_event=volume_z_entry)
        # Rows the fit actually used, after _training_mask's stealth-first fallbacks.
        n_train_labeled = model.train_rows
    else:
        train_labels = contrarian_build_labels(train_features, horizon)
        model = contrarian_train_on_features(train_features, train_labels, horizon=horizon)
        proba = model.predict_proba(test_features)
        if model.is_fallback:
            result = _flat_no_trade_result(len(test_features))
        else:
            result = run_backtest(
                test_features,
                proba,
                signal_threshold=signal_threshold,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                horizon=horizon,
                volume_z_entry=volume_z_entry,
            )
        event_fn = is_stretch_event
        n_train_labeled = int(
            (is_stretch_event(train_features) & train_labels.notna()).sum()
        )

    if model.is_fallback:
        accuracy, accuracy_n = None, 0
    else:
        accuracy, accuracy_n = directional_accuracy(
            test_features,
            proba,
            horizon=horizon,
            signal_threshold=signal_threshold,
            model_family=model_family,
            volume_z_event=volume_z_entry if model_family == "lightgbm" else None,
        )

    train_ds, train_de = feature_date_range(train_features)
    test_ds, test_de = feature_date_range(test_features)
    funnel = entry_funnel_counts(
        test_features,
        proba,
        signal_threshold=signal_threshold,
        volume_z_entry=volume_z_entry,
        is_event=event_fn,
        volume_confirms_on_event_bar=model_family != "lightgbm",
    )

    return FoldResult(
        fold_index=fold.fold_index,
        train_start=fold.train_start,
        train_end=fold.train_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
        train_date_start=train_ds,
        train_date_end=train_de,
        test_date_start=test_ds,
        test_date_end=test_de,
        train_bars=len(train_features),
        test_bars=len(test_features),
        fallback=model.is_fallback,
        total_trades=result.total_trades,
        sharpe_ratio=result.sharpe_ratio,
        sortino_ratio=result.sortino_ratio,
        cagr=result.cagr,
        max_drawdown=result.max_drawdown,
        win_rate=result.win_rate,
        directional_accuracy=accuracy,
        stretch_accuracy_samples=accuracy_n,
        train_labeled_events=n_train_labeled,
        funnel=funnel,
        equity_curve=list(result.equity_curve),
        trades=_trade_entry_dates(test_features, list(result.trades)),
    )


def chain_trade_markers(fold_results: list[FoldResult]) -> list[dict[str, Any]]:
    """Map per-fold trade entry bars onto the chained OOS equity curve."""
    markers: list[dict[str, Any]] = []
    global_offset = 0
    for fold_idx, fold in enumerate(fold_results):
        for trade in fold.trades:
            markers.append(
                {
                    "bar_index": global_offset + trade.entry_bar,
                    "won": trade.return_pct > 0,
                    "return_pct": float(trade.return_pct),
                    "entry_date": trade.entry_date,
                }
            )
        n = len(fold.equity_curve)
        if n == 0:
            continue
        if fold_idx == 0:
            global_offset = max(0, n - 1)
        else:
            global_offset += max(0, n - 1)
    return markers


def chain_equity_curves(fold_equities: list[list[float]]) -> list[float]:
    """Chain per-fold test equity curves into one continuous OOS series starting at 100."""
    combined: list[float] = []
    for curve in fold_equities:
        arr = np.asarray(curve, dtype=float)
        if arr.size == 0:
            continue
        if not combined:
            scale = 100.0 / arr[0] if arr[0] else 1.0
            combined = [float(x * scale) for x in arr]
            continue
        prev = combined[-1]
        base = arr[0]
        if base == 0:
            combined.extend([prev] * len(arr))
            continue
        scaled = prev * (arr / base)
        combined.extend(float(x) for x in scaled[1:])
    return combined


def aggregate_fold_results(fold_results: list[FoldResult]) -> BacktestResult:
    """Recompute headline metrics from chained OOS equity and pooled trade stats."""
    chained = chain_equity_curves([f.equity_curve for f in fold_results])
    if len(chained) < 2:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, chained)

    arr = np.asarray(chained, dtype=float)
    daily_returns = np.diff(arr) / arr[:-1]
    total_trades = sum(f.total_trades for f in fold_results)

    # Pool win rate only when trades exist (approximate from per-fold trade counts).
    weighted_wins = sum(f.win_rate * f.total_trades for f in fold_results if f.total_trades > 0)
    pooled_win_rate = weighted_wins / total_trades if total_trades > 0 else 0.0

    return BacktestResult(
        sharpe_ratio=sharpe(daily_returns),
        sortino_ratio=sortino(daily_returns),
        cagr=cagr(arr),
        max_drawdown=max_drawdown(arr),
        win_rate=pooled_win_rate,
        total_trades=total_trades,
        equity_curve=chained,
    )


def pooled_directional_accuracy(fold_results: list[FoldResult]) -> tuple[float | None, int]:
    samples = 0
    correct = 0.0
    for fold in fold_results:
        if fold.directional_accuracy is None or fold.stretch_accuracy_samples == 0:
            continue
        samples += fold.stretch_accuracy_samples
        correct += fold.directional_accuracy * fold.stretch_accuracy_samples
    if samples == 0:
        return None, 0
    return correct / samples, samples


def _fold_to_dict(fold: FoldResult) -> dict[str, Any]:
    return {
        "fold_index": fold.fold_index,
        "train_date_start": fold.train_date_start,
        "train_date_end": fold.train_date_end,
        "test_date_start": fold.test_date_start,
        "test_date_end": fold.test_date_end,
        "train_bars": fold.train_bars,
        "test_bars": fold.test_bars,
        "fallback": fold.fallback,
        "total_trades": fold.total_trades,
        "train_labeled_events": fold.train_labeled_events,
        "entry_signals": fold.funnel.get("entry_signals", 0),
        "test_stretch_events": fold.funnel.get("test_stretch_events", 0),
        "stretch_passes_proba": fold.funnel.get("stretch_passes_proba", 0),
        "stretch_passes_volume": fold.funnel.get("stretch_passes_volume", 0),
        "blocked_by_proba_only": fold.funnel.get("blocked_by_proba_only", 0),
        "blocked_by_volume_only": fold.funnel.get("blocked_by_volume_only", 0),
        "directional_accuracy": fold.directional_accuracy,
        "stretch_accuracy_samples": fold.stretch_accuracy_samples,
    }


def build_walk_forward_report(
    *,
    symbol: str,
    model_family: str,
    validation_mode: ValidationMode,
    window_start: date,
    window_end: date,
    load_start: date,
    bars_loaded: int,
    window_bars: int,
    fold_results: list[FoldResult],
    aggregate: BacktestResult,
    directional_accuracy: float | None,
    stretch_accuracy_samples: int,
    horizon: int,
    signal_threshold: float,
    volume_z_entry: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    purge_bars: int,
) -> dict[str, Any]:
    total_trades = aggregate.total_trades
    fallback_folds = sum(1 for f in fold_results if f.fallback)
    zero_signal_folds = sum(1 for f in fold_results if f.funnel.get("entry_signals", 0) == 0)

    if fallback_folds == len(fold_results):
        status = "fallback"
    elif total_trades == 0:
        status = "no_trades"
    else:
        status = "complete"

    mode_label = "Rolling window" if validation_mode == "rolling" else "Anchored expanding window"
    summary = (
        f"{mode_label}: {len(fold_results)} fold(s), {total_trades} total trade(s) across "
        f"{sum(f.test_bars for f in fold_results)} OOS bars."
    )

    warnings: list[str] = []
    if fallback_folds > 0:
        warnings.append(
            f"{fallback_folds} of {len(fold_results)} fold(s) used baseline fallback training; "
            "fallback folds simulate no trades."
        )
    if total_trades == 0:
        warnings.append(
            "No trades across all OOS folds — return metrics reflect a flat chained equity curve."
        )
    if zero_signal_folds == len(fold_results):
        event_label = "stealth" if model_family == "lightgbm" else "stretch"
        warnings.append(
            f"Every fold had zero entry signals ({event_label} + P(manifest) + volume). "
            "See per-fold funnel in the Folds table."
        )

    timeline = [
        {
            "phase": "data_load",
            "title": "Load market bars",
            "status": "ok",
            "detail": (
                f"Queried market_bar for {symbol.upper()} from {load_start.isoformat()} "
                f"through {window_end.isoformat()} ({LOOKBACK_DAYS}d lookback). "
                f"{window_bars} bars in selected window ({window_start.isoformat()} → {window_end.isoformat()})."
            ),
        },
        {
            "phase": "validation",
            "title": f"Walk-forward schedule ({validation_mode})",
            "status": "ok",
            "detail": (
                f"{mode_label}: train={LAB_TRAIN_BARS} bars (~2y), test={LAB_TEST_BARS} bars (~3mo), "
                f"step={LAB_STEP_BARS} bars, purge={purge_bars} bars between train and test. "
                f"Generated {len(fold_results)} fold(s). "
                f"Per-fold training used only pre-purge bars; test windows are strictly OOS with a "
                f"{purge_bars}-bar purge matching the {horizon}-bar label horizon."
            ),
        },
        {
            "phase": "simulate",
            "title": "Aggregate OOS simulation",
            "status": "warn" if total_trades == 0 else "ok",
            "detail": (
                f"Chained {len(fold_results)} test-window equity curves. "
                f"Sharpe {aggregate.sharpe_ratio:.2f}, CAGR {aggregate.cagr * 100:+.1f}%, "
                f"max drawdown {aggregate.max_drawdown * 100:.1f}%, trades {total_trades}."
                + (
                    # Whale Watcher labels are direction-agnostic (|move| >= threshold),
                    # so its accuracy measures manifestation, not direction.
                    f" Pooled {'manifestation' if model_family == 'lightgbm' else 'directional'} "
                    f"accuracy: {directional_accuracy:.1%} "
                    f"({stretch_accuracy_samples} {'stealth' if model_family == 'lightgbm' else 'stretch'} bars)."
                    if directional_accuracy is not None and stretch_accuracy_samples > 0
                    else ""
                )
            ),
        },
    ]

    model_name = (
        "Volume Profile Tracker (Whale Watcher)"
        if model_family == "lightgbm"
        else "Mean-Reversion Drop (Contrarian)"
    )
    spec_signal = WHALE_SIGNAL_THRESHOLD if model_family == "lightgbm" else 0.7
    spec_volume_z = 2.0 if model_family == "lightgbm" else 1.5

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "python",
        "model_name": model_name,
        "model_family": model_family,
        "symbol": symbol.upper(),
        "status": status,
        "summary": summary,
        "validation_mode": validation_mode,
        "configuration": {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "load_start": load_start.isoformat(),
            "validation_mode": validation_mode,
            "train_window_bars": LAB_TRAIN_BARS,
            "test_window_bars": LAB_TEST_BARS,
            "step_bars": LAB_STEP_BARS,
            "purge_bars": purge_bars,
            "fold_count": len(fold_results),
            "oos_bars": sum(f.test_bars for f in fold_results),
            "horizon_bars": horizon,
            "signal_threshold": signal_threshold,
            "volume_z_entry": volume_z_entry,
            "signal_threshold_spec_default": spec_signal,
            "volume_z_entry_spec_default": spec_volume_z,
            "threshold_note": (
                "Lab uses signal_threshold and volume_z_entry above. "
                f"Build spec defaults are {spec_signal} and {spec_volume_z}."
            ),
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        },
        "counts": {
            "bars_loaded": bars_loaded,
            "window_bars": window_bars,
            "fold_count": len(fold_results),
            "total_trades": total_trades,
            "fallback_folds": fallback_folds,
            "stretch_accuracy_samples": stretch_accuracy_samples,
        },
        "folds": [_fold_to_dict(f) for f in fold_results],
        "timeline": timeline,
        "warnings": warnings,
        "outcome": {
            "fallback": fallback_folds > 0,
            "sharpe_ratio": aggregate.sharpe_ratio,
            "sortino_ratio": aggregate.sortino_ratio,
            "cagr": aggregate.cagr,
            "max_drawdown": aggregate.max_drawdown,
            "win_rate": aggregate.win_rate,
            "directional_accuracy": directional_accuracy,
            "total_trades": total_trades,
        },
    }


def run_walk_forward_lab_backtest(
    window_features: pd.DataFrame,
    *,
    symbol: str,
    model_family: str,
    validation_mode: ValidationMode,
    window_start: date,
    window_end: date,
    load_start: date,
    bars_loaded: int,
    horizon: int | None = None,
    signal_threshold: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    volume_z_entry: float | None = None,
    train_bars: int | None = None,
    test_bars: int | None = None,
    step_bars: int | None = None,
    purge_bars: int | None = None,
) -> dict[str, Any]:
    family = model_family.strip().lower()
    if family == "lightgbm":
        horizon = WHALE_HORIZON if horizon is None else horizon
        signal_threshold = (
            LAB_WHALE_SIGNAL_THRESHOLD if signal_threshold is None else signal_threshold
        )
        stop_loss_pct = WHALE_STOP_LOSS_PCT if stop_loss_pct is None else stop_loss_pct
        take_profit_pct = WHALE_TAKE_PROFIT_PCT if take_profit_pct is None else take_profit_pct
        volume_z_entry = LAB_WHALE_VOLUME_Z_ENTRY if volume_z_entry is None else volume_z_entry
    else:
        horizon = CONTRARIAN_HORIZON if horizon is None else horizon
        signal_threshold = LAB_SIGNAL_THRESHOLD if signal_threshold is None else signal_threshold
        stop_loss_pct = CONTRARIAN_STOP_LOSS_PCT if stop_loss_pct is None else stop_loss_pct
        take_profit_pct = (
            CONTRARIAN_TAKE_PROFIT_PCT if take_profit_pct is None else take_profit_pct
        )
        volume_z_entry = LAB_VOLUME_Z_ENTRY if volume_z_entry is None else volume_z_entry

    train_bars = LAB_TRAIN_BARS if train_bars is None else train_bars
    test_bars = LAB_TEST_BARS if test_bars is None else test_bars
    step_bars = LAB_STEP_BARS if step_bars is None else step_bars
    purge_bars = horizon if purge_bars is None else purge_bars
    folds = generate_folds(
        len(window_features),
        validation_mode,
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        purge_bars=purge_bars,
    )
    fold_results = [
        _run_single_fold(
            window_features,
            fold,
            model_family=family,
            horizon=horizon,
            signal_threshold=signal_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            volume_z_entry=volume_z_entry,
        )
        for fold in folds
    ]

    aggregate = aggregate_fold_results(fold_results)
    accuracy, accuracy_n = pooled_directional_accuracy(fold_results)

    equity_norm = normalize_curve(aggregate.equity_curve)
    trade_markers = chain_trade_markers(fold_results)
    # Price curve: concatenate OOS closes from each fold.
    price_parts: list[float] = []
    for fold in folds:
        test_close = window_features.iloc[fold.test_start : fold.test_end]["close"]
        part = price_curve_from_close(test_close)
        if not price_parts:
            price_parts = part
        elif part:
            scale = price_parts[-1] / part[0] if part[0] else 1.0
            price_parts.extend(float(x * scale) for x in part[1:])

    report = build_walk_forward_report(
        symbol=symbol,
        model_family=model_family,
        validation_mode=validation_mode,
        window_start=window_start,
        window_end=window_end,
        load_start=load_start,
        bars_loaded=bars_loaded,
        window_bars=len(window_features),
        fold_results=fold_results,
        aggregate=aggregate,
        directional_accuracy=accuracy,
        stretch_accuracy_samples=accuracy_n,
        horizon=horizon,
        signal_threshold=signal_threshold,
        volume_z_entry=volume_z_entry,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        purge_bars=purge_bars,
    )

    any_fallback = any(f.fallback for f in fold_results)

    return {
        "symbol": symbol.upper(),
        "model_family": model_family,
        "validation_mode": validation_mode,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "fold_count": len(fold_results),
        "test_bars": sum(f.test_bars for f in fold_results),
        "fallback": any_fallback,
        "sharpe_ratio": aggregate.sharpe_ratio,
        "sortino_ratio": aggregate.sortino_ratio,
        "cagr": aggregate.cagr,
        "max_drawdown": aggregate.max_drawdown,
        "win_rate": aggregate.win_rate,
        "directional_accuracy": accuracy,
        "stretch_accuracy_samples": accuracy_n,
        "total_trades": aggregate.total_trades,
        "equity_curve": equity_norm,
        "price_curve": price_parts,
        "trade_markers": trade_markers,
        "folds": [_fold_to_dict(f) for f in fold_results],
        "report": report,
    }
