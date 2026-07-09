"""Shared helpers for Lab backtest and walk-forward modules."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

import pandas as pd

from ml.mean_reversion_drop import build_labels as contrarian_build_labels
from ml.mean_reversion_drop import is_stretch_event

LOOKBACK_DAYS = 120
LAB_SIGNAL_THRESHOLD: float = 0.50
LAB_VOLUME_Z_ENTRY: float = 0.75

# Whale Watcher Lab defaults. volume_z_entry is the stealth-event spike threshold
# (spec §2 default 2.0); lowering it flags more/looser accumulation events.
LAB_WHALE_SIGNAL_THRESHOLD: float = 0.50
LAB_WHALE_VOLUME_Z_ENTRY: float = 2.0


def bar_dates(frame: pd.DataFrame) -> pd.Series:
    if "bar_date" in frame.columns:
        return pd.to_datetime(frame["bar_date"]).dt.date
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"]).dt.date
    raise ValueError("bars frame has no bar_date/timestamp column")


def slice_by_date(
    frame: pd.DataFrame, dates: pd.Series, start: date | None, end: date | None
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    return frame.loc[mask].reset_index(drop=True)


def contrarian_directional_accuracy(
    test_features: pd.DataFrame,
    proba: pd.Series,
    *,
    horizon: int,
    signal_threshold: float,
) -> tuple[float | None, int]:
    labels = contrarian_build_labels(test_features, horizon)
    mask = is_stretch_event(test_features) & labels.notna()
    n = int(mask.sum())
    if n == 0:
        return None, 0
    y_true = labels.loc[mask].astype(int)
    y_pred = (proba.loc[mask] >= signal_threshold).astype(int)
    return float((y_true.to_numpy() == y_pred.to_numpy()).mean()), n


def whale_watcher_directional_accuracy(
    test_features: pd.DataFrame,
    proba: pd.Series,
    *,
    horizon: int,
    signal_threshold: float,
    volume_z_event: float | None = None,
) -> tuple[float | None, int]:
    """
    Manifestation accuracy on stealth bars: the Whale Watcher label is
    direction-agnostic (|forward move| >= threshold within the horizon), so this
    measures whether a move manifested, not whether its direction was called.
    """
    from ml.volume_profile_tracker import VOLUME_Z_EVENT
    from ml.volume_profile_tracker import build_labels as whale_build_labels
    from ml.volume_profile_tracker import is_stealth_event

    threshold = VOLUME_Z_EVENT if volume_z_event is None else volume_z_event
    labels = whale_build_labels(test_features, horizon)
    mask = is_stealth_event(test_features, volume_z_event=threshold) & labels.notna()
    n = int(mask.sum())
    if n == 0:
        return None, 0
    y_true = labels.loc[mask].astype(int)
    y_pred = (proba.loc[mask] >= signal_threshold).astype(int)
    return float((y_true.to_numpy() == y_pred.to_numpy()).mean()), n


def directional_accuracy(
    test_features: pd.DataFrame,
    proba: pd.Series,
    *,
    horizon: int,
    signal_threshold: float,
    model_family: str = "logistic_regression",
    volume_z_event: float | None = None,
) -> tuple[float | None, int]:
    if model_family == "lightgbm":
        return whale_watcher_directional_accuracy(
            test_features,
            proba,
            horizon=horizon,
            signal_threshold=signal_threshold,
            volume_z_event=volume_z_event,
        )
    return contrarian_directional_accuracy(
        test_features, proba, horizon=horizon, signal_threshold=signal_threshold
    )


def normalize_curve(values: list[float], *, base: float = 100.0) -> list[float]:
    if not values:
        return []
    start = values[0]
    if start == 0:
        return [base for _ in values]
    scale = base / start
    return [float(v * scale) for v in values]


def price_curve_from_close(close: pd.Series) -> list[float]:
    arr = close.reset_index(drop=True).to_numpy(dtype=float)
    if arr.size == 0:
        return []
    return normalize_curve([float(x) for x in arr])


def feature_date_range(features: pd.DataFrame) -> tuple[str | None, str | None]:
    if "bar_date" not in features.columns or features.empty:
        return None, None
    dates = pd.to_datetime(features["bar_date"], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def entry_funnel_counts(
    test_features: pd.DataFrame,
    proba: pd.Series,
    *,
    signal_threshold: float,
    volume_z_entry: float,
    is_event: Callable[[pd.DataFrame], pd.Series] | None = None,
    volume_confirms_on_event_bar: bool = True,
) -> dict[str, Any]:
    """Funnel counts; `test_stretch_events` holds stealth events for Whale Watcher."""
    event_fn = is_event if is_event is not None else is_stretch_event
    events = event_fn(test_features)
    proba_arr = proba.reset_index(drop=True)
    volume_z = test_features["volume_z"].reset_index(drop=True)

    n_events = int(events.sum())
    passes_proba = events & (proba_arr >= signal_threshold)
    if volume_confirms_on_event_bar:
        passes_volume = events & (volume_z > volume_z_entry)
    else:
        # Stealth events already require a volume spike somewhere in the event window.
        passes_volume = events
    entry_signals = events & passes_proba & passes_volume

    event_proba = proba_arr[events]
    event_vol = volume_z[events]

    blocked_proba_only = events & (~passes_proba) & passes_volume
    blocked_volume_only = events & passes_proba & (~passes_volume)

    return {
        "test_stretch_events": n_events,
        "stretch_passes_proba": int(passes_proba.sum()),
        "stretch_passes_volume": int(passes_volume.sum()),
        "entry_signals": int(entry_signals.sum()),
        "blocked_by_proba_only": int(blocked_proba_only.sum()),
        "blocked_by_volume_only": int(blocked_volume_only.sum()),
        "stretch_proba_max": float(event_proba.max()) if n_events else None,
        "stretch_proba_median": float(event_proba.median()) if n_events else None,
        "stretch_volume_z_max": float(event_vol.max()) if n_events else None,
        "stretch_volume_z_median": float(event_vol.median()) if n_events else None,
    }
