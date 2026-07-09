"""
Volume Profile Tracker, *The Whale Watcher* (LightGBM).

Build spec: `ml_models_deep_dive (claude).md` §2. Detects stealth accumulation —
volume expanding sharply while price stays static — then predicts whether the move
manifests within the forward horizon.

Feature/label engineering is pure pandas/numpy; `train` lazily imports lightgbm
(install via `pip install -e '.[ml]'`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "rsi",
    "macd_signal",
    "volume_z",
    "price_range_pct",
    "volume_bias",
)

_AUX_COLUMNS: Final[tuple[str, ...]] = ("close", "high", "low")

VOLUME_WINDOW: Final[int] = 20
EVENT_WINDOW: Final[int] = 3

VOLUME_Z_EVENT: Final[float] = 2.0
TRAIN_VOLUME_Z_MIN: Final[float] = 1.25
PRICE_RANGE_MAX: Final[float] = 0.015
# 5% over the 15-bar horizon: on liquid large-caps a 2% move is too easy (all-positive labels).
LABEL_MOVE_PCT: Final[float] = 0.05

DEFAULT_HORIZON: Final[int] = 15
DEFAULT_SIGNAL_THRESHOLD: Final[float] = 0.65
DEFAULT_STOP_LOSS_PCT: Final[float] = 0.06
DEFAULT_TAKE_PROFIT_PCT: Final[float] = 0.12


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Whale Watcher features from a market_bar-shaped frame.

    Requires `close`, `volume`, `rsi`, `macd_signal`, plus `high`/`low` (or derives
    range from close when missing). All rolling features are backward-looking.
    """
    required = {"close", "volume", "rsi", "macd_signal"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"build_features missing required columns: {sorted(missing)}")

    src = bars.reset_index(drop=True)
    close = pd.to_numeric(src["close"], errors="coerce")
    volume = pd.to_numeric(src["volume"], errors="coerce")
    if "high" in src.columns and "low" in src.columns:
        high = pd.to_numeric(src["high"], errors="coerce")
        low = pd.to_numeric(src["low"], errors="coerce")
    else:
        high = close
        low = close

    window_high = high.rolling(window=EVENT_WINDOW, min_periods=EVENT_WINDOW).max()
    window_low = low.rolling(window=EVENT_WINDOW, min_periods=EVENT_WINDOW).min()

    vol_mean = volume.rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean()
    vol_std = volume.rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).std()
    volume_z = (volume - vol_mean) / vol_std.replace(0, np.nan)
    volume_z_peak = volume_z.rolling(window=EVENT_WINDOW, min_periods=EVENT_WINDOW).max()

    window_start_close = close.shift(EVENT_WINDOW - 1)
    up_dev = (window_high - window_start_close) / window_start_close.replace(0, np.nan)
    down_dev = (window_start_close - window_low) / window_start_close.replace(0, np.nan)
    price_range_pct = np.maximum(up_dev.fillna(0.0), down_dev.fillna(0.0))

    prev_close = close.shift(1)
    up_bar = close >= prev_close
    down_bar = close < prev_close
    up_vol = volume.where(up_bar, 0.0).rolling(EVENT_WINDOW, min_periods=EVENT_WINDOW).sum()
    down_vol = volume.where(down_bar, 0.0).rolling(EVENT_WINDOW, min_periods=EVENT_WINDOW).sum()
    total_vol = up_vol + down_vol
    volume_bias = (up_vol - down_vol) / total_vol.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "rsi": pd.to_numeric(src["rsi"], errors="coerce"),
            "macd_signal": pd.to_numeric(src["macd_signal"], errors="coerce"),
            "volume_z": volume_z,
            "volume_z_peak": volume_z_peak,
            "price_range_pct": price_range_pct,
            "volume_bias": volume_bias,
            "close": close,
            "high": high,
            "low": low,
        }
    )
    if "bar_date" in src.columns:
        out["bar_date"] = pd.to_datetime(src["bar_date"], errors="coerce")
    elif "timestamp" in src.columns:
        out["bar_date"] = pd.to_datetime(src["timestamp"], errors="coerce")

    numeric_keep = list(FEATURE_COLUMNS) + list(_AUX_COLUMNS)
    out = out.dropna(subset=numeric_keep).reset_index(drop=True)
    return out


def is_stealth_event(
    features: pd.DataFrame, *, volume_z_event: float = VOLUME_Z_EVENT
) -> pd.Series:
    """
    Candidate filter (spec §2): volume z-score > `volume_z_event` (spec default 2)
    over the event window while OHLC stays inside ±1.5% of the window-start close.
    """
    if "volume_z_peak" in features.columns:
        vol_spike = features["volume_z_peak"] > volume_z_event
    else:
        vol_spike = features["volume_z"].rolling(EVENT_WINDOW, min_periods=EVENT_WINDOW).max() > (
            volume_z_event
        )
    quiet_price = features["price_range_pct"] <= PRICE_RANGE_MAX
    return vol_spike & quiet_price


def is_training_candidate(features: pd.DataFrame) -> pd.Series:
    """
    Looser filter for fitting: elevated volume in the window with a quiet price,
    without requiring the full stealth spike threshold used at strategy entry.
    """
    if "volume_z_peak" in features.columns:
        vol_elevated = features["volume_z_peak"] > TRAIN_VOLUME_Z_MIN
    else:
        vol_elevated = (
            features["volume_z"].rolling(EVENT_WINDOW, min_periods=EVENT_WINDOW).max()
            > TRAIN_VOLUME_Z_MIN
        )
    quiet_price = features["price_range_pct"] <= PRICE_RANGE_MAX
    return vol_elevated & quiet_price


def build_labels(
    features: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    *,
    move_pct: float = LABEL_MOVE_PCT,
) -> pd.Series:
    """
    Binary manifest label (spec §2): 1 if abs forward return >= move_pct within
    `horizon` bars after the event bar, else 0. Final `horizon` rows are NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    close = features["close"].reset_index(drop=True)
    manifested = pd.Series(False, index=close.index)
    for k in range(1, horizon + 1):
        future_close = close.shift(-k)
        fwd_ret = (future_close - close) / close.replace(0, np.nan)
        manifested = manifested | (fwd_ret.abs() >= move_pct)

    labels = manifested.astype(float)
    if len(labels) > horizon:
        labels.iloc[-horizon:] = np.nan
    else:
        labels.iloc[:] = np.nan
    labels.index = features.index
    return labels


@dataclass
class WhaleWatcherModel:
    """Trained Whale Watcher model with degenerate-data fallback."""

    booster: Any | None
    base_rate: float
    horizon: int
    pos_index: int = 1
    # Labeled rows the fit actually used (after _training_mask fallbacks), for reporting.
    train_rows: int = 0

    @property
    def is_fallback(self) -> bool:
        return self.booster is None

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """P(manifest) in [0, 1], aligned to `features.index`."""
        if self.booster is None:
            return pd.Series(self.base_rate, index=features.index, dtype=float)
        matrix = _model_input_matrix(features)
        if matrix.size == 0:
            return pd.Series(dtype=float)
        proba = self.booster.predict_proba(matrix)[:, self.pos_index]
        return pd.Series(proba, index=features.index, dtype=float)


def _model_input_matrix(features: pd.DataFrame) -> np.ndarray:
    return features.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float, copy=True)


def _quiet_price_mask(features: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Bars inside the ±1.5% consolidation band (used when stealth rows are too sparse)."""
    return (features["price_range_pct"] <= PRICE_RANGE_MAX) & labels.notna()


def _training_mask(features: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """
    Pick labeled rows for fitting: prefer stealth events (spec), then looser candidates.

    Ingested `market_bar` history often has far fewer stealth spikes than ad-hoc downloads,
    so we fall back to quiet-consolidation bars (still includes volume_z as a feature).
    """
    loose = (
        (features["price_range_pct"] <= PRICE_RANGE_MAX)
        & (features["volume_z_peak"] > 1.0)
        & labels.notna()
    )
    quiet = _quiet_price_mask(features, labels)
    for candidate_mask in (
        is_stealth_event(features) & labels.notna(),
        is_training_candidate(features) & labels.notna(),
        loose,
        quiet,
    ):
        y = labels.loc[candidate_mask]
        if y.size >= 4 and y.nunique() >= 2:
            return candidate_mask
    # Last resort: any mask with both classes (even tiny).
    for candidate_mask in (
        is_stealth_event(features) & labels.notna(),
        is_training_candidate(features) & labels.notna(),
        loose,
        quiet,
    ):
        y = labels.loc[candidate_mask]
        if y.size >= 2 and y.nunique() >= 2:
            return candidate_mask
    return is_training_candidate(features) & labels.notna()


def train_on_features(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    horizon: int = DEFAULT_HORIZON,
    random_state: int = 0,
) -> WhaleWatcherModel:
    """Fit LightGBM on labeled accumulation events (stealth-first, then looser candidates)."""
    mask = _training_mask(features, labels)

    x = _model_input_matrix(features.loc[mask])
    y = labels.loc[mask].astype(int).to_numpy()

    base_rate = float(y.mean()) if y.size else 0.5

    if y.size < 2 or np.unique(y).size < 2:
        return WhaleWatcherModel(
            booster=None, base_rate=base_rate, horizon=horizon, train_rows=int(y.size)
        )

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "lightgbm is required to train Whale Watcher: pip install -e '.[ml]'"
        ) from exc

    clf = LGBMClassifier(
        num_leaves=31,
        n_estimators=100,
        learning_rate=0.05,
        random_state=random_state,
        verbosity=-1,
    )
    clf.fit(x, y)
    pos_index = int(list(clf.classes_).index(1))
    return WhaleWatcherModel(
        booster=clf,
        base_rate=base_rate,
        horizon=horizon,
        pos_index=pos_index,
        train_rows=int(y.size),
    )


def train(
    bars: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    random_state: int = 0,
) -> WhaleWatcherModel:
    """Convenience: build features + labels from raw bars, then fit."""
    features = build_features(bars)
    labels = build_labels(features, horizon)
    return train_on_features(features, labels, horizon=horizon, random_state=random_state)
