"""
Mean-Reversion Drop, *The Contrarian* (Logistic Regression).

Build spec: `ml_models_deep_dive (claude).md` §1. The model asks a bounded-probability
question ("how likely is a snapback?") after price has statistically overextended, which
is exactly the regime logistic regression fits; its inputs are each near-monotonic with
reversion probability.

Feature/label engineering here is pure pandas/numpy and needs no database; only `train`
lazily imports scikit-learn (install via `pip install -e '.[ml]'`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

# The model's input vector: the "correlated features" from the spec's toolkit. All
# derive from existing market_bar columns (no new indicator/migration needed).
FEATURE_COLUMNS: Final[tuple[str, ...]] = ("rsi", "price_vs_sma", "volume_z", "macd_line")

# Columns build_features keeps beyond FEATURE_COLUMNS, needed for labeling + backtest.
_AUX_COLUMNS: Final[tuple[str, ...]] = ("close", "sma_20")

# Stretch/candidate thresholds (spec §1 strategy logic).
RSI_OVERSOLD: Final[float] = 30.0
RSI_OVERBOUGHT: Final[float] = 70.0
STRETCH_DISTANCE: Final[float] = 0.05  # |price_vs_sma| must exceed 5%
VOLUME_WINDOW: Final[int] = 20

# Backtest tie-in defaults (spec §1).
DEFAULT_HORIZON: Final[int] = 5
DEFAULT_SIGNAL_THRESHOLD: Final[float] = 0.7
DEFAULT_STOP_LOSS_PCT: Final[float] = 0.03
DEFAULT_TAKE_PROFIT_PCT: Final[float] = 0.04


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the Contrarian feature frame from a market_bar-shaped frame.

    Input must have `close, volume, sma_20, rsi, macd_line` (raw bars already carry these
    indicator columns; for offline OHLCV run compute_indicators first). Adds
    `price_vs_sma` and `volume_z`, drops warm-up rows where any model input is NaN, and
    returns a 0-indexed frame with FEATURE_COLUMNS + close + sma_20. A `bar_date` column
    is carried through when the input has `bar_date` or `timestamp`, so callers can split
    the feature frame by date without re-incurring indicator warm-up per window.

    All features are backward-looking, so computing them once over a full series and
    slicing afterward introduces no look-ahead leakage.
    """
    required = {"close", "volume", "sma_20", "rsi", "macd_line"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"build_features missing required columns: {sorted(missing)}")

    src = bars.reset_index(drop=True)
    close = pd.to_numeric(src["close"], errors="coerce")
    sma = pd.to_numeric(src["sma_20"], errors="coerce")
    volume = pd.to_numeric(src["volume"], errors="coerce")

    price_vs_sma = (close - sma) / sma.replace(0, np.nan)
    vol_mean = volume.rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean()
    vol_std = volume.rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).std()
    volume_z = (volume - vol_mean) / vol_std.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "rsi": pd.to_numeric(src["rsi"], errors="coerce"),
            "price_vs_sma": price_vs_sma,
            "volume_z": volume_z,
            "macd_line": pd.to_numeric(src["macd_line"], errors="coerce"),
            "close": close,
            "sma_20": sma,
        }
    )
    if "bar_date" in src.columns:
        out["bar_date"] = pd.to_datetime(src["bar_date"], errors="coerce")
    elif "timestamp" in src.columns:
        out["bar_date"] = pd.to_datetime(src["timestamp"], errors="coerce")

    numeric_keep = list(FEATURE_COLUMNS) + list(_AUX_COLUMNS)
    out = out.dropna(subset=numeric_keep).reset_index(drop=True)
    return out


def is_stretch_event(features: pd.DataFrame) -> pd.Series:
    """Candidate filter (spec §1): oversold/overbought RSI *and* a real >5% stretch."""
    rsi = features["rsi"]
    extreme_rsi = (rsi < RSI_OVERSOLD) | (rsi > RSI_OVERBOUGHT)
    real_stretch = features["price_vs_sma"].abs() > STRETCH_DISTANCE
    return extreme_rsi & real_stretch


def build_labels(features: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.Series:
    """
    Binary reversion label (spec §1): 1 if within `horizon` bars the close reverts at
    least 50% of the way back toward `sma_20` at the event bar.

    The final `horizon` rows lack a full forward window and are left NaN (unlabelable),
    the standard treatment for forward-looking horizon labels.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    close = features["close"].reset_index(drop=True)
    sma = features["sma_20"].reset_index(drop=True)
    deviation = close - sma
    half_deviation = 0.5 * deviation.abs()
    deviation_sign = np.sign(deviation)

    reverted = pd.Series(False, index=close.index)
    for k in range(1, horizon + 1):
        future_close = close.shift(-k)
        # If it moves in opposite direction of deviation past the 50% mark, it has reverted.
        reverted = reverted | ((future_close - sma) * deviation_sign <= half_deviation)

    labels = reverted.astype(float)
    if len(labels) > horizon:
        labels.iloc[-horizon:] = np.nan
    else:
        labels.iloc[:] = np.nan
    labels.index = features.index
    return labels


@dataclass
class ContrarianModel:
    """
    Trained Contrarian model. `pipeline` is None when training data was degenerate
    (too few rows or a single label class), in which case predictions fall back to the
    observed base reversion rate rather than crashing on small user-chosen windows.
    """

    pipeline: Any | None
    base_rate: float
    horizon: int
    pos_index: int = 1

    @property
    def is_fallback(self) -> bool:
        return self.pipeline is None

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Calibrated P(reversion) in [0, 1], aligned to `features.index`."""
        if self.pipeline is None:
            return pd.Series(self.base_rate, index=features.index, dtype=float)
        matrix = _model_input_matrix(features)
        if matrix.size == 0:
            return pd.Series(dtype=float)

        proba = self.pipeline.predict_proba(matrix)[:, self.pos_index]
        return pd.Series(proba, index=features.index, dtype=float)


def _model_input_matrix(features: pd.DataFrame) -> np.ndarray:
    """Copy feature columns and map raw values to monotonic stretch magnitudes."""
    matrix = features.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float, copy=True)
    if matrix.size:
        matrix[:, 0] = np.abs(matrix[:, 0] - 50.0)  # RSI distance from 50
        matrix[:, 1] = np.abs(matrix[:, 1])  # Absolute price stretch
        matrix[:, 3] = np.abs(matrix[:, 3])  # Absolute MACD
    return matrix


def train_on_features(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    horizon: int = DEFAULT_HORIZON,
    random_state: int = 0,
) -> ContrarianModel:
    """
    Fit a StandardScaler + LogisticRegression pipeline on stretch-event rows only.

    "Detect the candidate structurally, let the model judge it": the model is trained
    exclusively on bars where a stretch event already occurred, so it learns
    exhaustion-vs-continuation rather than "is this even a stretch." Takes a pre-built
    feature frame + aligned labels so callers can build features once over a full series
    and label/train only within the training window (no look-ahead into the test window).
    """
    mask = is_stretch_event(features) & labels.notna()

    x = _model_input_matrix(features.loc[mask])
    y = labels.loc[mask].astype(int).to_numpy()

    base_rate = float(y.mean()) if y.size else 0.5

    if y.size < 2 or np.unique(y).size < 2:
        return ContrarianModel(pipeline=None, base_rate=base_rate, horizon=horizon)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - import-time guard
        raise ImportError(
            "scikit-learn is required to train models: pip install -e '.[ml]'"
        ) from exc

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    pipeline.fit(x, y)
    pos_index = int(list(pipeline.named_steps["clf"].classes_).index(1))
    return ContrarianModel(
        pipeline=pipeline, base_rate=base_rate, horizon=horizon, pos_index=pos_index
    )


def train(
    bars: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    random_state: int = 0,
) -> ContrarianModel:
    """Convenience: build features + labels from raw bars, then fit (single-window use)."""
    features = build_features(bars)
    labels = build_labels(features, horizon)
    return train_on_features(features, labels, horizon=horizon, random_state=random_state)
