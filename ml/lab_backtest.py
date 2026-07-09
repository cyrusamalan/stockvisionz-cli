"""
Run a Lab backtest for supported model families over a user-selected date window.

Supported: Mean-Reversion Drop (logistic_regression) and Whale Watcher (lightgbm).
Validation uses walk-forward folds (rolling or anchored), not a single train/test split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

from ml.lab_shared import (
    LOOKBACK_DAYS,
    LAB_SIGNAL_THRESHOLD,
    LAB_VOLUME_Z_ENTRY,
    LAB_WHALE_SIGNAL_THRESHOLD,
    LAB_WHALE_VOLUME_Z_ENTRY,
    bar_dates,
    slice_by_date,
)
from ml.lab_walk_forward import ValidationMode, run_walk_forward_lab_backtest
from ml.mean_reversion_drop import (
    DEFAULT_HORIZON as CONTRARIAN_HORIZON,
    DEFAULT_STOP_LOSS_PCT as CONTRARIAN_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT as CONTRARIAN_TAKE_PROFIT_PCT,
    build_features as contrarian_build_features,
)
from ml.volume_profile_tracker import (
    DEFAULT_HORIZON as WHALE_HORIZON,
    DEFAULT_STOP_LOSS_PCT as WHALE_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT as WHALE_TAKE_PROFIT_PCT,
    build_features as whale_build_features,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_MODEL_FAMILIES: Final[frozenset[str]] = frozenset(
    {"logistic_regression", "lightgbm"}
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        for path in (REPO_ROOT / ".env", REPO_ROOT / "frontend" / ".env.local"):
            if path.is_file():
                load_dotenv(path, override=False)
    except ImportError:
        pass


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_validation_mode(value: str) -> ValidationMode:
    mode = value.strip().lower()
    if mode not in ("rolling", "anchored"):
        raise ValueError("validation_mode must be 'rolling' or 'anchored'")
    return mode  # type: ignore[return-value]


def _family_defaults(family: str) -> dict[str, Any]:
    if family == "lightgbm":
        return {
            "build_features": whale_build_features,
            "horizon": WHALE_HORIZON,
            "signal_threshold": LAB_WHALE_SIGNAL_THRESHOLD,
            "stop_loss_pct": WHALE_STOP_LOSS_PCT,
            "take_profit_pct": WHALE_TAKE_PROFIT_PCT,
            "volume_z_entry": LAB_WHALE_VOLUME_Z_ENTRY,
        }
    return {
        "build_features": contrarian_build_features,
        "horizon": CONTRARIAN_HORIZON,
        "signal_threshold": LAB_SIGNAL_THRESHOLD,
        "stop_loss_pct": CONTRARIAN_STOP_LOSS_PCT,
        "take_profit_pct": CONTRARIAN_TAKE_PROFIT_PCT,
        "volume_z_entry": LAB_VOLUME_Z_ENTRY,
    }


def run_lab_backtest(
    symbol: str,
    window_start: date,
    window_end: date,
    *,
    model_family: str = "logistic_regression",
    validation_mode: ValidationMode = "rolling",
    horizon: int | None = None,
    signal_threshold: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    volume_z_entry: float | None = None,
    bars: "pd.DataFrame | None" = None,
    database_url: str | None = None,
) -> dict[str, Any]:
    """
    Walk-forward Lab backtest inside [window_start, window_end].
    See ml.lab_walk_forward for rolling vs anchored fold schedules.
    """
    family = model_family.strip().lower()
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"model_family {model_family!r} is not supported yet; "
            f"supported: {sorted(SUPPORTED_MODEL_FAMILIES)}"
        )
    if window_end < window_start:
        raise ValueError("end date must be on or after start date")

    defaults = _family_defaults(family)
    horizon = defaults["horizon"] if horizon is None else horizon
    signal_threshold = (
        defaults["signal_threshold"] if signal_threshold is None else signal_threshold
    )
    stop_loss_pct = defaults["stop_loss_pct"] if stop_loss_pct is None else stop_loss_pct
    take_profit_pct = defaults["take_profit_pct"] if take_profit_pct is None else take_profit_pct
    volume_z_entry = defaults["volume_z_entry"] if volume_z_entry is None else volume_z_entry
    build_features = defaults["build_features"]

    load_start = window_start - timedelta(days=LOOKBACK_DAYS)
    if bars is None:
        url = database_url or os.environ.get("DATABASE_URL") or os.environ.get("NEON_PRIMARY_URL")
        if not url:
            raise ValueError("DATABASE_URL (or NEON_PRIMARY_URL) is required")

        from ml.market_bar_reader import load_market_bars

        bars = load_market_bars(symbol, load_start, window_end, database_url=url)
        if bars.empty:
            raise ValueError(f"no market_bar rows for {symbol.upper()} in the requested range")
    else:
        # Expect the same schema as market_bar_reader.read_market_bars (MARKET_BAR_COLUMNS).
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise ImportError("pandas is required to pass bars=...") from exc
        if not isinstance(bars, pd.DataFrame):
            raise ValueError("bars must be a pandas DataFrame")
        if bars.empty:
            raise ValueError("bars is empty")
        if "symbol" not in bars.columns:
            bars = bars.copy()
            bars["symbol"] = symbol.strip().upper()

    features = build_features(bars)
    dates = bar_dates(features)
    window_features = slice_by_date(features, dates, window_start, window_end)
    if len(window_features) < 10:
        raise ValueError("selected window has too few bars after indicator warm-up")

    return run_walk_forward_lab_backtest(
        window_features,
        symbol=symbol,
        model_family=family,
        validation_mode=validation_mode,
        window_start=window_start,
        window_end=window_end,
        load_start=load_start,
        bars_loaded=len(bars),
        horizon=horizon,
        signal_threshold=signal_threshold,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        volume_z_entry=volume_z_entry,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockvisionz-lab-backtest",
        description="Run a Lab walk-forward backtest for a supported model family",
    )
    parser.add_argument("--symbol", required=True, help="Ticker symbol")
    parser.add_argument("--start", required=True, help="Window start YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Window end YYYY-MM-DD")
    parser.add_argument(
        "--model-family",
        default="logistic_regression",
        help="Runnable model family (default: logistic_regression)",
    )
    parser.add_argument(
        "--validation-mode",
        default="rolling",
        choices=("rolling", "anchored"),
        help="Walk-forward mode: rolling (default) or anchored",
    )
    parser.add_argument("--database-url", default=None, help="Postgres URL (default: env)")
    parser.add_argument("--format", choices=("json",), default="json", help="Output format")
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=None,
        help="Override strategy signal threshold (P threshold to enter)",
    )
    parser.add_argument(
        "--volume-z-entry",
        type=float,
        default=None,
        help=(
            "Override the volume-z gate (Contrarian: event-bar volume confirm; "
            "Whale: stealth-event spike threshold)"
        ),
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=None,
        help="Override stop loss as fraction of entry price",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=None,
        help="Override take profit as fraction of entry price",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override label/trade horizon in bars",
    )
    return parser


def _validate_strategy_overrides(
    *,
    signal_threshold: float | None,
    volume_z_entry: float | None,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    horizon: int | None,
) -> None:
    if signal_threshold is not None and not 0.05 <= signal_threshold <= 0.99:
        raise ValueError("signal_threshold must be between 0.05 and 0.99")
    if volume_z_entry is not None and not 0.0 <= volume_z_entry <= 5.0:
        raise ValueError("volume_z_entry must be between 0 and 5")
    if stop_loss_pct is not None and not 0.005 <= stop_loss_pct <= 0.25:
        raise ValueError("stop_loss_pct must be between 0.005 and 0.25")
    if take_profit_pct is not None and not 0.005 <= take_profit_pct <= 0.5:
        raise ValueError("take_profit_pct must be between 0.005 and 0.5")
    if horizon is not None and not 1 <= horizon <= 60:
        raise ValueError("horizon must be between 1 and 60")


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_strategy_overrides(
            signal_threshold=args.signal_threshold,
            volume_z_entry=args.volume_z_entry,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            horizon=args.horizon,
        )
        payload = run_lab_backtest(
            args.symbol,
            _parse_date(args.start),
            _parse_date(args.end),
            model_family=args.model_family,
            validation_mode=_parse_validation_mode(args.validation_mode),
            database_url=args.database_url,
            horizon=args.horizon,
            signal_threshold=args.signal_threshold,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            volume_z_entry=args.volume_z_entry,
        )
    except (ValueError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
