"""
features.py - Feature computation from OHLCV data.

All computations are purely backward-looking (no future leakage).
Given a DataFrame and a reference date (the "as-of" row), returns
a flat dict of features suitable for JSON serialisation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _pct_return(series: pd.Series, n: int) -> float | None:
    """(price[t] / price[t-n]) - 1, or None if insufficient history."""
    if len(series) < n + 1:
        return None
    val = float(series.iloc[-1] / series.iloc[-(n + 1)] - 1)
    return round(val, 6)


def _rolling_vol(close: pd.Series, n: int = 20) -> float | None:
    """Annualised daily-return volatility over last *n* trading days."""
    if len(close) < n + 1:
        return None
    log_rets = np.log(close.iloc[-(n + 1):].values)
    daily_std = float(np.std(np.diff(log_rets), ddof=1))
    ann_vol = daily_std * math.sqrt(252)
    return round(ann_vol, 6)


def _sma(close: pd.Series, n: int) -> float | None:
    if len(close) < n:
        return None
    return round(float(close.iloc[-n:].mean()), 4)


def _volume_ratio(volume: pd.Series, n: int = 20) -> float | None:
    """Latest volume / mean volume over last n days."""
    if len(volume) < n + 1:
        return None
    avg = float(volume.iloc[-(n + 1):-1].mean())
    if avg == 0:
        return None
    ratio = float(volume.iloc[-1]) / avg
    return round(ratio, 4)


def compute_features(df: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, Any]:
    """Compute features for a single ticker at a given date.

    Parameters
    ----------
    df    : full OHLCV DataFrame (index = DatetimeIndex)
    as_of : the date whose features we want (must exist in df.index)

    Returns
    -------
    Flat dict with all feature values and the reference price/date.
    """
    # Slice history up to and including as_of – no future leakage
    hist = df[df.index <= as_of].copy()
    if hist.empty:
        raise ValueError(f"No data on or before {as_of}")

    close = hist["Close"]
    volume = hist["Volume"]

    last_date = hist.index[-1]
    last_close = float(close.iloc[-1])

    ret_1d = _pct_return(close, 1)
    ret_5d = _pct_return(close, 5)
    ret_20d = _pct_return(close, 20)

    vol_20d = _rolling_vol(close, 20)

    ma_20 = _sma(close, 20)
    ma_50 = _sma(close, 50)

    vol_ratio = _volume_ratio(volume, 20)

    # Price relative to moving averages (None if MA unavailable)
    price_to_ma20 = round(last_close / ma_20 - 1, 6) if ma_20 else None
    price_to_ma50 = round(last_close / ma_50 - 1, 6) if ma_50 else None

    return {
        "date": last_date.strftime("%Y-%m-%d"),
        "close": round(last_close, 4),
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "vol_20d": vol_20d,
        "ma_20": ma_20,
        "ma_50": ma_50,
        "price_to_ma20": price_to_ma20,
        "price_to_ma50": price_to_ma50,
        "volume_ratio": vol_ratio,
    }
