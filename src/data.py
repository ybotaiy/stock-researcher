"""
data.py - Deterministic market data ingestion via yfinance.

Fetches OHLCV data and caches locally under data/<ticker>/.
Re-fetching the same (ticker, start, end) always returns identical rows.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).parent.parent / "data"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    d = DATA_DIR / ticker
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{start}_{end}.csv"


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return daily OHLCV DataFrame for *ticker* between *start* and *end* (exclusive).

    Columns: Open, High, Low, Close, Volume  (all float64)
    Index  : DatetimeIndex (UTC-naive, daily frequency)

    Parameters
    ----------
    ticker    : e.g. "GOOGL"
    start     : ISO date string, e.g. "2023-01-01"
    end       : ISO date string, e.g. "2025-12-31"
    use_cache : Load from CSV if available, otherwise download and save.
    """
    path = _cache_path(ticker, start, end)

    if use_cache and path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for {ticker} ({start} -> {end})")

    # yfinance >=0.2 may return MultiIndex columns when a single ticker is passed
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    # Ensure float64 throughout
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype("float64")
    df["Volume"] = df["Volume"].astype("float64")

    if use_cache:
        df.to_csv(path)

    return df


def fetch_all(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper – returns {ticker: ohlcv_df}."""
    return {t: fetch_ohlcv(t, start, end, use_cache=use_cache) for t in tickers}
