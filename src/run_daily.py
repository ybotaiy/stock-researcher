"""
run_daily.py - Generate today's evidence packs and recommendations.

Usage:
    python -m src.run_daily [--strategy momentum|llm] [--date YYYY-MM-DD]

Outputs per ticker under runs/<timestamp>/:
    evidence_pack_<TICKER>.json
    recommendation_<TICKER>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Project root on sys.path so `python -m src.run_daily` works
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import fetch_ohlcv
from src.features import compute_features
from src.recommend import MomentumStrategy, LLMStrategy, build_evidence_pack

TICKERS = ["GOOGL", "TSLA", "GLD"]
RUNS_DIR = Path(__file__).parent.parent / "runs"


def _make_run_dir(as_of: str) -> Path:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = RUNS_DIR / f"daily_{as_of}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_daily(
    as_of: str | None = None,
    strategy_name: str = "momentum",
    lookback_days: int = 120,
) -> Path:
    """Generate evidence packs and recommendations for all tickers.

    Parameters
    ----------
    as_of         : ISO date string (defaults to yesterday)
    strategy_name : "momentum" or "llm"
    lookback_days : calendar days of history to fetch

    Returns
    -------
    Path to the run directory.
    """
    if as_of is None:
        as_of = (date.today() - timedelta(days=1)).isoformat()

    as_of_ts = pd.Timestamp(as_of)

    # Fetch enough history
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(as_of) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if strategy_name == "momentum":
        strategy = MomentumStrategy()
    elif strategy_name == "llm":
        strategy = LLMStrategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name!r}")

    run_dir = _make_run_dir(as_of)
    print(f"Run directory: {run_dir}")

    for ticker in TICKERS:
        print(f"  Processing {ticker} ...")
        df = fetch_ohlcv(ticker, start, end)

        # Use the latest available date on or before as_of
        available = df.index[df.index <= as_of_ts]
        if available.empty:
            print(f"    No data for {ticker} on or before {as_of}, skipping.")
            continue
        ref_date = available[-1]

        features = compute_features(df, ref_date)
        evidence = build_evidence_pack(ticker, features)

        # Save evidence pack
        ep_path = run_dir / f"evidence_pack_{ticker}.json"
        with open(ep_path, "w") as f:
            json.dump(evidence, f, indent=2)

        # Generate recommendation
        rec = strategy.recommend(evidence)
        rec_out = {
            "ticker": ticker,
            "as_of": as_of,
            **rec,
        }

        rec_path = run_dir / f"recommendation_{ticker}.json"
        with open(rec_path, "w") as f:
            json.dump(rec_out, f, indent=2)

        print(f"    Signal: {rec['signal']}  |  {rec['rationale'][:80]}")

    print(f"\nArtifacts saved to: {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Daily stock research run")
    parser.add_argument("--strategy", default="momentum", choices=["momentum", "llm"])
    parser.add_argument("--date", default=None, help="As-of date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()
    run_daily(as_of=args.date, strategy_name=args.strategy)


if __name__ == "__main__":
    main()
