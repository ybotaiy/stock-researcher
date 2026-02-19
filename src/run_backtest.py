"""
run_backtest.py - Execute walk-forward backtest and persist results.

Usage:
    python -m src.run_backtest [--strategy momentum|llm] \
                               [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Outputs under runs/<timestamp>_backtest/:
    trades.csv
    metrics.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import run_backtest
from src.logging import append_run_record


def _git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

TICKERS = ["GOOGL", "TSLA", "GLD"]
RUNS_DIR = Path(__file__).parent.parent / "runs"


def _make_run_dir(start: str, end: str, strategy: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = RUNS_DIR / f"backtest_{strategy}_{start}_{end}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest runner")
    parser.add_argument("--strategy", default="momentum", choices=["momentum", "llm"])
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument(
        "--skip-hold-prefilter",
        action="store_true",
        help="Pre-filter with momentum; only call LLM on non-HOLD candidate days.",
    )
    args = parser.parse_args()

    print(f"\nBacktest: {args.strategy.upper()} | {args.start} → {args.end}")
    print("=" * 60)

    trades, summary = run_backtest(
        tickers=TICKERS,
        start=args.start,
        end=args.end,
        strategy_name=args.strategy,
        skip_hold_prefilter=args.skip_hold_prefilter,
    )

    run_dir = _make_run_dir(args.start, args.end, args.strategy)

    # Save trades
    trades_path = run_dir / "trades.csv"
    trades.to_csv(trades_path, index=False)

    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n=== AGGREGATE METRICS ===")
    agg = summary["aggregate"]
    for k, v in agg.items():
        print(f"  {k:<25}: {v}")

    print("\n=== PER-TICKER METRICS ===")
    for ticker, m in summary["per_ticker"].items():
        print(f"\n  {ticker}")
        for k, v in m.items():
            print(f"    {k:<25}: {v}")

    version = _git_version()
    for ticker, m in summary["per_ticker"].items():
        append_run_record({
            "ticker": ticker,
            "date": args.start,
            "version": version,
            "run_type": "backtest",
            "strategy": args.strategy,
            "start_date": args.start,
            "end_date": args.end,
            **m,
        })

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
