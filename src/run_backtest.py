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

from src.backtest import run_backtest, confidence_analysis
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
    parser.add_argument(
        "--model", default=None,
        help="LLM model ID (default: claude-haiku-4-5-20251001). Only used with --strategy llm.",
    )
    parser.add_argument(
        "--critic", action="store_true",
        help="Run a second LLM critic pass to review and possibly downgrade each recommendation.",
    )
    parser.add_argument(
        "--critic-model", default=None,
        help="Claude model ID for the critic (default: claude-haiku-4-5-20251001).",
    )
    parser.add_argument(
        "--rec-cache", default=None,
        help=(
            "Path to a JSON file of cached primary-strategy recommendations "
            "(keyed by 'ticker/date'). If the file exists it is loaded before "
            "the run; the updated cache is always written to rec_cache.json in "
            "the run directory and also back to this path when provided. "
            "Pass a cache from a prior LLM run to skip re-calling the primary "
            "LLM when adding --critic on a second pass."
        ),
    )
    args = parser.parse_args()

    print(f"\nBacktest: {args.strategy.upper()} | {args.start} → {args.end}")
    if args.critic:
        print(f"  Critic: enabled (model={args.critic_model or 'claude-haiku-4-5-20251001'})")
    print("=" * 60)

    # Load recommendation cache for LLM runs (enables critic-only reruns).
    rec_cache: dict | None = None
    if args.strategy == "llm":
        rec_cache = {}
        if args.rec_cache:
            cache_path = Path(args.rec_cache)
            if cache_path.exists():
                with open(cache_path) as f:
                    rec_cache = json.load(f)
                print(f"  Loaded {len(rec_cache)} cached recommendations from {cache_path}")

    trades, summary = run_backtest(
        tickers=TICKERS,
        start=args.start,
        end=args.end,
        strategy_name=args.strategy,
        skip_hold_prefilter=args.skip_hold_prefilter,
        model=args.model,
        critic=args.critic,
        critic_model=args.critic_model,
        rec_cache=rec_cache,
    )

    run_dir = _make_run_dir(args.start, args.end, args.strategy)

    # Save trades
    trades_path = run_dir / "trades.csv"
    trades.to_csv(trades_path, index=False)

    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Persist recommendation cache for LLM runs so future critic reruns can
    # skip the primary LLM calls.
    if rec_cache is not None:
        cache_out = run_dir / "rec_cache.json"
        with open(cache_out, "w") as f:
            json.dump(rec_cache, f, indent=2)
        if args.rec_cache and Path(args.rec_cache) != cache_out:
            with open(args.rec_cache, "w") as f:
                json.dump(rec_cache, f, indent=2)

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
        if args.critic and not trades.empty and "critic_agree" in trades.columns:
            t = trades[trades["ticker"] == ticker]
            if not t.empty:
                agree_rate = t["critic_agree"].mean()
                print(f"    {'critic_agree_rate':<25}: {agree_rate:.2%}")

    # Confidence analysis
    if not trades.empty and "confidence" in trades.columns:
        print("\n=== CONFIDENCE ANALYSIS (aggregate) ===")
        ca = confidence_analysis(trades)
        for _, row in ca.iterrows():
            wr = f"{row['win_rate']:.0%}" if row["win_rate"] is not None else "n/a"
            ap = f"{row['avg_pnl_pct']:.4f}%" if row["avg_pnl_pct"] is not None else "n/a"
            sh = f"{row['sharpe']:.4f}" if row["sharpe"] is not None else "n/a"
            print(f"  {row['bucket']:<16}  n={row['n_trades']:>4}  win_rate={wr:>5}  avg_pnl={ap:>10}  sharpe={sh:>8}")

        print("\n=== CONFIDENCE ANALYSIS (per-ticker) ===")
        for ticker in TICKERS:
            t = trades[trades["ticker"] == ticker]
            if t.empty:
                continue
            print(f"\n  {ticker}")
            ca_t = confidence_analysis(t)
            for _, row in ca_t.iterrows():
                wr = f"{row['win_rate']:.0%}" if row["win_rate"] is not None else "n/a"
                ap = f"{row['avg_pnl_pct']:.4f}%" if row["avg_pnl_pct"] is not None else "n/a"
                sh = f"{row['sharpe']:.4f}" if row["sharpe"] is not None else "n/a"
                print(f"    {row['bucket']:<16}  n={row['n_trades']:>4}  win_rate={wr:>5}  avg_pnl={ap:>10}  sharpe={sh:>8}")

        # Save aggregate confidence analysis
        ca.to_csv(run_dir / "confidence_analysis.csv", index=False)

    version = _git_version()
    for ticker, m in summary["per_ticker"].items():
        record: dict = {
            "ticker": ticker,
            "date": args.start,
            "version": version,
            "run_type": "backtest",
            "strategy": args.strategy,
            "start_date": args.start,
            "end_date": args.end,
            **m,
        }
        if args.critic and not trades.empty and "critic_agree" in trades.columns:
            t = trades[trades["ticker"] == ticker]
            if not t.empty:
                record["critic_agree_rate"] = round(float(t["critic_agree"].mean()), 4)
        append_run_record(record)

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
