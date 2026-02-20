"""
build_golden_cases.py - Generate evals/golden_cases.jsonl.

For each ticker in TICKERS, samples N_SAMPLES dates evenly across the
backtest range (BACKTEST_START → BACKTEST_END), then records:
  - the evidence pack (features as-of that date)
  - the deterministic momentum signal (consistency label)
  - the oracle signal derived from the actual 5-day forward return (quality label)

Oracle convention: same T+1 entry / T+6 exit open prices used by the backtest.
  forward_return_5d > ORACLE_THRESHOLD  → oracle_signal = BUY
  forward_return_5d < -ORACLE_THRESHOLD → oracle_signal = SELL
  otherwise                             → oracle_signal = HOLD

Output: evals/golden_cases.jsonl   (one JSON object per line)

Usage:
    .venv/bin/python -m evals.build_golden_cases
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python -m evals.build_golden_cases` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import fetch_ohlcv
from src.features import compute_features
from src.recommend import MomentumStrategy, build_evidence_pack

# ── config ────────────────────────────────────────────────────────────────────
TICKERS = ["GOOGL", "TSLA", "GLD"]
BACKTEST_START = "2023-01-01"
BACKTEST_END = "2025-12-31"
N_SAMPLES = 30
MIN_HISTORY = 60  # bars required before first valid signal date
HOLD_DAYS = 5     # trading days between entry and exit (mirrors backtest.py)
ORACLE_THRESHOLD = 0.02  # same threshold as MomentumStrategy

# Extra calendar-day buffer to load warmup data (mirrors run_backtest logic)
_WARMUP_CAL_DAYS = int(MIN_HISTORY * 1.5) + 30  # ~120 days

OUTPUT = Path(__file__).parent / "golden_cases.jsonl"
# ──────────────────────────────────────────────────────────────────────────────


def _sample_evenly(seq: list, n: int) -> list:
    """Return n elements evenly spaced across seq (inclusive of both ends)."""
    if len(seq) <= n:
        return list(seq)
    indices = np.linspace(0, len(seq) - 1, n, dtype=int)
    # deduplicate while preserving order (linspace rounding can repeat on short lists)
    seen: set[int] = set()
    result = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            result.append(seq[idx])
    return result


def _compute_oracle(
    df: pd.DataFrame, signal_idx: int, all_days: list
) -> tuple[float, str]:
    """Return (forward_return_5d, oracle_signal) for a signal at *signal_idx*.

    Uses T+1 open as entry and T+6 open as exit, matching the backtest convention.
    Caller must ensure signal_idx + HOLD_DAYS + 1 < len(all_days).
    """
    entry_price = float(df.loc[all_days[signal_idx + 1], "Open"])
    exit_price = float(df.loc[all_days[signal_idx + 1 + HOLD_DAYS], "Open"])
    fwd_ret = round((exit_price - entry_price) / entry_price, 6)

    if fwd_ret > ORACLE_THRESHOLD:
        oracle = "BUY"
    elif fwd_ret < -ORACLE_THRESHOLD:
        oracle = "SELL"
    else:
        oracle = "HOLD"

    return fwd_ret, oracle


def build_cases(ticker: str, strategy: MomentumStrategy) -> list[dict]:
    """Return N_SAMPLES golden cases for *ticker*."""
    data_start = (
        pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=_WARMUP_CAL_DAYS)
    ).strftime("%Y-%m-%d")

    df = fetch_ohlcv(ticker, data_start, BACKTEST_END)

    all_days = df.index.tolist()
    backtest_start_ts = pd.Timestamp(BACKTEST_START)

    # Valid signal dates: inside the backtest window, have min_history bars,
    # AND have enough future bars for the oracle (T+1 entry, T+6 exit).
    valid_indexed = [
        (i, day)
        for i, day in enumerate(all_days)
        if i >= MIN_HISTORY
        and day >= backtest_start_ts
        and i + HOLD_DAYS + 1 < len(all_days)
    ]

    sampled = _sample_evenly(valid_indexed, N_SAMPLES)

    cases = []
    for signal_idx, ts in sampled:
        features = compute_features(df, ts)
        evidence = build_evidence_pack(ticker, features)
        rec = strategy.recommend(evidence)
        fwd_ret, oracle = _compute_oracle(df, signal_idx, all_days)
        cases.append(
            {
                "ticker": ticker,
                "date": ts.strftime("%Y-%m-%d"),
                "evidence_pack": evidence,
                "momentum_signal": rec["signal"],
                "momentum_confidence": rec["confidence"],
                "momentum_rationale": rec["rationale"],
                "forward_return_5d": fwd_ret,
                "oracle_signal": oracle,
            }
        )
    return cases


def main() -> None:
    strategy = MomentumStrategy()
    OUTPUT.parent.mkdir(exist_ok=True)

    total = 0
    with open(OUTPUT, "w") as fh:
        for ticker in TICKERS:
            print(f"  {ticker}: sampling {N_SAMPLES} dates from {BACKTEST_START} → {BACKTEST_END} …")
            cases = build_cases(ticker, strategy)
            for case in cases:
                fh.write(json.dumps(case) + "\n")
            total += len(cases)
            # Signal distribution summary
            def _dist(key: str) -> str:
                sigs = [c[key] for c in cases]
                d = {s: sigs.count(s) for s in ("BUY", "SELL", "HOLD")}
                return f"BUY={d['BUY']} SELL={d['SELL']} HOLD={d['HOLD']}"
            print(f"    {len(cases)} cases  momentum: {_dist('momentum_signal')}  oracle: {_dist('oracle_signal')}")

    print(f"\nTotal: {total} cases → {OUTPUT}")


if __name__ == "__main__":
    main()
