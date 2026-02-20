"""
backtest.py - Walk-forward historical backtest.

Trade mapping (deterministic):
  BUY  → enter at next open, exit at open 5 trading days later  (long)
  SELL → enter at next open, exit at open 5 trading days later  (short)
  HOLD → no position opened

No future leakage: features on day T use only data up to T.
The signal on day T triggers entry at open on day T+1 and exit at open on day T+6.

Outputs per run:
  - trades DataFrame  : one row per closed trade
  - metrics dict      : per-ticker and aggregate statistics
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.critic import critique_recommendation
from src.data import fetch_ohlcv
from src.features import compute_features
from src.recommend import MomentumStrategy, LLMStrategy, build_evidence_pack


HOLD_DAYS = 5  # trading days between entry and exit opens


def _get_strategy(strategy_name: str, model: str | None = None):
    if strategy_name == "momentum":
        return MomentumStrategy()
    elif strategy_name == "llm":
        return LLMStrategy() if model is None else LLMStrategy(model=model)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name!r}")


def run_ticker_backtest(
    ticker: str,
    df: pd.DataFrame,
    strategy_name: str = "momentum",
    min_history: int = 60,
    skip_hold_prefilter: bool = False,
    trade_start: pd.Timestamp | None = None,
    model: str | None = None,
    critic: bool = False,
    critic_model: str | None = None,
    rec_cache: dict | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest for a single ticker.

    Parameters
    ----------
    ticker         : ticker symbol
    df             : full OHLCV DataFrame
    strategy_name  : "momentum" or "llm"
    min_history    : minimum rows before first signal

    Returns
    -------
    trades : DataFrame with columns
        signal_date, entry_date, exit_date, signal, entry_price,
        exit_price, pnl_pct, ticker, strategy

    skip_hold_prefilter : bool
        When True, run MomentumStrategy first and skip the primary strategy
        call entirely when momentum says HOLD. Useful for reducing LLM API
        calls (only invoke the LLM on candidate days, not every trading day).
        Has no effect when strategy_name == "momentum".
    rec_cache : dict, optional
        Shared mutable dict used to cache primary strategy recommendations,
        keyed by ``"<ticker>/<signal_date>"``.  On a cache hit the dict
        value is used directly and the strategy is not called.  On a miss
        the strategy is called and the result is stored for future lookups.
        Pass the same dict across multiple ``run_ticker_backtest`` calls (or
        to ``run_backtest``) to amortise repeated LLM calls across tickers.
    """
    strategy = _get_strategy(strategy_name, model=model)
    prefilter = MomentumStrategy() if skip_hold_prefilter and strategy_name != "momentum" else None
    trading_days = df.index.tolist()

    trades = []

    for i, signal_date in enumerate(trading_days):
        # Need at least min_history bars before this date
        if i < min_history:
            continue
        # Need at least HOLD_DAYS+1 bars after this date for entry + exit
        if i + HOLD_DAYS + 1 >= len(trading_days):
            break
        # Only generate signals within the requested trading window
        if trade_start is not None and signal_date < trade_start:
            continue

        entry_idx = i + 1
        exit_idx = i + 1 + HOLD_DAYS

        entry_date = trading_days[entry_idx]
        exit_date = trading_days[exit_idx]

        # Compute features using only history up to signal_date
        features = compute_features(df, signal_date)
        evidence = build_evidence_pack(ticker, features)

        # Optional cheap pre-filter: skip expensive strategy call on HOLD days
        if prefilter is not None and prefilter.recommend(evidence)["signal"] == "HOLD":
            continue

        cache_key = f"{ticker}/{signal_date.strftime('%Y-%m-%d')}"
        if rec_cache is not None and cache_key in rec_cache:
            rec = rec_cache[cache_key]
        else:
            rec = strategy.recommend(evidence)
            if rec_cache is not None:
                rec_cache[cache_key] = rec
        signal = rec["signal"]

        # Optional critic pass
        critic_fields: dict = {}
        if critic:
            critic_out = critique_recommendation(
                evidence, rec,
                model=critic_model,
                client=getattr(strategy, "_client", None),
            )
            signal = critic_out["stance_after"]
            critic_fields = {
                "critic_agree": critic_out["critic_agree"],
                "confidence_before": critic_out["confidence_before"],
                "confidence_after": critic_out["confidence_after"],
                "stance_before": critic_out["stance_before"],
                "stance_after": critic_out["stance_after"],
            }

        if signal == "HOLD":
            continue

        entry_price = float(df.loc[entry_date, "Open"])
        exit_price = float(df.loc[exit_date, "Open"])

        if signal == "BUY":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SELL / short
            pnl_pct = (entry_price - exit_price) / entry_price

        trades.append(
            {
                "ticker": ticker,
                "strategy": strategy_name,
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "signal": signal,
                "confidence": round(rec["confidence"], 4),
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "pnl_pct": round(pnl_pct, 6),
                **critic_fields,
            }
        )

    return pd.DataFrame(trades)


def compute_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    """Aggregate trade-level metrics."""
    if trades.empty:
        return {"n_trades": 0}

    pnl = trades["pnl_pct"]
    winners = pnl[pnl > 0]
    losers = pnl[pnl <= 0]

    total_return = float((1 + pnl).prod() - 1)
    avg_trade = float(pnl.mean())
    win_rate = float(len(winners) / len(pnl))
    avg_win = float(winners.mean()) if len(winners) else 0.0
    avg_loss = float(losers.mean()) if len(losers) else 0.0
    profit_factor = (
        float(winners.sum() / abs(losers.sum()))
        if losers.sum() != 0
        else float("inf")
    )
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252 / HOLD_DAYS)) if pnl.std() > 0 else 0.0
    max_drawdown = _max_drawdown(pnl)

    return {
        "n_trades": int(len(trades)),
        "n_buy": int((trades["signal"] == "BUY").sum()),
        "n_sell": int((trades["signal"] == "SELL").sum()),
        "total_return_pct": round(total_return * 100, 4),
        "avg_trade_pct": round(avg_trade * 100, 4),
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(avg_win * 100, 4),
        "avg_loss_pct": round(avg_loss * 100, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 4),
    }


def _max_drawdown(pnl: pd.Series) -> float:
    equity = (1 + pnl).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def confidence_analysis(
    trades: pd.DataFrame,
    buckets: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Break down trade performance by confidence bucket.

    Parameters
    ----------
    trades  : trades DataFrame (must have ``confidence`` and ``pnl_pct`` columns)
    buckets : list of (lo, hi) pairs; default four buckets from 0 to 1

    Returns
    -------
    DataFrame with columns: bucket, n_trades, win_rate, avg_pnl_pct, sharpe
    """
    if trades.empty or "confidence" not in trades.columns:
        return pd.DataFrame(columns=["bucket", "n_trades", "win_rate", "avg_pnl_pct", "sharpe"])

    if buckets is None:
        buckets = [(0.0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)]

    rows = []
    for lo, hi in buckets:
        mask = (trades["confidence"] >= lo) & (trades["confidence"] < hi)
        subset = trades.loc[mask, "pnl_pct"]
        n = len(subset)
        if n == 0:
            rows.append({"bucket": f"[{lo:.2f},{hi:.2f})", "n_trades": 0,
                         "win_rate": None, "avg_pnl_pct": None, "sharpe": None})
            continue
        win_rate = float((subset > 0).sum() / n)
        avg_pnl = float(subset.mean())
        sharpe = (
            float(subset.mean() / subset.std() * np.sqrt(252 / HOLD_DAYS))
            if subset.std() > 0 else 0.0
        )
        rows.append({
            "bucket": f"[{lo:.2f},{hi:.2f})",
            "n_trades": n,
            "win_rate": round(win_rate, 4),
            "avg_pnl_pct": round(avg_pnl * 100, 4),
            "sharpe": round(sharpe, 4),
        })

    return pd.DataFrame(rows)


def run_backtest(
    tickers: list[str],
    start: str,
    end: str,
    strategy_name: str = "momentum",
    use_cache: bool = True,
    skip_hold_prefilter: bool = False,
    min_history: int = 60,
    model: str | None = None,
    critic: bool = False,
    critic_model: str | None = None,
    rec_cache: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run walk-forward backtest across all tickers.

    Returns
    -------
    all_trades : combined trades DataFrame
    summary    : dict with per-ticker metrics + aggregate
    """
    all_trades = []
    per_ticker: dict[str, Any] = {}

    # Fetch extra warmup history so min_history is satisfied even for short windows.
    # ~1.5 calendar days per trading day, plus a small buffer.
    warmup_calendar_days = int(min_history * 1.5) + 30
    data_start = (
        pd.Timestamp(start) - pd.Timedelta(days=warmup_calendar_days)
    ).strftime("%Y-%m-%d")

    for ticker in tickers:
        label = f"{strategy_name}" + ("+prefilter" if skip_hold_prefilter else "")
        print(f"  Backtesting {ticker} [{label}] ...")
        df = fetch_ohlcv(ticker, data_start, end, use_cache=use_cache)
        # Restrict signal generation to the requested trading window
        trade_start = pd.Timestamp(start)
        trades = run_ticker_backtest(
            ticker, df,
            strategy_name=strategy_name,
            skip_hold_prefilter=skip_hold_prefilter,
            trade_start=trade_start,
            model=model,
            critic=critic,
            critic_model=critic_model,
            rec_cache=rec_cache,
        )
        metrics = compute_metrics(trades)
        per_ticker[ticker] = metrics
        if not trades.empty:
            all_trades.append(trades)

    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    aggregate = compute_metrics(combined)

    summary = {
        "strategy": strategy_name,
        "start": start,
        "end": end,
        "tickers": tickers,
        "aggregate": aggregate,
        "per_ticker": per_ticker,
    }

    return combined, summary
