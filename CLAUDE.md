# Stock Researcher — Claude Code Project

## What this is
A stock watchlist research agent for **GOOGL, TSLA, GLD** with:
- Deterministic momentum baseline strategy
- LLM-based recommendation layer (claude-sonnet-4-6)
- Walk-forward backtest (no future leakage)

## Stack
- Python 3.9 in `.venv/` — always use `.venv/bin/python`
- `yfinance` for market data, `pandas`/`numpy` for features, `anthropic` SDK for LLM
- CSV cache under `data/<TICKER>/` keyed by (start, end)
- Outputs saved to `runs/` with timestamps

## Module layout
| File | Role |
|------|------|
| `src/data.py` | `fetch_ohlcv(ticker, start, end)` — yfinance + CSV cache |
| `src/features.py` | `compute_features(df, as_of)` — returns, vol, MAs, volume ratio |
| `src/recommend.py` | `MomentumStrategy`, `LLMStrategy`, `build_evidence_pack()` |
| `src/critic.py` | `critique_recommendation()` — second LLM pass to review and possibly downgrade a signal |
| `src/backtest.py` | `run_backtest()` — walk-forward, warmup window, prefilter support |
| `src/run_daily.py` | Daily artifact generator (evidence_pack + recommendation JSON) |
| `src/run_backtest.py` | CLI backtest runner |

## Key conventions
- Features are always computed with `df[df.index <= as_of]` — no lookahead
- Trade mapping: signal on day T → entry open T+1 → exit open T+6 (5 trading days)
- `min_history=60` bars required before first signal; `run_backtest()` auto-fetches warmup data before `start`
- LLM reads **only** fields from `evidence_pack.json` (enforced by system prompt)

## Common commands
```bash
# Daily run (momentum)
.venv/bin/python -m src.run_daily --date 2025-01-10 --strategy momentum

# Daily run (LLM)
.venv/bin/python -m src.run_daily --date 2025-01-10 --strategy llm

# Daily run with critic (momentum + LLM critic review)
.venv/bin/python -m src.run_daily --date 2025-01-10 --strategy momentum --critic

# Daily run with critic using a specific critic model
.venv/bin/python -m src.run_daily --date 2025-01-10 --strategy momentum --critic --critic-model claude-haiku-4-5-20251001

# Backtest — momentum, full range
.venv/bin/python -m src.run_backtest --strategy momentum --start 2023-01-01 --end 2025-12-31

# Backtest — LLM with momentum prefilter (~47% fewer API calls)
.venv/bin/python -m src.run_backtest --strategy llm --skip-hold-prefilter --start 2025-10-01 --end 2025-12-31

# Backtest — with critic enabled
.venv/bin/python -m src.run_backtest --strategy momentum --critic --start 2025-01-01 --end 2025-03-31
```

## Momentum strategy rules
- **BUY**: `ret_5d > 2%` AND `price > MA20`
- **SELL**: `ret_5d < -2%` AND `price < MA20`
- **Vol filter**: HOLD if 20d annualised vol > 60%

## LLM strategy
- Model: `claude-sonnet-4-6` (set in `LLMStrategy.__init__`)
- Requires `ANTHROPIC_API_KEY` in environment (stored in `.env`)
- Use `--skip-hold-prefilter` to cut API calls by ~47% and cost by ~$0.27 on the full 3-year run

## LLM critic (opt-in)
- Enabled via `--critic` flag on both `run_daily` and `run_backtest`
- Runs a second LLM call (`critique_recommendation()` in `src/critic.py`) after the primary signal
- Critic may downgrade signal to HOLD or reduce confidence; it never increases confidence
- Default critic model: `claude-haiku-4-5-20251001` (override with `--critic-model`)
- Artifacts: `critic_<TICKER>.json` saved alongside recommendation JSON in the run directory
- Log columns added when critic is enabled: `critic_agree`, `confidence_before`, `confidence_after`, `stance_before`, `stance_after`, `critic_input_tokens`, `critic_output_tokens`, `critic_estimated_cost_usd`

## Backtest results (momentum, 2023-01-01 → 2025-12-31)
| Ticker | Trades | Win rate | Sharpe | Profit factor |
|--------|--------|----------|--------|---------------|
| GOOGL  | 366    | 48.1%    | 0.01   | 1.00          |
| TSLA   | 315    | 54.3%    | 1.09   | 1.51          |
| GLD    | 214    | 56.1%    | 0.48   | 1.19          |
| **All**| **895**| **52.2%**| **0.61**| **1.28**    |

## Dev workflow
- After any code change, always run `.venv/bin/python -m pytest tests/ -v`

## GitHub
https://github.com/ybotaiy/stock-researcher
