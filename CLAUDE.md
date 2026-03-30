<!-- agent-context:begin -->
## Personal KB Context
Use the linked personal KB before planning or status updates.

Read first:
1. `.agent-context/profile/communication_style.md`
2. `.agent-context/profile/decision_preferences.md`
3. `.agent-context/profile/constraints.md`
4. `.agent-context/profile/personal_context.md`
5. `.agent-context/status/active_priorities.md`
6. `.agent-context/status/weekly_status.md`
7. `.agent-context/projects/current.md`

Canonical project file: `agent-context/projects/stock-researcher.md`

Rules:
- Keep durable project facts, blockers, decisions, and next steps in the canonical KB.
- If this repo changes project reality, update `.agent-context/projects/current.md` and the relevant KB status files.
- Do not create duplicate personal project summaries in this repo.
<!-- agent-context:end -->

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
| `src/recommend.py` | `MomentumStrategy`, `LLMStrategy`, `ClaudeCodeStrategy`, `build_evidence_pack()` |
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

# Daily run (Claude Code CLI — no API key needed)
.venv/bin/python -m src.run_daily --date 2025-01-10 --strategy claude-code

# Daily run (Claude Code CLI + critic)
.venv/bin/python -m src.run_daily --date 2025-01-10 --strategy claude-code --critic

# Backtest — Claude Code CLI with momentum prefilter
.venv/bin/python -m src.run_backtest --strategy claude-code --skip-hold-prefilter --start 2025-10-01 --end 2025-12-31
```

## Momentum strategy rules
- **BUY**: `ret_5d > 2%` AND `price > MA20`
- **SELL**: `ret_5d < -2%` AND `price < MA20`
- **Vol filter**: HOLD if 20d annualised vol > 60%

### Momentum confidence formula
Non-HOLD signals get a continuous confidence in `[0.50, 0.95]` via `_compute_momentum_confidence()`:
- **ret_strength** (wt 0.50): `min((|ret_5d| - threshold) / 0.10, 1.0)`
- **ma_strength** (wt 0.30): `min(|price_to_ma20| / 0.05, 1.0)`
- **vol_penalty** (wt 0.20): `1 - min(vol / vol_cap, 1.0)`
- **Final**: `0.50 + raw * 0.45`

HOLD signals always have `confidence = 0.5`.

## LLM strategy
- Model: `claude-sonnet-4-6` (set in `LLMStrategy.__init__`)
- Requires `ANTHROPIC_API_KEY` in environment (stored in `.env`)
- Use `--skip-hold-prefilter` to cut API calls by ~47% and cost by ~$0.27 on the full 3-year run

## Claude Code CLI strategy
- Uses `claude -p` subprocess instead of the Anthropic Python SDK
- **No `ANTHROPIC_API_KEY` needed** — uses Claude Code's own authentication
- `--json-schema` flag enforces structured output; `--output-format json` provides cost metadata
- Default model: `sonnet` (CLI alias); override with `--model`
- Token counts not available (CLI doesn't expose breakdown); cost comes from `total_cost_usd` in wrapper
- Performance note: subprocess overhead (~1-2s per call) makes this slower than direct API calls

## LLM critic (opt-in)
- Enabled via `--critic` flag on both `run_daily` and `run_backtest`
- Runs a second LLM call (`critique_recommendation()` in `src/critic.py`) after the primary signal
- Critic may downgrade signal to HOLD or reduce confidence; it never increases confidence
- Default critic model: `claude-haiku-4-5-20251001` (override with `--critic-model`)
- Artifacts: `critic_<TICKER>.json` saved alongside recommendation JSON in the run directory
- Log columns added when critic is enabled: `critic_agree`, `confidence_before`, `confidence_after`, `stance_before`, `stance_after`, `critic_input_tokens`, `critic_output_tokens`, `critic_estimated_cost_usd`

## Recommendation cache (skip primary LLM on critic reruns)

**Backtest** — `rec_cache.json` is auto-saved to every LLM backtest run directory.
Pass it back with `--rec-cache` to run the critic without re-calling the primary LLM:
```bash
# Step 1: generate recs (saves rec_cache.json automatically)
.venv/bin/python -m src.run_backtest --strategy llm --skip-hold-prefilter \
    --start 2025-10-01 --end 2025-12-31

# Step 2: critic-only rerun — zero primary LLM calls
.venv/bin/python -m src.run_backtest --strategy llm --critic --skip-hold-prefilter \
    --rec-cache runs/backtest_llm_2025-10-01_2025-12-31_<ts>/rec_cache.json \
    --start 2025-10-01 --end 2025-12-31
```

**Daily** — pass `--rec-dir` pointing to an existing daily run directory:
```bash
# Step 1: generate recs
.venv/bin/python -m src.run_daily --strategy llm --date 2025-01-10

# Step 2: critic-only rerun using saved recommendation_<TICKER>.json files
.venv/bin/python -m src.run_daily --strategy llm --critic \
    --rec-dir runs/daily_2025-01-10_<ts>/ --date 2025-01-10
```

Cache key format: `"<TICKER>/<YYYY-MM-DD>"` → recommendation dict.

## Backtest results (momentum, 2023-01-01 → 2025-12-31)
| Ticker | Trades | Win rate | Sharpe | Profit factor |
|--------|--------|----------|--------|---------------|
| GOOGL  | 366    | 48.1%    | 0.01   | 1.00          |
| TSLA   | 315    | 54.3%    | 1.09   | 1.51          |
| GLD    | 214    | 56.1%    | 0.48   | 1.19          |
| **All**| **895**| **52.2%**| **0.61**| **1.28**    |

## Confidence tracking
- `confidence` column is always present in `trades.csv` (both momentum and LLM)
- `confidence_analysis.csv` is saved to every backtest run directory with per-bucket breakdown
- Evals include `confidence_calibrated` grader and `overconfident_miss` scorecard bucket
- Confidence calibration summary in eval report compares oracle agreement rates for high (>=0.7) vs low (<0.7) confidence predictions

## Dev workflow
- After any code change, always run `.venv/bin/python -m pytest tests/ -v`

## GitHub
https://github.com/ybotaiy/stock-researcher
