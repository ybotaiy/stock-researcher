"""
evals.py - Deterministic graders for recommendation outputs.

Loads golden cases from evals/golden_cases.jsonl, runs a strategy on each
evidence pack, and applies graders to the (case, recommendation) pairs.

Graders
-------
signal_valid             : output signal is exactly BUY | SELL | HOLD
agrees_with_momentum     : signal matches the momentum ground-truth label
no_direction_flip        : signal is not the opposite of momentum (BUY↔SELL reversal)
rationale_not_empty      : rationale has at least 10 non-whitespace characters
rationale_cites_evidence : rationale mentions a field name or a numeric value
signal_matches_rationale : BUY rationale uses bullish language; SELL uses bearish

Usage
-----
    .venv/bin/python -m src.evals [--strategy momentum|llm] [--cases evals/golden_cases.jsonl] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommend import MomentumStrategy, LLMStrategy

# ── constants ─────────────────────────────────────────────────────────────────

VALID_SIGNALS = {"BUY", "SELL", "HOLD"}

# Evidence field names the LLM should cite in its rationale
_EVIDENCE_FIELDS = frozenset({
    "ret_1d", "ret_5d", "ret_20d",
    "vol_20d", "ma_20", "ma_50",
    "price_to_ma20", "price_to_ma50",
    "volume_ratio", "close",
})

_NUMERIC_RE = re.compile(r"\d+\.\d+")
_EVIDENCE_NUMERIC_FIELDS = frozenset({
    "ret_1d", "ret_5d", "ret_20d", "vol_20d",
    "price_to_ma20", "price_to_ma50", "volume_ratio",
    "close", "ma_20", "ma_50",
})

_BULLISH_WORDS = frozenset({
    "bullish", "positive", "above", "momentum", "upward", "gain", "strong", "buy",
})
_BEARISH_WORDS = frozenset({
    "bearish", "negative", "below", "decline", "weak", "downward", "loss", "sell", "drop",
})

DEFAULT_CASES = Path(__file__).parent.parent / "evals" / "golden_cases.jsonl"

# ── types ─────────────────────────────────────────────────────────────────────

GradeResult = dict[str, Any]   # {name: str, passed: bool, detail: str}
Grader = Callable[[dict, dict], GradeResult]


# ── graders ───────────────────────────────────────────────────────────────────

def signal_valid(case: dict, rec: dict) -> GradeResult:
    """Output signal is exactly one of BUY | SELL | HOLD."""
    sig = rec.get("signal", "")
    passed = sig in VALID_SIGNALS
    return {
        "name": "signal_valid",
        "passed": passed,
        "detail": f"got {sig!r}" if not passed else "",
    }


def agrees_with_momentum(case: dict, rec: dict) -> GradeResult:
    """Signal matches the momentum ground-truth label."""
    expected = case["momentum_signal"]
    actual = rec.get("signal", "")
    passed = actual == expected
    return {
        "name": "agrees_with_momentum",
        "passed": passed,
        "detail": f"expected={expected} got={actual}" if not passed else "",
    }


def no_direction_flip(case: dict, rec: dict) -> GradeResult:
    """Signal is not the active opposite of momentum (BUY↔SELL reversal)."""
    expected = case["momentum_signal"]
    actual = rec.get("signal", "")
    flipped = (
        (expected == "BUY" and actual == "SELL")
        or (expected == "SELL" and actual == "BUY")
    )
    return {
        "name": "no_direction_flip",
        "passed": not flipped,
        "detail": f"momentum={expected} but signal={actual}" if flipped else "",
    }


def rationale_not_empty(case: dict, rec: dict) -> GradeResult:
    """Rationale has at least 10 non-whitespace characters."""
    rationale = rec.get("rationale", "")
    length = len(rationale.strip())
    passed = length >= 10
    return {
        "name": "rationale_not_empty",
        "passed": passed,
        "detail": f"length={length}" if not passed else "",
    }


def rationale_cites_evidence(case: dict, rec: dict) -> GradeResult:
    """Rationale mentions at least one evidence field name or a numeric value."""
    rationale = rec.get("rationale", "").lower()
    field_hit = next((f for f in _EVIDENCE_FIELDS if f in rationale), None)
    number_hit = bool(re.search(r"\d+\.\d+", rationale))
    passed = field_hit is not None or number_hit
    if not passed:
        detail = "no field name or numeric value found"
    elif field_hit:
        detail = f"cites field '{field_hit}'"
    else:
        detail = "cites numeric value"
    return {
        "name": "rationale_cites_evidence",
        "passed": passed,
        "detail": detail,
    }


def out_of_range(case: dict, rec: dict) -> GradeResult:
    """Any float in the rationale exceeds 3× the max absolute evidence value."""
    ep = case.get("evidence_pack", {})
    ev_vals = [abs(ep[f]) for f in _EVIDENCE_NUMERIC_FIELDS if f in ep and ep[f] is not None]
    if not ev_vals:
        return {"name": "out_of_range", "passed": True, "detail": "no evidence values"}
    ceiling = max(ev_vals) * 3
    nums = [float(m) for m in _NUMERIC_RE.findall(rec.get("rationale", ""))]
    bad = [n for n in nums if n > ceiling]
    passed = not bad
    return {
        "name": "out_of_range",
        "passed": passed,
        "detail": f"implausible values: {bad}" if bad else "",
    }


def agrees_with_oracle(case: dict, rec: dict) -> GradeResult:
    """Signal matches the actual 5-day forward return direction."""
    oracle = case.get("oracle_signal")
    if oracle is None:
        return {"name": "agrees_with_oracle", "passed": True, "detail": "no oracle — skipped"}
    actual = rec.get("signal", "")
    passed = actual == oracle
    fwd = case.get("forward_return_5d")
    fwd_str = f" fwd={fwd:.2%}" if fwd is not None else ""
    return {
        "name": "agrees_with_oracle",
        "passed": passed,
        "detail": f"oracle={oracle} got={actual}{fwd_str}" if not passed else f"oracle={oracle}{fwd_str}",
    }


def signal_matches_rationale(case: dict, rec: dict) -> GradeResult:
    """BUY rationale contains bullish language; SELL rationale contains bearish language."""
    signal = rec.get("signal", "")
    words = set(re.findall(r"[a-z]+", rec.get("rationale", "").lower()))

    if signal == "BUY":
        matched = words & _BULLISH_WORDS
        passed = bool(matched)
        detail = "no bullish keywords" if not passed else f"found: {sorted(matched)}"
    elif signal == "SELL":
        matched = words & _BEARISH_WORDS
        passed = bool(matched)
        detail = "no bearish keywords" if not passed else f"found: {sorted(matched)}"
    else:
        # HOLD rationale varies too much to keyword-match
        passed = True
        detail = "HOLD — not checked"

    return {
        "name": "signal_matches_rationale",
        "passed": passed,
        "detail": detail,
    }


def confidence_calibrated(case: dict, rec: dict) -> GradeResult:
    """High-confidence predictions should agree with the oracle more often.

    Fails when confidence >= 0.7 AND the signal disagrees with oracle.
    HOLD signals or missing oracle are always a pass (nothing to validate).
    """
    signal = rec.get("signal", "HOLD")
    confidence = rec.get("confidence", 0.5)
    oracle = case.get("oracle_signal")

    if signal == "HOLD" or oracle is None:
        return {"name": "confidence_calibrated", "passed": True,
                "detail": "HOLD or no oracle", "confidence": confidence}

    agrees = signal == oracle
    passed = not (confidence >= 0.7 and not agrees)
    return {
        "name": "confidence_calibrated",
        "passed": passed,
        "detail": f"conf={confidence:.2f} oracle={oracle} signal={signal}",
        "confidence": confidence,
    }


ALL_GRADERS: list[Grader] = [
    signal_valid,
    agrees_with_momentum,
    agrees_with_oracle,
    no_direction_flip,
    rationale_not_empty,
    rationale_cites_evidence,
    out_of_range,
    signal_matches_rationale,
    confidence_calibrated,
]

# ── scorecard ─────────────────────────────────────────────────────────────────

_BUCKET_GRADER = {
    "schema_fail":        "signal_valid",
    "out_of_range":       "out_of_range",
    "ungrounded_claim":   "rationale_cites_evidence",
    "logic_conflict":     "signal_matches_rationale",
    "overconfident_miss": "confidence_calibrated",
}


def build_scorecard(report: dict) -> dict[str, int]:
    """Map grader failure counts to the four named failure buckets."""
    summary = report["grader_summary"]
    n = report["n_cases"]
    return {
        bucket: n - summary[grader]["passed"]
        for bucket, grader in _BUCKET_GRADER.items()
        if grader in summary
    }


# ── runner ────────────────────────────────────────────────────────────────────

def _load_cases(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _get_strategy(name: str, model: str | None = None):
    if name == "momentum":
        return MomentumStrategy()
    if name == "llm":
        return LLMStrategy() if model is None else LLMStrategy(model=model)
    raise ValueError(f"Unknown strategy: {name!r}")


def run_evals(
    strategy_name: str = "llm",
    cases_path: Path = DEFAULT_CASES,
    graders: list[Grader] | None = None,
    verbose: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    """Run all graders on every golden case.

    Parameters
    ----------
    strategy_name : strategy to evaluate ("momentum" or "llm")
    cases_path    : path to golden_cases.jsonl
    graders       : grader list (defaults to ALL_GRADERS)
    verbose       : print one line per case

    Returns
    -------
    report : dict with keys
        strategy, n_cases, grader_summary, per_ticker, confusion_vs_momentum
    """
    if graders is None:
        graders = ALL_GRADERS

    cases = _load_cases(cases_path)
    strategy = _get_strategy(strategy_name, model=model)

    results: list[dict] = []
    for case in cases:
        rec = strategy.recommend(case["evidence_pack"])
        grades = [g(case, rec) for g in graders]

        if verbose:
            fails = [g["name"] for g in grades if not g["passed"]]
            status = "ok" if not fails else f"FAIL({','.join(fails)})"
            print(
                f"  {case['ticker']} {case['date']}"
                f"  mom={case['momentum_signal']:<4}"
                f"  rec={rec.get('signal', '?'):<4}"
                f"  {status}"
            )

        results.append(
            {
                "ticker": case["ticker"],
                "date": case["date"],
                "momentum_signal": case["momentum_signal"],
                "oracle_signal": case.get("oracle_signal"),
                "rec_signal": rec.get("signal", ""),
                "rec_confidence": rec.get("confidence", 0.5),
                "grades": grades,
            }
        )

    return _build_report(strategy_name, results, graders)


# ── report helpers ────────────────────────────────────────────────────────────

def _build_report(
    strategy_name: str,
    results: list[dict],
    graders: list[Grader],
) -> dict[str, Any]:
    grader_names = [g.__name__ for g in graders]
    n = len(results)

    def _pass_count(subset: list[dict], gname: str) -> int:
        return sum(
            1 for r in subset
            if next(g for g in r["grades"] if g["name"] == gname)["passed"]
        )

    # Aggregate pass rates across all cases
    grader_summary: dict[str, Any] = {
        gname: {
            "passed": _pass_count(results, gname),
            "total": n,
            "pass_rate": round(_pass_count(results, gname) / n, 4) if n else 0.0,
        }
        for gname in grader_names
    }

    # Per-ticker pass rates for each grader
    tickers = sorted({r["ticker"] for r in results})
    per_ticker: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        subset = [r for r in results if r["ticker"] == ticker]
        per_ticker[ticker] = {
            gname: round(_pass_count(subset, gname) / len(subset), 4)
            for gname in grader_names
        }

    def _confusion(ref_key: str) -> dict[str, dict[str, int]]:
        matrix: dict[str, dict[str, int]] = {
            s: {"BUY": 0, "SELL": 0, "HOLD": 0} for s in VALID_SIGNALS
        }
        for r in results:
            ref = r.get(ref_key)
            rec = r["rec_signal"]
            if ref in matrix and rec in matrix[ref]:
                matrix[ref][rec] += 1
        return matrix

    # Confidence calibration analysis: compare oracle agreement rate
    # for high-confidence vs low-confidence non-HOLD predictions.
    non_hold = [r for r in results if r["rec_signal"] != "HOLD" and r.get("oracle_signal") is not None]
    high = [r for r in non_hold if r["rec_confidence"] >= 0.7]
    low = [r for r in non_hold if r["rec_confidence"] < 0.7]

    def _agree_rate(subset: list[dict]) -> float | None:
        if not subset:
            return None
        return round(sum(1 for r in subset if r["rec_signal"] == r["oracle_signal"]) / len(subset), 4)

    high_rate = _agree_rate(high)
    low_rate = _agree_rate(low)
    if high_rate is not None and low_rate is not None:
        delta = round(high_rate - low_rate, 4)
    else:
        delta = None

    confidence_calibration = {
        "low_n": len(low),
        "low_oracle_agree_rate": low_rate,
        "high_n": len(high),
        "high_oracle_agree_rate": high_rate,
        "calibration_delta": delta,
    }

    report = {
        "strategy": strategy_name,
        "n_cases": n,
        "grader_summary": grader_summary,
        "per_ticker": per_ticker,
        "confusion_vs_momentum": _confusion("momentum_signal"),
        "confusion_vs_oracle": _confusion("oracle_signal"),
        "confidence_calibration": confidence_calibration,
    }
    report["scorecard"] = build_scorecard(report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  Eval report  strategy={report['strategy']}  n={report['n_cases']}")
    print(f"{'=' * width}")

    print("\nGrader pass rates:")
    for gname, stats in report["grader_summary"].items():
        filled = int(stats["pass_rate"] * 20)
        bar = "#" * filled + "-" * (20 - filled)
        print(
            f"  {gname:<35} {stats['passed']:3}/{stats['total']}"
            f"  [{bar}]  {stats['pass_rate']:.0%}"
        )

    print("\nPer-ticker — agrees_with_momentum:")
    for ticker, grades in report["per_ticker"].items():
        rate = grades.get("agrees_with_momentum", 0.0)
        print(f"  {ticker:<6}  {rate:.0%}")

    def _print_confusion(title: str, matrix: dict) -> None:
        print(f"\n{title}  (row=reference  col=strategy):")
        print(f"  {'':6}  {'BUY':>5}  {'SELL':>5}  {'HOLD':>5}")
        for ref_sig in ("BUY", "SELL", "HOLD"):
            counts = matrix[ref_sig]
            print(f"  {ref_sig:<6}  {counts['BUY']:>5}  {counts['SELL']:>5}  {counts['HOLD']:>5}")

    _print_confusion("Confusion vs momentum", report["confusion_vs_momentum"])
    _print_confusion("Confusion vs oracle  ", report["confusion_vs_oracle"])

    print(f"\nSCORECARD  (failures out of {report['n_cases']} cases)")
    for bucket, count in report["scorecard"].items():
        print(f"  {bucket:<20}:  {count}")

    cal = report.get("confidence_calibration", {})
    if cal:
        print("\nConfidence calibration:")
        lo_n = cal.get("low_n", 0)
        hi_n = cal.get("high_n", 0)
        lo_rate = cal.get("low_oracle_agree_rate")
        hi_rate = cal.get("high_oracle_agree_rate")
        delta = cal.get("calibration_delta")
        lo_str = f"{lo_rate:.0%}" if lo_rate is not None else "n/a"
        hi_str = f"{hi_rate:.0%}" if hi_rate is not None else "n/a"
        print(f"  Low  (<0.7):  n={lo_n}  oracle_agree= {lo_str}")
        print(f"  High (>=0.7): n={hi_n}  oracle_agree= {hi_str}")
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            quality = "higher confidence -> more accurate" if delta >= 0 else "WARNING: confidence anti-correlated"
            print(f"  Delta: {sign}{delta:.1%}pp  ({quality})")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic evals on golden cases")
    parser.add_argument("--strategy", default="llm", choices=["momentum", "llm"])
    parser.add_argument(
        "--cases", default=str(DEFAULT_CASES), help="Path to golden_cases.jsonl"
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-case results")
    parser.add_argument(
        "--model", default=None,
        help="LLM model ID (default: claude-haiku-4-5-20251001). Only used with --strategy llm.",
    )
    args = parser.parse_args()

    report = run_evals(
        strategy_name=args.strategy,
        cases_path=Path(args.cases),
        verbose=args.verbose,
        model=args.model,
    )
    _print_report(report)


if __name__ == "__main__":
    main()
