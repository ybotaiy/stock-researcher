"""
Unit tests for src/critic.py — critique_recommendation().

All Anthropic API calls are mocked so no real network calls occur.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.logging as run_logging
from src.critic import critique_recommendation
from src.run_daily import run_daily


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_EVIDENCE = {
    "ticker": "GOOGL",
    "ret_5d": 0.03,
    "price_to_ma20": 0.02,
    "vol_20d": 0.20,
}

SAMPLE_REC_BUY = {
    "signal": "BUY",
    "confidence": 0.7,
    "rationale": "Strong momentum.",
    "strategy": "momentum",
}


def _make_mock_client(agree: bool, adjusted_signal: str, adjusted_confidence: float, critique: str = "Looks fine."):
    """Build a fake Anthropic client whose messages.create returns the given critic JSON."""
    response_json = json.dumps({
        "agree": agree,
        "adjusted_signal": adjusted_signal,
        "adjusted_confidence": adjusted_confidence,
        "critique": critique,
    })

    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=response_json)]
    mock_message.usage.input_tokens = 100
    mock_message.usage.output_tokens = 50
    # No cache tokens
    del mock_message.usage.cache_creation_input_tokens
    del mock_message.usage.cache_read_input_tokens

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    return mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCriticReturnKeys(unittest.TestCase):
    """critique_recommendation() returns all expected keys."""

    def test_all_keys_present(self):
        client = _make_mock_client(True, "BUY", 0.65)
        result = critique_recommendation(SAMPLE_EVIDENCE, SAMPLE_REC_BUY, client=client)

        expected_keys = {
            "critic_agree",
            "confidence_before",
            "confidence_after",
            "stance_before",
            "stance_after",
            "critique",
            "critic_input_tokens",
            "critic_output_tokens",
            "critic_estimated_cost_usd",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_basic_values(self):
        client = _make_mock_client(True, "BUY", 0.65)
        result = critique_recommendation(SAMPLE_EVIDENCE, SAMPLE_REC_BUY, client=client)

        self.assertTrue(result["critic_agree"])
        self.assertEqual(result["stance_before"], "BUY")
        self.assertEqual(result["stance_after"], "BUY")
        self.assertAlmostEqual(result["confidence_before"], 0.7)
        self.assertAlmostEqual(result["confidence_after"], 0.65)


class TestCriticConfidenceNeverIncreases(unittest.TestCase):
    """adjusted_confidence is always clamped to <= confidence_before."""

    def test_confidence_clamped_down(self):
        client = _make_mock_client(True, "BUY", 0.99)  # LLM tries to raise to 0.99
        result = critique_recommendation(SAMPLE_EVIDENCE, SAMPLE_REC_BUY, client=client)
        # confidence_before = 0.7, so after must be <= 0.7
        self.assertLessEqual(result["confidence_after"], result["confidence_before"])
        self.assertAlmostEqual(result["confidence_after"], 0.7)  # clamped to original

    def test_confidence_can_decrease(self):
        client = _make_mock_client(False, "HOLD", 0.4)
        result = critique_recommendation(SAMPLE_EVIDENCE, SAMPLE_REC_BUY, client=client)
        self.assertLessEqual(result["confidence_after"], result["confidence_before"])
        self.assertAlmostEqual(result["confidence_after"], 0.4)


class TestCriticInvalidSignalDefaultsToHold(unittest.TestCase):
    """If the LLM returns an unrecognised signal, default to HOLD."""

    def test_invalid_signal_becomes_hold(self):
        response_json = json.dumps({
            "agree": False,
            "adjusted_signal": "STRONG_BUY",  # not in SIGNAL_CHOICES
            "adjusted_confidence": 0.6,
            "critique": "Overconfident signal.",
        })
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=response_json)]
        mock_message.usage.input_tokens = 80
        mock_message.usage.output_tokens = 40
        del mock_message.usage.cache_creation_input_tokens
        del mock_message.usage.cache_read_input_tokens

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        result = critique_recommendation(SAMPLE_EVIDENCE, SAMPLE_REC_BUY, client=mock_client)
        self.assertEqual(result["stance_after"], "HOLD")


class TestCriticJsonParseFailure(unittest.TestCase):
    """On JSON parse failure, fall back to agree=True and keep original signal/confidence."""

    def test_fallback_on_bad_json(self):
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Sorry, I cannot answer that.")]
        mock_message.usage.input_tokens = 50
        mock_message.usage.output_tokens = 10
        del mock_message.usage.cache_creation_input_tokens
        del mock_message.usage.cache_read_input_tokens

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        result = critique_recommendation(SAMPLE_EVIDENCE, SAMPLE_REC_BUY, client=mock_client)

        self.assertTrue(result["critic_agree"])
        self.assertEqual(result["stance_after"], "BUY")
        self.assertAlmostEqual(result["confidence_after"], 0.7)


_CANNED_CRITIC_RESULT = {
    "critic_agree": True,
    "confidence_before": 0.7,
    "confidence_after": 0.65,
    "stance_before": "BUY",
    "stance_after": "BUY",
    "critique": "Evidence supports the signal.",
    "critic_input_tokens": 100,
    "critic_output_tokens": 50,
    "critic_estimated_cost_usd": 0.0001,
}


class TestRunDailyWithCritic(unittest.TestCase):
    """Integration: run_daily(critic=True) logs critic columns to run_log.csv."""

    def test_critic_columns_in_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run_log.csv"

            with (
                patch.object(run_logging, "LOG_PATH", log_path),
                patch("src.run_daily.critique_recommendation", return_value=_CANNED_CRITIC_RESULT),
            ):
                run_daily(as_of="2025-01-10", strategy_name="momentum", critic=True)

            self.assertTrue(log_path.exists())

            with open(log_path, newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertIn("critic_agree", row)
                self.assertIn("stance_before", row)
                self.assertIn("stance_after", row)
                self.assertIn("confidence_before", row)
                self.assertIn("confidence_after", row)


class TestRunDailyRecDir(unittest.TestCase):
    """run_daily(rec_dir=...) loads cached recommendation and skips strategy call."""

    def test_cached_rec_is_used_and_critic_receives_it(self):
        """When rec_dir contains recommendation_<TICKER>.json, strategy is not called."""
        canned_rec = {
            "ticker": "GOOGL",
            "as_of": "2025-01-10",
            "signal": "SELL",
            "confidence": 0.8,
            "rationale": "Cached bearish signal.",
            "strategy": "momentum",
        }

        with tempfile.TemporaryDirectory() as rec_dir:
            # Write a pre-existing recommendation for GOOGL only.
            (Path(rec_dir) / "recommendation_GOOGL.json").write_text(
                json.dumps(canned_rec)
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                log_path = Path(tmpdir) / "run_log.csv"

                calls: list[dict] = []

                def _fake_critic(evidence, recommendation, **_kwargs):
                    # Copy recommendation: run_daily mutates rec in-place after
                    # the critic returns, so we need a snapshot of the input.
                    calls.append({"evidence": evidence, "rec": dict(recommendation)})
                    return _CANNED_CRITIC_RESULT

                with (
                    patch.object(run_logging, "LOG_PATH", log_path),
                    patch("src.run_daily.critique_recommendation", side_effect=_fake_critic),
                ):
                    run_dir = run_daily(
                        as_of="2025-01-10",
                        strategy_name="momentum",
                        critic=True,
                        rec_dir=rec_dir,
                    )

                # Critic called for all 3 tickers.
                self.assertEqual(len(calls), 3)

                # For GOOGL the critic received the cached signal, not a freshly
                # computed momentum signal.
                googl_call = next(c for c in calls if c["evidence"]["ticker"] == "GOOGL")
                self.assertEqual(googl_call["rec"]["signal"], "SELL")
                self.assertAlmostEqual(googl_call["rec"]["confidence"], 0.8)

                # The saved recommendation_GOOGL.json should reflect the critic output.
                saved = json.loads((run_dir / "recommendation_GOOGL.json").read_text())
                # critic_agree=True so stance_after == stance_before == "BUY" per canned result;
                # confirm the file exists and has the expected key.
                self.assertIn("signal", saved)


class TestBacktestRecCache(unittest.TestCase):
    """run_ticker_backtest honours the rec_cache dict."""

    def test_cache_hit_skips_strategy_call(self):
        """A pre-populated rec_cache entry is returned without calling strategy.recommend."""
        import pandas as pd
        from src.backtest import run_ticker_backtest

        # Build a minimal OHLCV DataFrame with enough rows (>60 + 6).
        n = 80
        dates = pd.bdate_range("2024-01-01", periods=n)
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [101.0] * n,
                "Low": [99.0] * n,
                "Close": [100.0] * n,
                "Volume": [1_000_000] * n,
            },
            index=dates,
        )

        # The cache has entries for every candidate signal date keyed by
        # "GOOGL/<date>".  Strategy.recommend should never be called.
        signal_dates = [d.strftime("%Y-%m-%d") for d in dates]
        rec_cache = {
            f"GOOGL/{d}": {"signal": "BUY", "confidence": 0.9, "rationale": "cached"}
            for d in signal_dates
        }

        call_count = {"n": 0}

        class _SpyStrategy:
            def recommend(self, evidence):
                call_count["n"] += 1
                return {"signal": "HOLD", "confidence": 0.5, "rationale": "spy"}

        # Monkey-patch _get_strategy to return the spy.
        import src.backtest as bt_module
        orig = bt_module._get_strategy
        bt_module._get_strategy = lambda *a, **kw: _SpyStrategy()
        try:
            trades = run_ticker_backtest(
                "GOOGL", df,
                strategy_name="momentum",
                rec_cache=rec_cache,
            )
        finally:
            bt_module._get_strategy = orig

        # Strategy was never called because every date was in the cache.
        self.assertEqual(call_count["n"], 0)
        # All cached signals were BUY so we should have trades.
        self.assertFalse(trades.empty)


if __name__ == "__main__":
    unittest.main()
