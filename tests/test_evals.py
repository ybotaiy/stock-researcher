"""
Smoke test: run_evals(momentum) completes and returns a well-formed report
with the expected scorecard structure.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evals import run_evals


class TestEvalsEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_evals(strategy_name="momentum")

    def test_has_cases(self):
        self.assertGreater(self.report["n_cases"], 0)

    def test_grader_summary_keys(self):
        expected = {
            "signal_valid",
            "rationale_cites_evidence",
            "out_of_range",
            "signal_matches_rationale",
        }
        self.assertTrue(expected.issubset(self.report["grader_summary"].keys()))

    def test_scorecard_keys(self):
        self.assertEqual(
            set(self.report["scorecard"].keys()),
            {"schema_fail", "out_of_range", "ungrounded_claim", "logic_conflict"},
        )

    def test_scorecard_zero_schema_fail(self):
        self.assertEqual(self.report["scorecard"]["schema_fail"], 0)

    def test_scorecard_zero_out_of_range(self):
        self.assertEqual(self.report["scorecard"]["out_of_range"], 0)

    def test_per_ticker_present(self):
        self.assertEqual(set(self.report["per_ticker"].keys()), {"GOOGL", "TSLA", "GLD"})


if __name__ == "__main__":
    unittest.main()
