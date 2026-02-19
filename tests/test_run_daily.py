"""
Smoke test: run_daily writes exactly 3 records (one per ticker) to run_log.csv.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.logging as run_logging
from src.run_daily import run_daily


class TestRunDailyLogging(unittest.TestCase):
    def test_writes_three_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run_log.csv"

            # Redirect LOG_PATH to a temp file
            with patch.object(run_logging, "LOG_PATH", log_path):
                run_daily(as_of="2025-01-10", strategy_name="momentum")

            self.assertTrue(log_path.exists(), "run_log.csv was not created")

            with open(log_path, newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 3, f"Expected 3 records, got {len(rows)}")

            tickers = {r["ticker"] for r in rows}
            self.assertEqual(tickers, {"GOOGL", "TSLA", "GLD"})

            for row in rows:
                self.assertEqual(row["date"], "2025-01-10")
                self.assertIn(row["signal"], {"BUY", "SELL", "HOLD"})


if __name__ == "__main__":
    unittest.main()
