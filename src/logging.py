"""
logging.py - Persistent run log.

append_run_record(record) appends a dict as a row to runs/run_log.csv,
creating the file and writing headers on first call.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).parent.parent / "runs" / "run_log.csv"


def append_run_record(record: dict[str, Any]) -> None:
    """Append *record* as a row to runs/run_log.csv.

    The CSV is created with headers on first write; subsequent calls
    append without re-writing headers. New keys in *record* that are
    absent from the existing header row are silently dropped.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = LOG_PATH.exists()
    fieldnames = list(record.keys())

    if file_exists:
        with open(LOG_PATH, newline="") as f:
            existing_fields = next(csv.reader(f), [])
        if existing_fields:
            fieldnames = existing_fields

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
