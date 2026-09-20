import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.shared_analysis_window import filter_analysis_window


class AnalysisWindowTests(unittest.TestCase):
    def test_inclusive_final_day_and_future_records(self):
        frame = pd.DataFrame({
            "id": ["old", "last_day", "new_year", "later"],
            "date": ["2013-01-01", "2025-12-31T23:59:59.999Z", "2026-01-01", "2027-02-01"],
        })
        self.assertEqual(filter_analysis_window(frame)["id"].tolist(), ["old", "last_day"])
        self.assertEqual(len(frame), 4)

    def test_year_fallback_and_conflicting_dates(self):
        frame = pd.DataFrame({
            "id": ["year_only", "date_only", "future_date", "future_year", "unknown", "bad_year"],
            "year": [2025, None, 2025, 2026, None, 2025.5],
            "date": [None, "2025-12-31", "2026-01-01", "2025-01-01", "bad", None],
        })
        self.assertEqual(filter_analysis_window(frame)["id"].tolist(), ["year_only", "date_only"])

    def test_explicit_event_columns(self):
        frame = pd.DataFrame({"start_year": [2025, 2026, 2027, None]})
        self.assertEqual(filter_analysis_window(frame, year_col="start_year").index.tolist(), [0])
        dates = pd.DataFrame({"start_date": ["2025-12-31", "2026-01-01", None]})
        self.assertEqual(filter_analysis_window(dates, year_col=None, date_col="start_date").index.tolist(), [0])

    def test_missing_time_columns_raise(self):
        with self.assertRaisesRegex(ValueError, "Cannot enforce"):
            filter_analysis_window(pd.DataFrame({"id": ["x"]}))


if __name__ == "__main__":
    unittest.main()
