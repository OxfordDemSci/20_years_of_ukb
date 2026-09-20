import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import data_analysis_01_growth as growth


class GrowthCutoffTests(unittest.TestCase):
    def test_endpoint_arrays_remain_aligned_at_cutoff(self):
        spec = {"trials": {"id_col": "ids", "event_col": "dates", "record_col": "n"}}
        source = pd.DataFrame({
            "ids": [["keep", "future", "unknown"]],
            "dates": [["2025-12-31", "2026-01-01", None]], "n": [3],
        })
        result = growth.restrict_endpoint_dates(source, spec)
        self.assertEqual(result.iloc[0]["ids"], ["keep"])
        self.assertEqual(result.iloc[0]["dates"], ["2025-12-31"])
        self.assertEqual(result.iloc[0]["n"], 1)
        self.assertEqual(source.iloc[0]["n"], 3)
        self.assertEqual(result.attrs["endpoint_links_excluded_by_date"], {"trials": 2})

    def test_unresolved_ids_do_not_shift_dates_between_records(self):
        spec = {"grants": {"id_col": "ids", "event_col": "dates", "record_col": "n"}}
        source = pd.DataFrame({"ids": [["missing", "old", "future"]],
                               "grants__id": [["old", "future"]],
                               "dates": [[2025, 2026]], "n": [2]})
        result = growth.restrict_endpoint_dates(source, spec)
        self.assertEqual(result.iloc[0]["ids"], ["old"])
        self.assertEqual(result.iloc[0]["dates"], [2025])

    def test_load_filters_before_deriving_publication_indicators(self):
        source = pd.DataFrame({
            "id": ["eligible", "future", "later"], "year": [2025, 2026, 2027],
            "date": ["2025-12-31", "2026-01-01", "2027-01-01"],
            "altmetric": [1, 100, 100], "times_cited": [1, 100, 100],
            "authors_count": [1, 1, 1],
        })
        for spec in growth.ENDPOINT_SPECS.values():
            source[spec["id_col"]] = [[], [], []]
            source[spec["event_col"]] = [[], [], []]
            source[spec["record_col"]] = 0
        with patch.object(growth, "load_showcase", return_value=source):
            result = growth.load_growth_papers()
        self.assertEqual(result["id"].tolist(), ["eligible"])
        self.assertEqual(result.attrs["source_record_count"], 3)
        self.assertEqual(result.attrs["excluded_publication_records"], 2)


if __name__ == "__main__":
    unittest.main()
