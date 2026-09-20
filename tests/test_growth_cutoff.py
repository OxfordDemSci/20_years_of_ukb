import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import data_analysis_01_growth as growth


class GrowthCutoffTests(unittest.TestCase):
    def test_publication_flags_include_the_entire_final_day(self):
        source = pd.DataFrame({
            "publication_date": ["2025-12-31", "2025-12-31T23:59:59.999Z", "2026-01-01"],
            "research_org_types": [[], [], []],
            "research_org_countries": [[], [], []],
            "altmetric": [0, 0, 0], "open_access": [[], [], []],
            "type": ["article"] * 3,
        })
        result = growth.add_publication_indicators(source, pd.Timestamp("2025-12-31"), {})
        self.assertEqual(result["future_dated"].tolist(), [False, False, True])

    def test_endpoint_arrays_remain_aligned_at_cutoff(self):
        spec = {"trials": {"id_col": "ids", "event_col": "dates", "record_col": "n"}}
        source = pd.DataFrame({
            "ids": [["too_early", "first_day", "keep", "future", "unknown"]],
            "dates": [["2012-12-31", "2013-01-01", "2025-12-31", "2026-01-01", None]], "n": [5],
        })
        result = growth.restrict_endpoint_dates(source, spec)
        self.assertEqual(result.iloc[0]["ids"], ["first_day", "keep"])
        self.assertEqual(result.iloc[0]["dates"], ["2013-01-01", "2025-12-31"])
        self.assertEqual(result.iloc[0]["n"], 2)
        self.assertEqual(source.iloc[0]["n"], 5)
        self.assertEqual(result.attrs["endpoint_links_excluded_by_date"], {"trials": 3})

    def test_unresolved_ids_do_not_shift_dates_between_records(self):
        spec = {"grants": {"id_col": "ids", "event_col": "dates", "record_col": "n"}}
        source = pd.DataFrame({"ids": [["missing", "old", "future"]],
                               "grants__id": [["old", "future"]],
                               "dates": [[2025, 2026]], "n": [2]})
        result = growth.restrict_endpoint_dates(source, spec)
        self.assertEqual(result.iloc[0]["ids"], ["old"])
        self.assertEqual(result.iloc[0]["dates"], [2025])

    def test_fractional_years_are_not_rounded_into_the_window(self):
        spec = {"grants": {"id_col": "ids", "event_col": "dates", "record_col": "n"}}
        source = pd.DataFrame({
            "ids": [["first_year", "invalid_early", "invalid_late", "last_year"]],
            "dates": [[2013.0, 2013.5, "2025.5", "2025"]], "n": [4],
        })
        result = growth.restrict_endpoint_dates(source, spec)
        self.assertEqual(result.iloc[0]["ids"], ["first_year", "last_year"])

    def test_load_filters_before_deriving_publication_indicators(self):
        source = pd.DataFrame({
            "id": ["too_early", "first_year", "eligible", "future", "later"],
            "year": [2012, 2013, 2025, 2026, 2027],
            "date": ["2012-12-31", "2013-01-01", "2025-12-31", "2026-01-01", "2027-01-01"],
            "altmetric": [100, 1, 1, 100, 100], "times_cited": [100, 1, 1, 100, 100],
            "authors_count": [1] * 5,
        })
        for spec in growth.ENDPOINT_SPECS.values():
            source[spec["id_col"]] = [[] for _ in range(len(source))]
            source[spec["event_col"]] = [[] for _ in range(len(source))]
            source[spec["record_col"]] = 0
        with patch.object(growth, "load_showcase", return_value=source):
            result = growth.load_growth_papers()
        self.assertEqual(result["id"].tolist(), ["first_year", "eligible"])
        self.assertEqual(result.attrs["source_record_count"], 5)
        self.assertEqual(result.attrs["excluded_publication_records"], 3)


if __name__ == "__main__":
    unittest.main()
