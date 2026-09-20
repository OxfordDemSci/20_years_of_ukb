"""Growth estimates must distinguish publication counts, composition and missing data."""

import sys
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import data_analysis_02_content_panels as panels
from utils.data_analysis_02_content_growth import category_growth_table, rank_growth_tables


class ContentGrowthTests(unittest.TestCase):
    def data(self, **long_frames):
        """Supply genuine long-form classifications to the same builder as the notebook."""
        data = {key: {"available": False} for key in ("for", "rcdc", "topics")}
        for key, frame in long_frames.items():
            data[key] = panels._flow_block(frame, "category", key)
        return data

    def counts(self, year_counts):
        records = []
        for year, categories in year_counts.items():
            for category, count in categories.items():
                for index in range(count):
                    records.append({"id": f"{year}-{category}-{index}",
                                    "year": year, "category": category})
        return pd.DataFrame(records)

    def test_rcdc_growth_counts_papers_while_composition_uses_fractional_credit(self):
        # Both baseline publications have two tags; just one final publication does.
        records = [("a", 2018, "X"), ("a", 2018, "Y"),
                   ("b", 2018, "X"), ("b", 2018, "Y"),
                   ("c", 2024, "X"), ("d", 2024, "X"),
                   ("e", 2025, "X"), ("e", 2025, "Y"),
                   ("f", 2025, "X"), ("g", 2025, "X"),
                   ("h", 2025, "X"), ("h", 2025, "X")]
        long = pd.DataFrame(records, columns=["id", "year", "category"])
        result = category_growth_table(self.data(rcdc=long), min_baseline_papers=2)
        row = result.set_index("category").loc["X"]
        self.assertEqual(row.n_baseline, 2)
        self.assertEqual(row.n_end, 4)
        self.assertEqual(row.n_previous, 2)
        self.assertEqual(row.fold_change, 2)
        self.assertAlmostEqual(row.cagr_pct, 100 * (2 ** (1 / 7) - 1))
        self.assertEqual(row.growth_pct, 100)
        self.assertEqual(row.recent_growth_pct, 100)
        self.assertEqual(row.baseline_composition_pct, 50)
        self.assertEqual(row.end_composition_pct, 87.5)
        self.assertEqual(row.share_change_pp, 37.5)
        self.assertTrue(row.eligible_cagr)

    def test_small_and_zero_baselines_remain_visible_without_inflating_cagr_rank(self):
        long = self.counts({
            2018: {"Established": 10, "Tiny": 2, "Flat": 10, "Declining": 20},
            2024: {"Established": 15, "Tiny": 30, "New": 20, "Flat": 10, "Declining": 15},
            2025: {"Established": 20, "Tiny": 50, "New": 30, "Very small": 5,
                   "Flat": 10, "Declining": 10},
        })
        result = category_growth_table(self.data(**{"for": long}))
        rows = result.set_index("category")
        self.assertEqual(rows.loc["New", "n_baseline"], 0)
        self.assertTrue(np.isnan(rows.loc["New", "cagr_pct"]))
        self.assertTrue(np.isnan(rows.loc["New", "fold_change"]))
        self.assertTrue(np.isnan(rows.loc["New", "growth_pct"]))
        self.assertFalse(rows.loc["Tiny", "eligible_cagr"])
        self.assertFalse(rows.loc["New", "eligible_cagr"])
        self.assertFalse(rows.loc["Flat", "eligible_cagr"])
        self.assertFalse(rows.loc["Declining", "eligible_cagr"])
        ranked = rank_growth_tables(result)
        self.assertEqual(ranked["fastest"].category.tolist(), ["Established"])
        self.assertTrue({"New", "Tiny"}.issubset(set(ranked["share_gains"].category)))
        self.assertNotIn("Very small", ranked["share_gains"].category.tolist())
        self.assertTrue((ranked["share_gains"].share_change_pp > 0).all())

    def test_zero_previous_count_does_not_produce_infinite_recent_growth(self):
        long = self.counts({2018: {"X": 10}, 2024: {"Y": 10},
                            2025: {"X": 20, "Y": 10}})
        rows = category_growth_table(self.data(**{"for": long})).set_index("category")
        self.assertEqual(rows.loc["X", "n_previous"], 0)
        self.assertTrue(np.isnan(rows.loc["X", "recent_growth_pct"]))
        self.assertTrue(rows.loc["X", "eligible_cagr"])

    def test_entire_missing_classification_year_is_not_a_zero_count(self):
        long = self.counts({2017: {"X": 10}, 2025: {"X": 30}})
        row = category_growth_table(self.data(**{"for": long})).iloc[0]
        self.assertTrue(np.isnan(row.n_baseline))
        self.assertTrue(np.isnan(row.n_previous))
        self.assertTrue(np.isnan(row.cagr_pct))
        self.assertTrue(np.isnan(row.recent_growth_pct))
        self.assertTrue(np.isnan(row.share_change_pp))
        self.assertFalse(row.eligible_cagr)
        self.assertEqual(row.n_end, 30)
        missing_end = self.counts({2018: {"X": 10}, 2024: {"X": 20}})
        final = category_growth_table(self.data(**{"for": missing_end})).iloc[0]
        self.assertTrue(np.isnan(final.n_end))
        self.assertFalse(final.eligible_cagr)

    def test_ranks_restart_per_vocabulary_and_break_cagr_ties_deterministically(self):
        fields = self.counts({
            2018: {"Zeta": 10, "Alpha": 10, "Largest": 20, "No growth": 10},
            2025: {"Zeta": 20, "Alpha": 20, "Largest": 40, "No growth": 10},
        })
        topics = self.counts({2018: {"A topic": 10}, 2025: {"A topic": 30}})
        result = category_growth_table(self.data(**{"for": fields, "topics": topics}))
        ranked = rank_growth_tables(result, top_n=2)["fastest"]
        self.assertEqual(set(ranked.vocabulary), {"for", "topics"})
        field_ranks = ranked[ranked.vocabulary == "for"]
        self.assertEqual(field_ranks.category.tolist(), ["Largest", "Alpha"])
        self.assertEqual(field_ranks["rank"].tolist(), [1, 2])
        self.assertEqual(ranked.loc[ranked.vocabulary == "topics", "rank"].tolist(), [1])
        self.assertNotIn("No growth", ranked.category.tolist())

    def test_invalid_analysis_periods_and_thresholds_are_rejected(self):
        data = self.data(**{"for": self.counts({2018: {"X": 10}, 2025: {"X": 20}})})
        for kwargs in ({"baseline_year": 2012}, {"end_year": 2026},
                       {"baseline_year": 2025},
                       {"baseline_year": 2024, "end_year": 2023},
                       {"baseline_year": 2018.5}, {"end_year": 2025.5},
                       {"min_baseline_papers": 0}, {"min_baseline_papers": -1}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    category_growth_table(data, **kwargs)


if __name__ == "__main__":
    unittest.main()
