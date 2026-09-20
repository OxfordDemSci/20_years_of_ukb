import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import data_analysis_03_academic_impact_analysis as academic
from utils import data_analysis_03_academic_impact_dimensions_api as api
from utils import data_analysis_03_academic_impact_field_counts as counts
from utils import data_analysis_05_author_characteristics as authors


def publication_rows():
    return pd.DataFrame({
        "id": ["eligible", "future", "later", "conflicting_date"],
        "year": [2025, 2026, 2027, 2025],
        "date": ["2025-12-31", "2026-01-01", "2027-01-01", "2026-01-01"],
        "times_cited": [80, 10000, 10000, 10000],
        "category_for_2020": [
            [{"id": "field.1", "name": "4202 Epidemiology"}]
        ] * 4,
    })


class AcademicCutoffTests(unittest.TestCase):
    def test_both_count_arms_and_totals_ignore_future_years(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for arm in ("ukbb", "background"):
                frame = pd.DataFrame({
                    "year": [2025, 2026, 2027], "level": ["L4"] * 3,
                    "code": ["4202"] * 3, "for_label": ["Epidemiology"] * 3,
                    "n_papers": [2, 1000, 1000],
                })
                frame.to_parquet(directory / f"counts.for.{arm}.parquet")
                frame[["year", "n_papers"]].to_parquet(
                    directory / f"totals.for.{arm}.parquet")
                result, _ = academic.load_arm(
                    directory, "for", arm, "L4", 2015, 2030, verbose=False)
                totals = academic.load_side_table(
                    directory, "for", arm, "totals", ["year"], 2015, 2030)
                self.assertEqual(result["year"].tolist(), [2025])
                self.assertEqual(result["n_papers"].sum(), 2)
                self.assertEqual(totals["n_papers"].sum(), 2)

    def test_paper_metrics_filter_publications_but_preserve_snapshot_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "api_ukbb_records.json").write_text(
                json.dumps(publication_rows().to_dict("records")))
            pd.DataFrame({
                "year": [2025, 2026, 2027], "level": ["L4"] * 3,
                "code": ["4202"] * 3, "mean_cit": [20, 1, 1],
            }).to_parquet(directory / "api_whole.for.parquet")
            pd.DataFrame({
                "year": [2025, 2025], "level": ["L4", "L4"],
                "code": ["4202", "4202"], "percentile": [10, 50],
                "threshold": [40, 10],
            }).to_csv(directory / "field_thresholds.for.csv", index=False)
            result = academic.paper_impact(
                directory, year_min=2015, year_max=2030, verbose=False)
            self.assertEqual(result["id"].tolist(), ["eligible"])
            self.assertEqual(result.iloc[0]["times_cited"], 80)
            self.assertEqual(result.iloc[0]["n_mncs"], 4)

    def test_api_record_counting_applies_cutoff_before_weights(self):
        codes = pd.DataFrame({
            "level": ["L4"], "code": ["4202"], "name": ["Epidemiology"],
        }, index=["field.1"])
        frame, totals = api.records_to_counts(
            publication_rows().to_dict("records"), codes, "for", "ukbb", 2015, 2030)
        self.assertEqual(frame["year"].tolist(), [2025])
        self.assertEqual(frame["n_papers"].sum(), 1)
        self.assertEqual(frame["n_cit"].sum(), 80)
        self.assertEqual(totals["n_papers"].sum(), 1)

    def test_streaming_counts_reject_future_dates_as_well_as_years(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.parquet"
            frame = publication_rows()
            frame["category_for_2020"] = frame["category_for_2020"].map(json.dumps)
            frame.to_parquet(path)
            tally = counts.Tally(weights=["cit"])
            counts.count_file(str(path), tally, counts.CATEGORIES["for"], None,
                              "none", 10)
            self.assertEqual(tally.n_kept, 1)
            self.assertEqual({key[0] for key in tally.codes}, {2025})
            self.assertEqual(sum(value[2] for value in tally.codes.values()), 80)

    def test_author_source_audit_is_raw_but_analysis_excludes_future_records(self):
        frame = publication_rows()
        frame["altmetric"] = 0
        frame["authors_count"] = 1
        with patch.object(authors, "load_showcase", return_value=frame):
            source, eligible = authors.load_author_papers(last_complete_year=2030)
        self.assertEqual(len(source), 4)
        self.assertEqual(eligible["id"].tolist(), ["eligible"])
        self.assertEqual(eligible["times_cited"].sum(), 80)

    def test_contaminated_aggregate_author_metrics_are_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authors.csv"
            pd.DataFrame({
                "researcher_id": ["ur.1"], "author_name": ["Example"],
                "showcase_paper_count": [2], "total_dataset_citations": [10000],
                "mean_impact_metric": [10], "median_impact_metric": [10],
                "showcase_h_index": [2], "last_year": [2026],
            }).to_csv(path, index=False)
            with patch.object(authors, "LEGACY_AUTHOR_IMPACT", path):
                with self.assertRaisesRegex(ValueError, "Rerun 03_academic"):
                    authors._author_impact_from_legacy_summary(None)


if __name__ == "__main__":
    unittest.main()
