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
from utils import data_analysis_03_academic_impact_panels as panels
from utils import data_analysis_05_author_characteristics as authors


def publication_rows():
    return pd.DataFrame({
        "id": ["early", "start", "eligible", "future", "later", "conflicting_date",
               "conflicting_early_date"],
        "year": [2012, 2013, 2025, 2026, 2027, 2025, 2013],
        "date": ["2012-12-31", "2013-01-01", "2025-12-31", "2026-01-01",
                 "2027-01-01", "2026-01-01", "2012-12-31"],
        "times_cited": [10000, 20, 80, 10000, 10000, 10000, 10000],
        "category_for_2020": [
            [{"id": "field.1", "name": "4202 Epidemiology"}]
        ] * 7,
    })


class AcademicCutoffTests(unittest.TestCase):
    def test_both_count_arms_and_totals_keep_only_2013_through_2025(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for arm in ("ukbb", "background"):
                frame = pd.DataFrame({
                    "year": [2012, 2013, 2025, 2026, 2027], "level": ["L4"] * 5,
                    "code": ["4202"] * 5, "for_label": ["Epidemiology"] * 5,
                    "n_papers": [1000, 3, 2, 1000, 1000],
                })
                frame.to_parquet(directory / f"counts.for.{arm}.parquet")
                frame[["year", "n_papers"]].to_parquet(
                    directory / f"totals.for.{arm}.parquet")
                result, _ = academic.load_arm(
                    directory, "for", arm, "L4", 2000, 2030, verbose=False)
                totals = academic.load_side_table(
                    directory, "for", arm, "totals", ["year"], 2000, 2030)
                self.assertEqual(result["year"].tolist(), [2013, 2025])
                self.assertEqual(result["n_papers"].sum(), 5)
                self.assertEqual(totals["n_papers"].sum(), 5)

    def test_paper_metrics_filter_publications_but_preserve_snapshot_citations(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "api_ukbb_records.json").write_text(
                json.dumps(publication_rows().to_dict("records")))
            pd.DataFrame({
                "year": [2012, 2013, 2025, 2026, 2027], "level": ["L4"] * 5,
                "code": ["4202"] * 5, "mean_cit": [1, 10, 20, 1, 1],
            }).to_parquet(directory / "api_whole.for.parquet")
            pd.DataFrame({
                "year": [2025, 2025], "level": ["L4", "L4"],
                "code": ["4202", "4202"], "percentile": [10, 50],
                "threshold": [40, 10],
            }).to_csv(directory / "field_thresholds.for.csv", index=False)
            result = academic.paper_impact(
                directory, year_min=2000, year_max=2030, verbose=False)
            self.assertEqual(result["id"].tolist(), ["start", "eligible"])
            self.assertEqual(result["times_cited"].tolist(), [20, 80])
            self.assertEqual(result["n_mncs"].tolist(), [2, 4])
            # A missing early percentile reference does not drop the publication
            # or turn its unavailable score into zero.
            self.assertTrue(pd.isna(result.iloc[0]["n_top10f"]))

    def test_api_record_counting_applies_cutoff_before_weights(self):
        codes = pd.DataFrame({
            "level": ["L4"], "code": ["4202"], "name": ["Epidemiology"],
        }, index=["field.1"])
        frame, totals = api.records_to_counts(
            publication_rows().to_dict("records"), codes, "for", "ukbb", 2000, 2030)
        self.assertEqual(frame["year"].tolist(), [2013, 2025])
        self.assertEqual(frame["n_papers"].sum(), 2)
        self.assertEqual(frame["n_cit"].sum(), 100)
        self.assertEqual(totals["n_papers"].sum(), 2)

    def test_streaming_counts_reject_future_dates_as_well_as_years(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.parquet"
            frame = publication_rows()
            frame["category_for_2020"] = frame["category_for_2020"].map(json.dumps)
            frame.to_parquet(path)
            tally = counts.Tally(weights=["cit"])
            counts.count_file(str(path), tally, counts.CATEGORIES["for"], None,
                              "none", 10)
            self.assertEqual(tally.n_kept, 2)
            self.assertEqual({key[0] for key in tally.codes}, {2013, 2025})
            self.assertEqual(sum(value[2] for value in tally.codes.values()), 100)

    def test_author_source_audit_is_raw_but_analysis_excludes_future_records(self):
        frame = publication_rows()
        frame["altmetric"] = 0
        frame["authors_count"] = 1
        with patch.object(authors, "load_showcase", return_value=frame):
            source, eligible = authors.load_author_papers(last_complete_year=2030)
        self.assertEqual(len(source), 7)
        self.assertEqual(eligible["id"].tolist(), ["start", "eligible"])
        self.assertEqual(eligible["times_cited"].sum(), 100)

    def test_missing_early_references_do_not_move_the_study_start(self):
        totals = pd.DataFrame({
            "year": [2012, 2013, 2015, 2025, 2026],
            "n_papers": [10] * 5,
            "n_mncs_docs_total": [10] * 5,
            "n_top10f_docs_total": [10, 0, 10, 10, 10],
        })
        years, window, usable, _ = academic.citation_windows(
            {"ukbb": totals, "background": totals}, ["n_mncs", "n_top10f"],
            2000, 2030, 0.5, 0, verbose=False)
        self.assertEqual(window, (2013, 2025))
        self.assertTrue(usable)
        self.assertEqual(years["n_mncs"], [2013, 2015, 2025])
        self.assertEqual(years["n_top10f"], [2015, 2025])

    def test_retained_author_metrics_require_matching_window_provenance(self):
        # Asserted against the window analysis 03 ACTUALLY runs on, read from the panels
        # module rather than written out here. The guard's own defaults are the shared
        # corpus floor of 2013, which no caller wants: the only caller is analysis 03 and
        # D19 puts its floor at 2015, because 2014 has no measured cut-off in any field.
        # A literal year in this test is what let the two drift apart once already.
        expected = f"{panels.ANALYSIS_MIN}-{panels.ANALYSIS_MAX}"
        wrong = f"{panels.ANALYSIS_MIN - 2}-{panels.ANALYSIS_MAX}"
        self.assertEqual(expected, "2015-2025")
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "input_manifest_author_impact_notebook.csv"
            with self.assertRaisesRegex(FileNotFoundError, "Rerun"):
                academic.require_author_impact_window(
                    directory, panels.ANALYSIS_MIN, panels.ANALYSIS_MAX)
            pd.DataFrame({"role": ["analysis window"], "value": [wrong]}).to_csv(
                manifest, index=False)
            with self.assertRaisesRegex(ValueError, expected):
                academic.require_author_impact_window(
                    directory, panels.ANALYSIS_MIN, panels.ANALYSIS_MAX)
            pd.DataFrame({"role": ["analysis window"], "value": [expected]}).to_csv(
                manifest, index=False)
            academic.require_author_impact_window(
                directory, panels.ANALYSIS_MIN, panels.ANALYSIS_MAX)

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
