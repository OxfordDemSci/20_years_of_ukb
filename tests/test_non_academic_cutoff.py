"""Future outcomes and cached rows must not leak into pre-2026 analyses."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import shared_paths as P
from utils import data_analysis_04_non_academic_panels as panels
from utils import data_analysis_04_non_academic_collab_classifier as classifier
from utils.data_analysis_04_non_academic_sources import filter_endpoint_links


class NonAcademicCutoffTests(unittest.TestCase):
    def test_reverse_links_use_endpoint_ids_and_both_date_and_year(self):
        corpus = pd.DataFrame({
            "id": ["paper"],
            "patents__linked_ids": ['["future", "boundary", "unfetched", "conflict"]'],
            "patents__n_links": [4],
        })
        wide = pd.DataFrame({
            "patents__n_records": [3],
            "patents__id": [json.dumps(["boundary", "conflict", "future"])],
            "patents__publication_year": [json.dumps([2025, 2025, 2026])],
            "patents__publication_date": [json.dumps(["2025-12-31", "2026-01-01", None])],
        })
        result = filter_endpoint_links(corpus, wide=wide)
        self.assertEqual(result.loc[0, "patents__linked_ids"], ["boundary"])
        self.assertEqual(result.loc[0, "patents__n_links"], 1)
        self.assertEqual(corpus.loc[0, "patents__n_links"], 4)

    def test_reverse_links_limit_trial_starts_and_policy_years(self):
        corpus = pd.DataFrame({
            "clinical_trials__linked_ids": ['["old", "new"]'],
            "policy_documents__linked_ids": ['["old", "new"]'],
        })
        wide = pd.DataFrame({
            "clinical_trials__n_records": [2],
            "clinical_trials__id": ['["old", "new"]'],
            "clinical_trials__start_date": ['["2025-12-31", "2026-01-01"]'],
            "policy_documents__n_records": [2],
            "policy_documents__id": ['["old", "new"]'],
            "policy_documents__year": ['[2025, 2026]'],
        })
        result = filter_endpoint_links(corpus, wide=wide)
        for column in corpus:
            self.assertEqual(result.loc[0, column], ["old"])

    def test_trial_loader_excludes_future_starts_without_changing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.csv"
            pd.DataFrame({"id": ["old", "new", "unknown"],
                          "start_date": ["2025-12-31", "2026-01-01", None],
                          "study_type": ["Interventional"] * 3}).to_csv(path, index=False)
            original = path.read_bytes()
            with patch.object(P, "CT_CSV", path):
                result = panels.load_trials()
            self.assertEqual(result.id.tolist(), ["old"])
            self.assertEqual(path.read_bytes(), original)

    def test_patent_loader_uses_publication_date_not_earlier_priority_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patents_modularized_export.csv"
            pd.DataFrame({"id": ["old", "new"], "publication_year": [2025, 2025],
                          "publication_date": ["2025-12-31", "2026-01-01"],
                          "priority_year": [2020, 2020], "granted_year": [2025, 2026]}
                         ).to_csv(path, index=False)
            with patch.object(P, "PATENT", Path(tmp)):
                result = panels.load_patents()
            self.assertEqual(result.id.tolist(), ["old"])

    def test_altmetric_filters_papers_without_redating_snapshot_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alt.csv"
            pd.DataFrame({
                "DOI": ["old", "new", "outside-corpus"],
                "Publication Date": ["2025-12-31", "2026-01-01", "2025-01-01"],
                "Altmetric Attention Score": [987, 2000, 3000],
                "News mentions": [50, 60, 70], "Policy mentions": [20, 30, 40],
            }).to_csv(path, index=False)
            corpus = pd.DataFrame({"id": ["a", "b"], "year": [2025, 2025],
                                   "doi_clean": ["old", "new"], "times_cited": [123, 456]})
            with patch.object(P, "ALTMETRIC_CSV", path):
                result = panels.load_altmetric(corpus)
            self.assertEqual(result.id.tolist(), ["a"])
            self.assertEqual(result.iloc[0]["Altmetric Attention Score"], 987)
            self.assertEqual(result.iloc[0]["times_cited"], 123)

    def test_existing_classifier_cache_is_filtered_in_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "classified.csv"
            df = pd.DataFrame({"id": ["old", "new"], "year": [2025, 2026]})
            df.to_csv(path, index=False)
            original = path.read_bytes()
            with patch.object(classifier, "classify_institution_lists",
                              side_effect=AssertionError("API must not run")):
                result, ran = classifier.load_or_classify(df, out_path=path)
            self.assertFalse(ran)
            self.assertEqual(result.id.tolist(), ["old"])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
