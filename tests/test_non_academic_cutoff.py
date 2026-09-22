"""Outcomes and cached rows must stay inside the inclusive 2013–2025 window."""

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
from utils import data_analysis_04_non_academic_collab_helpers as collaboration
from utils.data_analysis_04_non_academic_sources import filter_endpoint_links


def _write_wide(path: Path, endpoint: str, fields: dict) -> Path:
    """A one-row stand-in for the wide corpus export, carrying one endpoint block.

    The real file stores each endpoint as parallel JSON arrays hanging off a publication
    row, with nested values double-encoded — a list or dict cell is itself a JSON string
    inside the array. The loaders read through `shared_showcase.endpoint_records`, so a
    fixture has to have that shape rather than be a flat table of records.
    """
    row = {}
    for name, values in fields.items():
        row[f"{endpoint}__{name}"] = [json.dumps([
            json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            for v in values
        ])]
    row[f"{endpoint}__n_records"] = [len(next(iter(fields.values())))]
    pd.DataFrame(row).to_parquet(path, index=False)
    return path


class NonAcademicCutoffTests(unittest.TestCase):
    def test_default_collaboration_metrics_include_2013(self):
        frame = pd.DataFrame({"year": [2012, 2013, 2025, 2026],
                              "non_academic_flag": [1, 1, 1, 1]})
        result = collaboration.build_yearly_metrics_table(frame).set_index("year")
        self.assertEqual(result.index.tolist(), list(range(2013, 2026)))
        self.assertEqual(result.loc[2013, "papers_total"], 1)
        self.assertEqual(result.papers_total.sum(), 2)

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
        # The trials arm reads the corpus's own `clinical_trials__*` block, not the side
        # CSV (repointed 2026-09-22, as D43 repointed patents), so the fixture is a wide
        # export rather than a CSV. `CT_CSV` is pointed at a path that does not exist so
        # the `mesh_leaf_ids` merge stays out of the window assertion.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wide(Path(tmp) / "corpus.parquet", "clinical_trials", {
                "id": ["old", "new", "unknown", "first", "before"],
                "start_date": ["2025-12-31", "2026-01-01", None,
                               "2013-01-01", "2012-12-31"],
                "study_type": ["Interventional"] * 5,
            })
            original = path.read_bytes()
            with patch.object(P, "SHOWCASE_PLUS", path), \
                    patch.object(P, "CT_CSV", Path(tmp) / "absent.csv"):
                result = panels.load_trials()
            self.assertEqual(result.id.tolist(), ["old", "first"])
            self.assertEqual(path.read_bytes(), original)

    def test_policy_loader_flattens_publisher_dicts_out_of_the_corpus_block(self):
        # `publisher_org_country` is DICT-valued. The endpoint reader parsed every nested
        # cell as a list, which returns [] for an object and silently emptied all four
        # publisher columns; this is the regression test for that.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wide(Path(tmp) / "corpus.parquet", "policy_documents", {
                "id": ["old", "new"],
                "year": [2025, 2026],
                "publisher_org": [{"id": "grid.1", "name": "WHO"}, {"id": "grid.2"}],
                "publisher_org_country": [{"id": "CH", "name": "Switzerland"}, {}],
            })
            with patch.object(P, "SHOWCASE_PLUS", path), \
                    patch.object(P, "POLICY_CSV", Path(tmp) / "absent.csv"):
                result = panels.load_policy()
            self.assertEqual(result.id.tolist(), ["old"])
            self.assertEqual(result.loc[0, "publisher_name"], "WHO")
            self.assertEqual(result.loc[0, "publisher_country"], "Switzerland")
            self.assertEqual(result.loc[0, "publisher_country_code"], "CH")

    def test_reverse_links_exclude_pre2013_outcomes_and_keep_first_day(self):
        corpus = pd.DataFrame({
            "patents__linked_ids": ['["first", "before", "conflict"]'],
            "clinical_trials__linked_ids": ['["first", "before"]'],
            "policy_documents__linked_ids": ['["first", "before"]'],
        })
        wide = pd.DataFrame({
            "patents__n_records": [3],
            "patents__id": ['["first", "before", "conflict"]'],
            "patents__publication_year": ['[2013, 2012, 2013]'],
            "patents__publication_date": ['["2013-01-01", "2012-12-31", "2012-12-31"]'],
            "clinical_trials__n_records": [2],
            "clinical_trials__id": ['["first", "before"]'],
            "clinical_trials__start_date": ['["2013-01-01", "2012-12-31"]'],
            "policy_documents__n_records": [2],
            "policy_documents__id": ['["first", "before"]'],
            "policy_documents__year": ['[2013, 2012]'],
        })
        result = filter_endpoint_links(corpus, wide=wide)
        for column in corpus:
            self.assertEqual(result.loc[0, column], ["first"])

    def test_patent_loader_uses_publication_date_not_earlier_priority_date(self):
        # Fixture repointed to the corpus block to match D43 (2026-09-20), which moved the
        # loader off `patents_modularized_export.csv`. The test had kept patching `P.PATENT`
        # and so had been asserting against the real 767-patent block since that decision.
        with tempfile.TemporaryDirectory() as tmp:
            fields = {name: [None, None] for name in panels.PATENT_ENDPOINT_FIELDS}
            fields.update({"id": ["old", "new"], "publication_year": [2025, 2025],
                           "publication_date": ["2025-12-31", "2026-01-01"],
                           "priority_year": [2020, 2020], "granted_year": [2025, 2026],
                           "assignee_countries": [[], []]})
            path = _write_wide(Path(tmp) / "corpus.parquet", "patents", fields)
            with patch.object(P, "SHOWCASE_PLUS", path):
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
            df = pd.DataFrame({"id": ["old", "new", "first", "before"],
                               "year": [2025, 2026, 2013, 2012]})
            df.to_csv(path, index=False)
            original = path.read_bytes()
            with patch.object(classifier, "classify_institution_lists",
                              side_effect=AssertionError("API must not run")):
                result, ran = classifier.load_or_classify(df, out_path=path)
            self.assertFalse(ran)
            self.assertEqual(result.id.tolist(), ["old", "first"])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
