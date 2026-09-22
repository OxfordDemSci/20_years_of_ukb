"""RCDC caches are tied to the eligible cohort, not an unverified legacy fit."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import data_analysis_04_non_academic_patents_rcdc_macro as macro
from utils import shared_patent_utils as patents


def cohort():
    return pd.DataFrame({
        "id": ["old-a", "old-b", "future", "before"],
        "publication_year": [2013, 2025, 2025, 2013],
        "publication_date": ["2013-01-01", "2025-12-31", "2026-01-01", "2012-12-31"],
        "category_rcdc": [[{"id": "A", "name": "Alpha"}],
                          [{"id": "B", "name": "Beta"}],
                          [{"id": "FUTURE", "name": "Future"}],
                          [{"id": "EARLIER", "name": "Earlier"}]],
    })


class PartitionCacheTests(unittest.TestCase):
    def test_linked_ukbb_paper_ids_have_deterministic_order(self):
        linked = patents.find_ukbb_papers(
            "['pub.3', 'pub.1', 'pub.2']",
            ["pub.2", "pub.3", "pub.1"],
        )
        self.assertEqual(linked, ["pub.1", "pub.2", "pub.3"])

    def test_boundary_and_changed_cohort_control_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "cluster_label_summary_louvain.csv"
            legacy.write_text("community,n_labels,top_labels,all_labels\n0,1,FUTURE,FUTURE\n")
            original = legacy.read_bytes()
            result, path = macro.load_or_build_analysis_partition(cohort(), cache_dir=tmp)
            self.assertEqual(set(result.all_labels), {"A", "B"})
            metadata = json.loads(path.with_suffix(".provenance.json").read_text())
            self.assertEqual(metadata["patents"], 2)
            self.assertEqual(metadata["analysis_start_date"], "2013-01-01")
            self.assertEqual(metadata["analysis_end_date"], "2025-12-31")
            with patch.object(macro, "run_louvain", side_effect=AssertionError("cache not reused")):
                _, same_path = macro.load_or_build_analysis_partition(cohort(), cache_dir=tmp)
            self.assertEqual(path, same_path)
            changed = cohort()
            changed.at[0, "category_rcdc"] = [{"id": "CHANGED", "name": "Changed"}]
            _, changed_path = macro.load_or_build_analysis_partition(changed, cache_dir=tmp)
            self.assertNotEqual(path, changed_path)
            self.assertEqual(legacy.read_bytes(), original)

    def test_tampered_partition_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, path = macro.load_or_build_analysis_partition(cohort(), cache_dir=tmp)
            path.write_text("community,n_labels,top_labels,all_labels\n0,1,FUTURE,FUTURE\n")
            rebuilt, _ = macro.load_or_build_analysis_partition(cohort(), cache_dir=tmp)
            self.assertEqual(set(rebuilt.all_labels), {"A", "B"})

    def test_reviewed_names_cannot_follow_changed_community_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "reviewed.csv"
            legacy.write_text("community,n_labels,top_labels,all_labels\n0,1,A,A\n")
            changed_partition = pd.DataFrame({"community": [0], "n_labels": [1],
                                              "top_labels": ["B"], "all_labels": ["B"]})
            with patch.object(macro, "load_or_build_analysis_partition",
                              return_value=(changed_partition, Path(tmp) / "new.csv")):
                with self.assertRaisesRegex(ValueError, "differs from the reviewed clusters"):
                    patents.prepare_rcdc_macro_context(cohort(), summary_csv=legacy)


if __name__ == "__main__":
    unittest.main()
