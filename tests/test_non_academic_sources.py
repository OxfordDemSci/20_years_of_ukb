"""The wide snapshot must preserve endpoint alignment and entity identity."""

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.data_analysis_04_non_academic_sources import recover_endpoint_records


def row(ids, titles, affiliations):
    return {
        "trials__n_records": len(ids),
        "trials__id": json.dumps(ids),
        "trials__title": json.dumps(titles),
        "trials__research_orgs": json.dumps(affiliations),
        "trials__linked_ids": '["unfetched_trial"]',
    }


class EndpointRecoveryTests(unittest.TestCase):
    def test_reconstructs_aligned_entities_and_deduplicates_repeated_records(self):
        org = {"name": "University", "types": ["Education"]}
        wide = pd.DataFrame([
            row(["a", "b"], ["First", "Second"], [[org], []]),
            row(["b"], ["Second"], [[]]),
        ])
        result = recover_endpoint_records(wide, "trials").set_index("id")
        self.assertEqual(result.index.tolist(), ["a", "b"])
        self.assertEqual(result.loc["a", "research_orgs"], [org])
        self.assertEqual(result.loc["b", "title"], "Second")


    def test_rejects_misaligned_arrays(self):
        wide = pd.DataFrame([row(["a", "b"], ["First"], [[], []])])
        with self.assertRaisesRegex(ValueError, "array length"):
            recover_endpoint_records(wide, "trials")


    def test_rejects_conflicting_repeated_records(self):
        wide = pd.DataFrame([row(["a"], ["First"], [[]]), row(["a"], ["Changed"], [[]])])
        with self.assertRaisesRegex(ValueError, "conflicting copies"):
            recover_endpoint_records(wide, "trials")


    def test_linkage_only_rows_do_not_become_entity_records(self):
        wide = pd.DataFrame({"trials__linked_ids": ['["a"]'], "trials__n_records": [0]})
        with self.assertRaisesRegex(ValueError, "linkage IDs are insufficient"):
            recover_endpoint_records(wide, "trials")

    def test_rejects_negative_counts_before_filtering_rows(self):
        invalid = row([], [], [])
        invalid["trials__n_records"] = -1
        wide = pd.DataFrame([row(["a"], ["First"], [[]]), invalid])
        with self.assertRaisesRegex(ValueError, "invalid record count"):
            recover_endpoint_records(wide, "trials")

    def test_zero_or_missing_count_cannot_hide_populated_arrays(self):
        for count in (0, None):
            with self.subTest(count=count):
                invalid = row(["b"], ["Second"], [[]])
                invalid["trials__n_records"] = count
                wide = pd.DataFrame([row(["a"], ["First"], [[]]), invalid])
                with self.assertRaisesRegex(ValueError, "zero/missing n_records"):
                    recover_endpoint_records(wide, "trials")

    def test_blank_fields_in_zero_record_rows_are_absent_metadata(self):
        empty = {"trials__n_records": 0, "trials__id": "", "trials__title": "",
                 "trials__research_orgs": ""}
        result = recover_endpoint_records(pd.DataFrame([row(["a"], ["First"], [[]]), empty]), "trials")
        self.assertEqual(result["id"].tolist(), ["a"])


if __name__ == "__main__":
    unittest.main()
