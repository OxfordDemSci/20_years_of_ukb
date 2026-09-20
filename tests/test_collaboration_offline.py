"""Analysis reruns must use saved classifications without contacting a provider."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import data_analysis_04_non_academic_collab_classifier as classifier


class OfflineClassificationTests(unittest.TestCase):
    def test_uncached_labels_do_not_call_provider_by_default(self):
        client = Mock()
        with self.assertRaisesRegex(FileNotFoundError, "API classification is disabled"):
            classifier.classify_institution_lists([["Example University"]], client=client)
        client.messages.create.assert_not_called()

    def test_complete_cache_rebuilds_classified_table_without_credentials(self):
        import pandas as pd

        result = {f"{name}_indices": [] for name in classifier.INDEX_FIELDS}
        result["academic_indices"] = [0]
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "cache.jsonl"
            output = Path(folder) / "classified.csv"
            cache.write_text(json.dumps({"key": ["Example University"], "result": result}) + "\n")
            source = pd.DataFrame({"id": ["pub.1"], "research_orgs": [[{"name": "Example University"}]]})
            with patch.object(classifier, "_resolve_api_key", side_effect=AssertionError("credential access")):
                classified, rebuilt = classifier.load_or_classify(source, output, cache_path=cache)
            self.assertTrue(rebuilt)
            self.assertTrue(classified.loc[0, "academic_flag"])
            self.assertEqual(json.loads(pd.read_csv(output).loc[0, "academic_indices"]), [0])

    def test_saved_labels_are_loaded_without_classification(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "classified.csv"
            output.write_text("id,academic_flag\npub.1,True\n")
            with patch.object(classifier, "classify_institution_lists", side_effect=AssertionError("classification")):
                frame, rebuilt = classifier.load_or_classify(None, output)
            self.assertFalse(rebuilt)
            self.assertEqual(frame.loc[0, "id"], "pub.1")

    def test_explicit_opt_in_allows_classification(self):
        result = {f"{name}_indices": [] for name in classifier.INDEX_FIELDS}
        with patch.object(classifier, "classify_batch", return_value=[result]) as classify:
            actual = classifier.classify_institution_lists(
                [["Example University"]], client=Mock(), allow_api=True,
            )
        self.assertEqual(actual, [result])
        classify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
