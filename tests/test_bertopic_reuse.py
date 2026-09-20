"""Completed topic results skip model work, without importing or fitting BERTopic."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import shared_paths as P
from utils.data_analysis_02_content_window import (
    find_existing_topic_results, write_topic_window_provenance,
)


class TopicResultsReuseTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.folder = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name, value in {
            "OUTPUT": self.folder / "output",
            "CONTENT": self.folder / "content",
            "TOPIC_ASSIGNMENTS": self.folder / "content" / "showcase_plus_id_topics.csv",
            "ACADEMIC_IMPACT": self.folder / "academic_impact",
        }.items():
            self.stack.enter_context(patch.object(P, name, value))
        self.output = P.OUTPUT / "bertopic"

    def results(self, path=None, *, script_schema=False):
        path = path or self.output / "showcase_plus_id_topics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"id": ["a", "b"], "topics": ["T0: genetics", "Outlier"]})
        if script_schema:
            frame = pd.DataFrame({"showcase_plus_id": ["a", "b"], "analysis_year": [2013, 2025], "topic": [0, -1]})
        frame.to_csv(path, index=False)
        write_topic_window_provenance(path, [2013, 2025])
        return path

    def test_absent_results_do_not_confuse_embeddings_with_final_outputs(self):
        self.output.mkdir(parents=True)
        (self.output / "embeddings.npy").write_bytes(b"intermediate cache")
        self.assertIsNone(find_existing_topic_results(self.output))

    def test_current_and_legacy_locations_are_recognised(self):
        legacy = self.results(P.TOPIC_ASSIGNMENTS)
        self.assertEqual(find_existing_topic_results(self.output), legacy)
        current = self.results()
        self.assertEqual(find_existing_topic_results(self.output), current)

    def test_script_assignment_schema_is_recognised(self):
        path = self.results(self.output / "tables" / "bertopic_document_topic_assignments.csv", script_schema=True)
        self.assertEqual(find_existing_topic_results(self.output), path)

    def test_invalid_existing_results_raise_instead_of_refitting(self):
        path = self.results()
        path.with_suffix(".analysis_window.json").unlink()
        before = path.read_bytes()
        with self.assertRaisesRegex(FileNotFoundError, "Outdated or unverified"):
            find_existing_topic_results(self.output)
        self.assertEqual(path.read_bytes(), before)
        pd.DataFrame({"id": ["a", "a"], "topics": ["T0", "T1"]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate assignments"):
            find_existing_topic_results(self.output)

    def test_entire_notebook_skips_dependencies_and_modelling_with_results(self):
        path = self.results()
        before = {p: p.read_bytes() for p in self.output.iterdir()}
        notebook = json.loads((ROOT / "src/data_analysis/02_content_1_bert_topic.ipynb").read_text())
        namespace = {}
        with patch("importlib.util.find_spec", side_effect=AssertionError("Dependency checks must be skipped")):
            with contextlib.redirect_stdout(io.StringIO()) as log:
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), "BERTopic notebook", "exec"), namespace)
        self.assertTrue(namespace["REUSE_TOPIC_RESULTS"])
        self.assertEqual(namespace["CACHED_TOPIC_RESULTS"], path)
        self.assertIn("[SKIP]", log.getvalue())
        self.assertNotIn("final_model", namespace)
        self.assertNotIn("embeddings", namespace)
        self.assertEqual({p: p.read_bytes() for p in self.output.iterdir()}, before)

    def test_script_returns_before_input_loading_and_fitting(self):
        from utils import data_analysis_02_content_bert_topic as runner

        self.results()
        with patch.object(runner, "prepare_input", side_effect=AssertionError("Must not prepare training input")) as prepare:
            with patch.object(runner, "fit_final_model", side_effect=AssertionError("Must not fit")) as fit:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = runner.run_bertopic(self.folder / "missing.parquet", self.output)
                self.assertIsNone(result)
                prepare.assert_not_called()
                fit.assert_not_called()
        with self.assertRaises(FileNotFoundError):
            runner.run_bertopic(self.folder / "missing.parquet", self.output, force=True)


if __name__ == "__main__":
    unittest.main()
