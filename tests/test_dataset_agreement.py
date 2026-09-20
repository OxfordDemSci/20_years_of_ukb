"""Agreement consolidation: date boundaries, saved labels and semantic reuse."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import data_analysis_00_agreement as A


def candidate(identifier, year, label=1, **extra):
    result = dict(id=identifier, year=year, title="UK Biobank genetics", abstract="Cohort research")
    for model in A.MODEL_NAMES:
        result[f"{model}_label"] = label
        result[f"{model}_parse_ok"] = True
    result.update(extra)
    return result


class DatasetAgreementTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_inclusive_dates_before_deduplication_and_blank_ids(self):
        frame = pd.DataFrame([
            candidate("repeat", 2012), candidate("repeat", 2013),
            candidate("end", 2025), candidate("future", 2026),
            candidate(None, 2020), candidate("", 2020),
            candidate("conflict", 2025, date="2026-01-01"),
        ])
        actual = A.prepare_candidates(frame)
        self.assertEqual(actual.id.tolist(), ["repeat", "end"])
        self.assertEqual(actual.year_int.tolist(), [2013, 2025])

    def test_invalid_saved_label_is_unparsed_not_false_or_true(self):
        frame = pd.DataFrame([candidate("a", 2024, qwen_label="unknown")])
        actual = A.prepare_candidates(frame).iloc[0]
        self.assertEqual(actual.n_models_parsed, 2)
        self.assertEqual(actual.n_true_votes, 2)
        self.assertFalse(actual.qwen_true)
        self.assertFalse(actual.qwen_false)
        self.assertIn("qwen=NA", actual.vote_signature)

    def test_missing_input_skips_without_reading_or_encoding(self):
        with patch.object(A, "resolve_combined_labels", return_value=None), \
             patch.object(pd, "read_csv", side_effect=AssertionError("unexpected read")), \
             contextlib.redirect_stdout(io.StringIO()):
            result = A.run_agreement()
        self.assertEqual(result["status"], "SKIP")

    def test_semantic_reuse_matches_ids_texts_groups_and_years(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            sample = pd.DataFrame({"id": ["a", "b"], "analysis_text": ["text a", "text b"],
                                   "binary_split": ["TRUE", "Rest"], "year_int": [2013, 2025]})
            sample.assign(semantic_x=[1., 2.], semantic_y=[3., 4.]).to_csv(
                folder / "semantic_sample_with_coordinates.csv", index=False)
            pd.DataFrame({"metric": ["SI_silhouette_index_cosine"], "value": [.2]}).to_csv(
                folder / "semantic_metrics.csv", index=False)
            self.assertIsNotNone(A._semantic_cache(folder, sample))
            for column, changed in (("id", "c"), ("analysis_text", "changed"),
                                    ("binary_split", "Rest"), ("year_int", 2024)):
                mismatch = sample.copy()
                mismatch.loc[0, column] = changed
                self.assertIsNone(A._semantic_cache(folder, mismatch))

    def test_all_non_model_analyses_run_and_export_with_semantic_disabled(self):
        rows = []
        for label in (0, 1):
            for index in range(12):
                rows.append(candidate(f"paper-{label}-{index}", 2013 + index, label,
                    title="genetic cardiovascular cohort" if label else "laboratory cancer oncology",
                    abstract="UK Biobank population participants inherited traits" if label else "Mouse animal experiments microscopy pathology"))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "labels.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            # Exercise figure construction while avoiding large publication exports in a unit test.
            with patch("matplotlib.figure.Figure.savefig") as savefig, \
                 patch.object(A, "_semantic_cache", return_value=None), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = A.run_agreement(source, output_dir=folder / "tables",
                                         figure_dir=folder / "figures", show_figures=False)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["semantic_status"], "SKIP")
            self.assertEqual([Path(path).name for path in result["figures"]], [
                "00_01_figure_candidate_agreement.png", "00_01_figure_candidate_agreement.pdf",
                "00_02_figure_candidate_text.png", "00_02_figure_candidate_text.pdf",
            ])
            self.assertEqual(savefig.call_count, 4)
            self.assertTrue(all(call.kwargs["dpi"] == 500 for call in savefig.call_args_list))
            self.assertEqual(len(list((folder / "figures").glob("*_caption.txt"))), 2)
            normalised = pd.read_csv(folder / "tables/combined_consensus_normalised.csv")
            self.assertEqual(len(normalised), 24)
            self.assertEqual(normalised.three_model_TRUE_agreement.sum(), 12)
            self.assertTrue((folder / "tables/tfidf_discriminative_terms.csv").is_file())
            self.assertTrue((folder / "tables/pairwise_model_agreement.csv").is_file())


if __name__ == "__main__":
    unittest.main()
