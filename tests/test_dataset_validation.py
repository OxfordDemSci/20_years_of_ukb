"""Validation reuse/preflight checks must never load or run tagging models."""
import builtins
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import data_analysis_00_validation as validation


class ValidationReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "tables"
        self.figures = self.root / "figures"
        self.original_import = builtins.__import__

    def tearDown(self):
        self.temp.cleanup()

    def no_ml_imports(self, name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "sentence_transformers", "huggingface_hub"}:
            raise AssertionError(f"Unexpected model import: {name}")
        return self.original_import(name, *args, **kwargs)

    def make_cache(self):
        self.output.mkdir()
        frame = pd.DataFrame({
            "id": ["a", "b", "c", "d"], "year": [2013, 2014, 2025, 2025],
            "True_label": [True, False, True, False],
            "qwen2_5_7b": [True, False, True, None],
            "sbert_minilm_sim": [True, True, False, False],
        })
        for prompt, _ in validation.PROMPTS:
            frame.to_csv(self.output / f"predictions_{prompt}.csv", index=False)
        return frame

    def run_cached(self):
        with patch("builtins.__import__", side_effect=self.no_ml_imports), \
             patch.object(validation, "_run_original_inference", side_effect=AssertionError("Model inference called")), \
             patch.object(validation, "_render_performance", return_value=[]), \
             patch.object(validation, "_render_agreement", return_value=[]) as render:
            result = validation.run_validation(self.output, figure_dir=self.figures)
        return result, render

    def test_valid_predictions_recompute_metrics_without_models(self):
        self.make_cache()
        source_bytes = {p: p.read_bytes() for p in self.output.glob("predictions_*.csv")}
        result, render = self.run_cached()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["n_papers"], 4)
        render.assert_called_once()
        self.assertEqual(len(result["summary"]), 12)
        qwen = result["summary"].query('model == "qwen2_5_7b"').iloc[0]
        self.assertEqual(qwen.accuracy, 1)
        self.assertEqual(qwen.parse_rate, .75)
        self.assertEqual(qwen.tp, 2)
        enc = result["summary"].query('model == "sbert_minilm_sim"').iloc[0]
        self.assertEqual(enc.accuracy, .5)
        self.assertIn("in-sample", enc.calibration_note)
        agreement = pd.read_csv(self.output / "pairwise_agreement_percent_p1_conservative.csv", index_col=0)
        self.assertEqual(agreement.loc["qwen2_5_7b", "sbert_minilm_sim"], 33.3)
        for path, before in source_bytes.items():
            self.assertEqual(path.read_bytes(), before)

    def test_missing_predictions_skip_without_inference(self):
        result, render = self.run_cached()
        self.assertEqual(result["status"], "SKIP")
        self.assertIn("No saved validation predictions", result["reason"])
        render.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_out_of_window_cache_is_rejected_before_plotting(self):
        frame = self.make_cache()
        frame.loc[0, "year"] = 2012
        frame.to_csv(self.output / "predictions_p1_conservative.csv", index=False)
        with self.assertRaisesRegex(ValueError, "2013–2025"):
            self.run_cached()
        self.assertFalse((self.output / "results_p1_conservative.csv").exists())

    def test_inconsistent_cached_ground_truth_is_rejected(self):
        frame = self.make_cache()
        frame.loc[0, "True_label"] = False
        frame.to_csv(self.output / "predictions_p6_real_five_shot.csv", index=False)
        with self.assertRaisesRegex(ValueError, "same evaluation papers/labels"):
            self.run_cached()

    def test_partial_cache_does_not_trigger_inference(self):
        self.make_cache()
        (self.output / "predictions_p2_balanced.csv").unlink()
        with patch.object(validation, "_run_original_inference", side_effect=AssertionError("Model inference called")):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                validation.run_validation(self.output, run_inference=True)

    def test_fresh_inference_requires_explicit_sample_sizes(self):
        with patch("builtins.__import__", side_effect=self.no_ml_imports):
            with self.assertRaisesRegex(ValueError, "UKB_VALIDATION_N_POS"):
                validation.run_validation(self.output, run_inference=True)

    def test_missing_inputs_fail_before_model_imports(self):
        with patch("builtins.__import__", side_effect=self.no_ml_imports):
            with self.assertRaisesRegex(FileNotFoundError, "Missing labelled validation CSV"):
                validation.run_validation(self.output, n_positive=1, n_negative=1,
                                          positive_csv=self.root / "absent_positive.csv",
                                          negative_csv=self.root / "absent_negative.csv", run_inference=True)

    def test_overlapping_source_ids_fail_before_model_imports(self):
        base = pd.DataFrame({"id": ["a", "b", "c", "d"], "year": [2013] * 4,
                             "title": ["title"] * 4, "abstract": ["abstract"] * 4})
        positive, negative = self.root / "positive.csv", self.root / "negative.csv"
        base.to_csv(positive, index=False)
        base.to_csv(negative, index=False)
        with patch("builtins.__import__", side_effect=self.no_ml_imports):
            with self.assertRaisesRegex(ValueError, "overlap"):
                validation.run_validation(self.output, positive_csv=positive, negative_csv=negative,
                                          n_positive=1, n_negative=1, run_inference=True)

    def test_out_of_window_source_rows_cannot_supply_prompt_examples(self):
        positive, negative = self.root / "positive.csv", self.root / "negative.csv"
        pd.DataFrame({"id": ["a", "b", "c", "d"], "year": [2012, 2014, 2025, 2026],
                      "title": ["title"] * 4, "abstract": ["abstract"] * 4}).to_csv(positive, index=False)
        pd.DataFrame({"id": ["e", "f", "g"], "year": [2013] * 3,
                      "title": ["title"] * 3, "abstract": ["abstract"] * 3}).to_csv(negative, index=False)
        with patch("builtins.__import__", side_effect=self.no_ml_imports):
            with self.assertRaisesRegex(ValueError, "2 valid 2013–2025 papers"):
                validation.run_validation(self.output, positive_csv=positive, negative_csv=negative,
                                          n_positive=1, n_negative=1, run_inference=True)

    def test_cache_is_checked_against_explicit_source_text(self):
        frame = self.make_cache().assign(title="Original title", abstract="Original abstract")
        for prompt, _ in validation.PROMPTS:
            frame.to_csv(self.output / f"predictions_{prompt}.csv", index=False)
        positive, negative = self.root / "positive.csv", self.root / "negative.csv"
        frame.loc[frame.True_label].to_csv(positive, index=False)
        frame.loc[~frame.True_label].to_csv(negative, index=False)
        with patch.object(validation, "_render_agreement", return_value=[]), \
             patch.object(validation, "_render_performance", return_value=[]), \
             patch("builtins.__import__", side_effect=self.no_ml_imports):
            result = validation.run_validation(self.output, figure_dir=self.figures,
                                               positive_csv=positive, negative_csv=negative,
                                               n_positive=2, n_negative=2)
            self.assertEqual(result["status"], "PASS")
            changed = pd.read_csv(positive)
            changed.loc[0, "abstract"] = "Changed source abstract"
            changed.to_csv(positive, index=False)
            with self.assertRaisesRegex(ValueError, "abstract values differ"):
                validation.run_validation(self.output, figure_dir=self.figures,
                                          positive_csv=positive, negative_csv=negative)

    def test_current_figure_inventory_excludes_stale_files(self):
        self.make_cache()
        self.figures.mkdir()
        (self.figures / "stale_heatmap.png").touch()
        current = [self.figures / f"{stem}.{ext}"
                   for stem in (validation.PERFORMANCE_STEM, validation.AGREEMENT_STEM)
                   for ext in ("png", "pdf")]
        with patch.object(validation, "_render_performance", return_value=current[:2]), \
             patch.object(validation, "_render_agreement", return_value=current[2:]):
            result = validation.run_validation(self.output, figure_dir=self.figures, show_figures=False)
        self.assertEqual(result["figure_files"], current)
        self.assertNotIn(self.figures / "stale_heatmap.png", result["files"])
        self.assertEqual(len(list(self.figures.glob("*_caption.txt"))), 2)
        self.assertFalse(list(self.output.glob("*_caption.txt")))

    def test_explicit_inference_reports_generated_predictions(self):
        positive, negative = self.root / "positive.csv", self.root / "negative.csv"
        source = pd.DataFrame({"id": list("abcdefgh"), "year": [2025] * 8,
                               "title": ["Title"] * 8, "abstract": ["Abstract"] * 8})
        source.iloc[:4].to_csv(positive, index=False)
        source.iloc[4:].to_csv(negative, index=False)
        def fake_inference(*args):
            frame = source.iloc[[0, 4]].copy()
            frame["True_label"] = [True, False]
            frame["qwen2_5_7b"] = [True, False]
            for prompt, _ in validation.PROMPTS:
                frame.to_csv(self.output / f"predictions_{prompt}.csv", index=False)
        with patch.object(validation, "_run_original_inference", side_effect=fake_inference), \
             patch.object(validation, "_render_performance", return_value=[]), \
             patch.object(validation, "_render_agreement", return_value=[]), \
             patch("builtins.__import__", side_effect=self.no_ml_imports):
            result = validation.run_validation(self.output, figure_dir=self.figures,
                                               positive_csv=positive, negative_csv=negative,
                                               n_positive=1, n_negative=1, run_inference=True)
        self.assertEqual(result["reason"], "Generated predictions by explicit inference")

    def test_cold_inference_no_longer_exports_individual_heatmaps(self):
        source = inspect.getsource(validation._run_original_inference)
        self.assertNotIn("save_agreement_heatmap", source)
        self.assertNotIn("savefig", source)


class ValidationFigureTests(unittest.TestCase):
    def test_missing_agreement_cells_are_blank_and_model_labels_readable(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        models = ["qwen2_5_7b", "sbert_minilm_sim"]
        agreement = pd.DataFrame([[1., .5], [.5, 1.]], index=models, columns=models)
        counts = pd.DataFrame([[10, 0], [0, 10]], index=models, columns=models)
        captured = []
        def capture(fig, *args, **kwargs):
            captured.append(fig)
            return []
        with patch.object(validation, "_save_publication_figure", side_effect=capture):
            validation._render_agreement({p: agreement for p, _ in validation.PROMPTS}, Path("/tmp"),
                                         comparison_counts={p: counts for p, _ in validation.PROMPTS},
                                         n_papers=10, show_figures=False)
        fig = captured[0]
        ax = fig.axes[0]
        array = ax.collections[0].get_array()
        self.assertEqual(int(np.ma.getmaskarray(array).sum()), 2)
        self.assertEqual(sum(t.get_text() == "—" for t in ax.texts), 2)
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ["Qwen2.5-7B", "MiniLM*"])
        self.assertEqual(ax.collections[0].get_clim(), (0, 1))
        self.assertIn("n = 10 / 10 papers", ax.texts[0].get_text())
        self.assertIn("in-sample", fig.texts[0].get_text())
        from utils.shared_style import finalize_figure
        finalize_figure(fig)
        fig.canvas.draw()
        self.assertFalse(ax._left_title.get_window_extent().overlaps(ax.texts[0].get_window_extent()))
        plt.close(fig)

    def test_performance_layout_has_four_metrics_and_no_zero_imputation(self):
        import matplotlib.pyplot as plt
        import numpy as np
        rows = [{"model": model, "prompt": prompt, "accuracy": .9, "precision": .8,
                 "recall": .7, "f1": .75, "parse_rate": .99}
                for prompt, _ in validation.PROMPTS for model in ["qwen2_5_7b", "sbert_minilm_sim"]]
        rows = rows[:-1]
        captured = []
        with patch.object(validation, "_save_publication_figure", side_effect=lambda fig, *a, **kw: captured.append(fig) or []):
            validation._render_performance(pd.DataFrame(rows), Path("/tmp"), n_papers=20,
                                           n_positive=10, show_figures=False)
        fig = captured[0]
        self.assertEqual([a.get_title(loc="left") for a in fig.axes[:4]],
                         ["A  ACCURACY", "B  PRECISION", "C  RECALL", "D  F1 SCORE"])
        for ax in fig.axes[:4]:
            self.assertEqual(ax.collections[0].get_clim(), (0, 1))
            self.assertEqual(int(np.ma.getmaskarray(ax.collections[0].get_array()).sum()), 1)
        self.assertIn("99.0–99.0%", fig.texts[0].get_text())
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
