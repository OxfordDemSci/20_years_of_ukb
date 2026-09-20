"""The merged dataset notebook retains the manuscript diagnostic panels."""
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
from utils import data_analysis_00_figures as F


def inputs():
    models = pd.DataFrame({
        "model": ["qwen", "llama3_8b", "mistral_7b"],
        "display_name": ["Qwen2.5-7B", "Llama3-8B", "Mistral-7B"],
        "true_percent_among_parsed": [20, 30, 21], "parse_rate_percent": [99, 100, 100],
    })
    yearly = pd.DataFrame({
        "year_int": [2013, 2014, 2025, 2026], "qwen_TRUE": [2, 3, 80, 9_999],
        "llama3_8b_TRUE": [3, 4, 100, 9_999], "mistral_7b_TRUE": [2, 4, 81, 9_999],
        "three_model_TRUE_agreement": [0, 2, 70, 9_999],
        "rest_NOT_three_model_TRUE_agreement": [10, 30, 400, 9_999],
        "three_model_TRUE_agreement_rate_percent": [0, 6.25, 14.89, 50],
    })
    categories = pd.DataFrame([
        {"binary_split": group, "category": name, "n_group": size, "percent_with_category": value}
        for group, size, values in [(F.TRUE_GROUP, 72, [90, 80, 55]), (F.REST_GROUP, 440, [3, 5, 35])]
        for name, value in zip(["explicit UKB", "genetics / genomics", "cohort / participants"], values)
    ])
    terms = pd.DataFrame({"term": [f"term {i}" for i in range(60)],
                          "difference_TRUE_minus_rest": np.r_[np.linspace(.03, .001, 30),
                                                               np.linspace(-.001, -.03, 30)]})
    semantics = pd.DataFrame({
        "binary_split": [F.TRUE_GROUP] * 3 + [F.REST_GROUP] * 3,
        "year_int": [2013, 2016, 2025, 2014, 2020, 2025],
        "semantic_x": [-.1, .1, .2, -.2, .2, .3], "semantic_y": [.1, -.1, .3, .2, .1, -.2],
    })
    metrics = pd.DataFrame({"metric": ["embedding_method", "SI_silhouette_index_cosine"],
                            "value": ["test embeddings", .123456]})
    groups = pd.DataFrame({"consensus_group": [F.TRUE_GROUP, "Three-model FALSE agreement", "One TRUE vote",
                                                "Two TRUE votes", "No TRUE votes / parsed non-positive"],
                           "n_candidates": [72, 400, 25, 14, 1]})
    return models, yearly, categories, terms, semantics, metrics, groups


class DatasetFigureRetentionTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_manuscript_panels_retain_data_and_current_semantic_metric(self):
        models, yearly, categories, _, semantics, metrics, _ = inputs()
        fig = F.plot_consensus_validation(models, yearly, categories, semantics, metrics)
        self.assertEqual(len(fig.axes), 4)
        annual, counts, keywords, semantic = fig.axes
        self.assertEqual(len(annual.lines), 3)
        self.assertEqual(annual.lines[0].get_xdata().tolist(), list(range(2013, 2026)))
        self.assertEqual(annual.lines[0].get_ydata()[-1], 80)
        self.assertTrue(np.isnan(annual.lines[0].get_ydata()[2]))  # Missing 2015 stays a gap.
        self.assertEqual(counts.get_yscale(), "log")
        self.assertTrue(np.isnan(counts.lines[0].get_ydata()[0]))  # Log axis omits zero.
        self.assertEqual(len(keywords.patches), len(categories))
        self.assertEqual(sum(len(points.get_offsets()) for points in semantic.collections), len(semantics))
        self.assertIn("0.1235", " ".join(text.get_text() for text in semantic.texts))
        self.assertNotIn("0.0386", " ".join(text.get_text() for text in semantic.texts))

    def test_missing_semantic_panel_is_explicit_and_never_simulated(self):
        models, yearly, categories, *_ = inputs()
        fig = F.plot_consensus_validation(models, yearly, categories)
        semantic = fig.axes[3]
        self.assertEqual(len(semantic.collections), 0)
        self.assertIn("incomplete", " ".join(text.get_text() for text in semantic.texts))
        self.assertEqual(len(fig.axes[0].lines), 3)

    def test_semantic_data_outside_window_cannot_use_current_metric(self):
        models, yearly, categories, _, semantics, metrics, _ = inputs()
        semantics.loc[0, "year_int"] = 2026
        with self.assertRaisesRegex(ValueError, "2013–2025"):
            F.plot_consensus_validation(models, yearly, categories, semantics, metrics)

    def test_original_25_term_views_and_detailed_consensus_groups_are_retained(self):
        _, _, categories, terms, _, _, groups = inputs()
        text_fig = F.plot_candidate_text(categories, terms)
        self.assertEqual([len(ax.patches) for ax in text_fig.axes[1:]], [25, 25])
        group_ax = F.plot_consensus_groups(groups).axes[0]
        self.assertEqual(len(group_ax.patches), len(groups))
        np.testing.assert_equal([bar.get_width() for bar in group_ax.patches],
                                sorted(groups.n_candidates, reverse=True))
        self.assertTrue(any("some labels unparsed" in tick.get_text() for tick in group_ax.get_yticklabels()))

    def test_exports_distinguish_complete_from_incomplete_and_keep_existing_figures(self):
        models, yearly, categories, terms, semantics, metrics, groups = inputs()
        votes = pd.DataFrame({"n_true_votes": [0, 1, 2, 3], "n_candidates": [401, 25, 14, 72]})
        agreement = pd.DataFrame(np.eye(3), index=models.model, columns=models.model)
        sample = pd.DataFrame({"binary_split": [F.TRUE_GROUP, F.REST_GROUP]})
        for complete in (False, True):
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as directory:
                with patch("matplotlib.figure.Figure.savefig") as save:
                    files = F.render_agreement_figures(
                        model_summary=models, vote_distribution=votes, agreement_matrix=agreement,
                        yearly=yearly, category_summary=categories, tfidf_terms=terms, tfidf_data=sample,
                        group_distribution=groups, semantic_data=semantics if complete else None,
                        semantic_metrics=metrics if complete else None, figure_dir=directory, show_figures=False)
                stems = {path.stem for path in files}
                self.assertIn("00_01_figure_candidate_agreement", stems)
                self.assertIn("00_02_figure_candidate_text", stems)
                self.assertIn("00_07_figure_consensus_groups", stems)
                self.assertEqual("00_05_figure_semantic_diagnostics" in stems, complete)
                suffix = "" if complete else "_incomplete"
                stem = "00_06_figure_consensus_validation" + suffix
                self.assertIn(stem, stems)
                caption = (Path(directory) / f"{stem}_caption.txt").read_text()
                self.assertIn("test embeddings" if complete else "incomplete", caption)
                self.assertTrue(all(call.kwargs["dpi"] == 500 for call in save.call_args_list))
                self.assertEqual({path.suffix for path in files}, {".png", ".pdf"})


if __name__ == "__main__":
    unittest.main()
