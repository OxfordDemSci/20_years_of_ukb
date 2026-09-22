"""The held-out plotting extraction preserves all results and figure families."""
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
from utils import data_analysis_00_heldout_figures as H
from utils.shared_style import blue_cream_red_colormap, load_style


class HeldoutFigureTests(unittest.TestCase):
    def setUp(self):
        self.prompts = {f"p{i}": f"Prompt {i}" for i in range(6)}
        self.models = {f"m{i}": f"Model {i}" for i in range(6)}
        values = np.full((6, 6), .75)
        np.fill_diagonal(values, 1)
        self.agreements = {
            p: pd.DataFrame(values, index=self.models, columns=self.models) for p in self.prompts
        }
        self.metrics = pd.DataFrame([
            {"prompt": p, "model": m, "model_display": self.models[m],
             "precision": .8, "recall": .7, "f1": .7467}
            for p in self.prompts for m in self.models
        ])
        self.primary = self.metrics.loc[self.metrics.prompt.eq("p1")]
        self.matrix = np.array([[70, 10], [20, 60]])
        self.lookup = {(p, m): self.matrix.copy() for p in self.prompts for m in self.models}

    def tearDown(self):
        plt.close("all")

    def test_common_typography_palette_and_unaltered_heatmap_values(self):
        fig = H.plot_heldout_agreement(self.agreements, self.models)
        style = load_style("00_dataset", activate=False)
        for ax, letter in zip(fig.axes[:6], "ABCDEF"):
            self.assertEqual(ax.get_title(loc="left"), letter)
            self.assertEqual(ax._left_title.get_fontsize(), style["title_fs"])
            self.assertTrue(all(t.get_fontsize() == style["tick_fs"] for t in ax.get_xticklabels()))
            image = ax.collections[0]
            array = image.get_array()
            mask = np.triu(np.ones((6, 6), dtype=bool), k=1)
            np.testing.assert_array_equal(np.ma.getmaskarray(array), mask)
            np.testing.assert_allclose(array.compressed(), self.agreements["p0"].to_numpy()[~mask])
            self.assertEqual(len(ax.texts), 21)
            self.assertEqual(len(ax.patches), 21)
            self.assertTrue(all(p.get_edgecolor() == (0, 0, 0, 1) for p in ax.patches))
            self.assertTrue(all(t.tick1line.get_visible() and t.tick1line.get_markersize() > 0
                                for t in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks()))
            np.testing.assert_allclose(image.cmap(np.linspace(0, 1, 7)),
                                       blue_cream_red_colormap()(np.linspace(0, 1, 7)))
            self.assertEqual(image.get_clim(), (.75, 1))
        self.assertIsNone(fig._suptitle)

    def test_confusion_panels_keep_counts_row_percentages_and_classifier_labels(self):
        fig = H.plot_heldout_confusion(self.lookup, ["p1"], self.models, self.prompts, self.models)
        ax = fig.axes[0]
        np.testing.assert_allclose(ax.images[0].get_array(), self.matrix / 80)
        self.assertEqual(ax.images[0].get_clim(), (.125, .875))
        self.assertEqual([t.get_text() for t in ax.texts],
                         ["70\n(87.5%)", "10\n(12.5%)", "20\n(25.0%)", "60\n(75.0%)"])
        self.assertEqual([a.get_title(loc="left") for a in fig.axes[:6]], list("ABCDEF"))
        self.assertEqual([a.get_xlabel() for a in fig.axes[:6]], list(self.models.values()))
        self.assertEqual(ax.images[0].cmap.name, blue_cream_red_colormap().name)
        self.assertEqual(H._letter(35), "AJ")

    def test_agreement_scale_uses_all_visible_panels_and_larger_colourbar_text(self):
        self.agreements["p5"].iloc[5, 0] = .2
        self.agreements["p0"].iloc[0, 5] = .01  # Hidden upper triangle.
        self.agreements["p1"].iloc[4, 0] = np.nan
        fig = H.plot_heldout_agreement(self.agreements, self.models)
        for ax in fig.axes[:6]:
            self.assertEqual(ax.collections[0].get_clim(), (.2, 1))
        style = load_style("00_dataset", activate=False)
        self.assertEqual(fig.axes[-1].yaxis.label.get_fontsize(), style["colorbar_label_fs"])
        np.testing.assert_allclose(fig.axes[-1].get_yticks()[[0, -1]], [.2, 1])

    def test_all_sixteen_figures_have_inline_captions_without_caption_files(self):
        with tempfile.TemporaryDirectory() as directory, patch("matplotlib.figure.Figure.savefig"), \
                patch("utils.data_analysis_00_figures.display_figure") as show:
            files = H.render_heldout_agreement(self.agreements, self.prompts, self.models,
                                               directory, show_figures=True)
            files += H.render_heldout_performance(
                self.metrics, self.primary, self.prompts, self.models,
                self.prompts, self.models, "p1", directory, show_figures=True,
            )
            files += H.render_heldout_confusion(self.lookup, self.prompts, self.models,
                                               self.prompts, self.models, directory, show_figures=True)
            self.assertEqual(len(files), 32)
            self.assertEqual(len({p.stem for p in files}), 16)
            self.assertEqual({p.suffix for p in files}, {".png", ".pdf"})
            self.assertFalse(list(Path(directory).glob("*_caption.txt")))
            self.assertEqual(show.call_count, 16)
            captions = [call.args[2] for call in show.call_args_list]
            self.assertTrue(all(captions))
            self.assertTrue(any("A, Precision; B, Recall; C, F1" in c for c in captions))

    def test_performance_panels_share_observed_range(self):
        fig = H.plot_heldout_performance(self.metrics, self.prompts, self.models,
                                         self.prompts, self.models)
        for ax in fig.axes[:3]:
            self.assertEqual(ax.collections[0].get_clim(), (.7, .8))
        self.assertEqual(fig.axes[-1].yaxis.label.get_fontsize(), 16)


if __name__ == "__main__":
    unittest.main()
