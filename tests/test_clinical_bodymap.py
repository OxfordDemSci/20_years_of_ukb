"""The shared clinical schematic is vector-native, legible and palette-consistent."""

import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.text import Annotation, Text
import numpy as np
import pandas as pd
import pytest

from utils import data_analysis_04_non_academic_figures as F
from utils import data_analysis_04_non_academic_panels as N
from utils import shared_style as S


@pytest.fixture
def trial_data():
    S.load_style("04_non_academic_panels")
    counts = pd.Series([22, 23, 18, 44, 48, 22, 9, 13, 12, 21, 11],
                       index=N.ICD_BODY_POS, dtype="int64")
    yield {"trials": {"icd_chapters": counts, "icd_mapped_trials": 100,
                      "papers_per_trial": pd.Series([1] * 120)}}
    plt.close("all")


def test_clinical_figure_palette_and_left_shift_preserve_body_geometry(trial_data):
    trial_data["trials"].update({
        "stage_by_type": pd.DataFrame({"Interventional": [6, 41], "Observational": [7, 39]},
                                        index=["Planned", "Ongoing"]),
        "rcdc_disease": pd.Series([57, 39], index=["Cardiovascular", "Obesity"]),
        "enrollment": pd.Series([10, 100, 160, 1000, 10000]),
        "country_sector": pd.DataFrame({"Academia": [20, 10], "Industry": [4, 2]},
                                         index=["United States", "United Kingdom"]),
    })
    with patch.object(N, "savefig") as save:
        fig = N.figure_si_trials(trial_data)
    save.assert_called_once_with(fig, "04_03_supplementary_figure_02_clinical_trials")
    for _ in range(2):
        F.finalize_figure(fig)
        fig.canvas.draw()
    a, b, c, d, e = fig.axes
    assert {to_hex(bar.get_facecolor()).upper() for bar in a.patches} == set(S.palette("steel_blue", "red"))
    assert {to_hex(bar.get_facecolor()).upper() for bar in c.patches} == {S.palette("steel_blue")}
    assert {to_hex(bar.get_facecolor()).upper() for bar in d.patches} == {S.palette("cream")}
    original = b.get_subplotspec().get_position(fig)
    shifted = b.get_position(original=True)
    assert shifted.x0 == pytest.approx(original.x0 - .04)
    np.testing.assert_allclose([shifted.y0, shifted.width, shifted.height],
                               [original.y0, original.width, original.height])
    assert b.get_aspect() == 1
    assert b.get_xlim() == (-.3, 10.3)
    assert b.get_ylim() == (.05, 14.25)
    np.testing.assert_array_equal(sorted(bar.get_width() for bar in c.patches), [39, 57])


@pytest.mark.parametrize("figsize", [(6.2, 8), (8, 9)])
def test_bodymap_vector_palette_proportions_and_label_clearance(trial_data, figsize):
    counts = trial_data["trials"]["icd_chapters"]
    original = counts.copy(deep=True)
    fig, ax = plt.subplots(figsize=figsize)
    N.draw_trial_bodymap(ax, trial_data)
    for _ in range(2):
        F.finalize_figure(fig)
        fig.canvas.draw()
    assert not ax.images
    assert {patch.get_gid() for patch in ax.patches} == {"trial-body-outline", "trial-body-head"}
    for patch in ax.patches:
        assert to_hex(patch.get_facecolor()) == "#ffffff"
        assert to_hex(patch.get_edgecolor()) == S.palette("navy").lower()
    dots = [c for c in ax.collections if (c.get_gid() or "").startswith("trial-chapter:")]
    assert len(dots) == len(counts)
    for dot, (chapter, value) in zip(dots, counts.items()):
        assert dot.get_sizes().tolist() == [8 * value]
        assert to_hex(dot.get_facecolors()[0]) == S.palette("steel_blue").lower()
        assert dot.get_facecolors()[0, 3] == 1
        np.testing.assert_array_equal(dot.get_offsets(), [N.ICD_BODY_POS[chapter][:2]])
    labels = [text for text in ax.texts if isinstance(text, Annotation)]
    assert len(labels) == len(counts)
    renderer = fig.canvas.get_renderer()
    boxes = [Text.get_window_extent(text, renderer) for text in labels]
    for index, (text, box) in enumerate(zip(labels, boxes)):
        assert text.get_size() == 12
        assert text.get_fontfamily() == ["Helvetica"]
        assert "trials" in text.get_text()
        assert ax.bbox.contains(*box.p0) and ax.bbox.contains(*box.p1)
        assert not any(box.overlaps(other) for other in boxes[index + 1:])
    for text in labels:
        if text.arrow_patch is not None:
            assert text.arrow_patch.get_connectionstyle().rad == 0
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["10", "30", "50"]
    assert [handle.get_sizes().item() for handle in legend.legend_handles] == [80, 240, 400]
    legend_box = legend.get_window_extent(renderer)
    for handle in legend.legend_handles:
        centre = handle.get_offset_transform().transform(handle.get_offsets())[0]
        radius = np.sqrt(handle.get_sizes()[0]) * fig.dpi / 144
        assert legend_box.contains(*(centre - radius))
        assert legend_box.contains(*(centre + radius))
    caption = F._caption(fig, "ct_icd_bodymap")
    assert "100 of 120 trials" in caption
    assert "directly proportional" in caption and "schematic" in caption
    assert "systemic" in caption
    pd.testing.assert_series_equal(counts, original)


def test_bodymap_handles_zero_and_missing_counts_without_false_bubbles(trial_data):
    trial_data["trials"]["icd_chapters"] = pd.Series(
        [0, np.nan], index=list(N.ICD_BODY_POS)[:2])
    fig, ax = plt.subplots(figsize=(7, 9))
    N.draw_trial_bodymap(ax, trial_data)
    F.finalize_figure(fig)
    fig.canvas.draw()
    assert not any((c.get_gid() or "").startswith("trial-chapter:") for c in ax.collections)


def test_trial_stage_colours_remain_exact_palette_values_after_finalization():
    S.load_style("04_non_academic_panels")
    frame = pd.DataFrame({"Interventional": [35, 41], "Observational": [25, 39]},
                         index=["Completed", "Ongoing"])
    fig, ax = plt.subplots()
    N.draw_trial_stage(ax, {"trials": {"stage_by_type": frame}})
    F.finalize_figure(fig)
    expected = S.palette("steel_blue", "red")
    assert [to_hex(c.patches[0].get_facecolor()).upper() for c in ax.containers] == expected
    assert [to_hex(h.get_facecolor()).upper() for h in ax.get_legend().legend_handles] == expected
    plt.close(fig)


def test_notebook_reuses_bodymap_without_changing_mapping_or_export_name():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "c83c8411")["source"])
    assert "def trials_by_icd_chapter" in source
    assert "icd_counts, n_icd = trials_by_icd_chapter(df_ct)" in source
    assert "draw_trial_bodymap(ax," in source
    assert 'name="ct_icd_bodymap"' in source
    assert "def _draw_body" not in source
    assert "BODY_COLOR" not in source
