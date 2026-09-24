"""Attention scatters retain the data, not overlapping bubbles and statistic boxes."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from utils import data_analysis_04_non_academic_figures as F
from utils import data_analysis_04_non_academic_panels as N
from utils import shared_paths as P
from utils import shared_style as S


@pytest.fixture
def attention_data():
    S.load_style("04_non_academic_panels")
    rng = np.random.default_rng(36)
    mentions = np.maximum(1, np.round(10 ** rng.uniform(0, 2.8, 600)))
    scores = np.maximum(1, mentions * rng.lognormal(1.2, .5, len(mentions)))
    frame = pd.DataFrame({"substantive": mentions,
                          "Altmetric Attention Score": scores,
                          "times_cited": rng.integers(0, 3000, len(mentions)).astype(float)})
    frame.loc[:1, "times_cited"] = [0, np.nan]
    top = pd.DataFrame({"substantive": [1125, 820],
                        "Altmetric Attention Score": [15564, 6068],
                        "times_cited": [1539, 88],
                        "first_author": ["Douaud", "Chieng"], "year": [2022, 2022],
                        "journal": ["Nature", "European Journal of Preventive Cardiology"]})
    frame = pd.concat([frame, top], ignore_index=True)
    yield {"altmetric": {
        "scatter": frame, "scatter_top": top,
        "score_distribution": frame["Altmetric Attention Score"].copy(),
        "mentions_by_year": pd.DataFrame({"News mentions": [50, 500, 900],
                                          "Policy mentions": [2, 10, 8]}, index=[2013, 2019, 2025]),
        "coverage_by_year": pd.DataFrame({"pct_news": [60., 35., 25.],
                                           "pct_policy": [40., 10., 1.]}, index=[2013, 2019, 2025]),
    }}
    plt.close("all")


def assert_scatter_style(ax, color=N.ATTENTION_PRIMARY, *, outlined=True):
    assert ax.get_xscale() == ax.get_yscale() == "log"
    for side, spine in ax.spines.items():
        assert spine.get_visible() == (not outlined or side in ("left", "bottom"))
    assert ax.xaxis.label.get_size() == ax.yaxis.label.get_size() == 14
    assert ax.xaxis.label.get_fontfamily() == ["Helvetica"]
    assert all(t.get_size() == 12 for t in ax.get_xticklabels() + ax.get_yticklabels())
    assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    assert ax.get_legend() is None
    cloud = ax.collections[0]
    assert cloud.get_sizes().tolist() == [24 if outlined else 14]
    assert to_hex(cloud.get_facecolors()[0]) == S.palette(color).lower()
    if outlined:
        np.testing.assert_array_equal(cloud.get_edgecolors(), [[0, 0, 0, 1]])
        np.testing.assert_allclose(cloud.get_linewidths(), [.3])
        assert cloud.get_facecolors()[0, 3] == .65
    assert cloud.get_rasterized()


@pytest.mark.parametrize("figsize", [(7, 6.5), (6.5, 4)])
def test_mentions_points_and_readable_callouts_survive_finalization(attention_data, figsize):
    frame = attention_data["altmetric"]["scatter"]
    original = frame.copy(deep=True)
    fig, ax = plt.subplots(figsize=figsize)
    N.draw_altmetric_scatter(ax, attention_data)
    for _ in range(2):
        F.finalize_figure(fig)
        fig.canvas.draw()
    assert_scatter_style(ax)
    np.testing.assert_array_equal(ax.collections[0].get_offsets(),
                                  frame[["substantive", "Altmetric Attention Score"]])
    top = attention_data["altmetric"]["scatter_top"]
    np.testing.assert_array_equal(ax.collections[1].get_offsets(),
                                  top[["substantive", "Altmetric Attention Score"]])
    assert to_hex(ax.collections[1].get_facecolors()[0]) == S.palette(N.ATTENTION_SECONDARY).lower()
    assert ax.collections[1].get_sizes().tolist() == [70]
    np.testing.assert_array_equal(ax.collections[1].get_edgecolors(), [[0, 0, 0, 1]])
    np.testing.assert_allclose(ax.collections[1].get_linewidths(), [.65])
    assert {t.get_text() for t in ax.texts} == {"Douaud et al. (2022)", "Chieng et al. (2022)"}
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    boxes = [t.get_bbox_patch().get_window_extent(renderer) for t in ax.texts]
    assert not boxes[0].overlaps(boxes[1])
    points = ax.transData.transform(ax.collections[0].get_offsets())
    for label, box in zip(ax.texts, boxes):
        assert bounds.contains(*box.p0) and bounds.contains(*box.p1)
        assert not any(box.contains(*point) for point in points)
        assert label.get_size() == 12
        assert abs(label.arrow_patch.get_connectionstyle().rad) == .20
        assert label.arrow_patch.shrinkA == 6
        assert label.arrow_patch.shrinkB == 14
        assert label.arrow_patch.get_arrowstyle().arrow == "->"
        assert np.linalg.norm(label.get_position()) >= 56 - 1e-8
        assert to_hex(label.get_bbox_patch().get_facecolor()) == "#ffffff"
    pd.testing.assert_frame_equal(frame, original)


def test_scatter_caption_keeps_statistics_without_repeating_paper_details(attention_data):
    fig, ax = plt.subplots(figsize=(7, 6))
    N.draw_altmetric_scatter(ax, attention_data)
    caption = F._caption(fig, P.MAIN_FIGURE_STEMS[6])
    assert "each uniform small point" in caption
    assert "news and policy mentions" in caption
    assert "602 publications" in caption
    assert "mean attention score" in caption and "median" in caption and "maximum 15,564" in caption
    assert "European Journal of Preventive Cardiology" not in caption
    assert "Highlighted publications:" not in caption
    assert len(caption.split()) <= 258
    assert "bubble" not in caption.lower()


def test_five_curved_arrows_clear_markers_text_and_each_other(attention_data):
    lower = pd.DataFrame({"substantive": [11, 194], "Altmetric Attention Score": [4., 545.],
                          "first_author": ["Yip", "Levy"], "year": [2021, 2020],
                          "times_cited": [10, 20], "journal": ["Journal A", "Journal B"]})
    attention_data["altmetric"]["scatter_lower"] = lower
    central = pd.DataFrame({"substantive": [12], "Altmetric Attention Score": [1738.],
                            "first_author": ["Hill"], "year": [2019],
                            "times_cited": [100], "journal": ["Nature Communications"]})
    attention_data["altmetric"]["scatter_center"] = central
    attention_data["altmetric"]["scatter"] = pd.concat(
        [attention_data["altmetric"]["scatter"], lower, central], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    N.draw_altmetric_scatter(ax, attention_data)
    for dpi in (100, 200):
        fig.set_dpi(dpi)
        F.finalize_figure(fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [text.get_bbox_patch().get_window_extent(renderer) for text in ax.texts]
        assert len(boxes) == 5
        curves = [np.concatenate([segment(np.linspace(0, 1, 201))
                                  for segment, _ in text.arrow_patch.get_path().iter_bezier()])
                  for text in ax.texts]
        for curve in curves:
            for box in boxes:
                assert not np.any((curve[:, 0] >= box.x0) & (curve[:, 0] <= box.x1)
                                  & (curve[:, 1] >= box.y0) & (curve[:, 1] <= box.y1))
            for collection in ax.collections:
                points = ax.transData.transform(collection.get_offsets())
                radius = (np.sqrt(collection.get_sizes()[0]) / 2 + collection.get_linewidths()[0] / 2) * dpi / 72
                assert np.min(np.linalg.norm(curve[:, None, :] - points[None, :, :], axis=2)) > radius
        for i, curve in enumerate(curves):
            for other in curves[i+1:]:
                assert np.min(np.linalg.norm(curve[:, None, :] - other[None, :, :], axis=2)) > dpi / 72
        for text in ax.texts:
            if text.get_text().startswith(("Yip", "Levy")):
                assert text.get_position()[1] < 0
    caption = F._caption(fig, P.MAIN_FIGURE_STEMS[6])
    assert "lowest-ratio and most-mentioned" in caption
    assert "five to thirty" in caption
    assert "Hill et al. (2019)" in {text.get_text() for text in ax.texts}
    assert len(caption.split()) <= 258


def test_citation_scatter_keeps_its_cohort_and_moves_statistics_to_caption(attention_data):
    frame = attention_data["altmetric"]["scatter"]
    original = frame.copy(deep=True)
    expected = frame[frame.times_cited > 0]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    N.draw_altmetric_vs_citations(ax, attention_data)
    F.finalize_figure(fig)
    fig.canvas.draw()
    assert_scatter_style(ax, N.ATTENTION_NEWS, outlined=False)
    np.testing.assert_array_equal(ax.collections[0].get_offsets(),
                                  expected[["times_cited", "Altmetric Attention Score"]])
    assert len(ax.collections) == 1 and not ax.texts
    rho = expected[["times_cited", "Altmetric Attention Score"]].corr(method="spearman").iat[0, 1]
    caption = F._caption(fig, "04_05_supplementary_figure_04_altmetric")
    assert f"{len(expected):,} publications" in caption
    assert f"Spearman rho = {rho:.2f}" in caption
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("draw", [N.draw_altmetric_scatter, N.draw_altmetric_vs_citations])
def test_empty_and_single_publication_scatters(attention_data, draw, count):
    attention_data["altmetric"]["scatter"] = attention_data["altmetric"]["scatter"].tail(count)
    attention_data["altmetric"]["scatter_top"] = None
    fig, ax = plt.subplots(figsize=(6, 4))
    draw(ax, attention_data)
    F.finalize_figure(fig)
    fig.canvas.draw()
    color = N.ATTENTION_NEWS if draw is N.draw_altmetric_vs_citations else N.ATTENTION_PRIMARY
    assert_scatter_style(ax, color, outlined=draw is N.draw_altmetric_scatter)
    assert len(ax.collections[0].get_offsets()) == count
    assert np.isfinite([*ax.get_xlim(), *ax.get_ylim()]).all()


def test_supplementary_attention_is_square_with_exact_palette_and_major_only_grids(attention_data):
    originals = {key: value.copy(deep=True) for key, value in attention_data["altmetric"].items()}
    fig = N.figure_si_altmetric(attention_data, save=False)
    for _ in range(2):
        F.finalize_figure(fig)
        fig.canvas.draw()
    a, b, c, d, twin = fig.axes
    assert fig.get_figwidth() == fig.get_figheight()
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["A", "B", "C", "D", ""]
    np.testing.assert_allclose(b.get_position().bounds, twin.get_position().bounds)
    expected_grid = (S._resolve(None)["grid_linestyle"], S._resolve(None)["grid_linewidth"],
                     S._resolve(None)["grid_color"], S._resolve(None)["grid_alpha"])
    renderer = fig.canvas.get_renderer()
    for ax in (a, b, c, d):
        assert ax.bbox.width == pytest.approx(ax.bbox.height)
        assert ax.xaxis.label.get_size() == ax.yaxis.label.get_size() == 14
        assert all(t.get_size() == 12 for t in ax.get_xticklabels() + ax.get_yticklabels())
        assert all(spine.get_visible() for spine in ax.spines.values())
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks():
                grid = tick.gridline
                assert grid.get_visible()
                assert (grid.get_linestyle(), grid.get_linewidth(), grid.get_color(), grid.get_alpha()) == expected_grid
            assert not any(tick.gridline.get_visible() for tick in axis.get_minor_ticks())
        box = ax.get_tightbbox(renderer)
        assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1)
    assert not any(line.get_visible() for line in twin.get_xgridlines() + twin.get_ygridlines())
    blue, red = S.palette("steel_blue", "red")
    assert {to_hex(bar.get_facecolor()).upper() for bar in a.patches} == {blue}
    assert to_hex(a.lines[0].get_color()).upper() == red
    assert a.lines[0].get_xdata()[0] == originals["score_distribution"].median()
    assert a.get_legend().get_texts()[0].get_text().startswith("Median:")
    assert to_hex(b.lines[0].get_color()).upper() == blue
    assert to_hex(twin.lines[0].get_color()).upper() == red
    assert [to_hex(line.get_color()).upper() for line in c.lines] == [blue, red]
    assert [line.get_marker() for line in c.lines] == ["o", "s"]
    assert to_hex(d.collections[0].get_facecolors()[0]).upper() == blue
    for ax in (b, c):
        legend = ax.get_legend()
        assert [to_hex(h.get_color()).upper() for h in legend.legend_handles] == [blue, red]
        assert all(text.get_size() == 12 for text in legend.get_texts())
    np.testing.assert_array_equal(b.lines[0].get_ydata(), originals["mentions_by_year"]["News mentions"])
    np.testing.assert_array_equal(twin.lines[0].get_ydata(), originals["mentions_by_year"]["Policy mentions"])
    np.testing.assert_array_equal(c.lines[0].get_ydata(), originals["coverage_by_year"]["pct_news"])
    np.testing.assert_array_equal(c.lines[1].get_ydata(), originals["coverage_by_year"]["pct_policy"])
    for key, value in originals.items():
        if isinstance(value, pd.Series):
            pd.testing.assert_series_equal(attention_data["altmetric"][key], value)
        else:
            pd.testing.assert_frame_equal(attention_data["altmetric"][key], value)


def test_supplementary_attention_preserves_square_canvas_when_exported(attention_data):
    with patch.object(N, "savefig") as save:
        fig = N.figure_si_altmetric(attention_data)
    save.assert_called_once_with(fig, "04_05_supplementary_figure_04_altmetric", bbox_inches=None)


def test_main_scatter_and_overlap_keep_distinct_spine_styles(attention_data):
    frame = pd.DataFrame([[100., 20.], [40., 100.]],
                         index=["University/HEI", "UK company"],
                         columns=["University/HEI", "UK company"])
    attention_data["collaboration"] = {
        "flag_overlap": frame,
        "flag_totals": pd.Series([200, 100], index=frame.index),
    }
    fig, (e, f) = plt.subplots(1, 2, figsize=(14, 7))
    N.draw_altmetric_scatter(e, attention_data)
    N.draw_collab_flag_overlap(f, attention_data)
    for _ in range(2):
        F.finalize_figure(fig)
        fig.canvas.draw()
    assert_scatter_style(e)
    for spine in f.spines.values():
        assert spine.get_visible()
        assert to_hex(spine.get_edgecolor()) == "#000000"
        assert spine.get_linewidth() == 1
    np.testing.assert_array_equal(f.images[0].get_array(), frame.to_numpy())
    np.testing.assert_allclose(f.images[0].get_cmap()(np.linspace(0, 1, 5)),
                               S.blue_cream_red_colormap()(np.linspace(0, 1, 5)))
