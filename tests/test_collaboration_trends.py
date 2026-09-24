"""Keep both collaboration definitions intact when composing their figures."""
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest

from utils import data_analysis_04_non_academic_collab_helpers as H


def sector_data():
    records = [
        {"Hospital/Clinical": 5},
        {"UK company": 1, "University/HEI": 3},
        {"Hospital/Clinical": 2, "Company (non-UK)": 2},
        {},
        {"Government/Public": 3, "Research institute/Centre": 3},
        {"Other/Unknown": 1},
        {"University/HEI": 1, "UK company": 1},
        {"Hospital/Clinical": 1},
    ]
    df = pd.DataFrame({"year": [2019, 2020, 2020, 2021, 2022, 2022, 2022, 2023],
                       "collaborator_sector_counts": records})
    for label in H.NON_ACADEMIC_SECTOR_LABELS:
        df[H.sector_flag_col(label)] = [int(row.get(label, 0) > 0) for row in records]
    return df


def test_combined_counts_priority_palette_and_legend_are_preserved():
    H.apply_project_plot_style()
    df = sector_data()
    original = df.copy(deep=True)
    with patch.object(H, "show_figures") as show, patch.object(H, "save_figure_file") as save:
        fig = H.plot_collaboration_trends(df, 2020, 2022)
    show.assert_not_called()
    save.assert_not_called()
    fig.canvas.draw()
    a, b = fig.axes
    assert [ax.get_title(loc="left") for ax in (a, b)] == ["A", "B"]
    assert all(ax._left_title.get_size() == 20 for ax in (a, b))
    assert all(ax.yaxis.label.get_size() == 13 for ax in (a, b))
    assert a.get_position().x1 < b.get_position().x0
    assert a.get_position().y0 == pytest.approx(b.get_position().y0)
    assert a.get_position().y1 == pytest.approx(b.get_position().y1)
    assert a.get_xlim()[0] < 2020 and a.get_xlim()[1] > 2022  # whole end markers
    assert a.get_legend() is None and b.get_legend() is None
    lines = {line.get_label(): line for line in a.lines}
    assert set(lines) == set(H.NON_ACADEMIC_SECTOR_LABELS) - {"University/HEI"}
    for label, line in lines.items():
        expected = df.loc[df[H.sector_flag_col(label)].eq(1)].groupby("year").size()
        expected = expected.reindex([2020, 2021, 2022], fill_value=0).cumsum()
        np.testing.assert_array_equal(line.get_xdata(), [2020, 2021, 2022])
        np.testing.assert_array_equal(line.get_ydata(), expected)
        if label == "Other/Unknown":
            assert line.get_color() == "black" and line.get_linestyle() == ":"
            assert line.get_markerfacecolor() == "white"
        else:
            assert to_rgba(line.get_color()) == to_rgba(H.SECTOR_COLORS[label])

    # Highest count wins before the priority rule is used to break a tie.
    expected_primary = {
        "No taxonomy collaborator": [0, 1, 0], "UK company": [0, 0, 1],
        "Company (non-UK)": [1, 0, 0], "Hospital/Clinical": [0, 0, 0],
        "University/HEI": [1, 0, 0], "Government/Public": [0, 0, 1],
        "Research institute/Centre": [0, 0, 0], "Nonprofit/Charity": [0, 0, 0],
        "Other/Unknown": [0, 0, 1],
    }
    stack = []
    for area in b.collections:
        label = area.get_label()
        vertices = area.get_paths()[0].vertices
        heights = [np.ptp(vertices[vertices[:, 0] == year, 1]) for year in (2020, 2021, 2022)]
        np.testing.assert_array_equal(heights, expected_primary[label])
        stack.append(heights)
        expected_color = H.SECTOR_COLORS.get(label, H.palette("cream"))
        np.testing.assert_array_equal(area.get_facecolor()[0], to_rgba(expected_color))
    np.testing.assert_array_equal(np.sum(stack, axis=0), [2, 1, 3])
    assert b.collections[0].get_hatch() == "///"

    legend, = fig.legends
    assert legend._ncols == 3
    assert [text.get_text() for text in legend.get_texts()] == list(expected_primary)
    assert {text.get_size() for text in legend.get_texts()} == {11}
    # Both symbols in a tuple must match the actual line and filled area.
    legend_lines = legend.findobj(Line2D)
    legend_boxes = legend.findobj(Rectangle)
    assert len(legend_lines) == 7 and len(legend_boxes) == 9
    assert {to_rgba(line.get_color()) for line in legend_lines} == {
        to_rgba(line.get_color()) for line in lines.values()}
    assert sorted(box.get_facecolor() for box in legend_boxes) == sorted(
        tuple(area.get_facecolor()[0]) for area in b.collections)
    assert legend_boxes[0].get_hatch() == "///"
    renderer = fig.canvas.get_renderer()
    label_bottom = min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in (a, b))
    gap = (label_bottom - legend.get_window_extent(renderer).y1) / fig.dpi
    assert 0 <= gap < .3
    assert "must not be summed" in fig._ukb_caption
    assert "sum to all eligible publications" in fig._ukb_caption
    pd.testing.assert_frame_equal(df, original)
    plt.close(fig)


def test_embedding_each_original_plot_does_not_publish_or_create_figures():
    H.apply_project_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    existing = plt.get_fignums()
    with patch.object(H, "show_figures") as show, patch.object(H, "save_figure_file") as save:
        for function, ax in zip((H.plot_cumulative_by_type, H.plot_collaboration_mix_stacked_area), axes):
            returned, axis = function(sector_data(), 2020, 2022, ax=ax, legend=False)
            assert returned is fig and axis is ax
    assert plt.get_fignums() == existing
    show.assert_not_called()
    save.assert_not_called()
    plt.close(fig)


def test_combined_figure_requires_taxonomy_and_a_valid_window():
    with pytest.raises(ValueError, match="start year"):
        H.plot_collaboration_trends(sector_data(), 2022, 2020)
    with pytest.raises(ValueError, match="sector-taxonomy flags"):
        H.plot_collaboration_trends(pd.DataFrame({"year": [2020]}))
    with pytest.raises(ValueError, match="sector counts"):
        H.plot_collaboration_trends(sector_data().drop(columns="collaborator_sector_counts"))


def test_notebook_publishes_the_combined_trends_once():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    code = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
    assert code.count("h.plot_collaboration_trends(") == 1
    assert "h.plot_cumulative_by_type(" not in code
    assert "h.plot_collaboration_mix_stacked_area(" not in code
    assert "show_figures(h.COLLABORATION_TRENDS_EXPORT, caption=fig._ukb_caption)" in code
    assert H.COLLABORATION_TRENDS_EXPORT == "collaboration_cumulative_and_annual_mix"


def test_sector_comparison_aligns_counts_and_preserves_palette_and_inputs():
    H.apply_project_plot_style()
    summary = pd.DataFrame({
        "sector": H.NON_ACADEMIC_SECTOR_LABELS,
        "institution_mentions": [100, 40, 10, 30, 8, 5, 2, 0],
        # Deliberately a different ranking: both panels must retain the same rows.
        "papers": [20, 35, 6, 5, 8, 4, 2, 0],
    })
    original = summary.copy(deep=True)
    with patch.object(H, "show_figures") as show, patch.object(H, "save_figure_file") as save:
        fig = H.plot_sector_breakdown_comparison(summary)
    show.assert_not_called()
    save.assert_not_called()
    for _ in range(2):
        H.finalize_figure(fig)
        fig.canvas.draw()
    a, b = fig.axes
    ranked = summary.sort_values("institution_mentions", ascending=False, kind="stable")
    assert [ax.get_title(loc="left") for ax in (a, b)] == ["A", "B"]
    assert [ax.get_xlabel() for ax in (a, b)] == ["Institution mentions", "Publications"]
    assert a.get_shared_y_axes().joined(a, b)
    assert a.get_ylim() == b.get_ylim() and a.yaxis_inverted()
    assert [text.get_text() for text in a.get_yticklabels()] == ranked["sector"].tolist()
    assert not b.get_yticklabels()
    assert a.get_position().y0 == pytest.approx(b.get_position().y0)
    assert a.get_position().y1 == pytest.approx(b.get_position().y1)
    renderer = fig.canvas.get_renderer()
    assert a.get_tightbbox(renderer).x1 < b.get_tightbbox(renderer).x0
    for ax, metric in zip((a, b), ("institution_mentions", "papers")):
        assert ax._left_title.get_size() == 26
        assert ax.xaxis.label.get_size() == 17
        assert all(text.get_size() == 13 for text in ax.get_xticklabels() + ax.get_yticklabels())
        for side, spine in ax.spines.items():
            assert spine.get_visible() == (side in ("left", "bottom"))
            assert spine.get_linewidth() == 1
        assert ax.get_axisbelow() is True
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks():
                assert tick.gridline.get_visible()
                assert tick.gridline.get_linestyle() == "--"
                assert tick.gridline.get_alpha() == .2
                assert tick.gridline.get_linewidth() == .5
            assert not any(tick.gridline.get_visible() for tick in axis.get_minor_ticks())
        np.testing.assert_array_equal([bar.get_width() for bar in ax.patches], ranked[metric])
        assert [text.get_text() for text in ax.texts] == [f"{value:,.0f}" for value in ranked[metric]]
        for bar, text in zip(ax.patches, ax.texts):
            assert text.get_fontsize() == 13
            np.testing.assert_allclose(text.xy, (bar.get_width(), bar.get_y() + bar.get_height() / 2))
            assert text.get_verticalalignment() == "center"
            label_box = text.get_window_extent(renderer)
            assert ax.bbox.contains(*label_box.p0) and ax.bbox.contains(*label_box.p1)
        for bar, sector in zip(ax.patches, ranked["sector"]):
            assert bar.get_facecolor() == to_rgba(H.SECTOR_COLORS[sector])
        box = ax.get_tightbbox(renderer)
        assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1)
    assert "rather than unique organisations" in fig._ukb_caption
    assert "counted once within a sector" in fig._ukb_caption
    assert "not fractionally allocated" in fig._ukb_caption
    pd.testing.assert_frame_equal(summary, original)
    plt.close(fig)


def test_sector_comparison_keeps_zero_count_sectors_visible():
    summary = pd.DataFrame({"sector": H.NON_ACADEMIC_SECTOR_LABELS,
                            "institution_mentions": 0, "papers": 0})
    fig = H.plot_sector_breakdown_comparison(summary)
    fig.canvas.draw()
    for ax in fig.axes:
        assert len(ax.patches) == len(summary)
        assert ax.get_xlim() == (0., 1.)
        assert all(bar.get_width() == 0 for bar in ax.patches)
        assert [text.get_text() for text in ax.texts] == ["0"] * len(summary)
    plt.close(fig)


def test_sector_comparison_handles_empty_or_incomplete_summaries_without_plotting():
    existing = plt.get_fignums()
    assert H.plot_sector_breakdown_comparison(pd.DataFrame()) is None
    with pytest.raises(ValueError, match="missing columns: papers"):
        H.plot_sector_breakdown_comparison(pd.DataFrame({"sector": ["UK company"],
                                                         "institution_mentions": [2]}))
    assert plt.get_fignums() == existing


def test_notebook_publishes_one_combined_sector_breakdown():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    code = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
    assert code.count("h.plot_sector_breakdown_comparison(") == 1
    assert "h.plot_non_academic_sector_breakdown(" not in code
    assert "show_figures(h.SECTOR_BREAKDOWN_EXPORT, caption=fig._ukb_caption)" in code
    assert H.SECTOR_BREAKDOWN_EXPORT == "collaboration_sector_mentions_and_papers"
