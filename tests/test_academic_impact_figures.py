"""Style, denominator and display contracts for the academic-impact figures."""

from itertools import combinations
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import pytest

from utils import data_analysis_03_academic_impact_figures as figures
from utils import data_analysis_03_academic_impact_panels as panels
from utils import shared_style as style
from utils.shared_paths import ArtifactRegistry


@pytest.fixture(autouse=True)
def academic_style():
    style.load_style("03_academic_impact")
    yield
    plt.close("all")


@pytest.fixture
def papers():
    return pd.DataFrame({
        "id": [str(i) for i in range(120)],
        "year": np.tile(np.arange(2013, 2025), 10),
        "times_cited": np.arange(120),
        "citations_per_year": np.arange(120) / 3,
        "field_citation_ratio": np.arange(120) / 10,
        "relative_citation_ratio": np.arange(120) / 12,
    })


def test_snapshot_stock_and_concentration_preserve_denominators(papers):
    fig = figures.cumulative_citation_stock(papers)
    data = fig._ukb_tables["fig02_cumulative_output_and_citation_stock.csv"]
    assert data.cumulative_papers.iloc[-1] == len(papers)
    assert data.cumulative_citations.iloc[-1] == papers.times_cited.sum()
    assert data.cumulative_citation_share.iloc[-1] == pytest.approx(100)
    assert "not a historical time series" in fig._ukb_caption
    fig = figures.citation_concentration(papers)
    checkpoints = fig._ukb_tables["fig05_citation_concentration_checkpoints.csv"]
    ordered = papers.times_cited.sort_values(ascending=False)
    for row in checkpoints.itertuples():
        n = int(np.ceil(len(papers) * row.top_paper_percent / 100))
        assert row.citation_percent == pytest.approx(100 * ordered.head(n).sum() / ordered.sum())


@pytest.mark.parametrize("figure_function,legend_fontsize", [
    (figures.publication_contribution, 14), (figures.activity_over_time, 14),
])
def test_field_legend_sits_close_below_axis_labels_with_equal_columns(figure_function, legend_fontsize):
    labels = ["Epidemiology", "Clinical Sciences", "Genetics", "Biological Psychology",
              "Cardiovascular Medicine and Haematology", "Public Health",
              "Oncology and Carcinogenesis", "Health Services and Systems"]
    counts = pd.DataFrame(np.arange(104).reshape(13, 8),
                          index=range(2013, 2026), columns=labels)
    context = {"TOP_LABELS": labels, "FIELD_COLORS": dict(zip(labels, panels.field_palette(8))),
               "ukbb_ts": counts, "share_ts": counts / 100}
    codes = [str(i) for i in range(8)]
    activity = pd.Series(np.geomspace(.2, 12, len(codes)), index=codes)
    publications = pd.DataFrame([
        {"year": year, "code": code, "for_label": label, "n_papers": 30}
        for year in range(2015, 2026) for code, label in zip(codes, labels)
    ])
    context.update({"TOP_CODES": codes, "ACTIVITY": activity,
                    "ACTIVITY_LABEL": "within its own L2 division",
                    "VALUE": "n_papers", "FRAC_COL": "n_papers",
                    "ukbb": publications, "whole": publications.copy(),
                    "activity_index": lambda window: activity})
    fig = figure_function(context)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bottom = min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in fig.axes)
    legend_bounds = fig.legends[0].get_window_extent(renderer)
    assert all(text.get_fontsize() == legend_fontsize for text in fig.legends[0].get_texts())
    assert fig.bbox.x0 < legend_bounds.x0 < legend_bounds.x1 < fig.bbox.x1
    legend_top = legend_bounds.y1
    gap_points = (label_bottom - legend_top) * 72 / fig.dpi
    assert 6 <= gap_points <= 18
    columns = fig.legends[0]._legend_handle_box.get_children()
    heights = [column.get_window_extent(renderer).height for column in columns]
    assert max(heights) - min(heights) < .01
    for row_index in range(2):
        positions = [column.get_children()[row_index].get_window_extent(renderer).y0
                     for column in columns]
        assert max(positions) - min(positions) < .01


def test_citation_measure_legends_are_inside_requested_corners():
    codes = [str(i) for i in range(8)]
    labels = [f"Field {i}" for i in range(8)]
    metrics = ["n_top10f", "n_top50f", "n_mncs"]
    impact = pd.DataFrame({f"rci_{metric[2:]}": np.linspace(1, 4, 8)
                           for metric in metrics}, index=codes)
    annual = pd.DataFrame({code: [3, 2, 1] for code in codes}, index=[2016, 2020, 2025])
    context = {"CITE_HEADLINE": "n_mncs", "IMPACT": impact, "TOP_CODES": codes,
               "TOP_LABELS": labels, "RCI_WEIGHTS": metrics,
               "FIELD_COLORS": dict(zip(labels, panels.field_palette(8))),
               "OVERALL": {metric[2:]: 2 for metric in metrics},
               "label_of": dict(zip(codes, labels)).get,
               "rci_timeseries": lambda metric, minimum: annual}
    fig = figures.citation_measure_comparison(context)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, location in zip(fig.axes, [4, 1]):  # Lower right, then upper right.
        legend = ax.get_legend()
        assert legend._loc == location
        bounds = legend.get_window_extent(renderer)
        plot = ax.get_window_extent(renderer)
        assert bounds.x0 > plot.x0
        assert plot.y0 < bounds.y0 < bounds.y1 < plot.y1
        assert 0 < (plot.x1 - bounds.x1) * 72 / fig.dpi < 12
        vertical_gap = bounds.y0 - plot.y0 if location == 4 else plot.y1 - bounds.y1
        assert 0 < vertical_gap * 72 / fig.dpi < 12
    assert not fig.legends
    assert fig.subplotpars.bottom < .15


def test_top_decile_pool_uses_white_bars_and_large_close_legend():
    codes = [str(i) for i in range(8)]
    labels = [f"Field {i}" for i in range(8)]
    whole = pd.DataFrame([
        {"year": year, "code": code, "n_top10f": 100, "n_top10f_docs": 100}
        for year in [2015, 2020, 2025] for code in codes
    ])
    ukbb = whole.assign(n_top10f=5, n_top10f_docs=5)
    context = {"TOP10_COL": "n_top10f", "TOP_CODES": codes, "TOP_LABELS": labels,
               "FIELD_COLORS": dict(zip(labels, panels.field_palette(8))),
               "whole": whole, "ukbb": ukbb}
    fig = figures.top_decile_pool(context)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [ax for ax in fig.axes if ax.containers]
    assert len(axes) == 8
    for ax in axes:
        for bar in ax.containers[0]:
            assert bar.get_facecolor() == (1, 1, 1, 1)
            assert bar.get_edgecolor() == (0, 0, 0, 1)
            assert bar.get_height() == 95
            assert bar.get_y() == 5
    legend = fig.legends[0]
    assert all(text.get_fontsize() == 14 for text in legend.get_texts())
    assert legend.legend_handles[0].get_facecolor() == (1, 1, 1, 1)
    assert to_hex(legend.legend_handles[1].get_facecolor()) == to_hex(style.palette("steel_blue"))
    for ax in axes:
        for container, handle in zip(ax.containers, legend.legend_handles[:2]):
            for bar in container:
                assert bar.get_facecolor() == handle.get_facecolor()
                assert bar.get_edgecolor() == handle.get_edgecolor()
    share_axes = [ax for ax in fig.axes if ax not in axes]
    assert len(share_axes) == 8
    line_handle = legend.legend_handles[2]
    assert to_hex(line_handle.get_color()) == to_hex(style.palette("red"))
    for ax in share_axes:
        line, = ax.lines
        assert line.get_color() == line_handle.get_color()
        assert line.get_marker() == line_handle.get_marker()
        assert line.get_linestyle() == line_handle.get_linestyle()
        assert line.get_markeredgecolor() == line_handle.get_markeredgecolor()
        np.testing.assert_allclose(line.get_ydata(), 5)
    legend_bounds = legend.get_window_extent(renderer)
    assert fig.bbox.x0 < legend_bounds.x0 < legend_bounds.x1 < fig.bbox.x1
    label_bottom = min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in axes)
    assert 6 <= (label_bottom - legend_bounds.y1) * 72 / fig.dpi <= 18
    summary = fig._ukb_tables["top_decile_pool_summary.csv"]
    assert (summary.ukbb_share_pct == 5).all()


@pytest.fixture
def main_impact_context():
    labels = ["Epidemiology", "Clinical Sciences", "Genetics", "Biological Psychology",
              "Cardiovascular Medicine and Haematology", "Public Health",
              "Oncology and Carcinogenesis", "Health Services and Systems", "Statistics"]
    codes = [str(i) for i in range(len(labels))]
    years = np.arange(2015, 2026)
    shares = pd.DataFrame(np.linspace(.1, 1, len(years))[:, None]
                          * np.array([8, .2, 3, 1.5, 1.2, 1, .1, .08, .06]),
                          index=years, columns=codes)
    whole = pd.DataFrame([
        {"year": year, "code": code, "for_label": label, "n_papers": 1000,
         "n_top10f": 1000, "n_top10f_docs": 1000, "n_top50f": 1000, "n_top50f_docs": 1000}
        for year in years for code, label in zip(codes, labels)
    ])
    ukbb = whole.assign(n_papers=30, n_top10f=4.1, n_top10f_docs=30,
                        n_top50f=15, n_top50f_docs=30)
    activity = pd.Series(np.geomspace(.3, 12, len(codes)), index=codes)
    impact = pd.DataFrame({"ukbb_papers": np.arange(900, 0, -100), "activity_x": activity,
                           "rci_mncs": np.linspace(1.4, 4.2, len(codes))}, index=codes)

    def share_timeseries(metric, codes=None, denominator=None):
        factor = .2 if metric == "n_papers" else .08 if metric == "n_top50f" else .03
        frame = shares if denominator is None and metric == "n_top10f" else shares * factor
        return frame if codes is None else frame[codes]

    return {"LEVEL": "L4", "TOP_CODES": codes[:8], "TOP_LABELS": labels[:8],
            "FIELD_COLORS": dict(zip(labels[:8], panels.field_palette(8))),
            "CITE_HEADLINE": "n_mncs", "TOP10_COL": "n_top10f", "TOP50_COL": "n_top50f",
            "CITE_OK": True, "MEASURES": ["n_papers", "n_top10f", "n_top50f", "n_mncs"],
            "CITE_YEARS": {"n_top10f": years}, "CITE_WIN": (2015, 2025),
            "ukbb": ukbb, "whole": whole, "IMPACT": impact, "ACTIVITY": activity,
            "VALUE": "n_papers", "ACTIVITY_LABEL": "within its own L2 division",
            "OVERALL": {"mncs": 2.5}, "label_of": dict(zip(codes, labels)).get,
            "share_timeseries": share_timeseries}


def test_main_impact_uses_palette_and_keeps_enlarged_legends_clear(main_impact_context):
    style.load_style("03_academic_impact_panels")
    fig = panels.figure_main(main_impact_context, save=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    c = fig.axes[2]
    legend = c.get_legend()
    assert not c.texts  # The pooled-share value belongs in the key, not over data.
    assert "UK Biobank overall (0.41%)" in [t.get_text().replace("\n", " ")
                                           for t in legend.get_texts()]
    for key in (legend, fig.legends[0]):
        assert all(t.get_fontsize() == 14 for t in key.get_texts())
        bounds = key.get_window_extent(renderer)
        assert fig.bbox.x0 < bounds.x0 < bounds.x1 < fig.bbox.x1
    bounds = legend.get_window_extent(renderer)
    label_bottom = c.xaxis.label.get_window_extent(renderer).y0
    assert 6 <= (label_bottom - bounds.y1) * 72 / fig.dpi <= 12
    assert bounds.y0 > max(ax.get_tightbbox(renderer).y1 for ax in fig.axes[3:7]) + 10
    assert c._left_title.get_window_extent(renderer).y1 < fig.axes[1].get_tightbbox(renderer).y0
    columns = legend._legend_handle_box.get_children()
    boxes = [column.get_window_extent(renderer) for column in columns]
    assert boxes[0].x1 + 10 < boxes[1].x0
    assert boxes[0].height == pytest.approx(boxes[1].height)
    for text, handle in zip(legend.get_texts()[:-1], legend.legend_handles[:-1]):
        label = text.get_text().replace("\n", " ")
        assert handle.get_color() == main_impact_context["FIELD_COLORS"][label]
        assert any(line.get_color() == handle.get_color()
                   and line.get_marker() == handle.get_marker() for line in c.lines)
    for ax in fig.axes:
        for artist in ax.collections:
            assert all(to_hex(color) in {to_hex(c) for c in style.PALETTE}
                       for color in artist.get_facecolors())
    patches = fig.legends[0].findobj(Rectangle)
    for alpha in (.85, .18):
        colors = {to_hex(patch.get_facecolor()) for patch in patches if patch.get_alpha() == alpha}
        assert colors == {to_hex(color) for color in main_impact_context["FIELD_COLORS"].values()}


def test_survival_includes_measured_zeros_and_caption_names_fallback(papers):
    papers["field_citation_ratio"] = np.nan
    fig = figures.citation_survival(papers)
    assert "relative citation ratio" in fig._ukb_caption
    table = fig._ukb_tables["fig07_normalised_citation_thresholds.csv"]
    assert table.percent_at_or_above.iloc[0] == pytest.approx(
        100 * papers.relative_citation_ratio.ge(1).mean())


def test_combined_concentration_and_cohorts_preserves_both_analyses(papers):
    concentration = figures.citation_concentration(papers)
    cohorts = figures.citation_cohorts(papers, 2025, 2013)
    combined = figures.citation_concentration_and_cohorts(papers, 2025, 2013)
    expected_tables = {**concentration._ukb_tables, **cohorts._ukb_tables}
    assert combined._ukb_tables.keys() == expected_tables.keys()
    for name, frame in expected_tables.items():
        pd.testing.assert_frame_equal(combined._ukb_tables[name], frame)
    assert len(combined.axes) == 2
    left, right = combined.axes
    assert left.get_title(loc="left") == "A"
    assert right.get_title(loc="left") == "B"
    assert left.get_position().y0 == pytest.approx(right.get_position().y0)
    assert left.get_position().y1 == pytest.approx(right.get_position().y1)
    assert left.get_position().x1 < right.get_position().x0
    np.testing.assert_array_equal(left.lines[0].get_xydata(), concentration.axes[0].lines[0].get_xydata())
    assert [t.get_text() for t in right.get_xticklabels()] == [
        t.get_text() for t in cohorts.axes[0].get_xticklabels()]
    assert combined._ukb_caption == f"(A) {concentration._ukb_caption} (B) {cohorts._ukb_caption}"


def test_citation_overview_preserves_all_four_analyses_in_a_two_by_two_grid(papers):
    codes = np.repeat(["1000", "2000", "3000"], [60, 40, 20])
    long = pd.DataFrame({
        "id": papers.id, "for_code": codes, "for_name": [f"Field {code}" for code in codes],
        "fractional_papers": 1., "fractional_citations": papers.times_cited,
    })
    original = [figures.disciplinary_composition(papers, long, 2),
                figures.cumulative_citation_stock(papers),
                figures.field_citation_stock(long, 2), figures.citation_survival(papers)]
    combined = figures.citation_overview(papers, long, 2)
    expected_tables = {name: table for fig in original for name, table in fig._ukb_tables.items()}
    assert combined._ukb_tables.keys() == expected_tables.keys()
    for name, table in expected_tables.items():
        pd.testing.assert_frame_equal(combined._ukb_tables[name], table)
    assert len(combined.axes) == 5  # Four panels plus C's dedicated colourbar.
    a, b, c, d, colorbar = combined.axes
    for ax, letter, separate in zip([a, b, c, d], "ABCD", original):
        assert ax.get_title(loc="left") == letter
        assert ax._left_title.get_fontsize() == 20
        assert ax.get_xlabel() == separate.axes[0].get_xlabel()
        assert ax.get_ylabel() == separate.axes[0].get_ylabel()
        assert f"({letter}) {separate._ukb_caption}" in combined._ukb_caption
    for row in [(a, b), (c, d)]:
        assert row[0].get_position().y0 == pytest.approx(row[1].get_position().y0)
        assert row[0].get_position().y1 == pytest.approx(row[1].get_position().y1)
    assert a.get_position().x0 == pytest.approx(c.get_position().x0)
    assert a.get_position().width == pytest.approx(c.get_position().width)
    assert c.get_position().x1 < colorbar.get_position().x0 < colorbar.get_position().x1 < d.get_position().x0
    for index in (0, 2):
        np.testing.assert_allclose([bar.get_width() for bar in combined.axes[index].patches],
                                   [bar.get_width() for bar in original[index].axes[0].patches])
    for index in (1, 3):
        for line, before in zip(combined.axes[index].lines, original[index].axes[0].lines):
            np.testing.assert_array_equal(line.get_xydata(), before.get_xydata())
    assert colorbar._colorbar.cmap.name == style.blue_cream_red_colormap().name


def test_academic_styles_and_field_cycles_use_the_shared_palette():
    for name in ("03_academic_impact", "03_academic_impact_panels"):
        assert style.load_style(name)["colors"] == style.PALETTE
    assert set(panels.field_palette(25)) == set(style.PALETTE)
    labels = [f"Field {i}" for i in range(25)]
    context = {"FIELD_COLORS": dict(zip(labels[:8], panels.field_palette(8))),
               "TOP_LABELS": labels[:8]}
    styles = panels.field_line_styles(context, labels)
    assert set(s["color"] for s in styles.values()) <= set(style.PALETTE)
    assert len({(s["color"], s["marker"], s["linestyle"]) for s in styles.values()}) == len(labels)
    for label, handle in zip(context["TOP_LABELS"], figures._field_handles(context)):
        assert handle.get_color() == styles[label]["color"]
        assert handle.get_marker() == styles[label]["marker"]
        assert handle.get_linestyle() == styles[label]["linestyle"]


@pytest.mark.parametrize("figure_function,panel", [
    (figures.citation_cohorts, 0), (figures.citation_concentration_and_cohorts, 1),
])
def test_cohort_violins_use_distinct_project_palette_colours(papers, figure_function, panel):
    fig = figure_function(papers)
    bodies = fig.axes[panel].collections
    colors = [to_hex(body.get_facecolor()[0]) for body in bodies]
    assert colors == [to_hex(color) for color in style.palette("red", "cream", "steel_blue", "green")]
    assert all(body.get_alpha() == 1 for body in bodies)
    assert all(to_hex(body.get_edgecolor()[0]) == "#000000" for body in bodies)
    assert all(to_hex(box.get_facecolor()) == "#ffffff" for box in fig.axes[panel].patches)


def test_field_credit_totals_are_not_changed(papers):
    long = pd.DataFrame({
        "id": ["0", "0", "1"], "for_code": ["1000", "2000", "1000"],
        "for_name": ["Field one", "Field two", "Field one"],
        "fractional_papers": [.5, .5, 1], "fractional_citations": [5, 5, 20],
    })
    fig = figures.disciplinary_composition(papers, long)
    table = fig._ukb_tables["fig01_disciplinary_composition.csv"]
    assert table.fractional_paper_count.sum() == 2
    assert table.percent_of_corpus.sum() == pytest.approx(200 / len(papers))
    fig = figures.field_citation_stock(long)
    table = fig._ukb_tables["fig04_field_citation_impact.csv"]
    assert table.fractional_citations.sum() == 30
    cb = fig.axes[-1]._colorbar
    assert (cb.vmin, cb.vmax) == (.5, 1.5)


def test_diagnostic_figures_have_letters_and_helvetica(papers):
    for function in (figures.cumulative_citation_stock, figures.citation_concentration,
                     figures.citation_cohorts, figures.citation_survival):
        fig = function(papers)
        assert fig.axes[0].get_title(loc="left") == "A"
        assert fig.axes[0]._left_title.get_fontsize() == 20
        assert fig.axes[0].xaxis.label.get_fontfamily() == ["Helvetica"]
        assert fig._ukb_caption
        assert not fig.texts  # No prose footnotes or titles inside the artwork.
        for ax in fig.axes:
            assert not ax.get_title()
            for line in ax.get_xgridlines() + ax.get_ygridlines():
                if line.get_visible():
                    assert line.get_linestyle() == "--"


def test_publishing_registers_tables_and_displays_after_save(tmp_path):
    fig, ax = plt.subplots()
    fig._ukb_caption = "A specific caption."
    fig._ukb_tables = {"values.csv": pd.DataFrame({"x": [1]})}
    registry = ArtifactRegistry(tmp_path)
    events = []
    with patch.object(style, "savefig", side_effect=lambda *a, **k: events.append("save") or []) as save:
        with patch.object(style, "display_figure", side_effect=lambda *a: events.append("display")):
            figures.publish(fig, "example", registry, subdirectory="citation_diagnostics")
    assert events == ["save", "display"]
    assert save.call_args.kwargs["style"]["savedir"].name == "citation_diagnostics"
    assert registry.table_paths == [tmp_path / "values.csv"]
    assert not plt.fignum_exists(fig.number)


def test_callouts_use_curves_without_covering_points_or_each_other():
    fig, ax = plt.subplots(figsize=(10, 7))
    points = np.array([[1, 1], [1.05, 1.1], [2, 1.8], [.5, 2.1]])
    ax.set(xscale="log", yscale="log", xlim=(.1, 5), ylim=(.3, 5))
    ax.scatter(points[:, 0], points[:, 1], s=90)
    labels = style.scatter_callouts(ax, points, ["Field one", "Field two", "Field three", "Field four"])
    fig.canvas.draw()
    boxes = [t.get_bbox_patch().get_window_extent() for t in labels]
    for a, b in combinations(boxes, 2):
        assert not a.overlaps(b)
    for box in boxes:
        for point in ax.transData.transform(points):
            assert not box.contains(*point)
    assert all(t.arrow_patch.get_connectionstyle().rad != 0 for t in labels)


def test_panel_assembly_reuses_context_without_loading_field_arms():
    context = {"LEVEL": "L4", "CUTS": None, "ukbb": pd.DataFrame({"year": [2025], "n": [1]}),
               "whole": pd.DataFrame({"year": [2025], "n": [10]}),
               "ACTIVITY": None, "CODE_LABEL": {}, "FRAC_COL": "n"}
    with patch.object(panels.AI, "build", side_effect=AssertionError("Duplicate build")):
        with patch.object(panels, "_author_arm", return_value={}):
            with patch.object(panels, "_growth_table", return_value=(None, None, None, 1)):
                with patch.object(panels, "_cuts_table", side_effect=AssertionError("Duplicate cut-offs")):
                    result = panels.build_panel_data(verbose=False, context=context)
    assert result["ukbb_overall_share"] == 10
    assert "author" not in context
