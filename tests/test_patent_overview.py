"""The patent overview embeds the complete network without a duplicate export."""
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_hex
from matplotlib.text import Annotation
import networkx as nx
import numpy as np
import pandas as pd

from utils import shared_patent_utils as P
from utils import shared_style as S


def patent_inputs():
    frame = pd.DataFrame({
        "publication_date": ["2020-01-01"] * 6 + ["2021-01-01"] * 6,
        "legal_status": ["Active", "Pending", "Ceased"] * 4,
        "topic_count": [0, 2, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3],
    })
    countries = pd.DataFrame({"iso2": ["US", "GB", "FR"], "count": [8, 6, 2]})
    pivot = pd.DataFrame([[4., 1., 0.], [2., 0., 1.], [0., 1., 1.]],
                         index=["a", "b", "c"], columns=["US", "GB", "FR"])
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 8), ("a", "c", 5), ("b", "c", 6)])
    labels = {"a": "Biomedical and Clinical Sciences", "b": "Chemical Sciences",
              "c": "Information and Computing Sciences"}
    return frame, countries, pivot, graph, labels


def test_supplementary_patent_heatmap_keeps_all_four_spines_after_export_styling():
    from utils import data_analysis_04_non_academic_figures as F
    from utils import data_analysis_04_non_academic_panels as N

    S.load_style("04_non_academic_panels")
    frame = pd.DataFrame([[38., 25., 0.], [23., np.nan, 20.]],
                         index=["Biomedical and Clinical Sciences", "Biological Sciences"],
                         columns=["US", "CN", "GB"])
    original = frame.copy(deep=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    N.draw_patent_country_topics(ax, {"patents": {"country_topic_pct": frame.T}})
    for _ in range(2):
        F.finalize_figure(fig)
        fig.canvas.draw()
        assert all(spine.get_visible() and spine.get_linewidth() == 1
                   and to_hex(spine.get_edgecolor()) == "#000000"
                   for spine in ax.spines.values())
        image = ax.images[0]
        expected_mask = (frame.to_numpy() == 0) | ~np.isfinite(frame.to_numpy())
        np.testing.assert_array_equal(np.ma.getmaskarray(image.get_array()), expected_mask)
        rgba = image.to_rgba(image.get_array())
        np.testing.assert_array_equal(rgba[expected_mask], [[1., 1., 1., 1.]] * expected_mask.sum())
        assert not any(np.all(color == [1., 1., 1., 1.]) for color in rgba[~expected_mask])
        assert {text.get_position() for text in ax.texts} == {(0, 0), (1, 0), (0, 1), (2, 1)}
    np.testing.assert_array_equal(ax.images[0].get_array().data, frame.to_numpy())
    assert [label.get_text() for label in ax.get_xticklabels()] == frame.columns.tolist()
    assert ax.get_xlabel() == "Assignee country"
    pd.testing.assert_frame_equal(frame, original)
    plt.close(fig)


def test_patent_overview_layout_counts_styles_and_inputs_are_preserved():
    S.load_style("04_non_academic_02_patents")
    frame, countries, pivot, graph, labels = patent_inputs()
    original = frame.copy(deep=True)
    original_pivot = pivot.copy(deep=True)
    edges = list(graph.edges(data=True))
    with patch.object(P, "show_figures") as show, patch.object(P, "save_figure_file") as save:
        fig = P.plot_patent_overview(frame, countries, pivot, graph, labels)
    show.assert_not_called()
    save.assert_not_called()
    for _ in range(2):
        P.finalize_figure(fig)
        fig.canvas.draw()
    a, b, c, d, e, colorbar = fig.axes
    assert [ax.get_title(loc="left") for ax in (a, b, c, d, e)] == list("ABCDE")
    for ax in (a, b, c, d, e):
        assert ax._left_title.get_size() == 26
        assert ax.xaxis.label.get_size() == ax.yaxis.label.get_size() == 17
        assert all(text.get_size() == 13 for text in ax.get_xticklabels() + ax.get_yticklabels())
        assert all(text.get_size() == 12 for text in ax.texts)
        assert ax.axison
        assert all(spine.get_linewidth() == 1
                   and to_hex(spine.get_edgecolor()) == "#000000"
                   for spine in ax.spines.values())
    for ax in (a, b, c):
        assert {side: spine.get_visible() for side, spine in ax.spines.items()} == {
            "left": True, "bottom": True, "top": False, "right": False}
    assert not any(spine.get_visible() for spine in d.spines.values())
    assert all(spine.get_visible() for spine in e.spines.values())
    assert not len(d.get_xticks()) and not len(d.get_yticks())
    assert all(text.get_size() == 13 for ax in (a, c) for text in ax.get_legend().get_texts())
    assert [text.get_text() for text in c.get_legend().get_texts()] == ["Mean: 2.25", "Median: 2"]
    # Three distributions above a wider network and heatmap, with no overlap.
    assert d.get_position().width > b.get_position().width
    assert e.get_position().width > c.get_position().width
    assert d.get_position().y1 < a.get_position().y0
    assert e.get_position().x0 > d.get_position().x1
    assert sum(bar.get_height() for bar in a.patches) == len(frame)
    assert [bar.get_height() for bar in b.patches] == [8, 6, 2]
    assert sum(bar.get_height() for bar in c.patches) == len(frame)
    assert c.patches[0].get_height() == 1  # unclassified patents are not dropped
    for ax in (b, c):
        assert {to_hex(bar.get_facecolor()) for bar in ax.patches} == {S.palette("steel_blue").lower()}
    for handle, text in zip(a.get_legend().legend_handles, a.get_legend().get_texts()):
        expected = {"Active": "Active", "Pending": "Application Pending", "Ceased": "Application Ceased"}
        assert to_hex(handle.get_facecolor()).upper() == P.PATENT_STATUS_COLORS[expected[text.get_text()]]

    nodes = next(collection for collection in d.collections if isinstance(collection, PathCollection))
    links = next(collection for collection in d.collections if isinstance(collection, LineCollection))
    assert len(nodes.get_offsets()) == graph.number_of_nodes()
    assert len(links.get_segments()) == graph.number_of_edges()
    np.testing.assert_array_equal(nodes.get_sizes(), [300 + 150 * graph.degree(node) for node in graph])
    callouts = [text for text in d.texts if isinstance(text, Annotation)]
    assert {text.get_text().replace("\n", " ") for text in callouts} == set(labels.values())
    edge_labels = {text.get_text() for text in d.texts if not isinstance(text, Annotation)}
    assert edge_labels == {"8", "5", "6"}
    for text in callouts:
        box = text.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
        assert d.bbox.contains(*box.p0) and d.bbox.contains(*box.p1)

    image = e.images[0]
    expected = pivot.div(pivot.sum(axis=0), axis=1).to_numpy() * 100
    np.testing.assert_allclose(image.get_array().data, expected)
    np.testing.assert_array_equal(image.get_array().mask, expected == 0)
    np.testing.assert_array_equal(image.cmap(image.norm(image.get_array()))[expected == 0], np.ones((3, 4)))
    assert image.get_clim() == (expected[expected > 0].min(), expected.max())
    assert image.cmap.name == "blue_cream_red"
    assert colorbar.yaxis.label.get_size() == 17
    assert all(text.get_size() == 13 for text in colorbar.get_yticklabels())
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        box = ax.get_tightbbox(renderer)
        assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1)
    assert "at least 5 patent records" in fig._ukb_caption
    assert "over the displayed topics" in fig._ukb_caption
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(pivot, original_pivot)
    assert list(graph.edges(data=True)) == edges
    plt.close(fig)


def test_embedded_network_never_creates_another_figure_or_overwrites_caption():
    S.load_style("04_non_academic_02_patents")
    *_, graph, labels = patent_inputs()
    fig, ax = plt.subplots(figsize=(9, 6))
    fig._ukb_caption = "Combined figure caption"
    existing = plt.get_fignums()
    with patch.object(P, "show_figures") as show, patch.object(P, "save_figure_file") as save:
        returned, axis = P.plot_topic_cooccurrence_network(graph, labels, ax=ax, savefile="unused.pdf")
    assert returned is fig and axis is ax
    assert plt.get_fignums() == existing
    assert fig._ukb_caption == "Combined figure caption"
    show.assert_not_called()
    save.assert_not_called()
    plt.close(fig)


def test_overview_handles_no_network_edges_or_country_topics():
    S.load_style("04_non_academic_02_patents")
    frame, countries, _, _, labels = patent_inputs()
    fig = P.plot_patent_overview(frame, countries, pd.DataFrame(), nx.Graph(), labels)
    assert len(fig.axes) == 5
    assert not fig.axes[3].collections
    assert fig.axes[3].axison
    assert not any(spine.get_visible() for spine in fig.axes[3].spines.values())
    plt.close(fig)


def test_notebook_renders_network_once_in_overview_with_stable_export_name():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    code = {cell["id"]: "".join(cell["source"]) for cell in nb["cells"]}
    assert "build_topic_cooccurrence_network(" in code["44aa8574"]
    assert "plot_topic_cooccurrence_network(" not in code["44aa8574"]
    assert "plot_patent_overview(" in code["f1bd2b23"]
    assert "show_figures(patent.PATENT_OVERVIEW_EXPORT, caption=fig._ukb_caption)" in code["f1bd2b23"]
    assert P.PATENT_OVERVIEW_EXPORT == (
        "patent_counts_by_filing_status_and_year_top_countries_by_assignee_occurrences_"
        "distribution_of_topics_per_patent_patent_topics_by_country"
    )


def test_country_topic_summary_preserves_counts_and_uses_requested_styles():
    S.load_style("04_non_academic_02_patents")
    _, _, pivot, _, labels = patent_inputs()
    # A country's dominant field may be outside the displayed top fields.
    topics = pd.DataFrame({
        "country": ["US", "US", "GB", "GB", "FR"],
        "topic_code": ["a", "b", "a", "d", "c"],
        "count": [4., 2., 1., 6.5, 1.],
    })
    labels["d"] = "An additional field outside the heatmap"
    original_pivot, original_topics = pivot.copy(deep=True), topics.copy(deep=True)
    with patch.object(P, "show_figures") as show, patch.object(P, "save_figure_file") as save:
        fig = P.plot_country_topic_summary(pivot, topics, ["US", "GB", "FR"], labels)
    show.assert_not_called()
    save.assert_not_called()
    for _ in range(2):
        P.finalize_figure(fig)
        fig.canvas.draw()
    heatmap, bars, colorbar = fig.axes
    assert [ax.get_title(loc="left") for ax in (heatmap, bars)] == list("AB")
    assert heatmap.get_subplotspec().rowspan.start == bars.get_subplotspec().rowspan.start == 0
    assert heatmap.get_subplotspec().colspan.start == 0
    assert bars.get_subplotspec().colspan.start == 1
    for ax in (heatmap, bars):
        assert ax.xaxis.label.get_size() == ax.yaxis.label.get_size() == 17
        assert all(label.get_size() == 13 for label in ax.get_xticklabels() + ax.get_yticklabels())
        assert all(text.get_size() == 12 for text in ax.texts)
    assert all(spine.get_visible() and spine.get_linewidth() == 1
               and to_hex(spine.get_edgecolor()) == "#000000"
               for spine in heatmap.spines.values())
    assert len(heatmap._ukb_cell_edge_artists) == 2
    expected = pivot.div(pivot.sum(axis=0), axis=1).to_numpy() * 100
    image = heatmap.images[0]
    np.testing.assert_allclose(image.get_array().data, expected)
    np.testing.assert_array_equal(image.get_array().mask, expected == 0)
    np.testing.assert_array_equal(image.cmap(image.norm(image.get_array()))[expected == 0],
                                  np.ones((3, 4)))
    assert image.cmap.name == "blue_cream_red"
    assert {to_hex(bar.get_facecolor()) for bar in bars.patches} == {S.palette("steel_blue").lower()}
    assert [bar.get_width() for bar in bars.patches] == [1., 4., 6.5]
    assert [text.get_text() for text in bars.texts] == ["1.00", "4.00", "6.50"]
    assert "UK: An additional field" in bars.get_yticklabels()[-1].get_text()
    assert [label.get_text() for label in heatmap.get_xticklabels()] == ["US", "UK", "FR"]
    assert colorbar.yaxis.label.get_size() == 15
    renderer = fig.canvas.get_renderer()
    assert heatmap.get_tightbbox(renderer).x1 < bars.get_tightbbox(renderer).x0
    assert colorbar.get_tightbbox(renderer).x1 < bars.get_tightbbox(renderer).x0
    for ax in fig.axes:
        box = ax.get_tightbbox(renderer)
        assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1)
    assert "over the displayed topics" in fig._ukb_caption
    assert "using all recorded topics" in fig._ukb_caption
    assert "zero cells are white" in fig._ukb_caption
    pd.testing.assert_frame_equal(pivot, original_pivot)
    pd.testing.assert_frame_equal(topics, original_topics)
    plt.close(fig)


def test_embedded_dominant_topics_does_not_display_or_export():
    S.load_style("04_non_academic_02_patents")
    topics = pd.DataFrame({"country": ["GB"], "topic_code": ["a"], "count": [2.]})
    fig, ax = plt.subplots()
    existing = plt.get_fignums()
    with patch.object(P, "show_figures") as show, patch.object(P, "save_figure_file") as save:
        returned, axis = P.plot_country_dominant_topics(topics, ["GB"], ax=ax, savefile="unused.pdf")
    assert returned is fig and axis is ax
    assert plt.get_fignums() == existing
    show.assert_not_called()
    save.assert_not_called()
    plt.close(fig)


def test_country_topic_status_summary_combines_three_panels_without_changing_data():
    S.load_style("04_non_academic_02_patents")
    frame, _, pivot, _, labels = patent_inputs()
    original = frame.copy(deep=True)
    original_pivot = pivot.copy(deep=True)
    topics = pivot.rename_axis("topic_code").reset_index().melt(
        id_vars="topic_code", var_name="country", value_name="count")
    original_topics = topics.copy(deep=True)
    with patch.object(P, "show_figures") as show, patch.object(P, "save_figure_file") as save:
        fig = P.plot_country_topic_summary(pivot, topics, list(pivot.columns), labels,
                                           df_patent=frame)
    show.assert_not_called()
    save.assert_not_called()
    for _ in range(2):
        P.finalize_figure(fig)
        fig.canvas.draw()
    heat, bars, status, colorbar = fig.axes
    assert [ax.get_title(loc="left") for ax in (heat, bars, status)] == list("ABC")
    assert list(status.get_subplotspec().rowspan) == [1]
    assert list(status.get_subplotspec().colspan) == [0, 1]
    assert all(spine.get_visible() for spine in heat.spines.values())
    expected = pivot.div(pivot.sum(axis=0), axis=1).to_numpy() * 100
    np.testing.assert_allclose(heat.images[0].get_array().data, expected)
    np.testing.assert_array_equal(heat.images[0].get_array().mask, expected == 0)
    np.testing.assert_array_equal([bar.get_width() for bar in bars.patches], [1., 1., 4.])
    np.testing.assert_array_equal(
        np.sum([[bar.get_height() for bar in group] for group in status.containers], axis=0),
        [6, 6])
    legend = status.get_legend()
    for group, handle, label in zip(status.containers, legend.legend_handles, legend.get_texts()):
        color = P.PATENT_STATUS_COLORS[label.get_text()]
        assert to_hex(handle.get_facecolor()).upper() == color
        assert all(to_hex(bar.get_facecolor()).upper() == color for bar in group)
        assert color != S.palette("green")
        assert label.get_fontsize() == 12
    assert status.get_xlabel() == "Publication year"
    assert status.get_ylabel() == "Number of patents"
    assert "(C) Patent records by publication year" in fig._ukb_caption
    assert "recorded at extraction" in fig._ukb_caption
    renderer = fig.canvas.get_renderer()
    assert status.get_tightbbox(renderer).y1 < min(
        heat.get_tightbbox(renderer).y0, bars.get_tightbbox(renderer).y0)
    assert colorbar.get_tightbbox(renderer).x1 < bars.get_tightbbox(renderer).x0
    for ax in fig.axes:
        box = ax.get_tightbbox(renderer)
        assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(pivot, original_pivot)
    pd.testing.assert_frame_equal(topics, original_topics)
    plt.close(fig)


def test_country_topic_status_summary_retains_status_when_topic_data_is_empty():
    S.load_style("04_non_academic_02_patents")
    frame, *_ = patent_inputs()
    fig = P.plot_country_topic_summary(pd.DataFrame(), pd.DataFrame(), [], df_patent=frame)
    assert len(fig.axes) == 3
    assert sum(bar.get_height() for bar in fig.axes[2].patches) == len(frame)
    assert "No country-topic data" in [text.get_text() for text in fig.axes[0].texts]
    plt.close(fig)


def test_country_topic_summary_handles_empty_input_without_creating_a_figure():
    existing = plt.get_fignums()
    assert P.plot_country_topic_summary(pd.DataFrame(), pd.DataFrame(), []) is None
    assert plt.get_fignums() == existing


def test_notebook_displays_country_topic_summary_once():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    cell = next(cell for cell in nb["cells"] if cell["id"] == "a979d05e")
    source = "".join(cell["source"])
    assert "plot_country_topic_summary(" in source
    assert "plot_country_topic_heatmap(" not in source
    assert "plot_country_dominant_topics(" not in source
    assert source.count("show_figures(") == 1
    assert "show_figures(patent.PATENT_COUNTRY_TOPICS_EXPORT, caption=fig._ukb_caption)" in source
    assert "df_patent=df_with_iso" in source
    assert P.PATENT_COUNTRY_TOPICS_EXPORT == "patents_country_topics_dominant_topics_and_legal_status"
    duplicate = next(cell for cell in nb["cells"] if cell["id"] == "073c11ce")
    assert "show_figures" not in "".join(duplicate["source"])
    assert "plot_filing_status_over_time" not in "".join(duplicate["source"])
    assert not duplicate.get("outputs", [])
