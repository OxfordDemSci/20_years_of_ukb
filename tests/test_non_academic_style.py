"""Presentation contracts shared by the non-academic figure family."""
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
import numpy as np
import pandas as pd
import pytest

from utils import shared_style as S
from utils import data_analysis_04_non_academic_figures as F


def test_all_sections_inherit_manuscript_typography_and_palette():
    for section in ("panels", "01_clinical_trials", "02_patents", "03_altmetric", "04_collaboration"):
        style = S.load_style("04_non_academic_" + section)
        assert style["colors"] == S.NON_ACADEMIC_PALETTE
        assert [style[k] for k in ("panel_label_fs", "label_fs", "tick_fs", "legend_fs")] == [20, 13, 11, 11]
        assert style["formats"] == ["pdf", "png"]


def test_heatmaps_keep_directionality_observed_limits_and_cell_edges():
    fig, axes = plt.subplots(1, 2)
    values = np.array([[100., 20.], [45., 100.]])
    norm = Normalize(0, 100)
    images = [ax.imshow(values - i * 5, norm=norm) for i, ax in enumerate(axes)]
    cb = fig.colorbar(images[-1], ax=axes)
    cb.set_label("Publications (%)", fontsize=6)
    for ax in axes:
        for (i, j), v in np.ndenumerate(values):
            ax.text(j, i, str(v), ha="center")
    F.finalize_figure(fig)
    F.finalize_figure(fig)
    assert all(im.get_clim() == (15., 100.) for im in images)
    np.testing.assert_array_equal(images[0].get_array(), values)
    assert all(len(ax.collections) == 2 for ax in axes)  # idempotent cell borders
    assert cb.ax.yaxis.label.get_size() == 13
    assert cb.get_ticks()[0] == 15 and cb.get_ticks()[-1] == 100
    assert images[0].cmap.name == "blue_cream_red"
    plt.close(fig)


def test_titles_twins_and_legend_fonts():
    fig, axes = plt.subplots(1, 2)
    S.set_title(axes[0], "A Descriptive title")
    twin = axes[1].twinx()
    axes[0].plot([1, 2], label="Two\nlines")
    axes[0].legend(fontsize=6)
    F.finalize_figure(fig)
    assert [ax.get_title(loc="left") for ax in axes] == ["A", "B"]
    assert twin.get_title(loc="left") == ""
    assert all(ax._left_title.get_fontsize() == 20 for ax in axes)
    assert axes[0].get_legend().get_texts()[0].get_fontsize() == 11
    plt.close(fig)


def test_heatmap_border_opt_out_and_font_overrides_survive_finalization():
    fig, axes = plt.subplots(1, 2)
    values = np.ma.masked_equal([[0., 4.], [8., 16.]], 0)
    images = [ax.imshow(values) for ax in axes]
    colorbar = fig.colorbar(images[1], ax=axes[1])
    colorbar.set_label("Contributions")
    for ax in axes:
        ax.text(1, 0, "4", ha="center")
    F.finalize_figure(fig)
    assert [len(ax.collections) for ax in axes] == [2, 2]
    target = axes[1]
    target._ukb_heatmap_edges = False
    target._ukb_tick_fs = 13
    target._ukb_annotation_fs = 12
    colorbar.ax._ukb_tick_fs = 13
    colorbar.ax._ukb_label_fs = 15
    for _ in range(3):
        F.finalize_figure(fig)
        S.finalize_figure(fig)  # the shared file exporter also finalizes
    fig.canvas.draw()
    assert [len(ax.collections) for ax in axes] == [2, 0]
    assert axes[0].texts[0].get_fontsize() == 10
    assert target.texts[0].get_fontsize() == 12
    assert {text.get_size() for text in target.get_xticklabels()} == {13}
    assert {text.get_size() for text in colorbar.ax.get_yticklabels()} == {13}
    assert colorbar.ax.yaxis.label.get_size() == 15
    np.testing.assert_array_equal(images[1].get_array().data, values.data)
    np.testing.assert_array_equal(images[1].get_array().mask, values.mask)
    assert images[1].get_clim() == (4., 16.)
    np.testing.assert_array_equal(images[1].cmap(images[1].norm(values))[0, 0], [1, 1, 1, 1])
    # Re-enabling borders should add them once, without affecting another panel.
    target._ukb_heatmap_edges = True
    F.finalize_figure(fig)
    F.finalize_figure(fig)
    assert [len(ax.collections) for ax in axes] == [2, 2]
    plt.close(fig)


def test_retained_company_heatmap_has_spines_ticks_white_zeros_and_no_cell_borders():
    from utils import data_analysis_04_non_academic_collab_helpers as H
    H.apply_project_plot_style()
    company_year = pd.DataFrame({"org": [f"Company {i} Biomedical Research" for i in range(15)],
                                2020: np.arange(15), 2021: np.arange(15) + 1,
                                2022: np.arange(15) + 2})
    original = company_year.copy(deep=True)
    with patch.object(H, "show_figures") as show:
        H.plot_top_company_heatmap(company_year)
    show.assert_called_once()
    assert show.call_args.args == ('collaboration_top_company_heatmap',)
    assert "Zero cells are white" in show.call_args.kwargs["caption"]
    fig = plt.gcf()
    for _ in range(2):
        F.finalize_figure(fig)
        S.finalize_figure(fig)
    fig.canvas.draw()
    heatmap, colorbar = fig.axes
    assert heatmap.get_title(loc="left") == "A"
    assert heatmap._left_title.get_size() == 26
    assert not heatmap.collections
    assert all(spine.get_visible() and spine.get_linewidth() == 1
               and to_hex(spine.get_edgecolor()) == "#000000"
               for spine in heatmap.spines.values())
    for axis in (heatmap.xaxis, heatmap.yaxis):
        assert axis.label.get_size() == 17
        for tick in axis.get_major_ticks():
            assert tick.tick1line.get_visible() and tick.tick2line.get_visible()
            assert tick.tick1line.get_markersize() > 0 and tick.tick2line.get_markersize() > 0
    assert not any(line.get_visible() for line in heatmap.get_xgridlines() + heatmap.get_ygridlines())
    assert {text.get_size() for text in heatmap.texts} == {12}
    assert {text.get_size() for text in heatmap.get_xticklabels() + heatmap.get_yticklabels()} == {13}
    assert {text.get_size() for text in colorbar.get_yticklabels()} == {13}
    assert colorbar.yaxis.label.get_size() == 17
    image = heatmap.images[0]
    np.testing.assert_array_equal(image.get_array().data, company_year.drop(columns="org").values)
    np.testing.assert_array_equal(image.get_array().mask, image.get_array().data <= 0)
    np.testing.assert_array_equal(image.cmap(image.norm(image.get_array()))[0, 0], [1, 1, 1, 1])
    assert image.cmap.name == "blue_cream_red" and image.get_clim() == (1., 16.)
    assert image.get_interpolation() == "nearest"
    boxes = [text.get_window_extent(fig.canvas.get_renderer()) for text in heatmap.get_yticklabels()]
    assert all(lower.y1 < upper.y0 for upper, lower in zip(boxes, boxes[1:]))
    for ax in fig.axes:
        box = ax.get_tightbbox(fig.canvas.get_renderer())
        assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1)
    pd.testing.assert_frame_equal(company_year, original)
    plt.close(fig)


def test_collaboration_publication_figure_keeps_only_all_sector_citations_without_warnings(tmp_path):
    import warnings
    from utils import data_analysis_04_non_academic_collab_helpers as H
    H.apply_project_plot_style()
    df = pd.DataFrame({"year": [2020, 2021, 2022], "times_cited": [3, 8, 20]})
    for i, sector in enumerate(H.NON_ACADEMIC_SECTOR_LABELS):
        df[H.sector_flag_col(sector)] = [1, int(i % 2 == 0), 1]
    original = df.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", matplotlib.MatplotlibDeprecationWarning)
        warnings.simplefilter("error", PendingDeprecationWarning)
        with patch.object(H, "build_company_collaborator_churn_table") as churn, \
             patch.object(H, "build_top_company_year_authorship_matrix") as matrix, \
             patch.object(H, "show_figures") as show, patch.object(H, "save_figure_file") as save:
            paths = H.plot_publication_figure(df, start_year=2020, end_year=2022,
                                              save_path=tmp_path / "non_academic_collab_publication_figure")
    churn.assert_not_called()
    matrix.assert_not_called()
    assert set(paths) == {"pdf", "png"} and save.call_count == 2
    show.assert_called_once()
    assert "Cumulative" not in show.call_args.kwargs["caption"]
    assert "memberships overlap" in show.call_args.kwargs["caption"]
    fig = plt.gcf()
    F.finalize_figure(fig)
    S.finalize_figure(fig)
    fig.canvas.draw()
    ax, = fig.axes
    assert ax.get_title(loc="left") == "A"
    assert not ax.images and not ax.containers
    assert len(ax.patches) == len(H.NON_ACADEMIC_SECTOR_LABELS)
    medians = [line for line in ax.lines if line.get_linewidth() == 1.5]
    assert len(medians) == len(H.NON_ACADEMIC_SECTOR_LABELS)
    for box, text, median, sector in zip(ax.patches, ax.get_yticklabels(), medians,
                                          H.NON_ACADEMIC_SECTOR_LABELS):
        assert to_hex(box.get_facecolor()) == to_hex(H.SECTOR_COLORS[sector])
        assert box.get_alpha() == 1
        values = np.log10(df.loc[df[H.sector_flag_col(sector)].eq(1), "times_cited"] + 1)
        np.testing.assert_allclose(median.get_xdata(), np.median(values))
        assert f"(n={len(values)})" in text.get_text()
    assert ax._left_title.get_size() == 26
    assert ax.xaxis.label.get_size() == ax.yaxis.label.get_size() == 17
    assert {side: spine.get_visible() for side, spine in ax.spines.items()} == {
        "left": True, "bottom": True, "top": False, "right": False}
    pd.testing.assert_frame_equal(df, original)
    plt.close(fig)


def test_earlier_citation_comparison_uses_current_boxplot_arguments_and_exact_palette():
    import warnings
    from utils import data_analysis_04_non_academic_collab_helpers as H
    H.apply_project_plot_style()
    df = pd.DataFrame({"times_cited": [0, 3, 8], "any_sector_collab_flag": [0, 1, 1]})
    sectors = ("Hospital/Clinical", "University/HEI", "Company (non-UK)", "UK company")
    for sector in sectors:
        df[H.sector_flag_col(sector)] = [0, 1, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("error", matplotlib.MatplotlibDeprecationWarning)
        warnings.simplefilter("error", PendingDeprecationWarning)
        with patch.object(H, "show_figures"):
            H.plot_citation_distribution_by_group(df)
    fig = plt.gcf()
    expected = [S.palette("cream"), *[H.SECTOR_COLORS[sector] for sector in sectors]]
    assert [to_hex(box.get_facecolor()) for box in fig.axes[0].patches] == [color.lower() for color in expected]
    assert all(box.get_alpha() == 1 for box in fig.axes[0].patches)
    plt.close(fig)


def test_collaboration_company_heatmaps_are_duplicates_in_paper_contribution_mode():
    from utils import data_analysis_04_non_academic_collab_helpers as H
    companies = [["Company A", "Company B"], ["Company A"], ["Company C"], ["Company B"]]
    df = pd.DataFrame({"year": [2020, 2021, 2021, 2022],
                       "company_institutions_norm": companies,
                       "company_authorship_mentions": companies})
    papers = H.build_top_company_year_matrix(df, start_year=2020, end_year=2022)
    contributions = H.build_top_company_year_authorship_matrix(df, start_year=2020, end_year=2022)
    pd.testing.assert_frame_equal(papers.set_index("org").sort_index(),
                                  contributions.set_index("org").sort_index())


def test_clinical_heatmap_has_full_spines_larger_labels_white_zeros_and_panel_gap():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    source = "".join(next(c for c in nb["cells"] if c["id"] == "701c21a5")["source"])
    draw = source[source.index("n = len(leaves)"):source.index('savefig(fig, "ct_mesh_leaf_condition_icd_heatmaps")')]
    style = S.load_style("04_non_academic_01_clinical_trials")
    first = np.array([[0., 25., 100.], [50., 0., 75.]])
    second = np.array([[100., 0., 10.], [0., 25., 50.]])
    import textwrap
    namespace = dict(plt=plt, np=np, textwrap=textwrap, STYLE=style,
                     leaves=["Disease one", "Disease two"],
                     leaf_trials={"Disease one": range(4), "Disease two": range(6)},
                     chapters=["I Chapter", "II Chapter", "III Chapter"],
                     conditions=["Condition one", "Condition two", "Condition three"],
                     H_icd=pd.DataFrame(first), H_cond=pd.DataFrame(second),
                     blue_cream_red_colormap=S.blue_cream_red_colormap, set_title=S.set_title)
    exec(draw, namespace)
    fig = namespace["fig"]
    F.finalize_figure(fig)
    F.finalize_figure(fig)
    fig.canvas.draw()
    left, right = fig.axes[:2]
    assert right.get_position().x0 - left.get_position().x1 >= .04
    for ax in (left, right):
        assert all(spine.get_visible() for spine in ax.spines.values())
        assert all(to_hex(spine.get_edgecolor()) == "#000000" for spine in ax.spines.values())
        assert all(spine.get_linewidth() == 1.0 for spine in ax.spines.values())
        assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 17
        assert all(label.get_fontsize() == 13 for label in ax.get_xticklabels() + ax.get_yticklabels())
    colorbar = right.images[0].colorbar
    assert colorbar.ax.yaxis.label.get_fontsize() == 16
    assert all(label.get_fontsize() == 12 for label in colorbar.ax.get_yticklabels())
    images = [left.images[0], right.images[0]]
    assert images[0].norm is images[1].norm
    assert all(im.get_clim() == (10., 100.) for im in images)
    for image, original in zip(images, (first, second)):
        displayed = image.get_array()
        np.testing.assert_array_equal(displayed.data, original[::-1])
        np.testing.assert_array_equal(np.ma.getmaskarray(displayed), original[::-1] == 0)
        rgba = image.cmap(image.norm(displayed))
        np.testing.assert_array_equal(rgba[displayed.mask], np.ones((2, 4)))
        assert len(image.axes.collections) == 2  # cell edges survive repeat finalization
        assert sum(text.get_text() == '-' for text in image.axes.texts) == 2
    np.testing.assert_array_equal(namespace["H_icd"].to_numpy(), first)
    np.testing.assert_array_equal(namespace["H_cond"].to_numpy(), second)
    plt.close(fig)


def test_legend_handles_keep_their_own_transform_and_match_lines():
    fig, ax = plt.subplots()
    line, = ax.plot([1, 2], [1, 2], color="white", marker="o", label="Other/Unknown")
    legend = ax.legend()
    handle = legend.legend_handles[0]
    transform = handle.get_transform()
    F.finalize_figure(fig)
    fig.canvas.draw()
    assert handle.get_transform() is transform
    assert handle.get_color() == line.get_color() == "black"
    assert handle.get_linestyle() == line.get_linestyle() == ":"
    assert handle.get_markeredgecolor() == "black"
    assert handle.get_markersize() == line.get_markersize() == 7
    plt.close(fig)


def test_field_and_concept_rankings_refresh_palette_and_keep_larger_text_on_export():
    import textwrap
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    by_id = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    source = by_id["f3a1dc2e"]
    setup = source[:source.index("import textwrap")]
    draw = source[source.index("# wrap long labels"):source.index('savefig(fig, "ct_ukbb_papers_fields_concepts"')]
    fields = pd.Series([35.3, 27.7, 24.2, 19., 13.3, 7.5, 6.3, 5.7, 5.3, 2.8],
                       index=["Clinical Sciences", "Epidemiology", "Genetics", "Public Health",
                              "Cardiovascular Medicine and Haematology", "Biological Psychology",
                              "Oncology and Carcinogenesis", "Nutrition and Dietetics",
                              "Health Services and Systems", "Neurosciences"])
    concepts = pd.Series([.71, .63, .55, .54, .52, .48, .43, .41, .40, .38],
                         index=["variants", "loci", "genome-wide association studies",
                                "cardiovascular disease", "fat", "cancer", "physical activity",
                                "phenotype", "liver fat", "traits"])
    namespace = dict(plt=plt, np=np, textwrap=textwrap, set_title=S.set_title,
                     STYLE={"c_primary": "purple", "c_green": "olive"},
                     for_frac=fields.copy(), con_frac=concepts.copy(), n_for=158, n_con=166)
    exec(by_id["8e6e07a8"], namespace)
    exec(setup + "\n" + draw, namespace)
    fig = namespace["fig"]
    F.finalize_figure(fig)
    F.finalize_figure(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert namespace["STYLE"]["colors"] == S.NON_ACADEMIC_PALETTE
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["A", "B"]
    for ax, values, color in zip(fig.axes, (fields, concepts), S.palette("red", "steel_blue")):
        assert {to_hex(bar.get_facecolor()).upper() for bar in ax.patches} == {color}
        np.testing.assert_array_equal([bar.get_width() for bar in ax.patches], sorted(values))
        assert ax.xaxis.label.get_fontsize() == 17
        assert all(t.get_fontsize() == 14 for t in ax.get_xticklabels() + ax.get_yticklabels())
        assert all(t.get_fontsize() == 13 for t in ax.texts)
        assert all(t.get_fontfamily() == ["Helvetica"] for t in ax.texts)
        boxes = [t.get_window_extent(renderer) for t in ax.get_yticklabels()]
        assert not any(a.overlaps(b) for a, b in zip(boxes, boxes[1:]))
        for text in [ax.xaxis.label, *ax.get_yticklabels(), *ax.texts]:
            box = text.get_window_extent(renderer)
            assert fig.bbox.x0 <= box.x0 < box.x1 <= fig.bbox.x1
            assert fig.bbox.y0 <= box.y0 < box.y1 <= fig.bbox.y1
        for text in ax.texts:
            assert text.get_window_extent(renderer).x1 <= ax.bbox.x1
    plt.close(fig)


def test_field_concept_plot_reloads_project_blue_and_draws_white_zeros_with_space():
    import textwrap
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    source = "".join(next(c for c in nb["cells"] if c["id"] == "e32778ed")["source"])
    setup = source[:source.index("# =============================================================================")]
    draw = source[source.index("n = len(fields)"):source.index('savefig(fig, "ct_field_concept_heatmap")')]
    fields = ["Clinical Sciences", "Genetics", "Public Health"]
    values = np.array([[0., 10., 50.], [25., 0., 75.], [0., 20., 0.]])
    weights = pd.Series([35.3, 24.2, 19.0], index=fields)
    namespace = dict(np=np, plt=plt, textwrap=textwrap, STYLE={"c_primary": "#274668"},
                     fields=fields, concepts=["Concept one", "Concept two", "Concept three"],
                     field_frac=weights, heat=pd.DataFrame(values, index=fields),
                     ylabels=fields, set_title=S.set_title)
    exec(setup + "\n" + draw, namespace)
    fig = namespace["fig"]
    F.finalize_figure(fig)
    F.finalize_figure(fig)
    fig.canvas.draw()
    left, right = fig.axes[:2]
    assert right.get_position().x0 - left.get_position().x1 >= .05
    assert {side: spine.get_visible() for side, spine in left.spines.items()} == {
        "left": True, "bottom": True, "top": False, "right": False}
    assert all(spine.get_visible() for spine in right.spines.values())
    for ax, annotation_size in ((left, 13), (right, 12)):
        assert all(to_hex(spine.get_edgecolor()) == "#000000" for spine in ax.spines.values())
        assert all(spine.get_linewidth() == 1.0 for spine in ax.spines.values())
        assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 17
        assert all(t.get_fontsize() == 14 for t in ax.get_xticklabels() + ax.get_yticklabels())
        assert all(t.get_fontsize() == annotation_size for t in ax.texts)
        np.testing.assert_array_equal(ax.get_ylim(), (-.5, len(fields) - .5))
    colorbar = right.images[0].colorbar
    assert colorbar.ax.yaxis.label.get_fontsize() == 16
    assert all(t.get_fontsize() == 13 for t in colorbar.ax.get_yticklabels())
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for text in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
            if text.get_text():
                box = text.get_window_extent(renderer)
                assert fig.bbox.x0 <= box.x0 < box.x1 <= fig.bbox.x1
                assert fig.bbox.y0 <= box.y0 < box.y1 <= fig.bbox.y1
    expected_blue = S.palette("steel_blue")
    assert {to_hex(bar.get_facecolor()).upper() for bar in left.patches} == {expected_blue}
    np.testing.assert_array_equal([bar.get_width() for bar in left.patches], weights.values[::-1])
    image = right.images[0]
    displayed = image.get_array()
    np.testing.assert_array_equal(displayed.data, values[::-1])
    np.testing.assert_array_equal(displayed.mask, values[::-1] == 0)
    np.testing.assert_array_equal(image.cmap(image.norm(displayed))[displayed.mask], np.ones((4, 4)))
    assert image.get_clim() == (10., 75.)
    assert sum(text.get_text() == "-" for text in right.texts) == 4
    assert len(right.collections) == 2
    assert [ax.get_title(loc="left") for ax in (left, right)] == ["A", "B"]
    np.testing.assert_array_equal(namespace["heat"].to_numpy(), values)
    plt.close(fig)


@pytest.mark.parametrize("cached_palette", [False, True])
def test_timeliness_annotation_and_all_panel_colours_follow_current_data_and_palette(cached_palette):
    import textwrap
    from matplotlib.text import Annotation
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    source = "".join(next(c for c in nb["cells"] if c["id"] == "24a133cf")["source"])
    papers = pd.DataFrame({"id": ["a", "b", "c", "d"],
                           "year": [2015, 2016, 2017, 2018],
                           "times_cited": [100, 154, 154, 220],
                           "journal.title": ["Nature", "Nature", "JAMA", "Clinical Journal"]})
    trials = pd.DataFrame({"start_year": [2018, 2020, 2028, 2014],
                           "n_ukbb": [2, 1, 1, 1],
                           "brief_title": ["Two-paper trial", "Trial two", "Trial three", "Trial four"],
                           "pids": [["a", "b"], ["c"], ["d"], ["a"]]})
    blue, cream, red = S.palette("steel_blue", "cream", "red")
    renderer = F.trial_papers_overview_figure

    def render(*args, **kwargs):
        figure = renderer(*args, **kwargs)
        if cached_palette:
            for ax in figure.axes:
                for bar in ax.patches:
                    bar.set_facecolor(S.palette("green"))
        return figure

    for shift, expected_label in ((0, "Median: three years"), (-1, "Median: 2 years")):
        shifted = trials.copy()
        shifted["start_year"] += shift
        namespace = dict(np=np, pd=pd, plt=plt, textwrap=textwrap, ct_papers=papers.copy(),
                         df_ct=shifted, STYLE={"c_primary": "#274668"},
                         set_title=S.set_title, savefig=lambda *args, **kwargs: None,
                         ANALYSIS_START_YEAR=2013, ANALYSIS_END_YEAR=2025)
        with (patch.object(F, "show_figures"),
              patch.object(F, "trial_papers_overview_figure", side_effect=render)):
            exec(source, namespace)
        fig = namespace["fig"]
        F.finalize_figure(fig)
        F.finalize_figure(fig)
        fig.canvas.draw()
        assert len(fig.axes) == 5
        assert [ax.get_title(loc="left") for ax in fig.axes] == list("ABCDE")
        assert list(fig.axes[-1].get_subplotspec().rowspan) == [0, 1]
        for ax, color in zip(fig.axes, (blue, cream, blue, cream, red)):
            assert {to_hex(bar.get_facecolor()).upper() for bar in ax.patches} == {color}
            assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 17
            assert all(t.get_fontsize() == 14 for t in ax.get_xticklabels() + ax.get_yticklabels())
        middle = fig.axes[3]
        annotation, = [t for t in middle.texts if isinstance(t, Annotation) and t.arrow_patch is not None]
        assert annotation.get_text() == expected_label
        assert annotation.xy[0] == namespace["lags_pos"].median() == 3 + shift
        assert to_hex(middle.lines[0].get_color()).upper() == S.palette("red")
        citation_annotation, = fig.axes[0].texts
        assert citation_annotation.get_text() == "Median: 154"
        assert citation_annotation.xy[0] == papers["times_cited"].median()
        assert sum(bar.get_height() for bar in fig.axes[0].patches) == len(papers)
        assert sum(bar.get_height() for bar in fig.axes[1].patches) == len(trials)
        np.testing.assert_array_equal(fig.axes[1].get_xticks(), [1, 2])
        assert sum(bar.get_height() for bar in fig.axes[2].patches) == len(papers)
        assert sum(bar.get_height() for bar in middle.patches) == len(namespace["lags_pos"])
        assert sum(bar.get_width() for bar in fig.axes[4].patches) == len(papers)
        callouts = [t for ax in fig.axes for t in ax.texts
                    if isinstance(t, Annotation) and t.arrow_patch is not None]
        assert len(callouts) == 3
        assert callouts[1].xy[0] > trials["n_ukbb"].max()  # arrow clears the last count label
        for annotation in callouts:
            assert annotation.get_fontsize() == 13
            assert to_hex(annotation.arrow_patch.get_edgecolor()).upper() == S.palette("navy")
            box = annotation.get_bbox_patch()
            assert to_hex(box.get_facecolor()) == "#ffffff"
            assert to_hex(box.get_edgecolor()) == "#000000"
            assert box.get_linewidth() == box.get_alpha() == 1
            assert box.get_boxstyle().pad == .35
            bounds = box.get_window_extent(fig.canvas.get_renderer())
            assert annotation.axes.bbox.contains(bounds.x0, bounds.y0)
            assert annotation.axes.bbox.contains(bounds.x1, bounds.y1)
        pd.testing.assert_frame_equal(namespace["ct_papers"], papers)
        pd.testing.assert_frame_equal(namespace["df_ct"], shifted)
        plt.close(fig)


def test_combined_trial_paper_overview_handles_empty_positive_lags():
    fig = F.trial_papers_overview_figure(
        pd.Series([0., 0.]), pd.Series([2], index=[1]), None,
        pd.Series([2], index=[2020]), pd.Series(dtype=float),
        pd.Series([2], index=["Journal one"]),
        style=S.load_style("04_non_academic_01_clinical_trials"), start_year=2013, end_year=2025,
    )
    fig.canvas.draw()
    assert any(t.get_text() == "No positive lags" for t in fig.axes[3].texts)
    assert not fig.axes[3].lines
    assert fig.axes[0].texts[0].get_text() == "Median: 0"
    assert fig.axes[0].lines[0].get_xdata()[0] == 1
    plt.close(fig)


def test_trial_paper_overview_is_displayed_once_with_one_export_and_caption():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    by_id = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    assert "show_figures" not in by_id["21d3e3d0"]
    assert "plt.subplots" not in by_id["21d3e3d0"]
    assert 'savefig(fig, TRIAL_PAPERS_EXPORT, STYLE)' in by_id["24a133cf"]
    assert by_id["24a133cf"].count("show_figures()") == 1
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert '"ct_ukbb_papers_impact"' not in code
    assert '"ct_papers_timeliness"' not in code
    assert all(f"({letter})" in F.CAPTIONS[F.TRIAL_PAPERS_EXPORT] for letter in "ABCDE")


def test_clinical_and_patent_semantic_colours_are_shared():
    from utils import data_analysis_04_non_academic_panels as panels
    assert panels._trial_sector_colors() == F.TRIAL_SECTOR_COLORS
    assert F.PATENT_STATUS_COLORS["Active"] == S.NON_ACADEMIC_PALETTE[6]
    assert F.SECTOR_COLORS["Other/Unknown"] == "white"


def test_clinical_status_and_annual_bars_use_shared_palette_with_matching_legends():
    from utils import data_analysis_04_non_academic_panels as panels
    style = S.load_style("04_non_academic_01_clinical_trials")
    expected = {"Interventional": S.palette("steel_blue"), "Observational": S.palette("red")}
    assert style["trial_type_colors"] == panels._trial_type_colors() == expected
    assert style["c_gold"] == S.palette("cream")
    assert S._resolve(None) is style  # looking up colours must not switch the active style

    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    by_id = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    frame = pd.DataFrame({
        "overall_status": ["Completed", "Recruiting", "Completed", "Not yet recruiting"],
        "study_type": ["Interventional", "Interventional", "Observational", "Observational"],
        "start_year": [2014, 2020, 2020, 2025],
    })
    with (patch.object(F, "show_figures") as show, patch.object(F, "savefig") as save,
          patch("IPython.display.display") as display):
        namespace = dict(np=np, pd=pd, plt=plt, STYLE=style, df_ct=frame.copy(),
                         set_title=S.set_title, savefig=save, display=display,
                         ANALYSIS_START_YEAR=2013, ANALYSIS_END_YEAR=2025)
        before = set(plt.get_fignums())
        exec(by_id["36b5d875"], namespace)
        assert set(plt.get_fignums()) == before
        save.assert_not_called()
        show.assert_not_called()
        display.assert_not_called()
        exec(by_id["7d4d906b"], namespace)
        fig = namespace["fig"]
        assert set(plt.get_fignums()) - before == {fig.number}
        assert len(fig.axes) == 2
        save.assert_called_once_with(fig, "ct_combined_figure", style)
        show.assert_called_once_with()
        display.assert_called_once()
        table = display.call_args.args[0]
        assert table["Total"].sum() == len(frame)
        pd.testing.assert_series_equal(table["Total"], table.drop(columns="Total").sum(axis=1),
                                       check_names=False)
    F.finalize_figure(fig)
    ax = fig.axes[0]
    for container in ax.containers:
        assert all(to_hex(bar.get_facecolor()).upper() == expected[container.get_label()]
                   for bar in container.patches)
    legend = ax.get_legend()
    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        assert to_hex(handle.get_facecolor()).upper() == expected[text.get_text()]
    annual = fig.axes[1].patches
    assert all(to_hex(bar.get_facecolor()).upper() == S.palette("cream") for bar in annual)
    assert sum(bar.get_height() for bar in annual) == len(frame)
    source = "\n".join(by_id.values())
    assert "ct_status_stage" not in source
    assert source.count("def plot_grouped_status(") == 1
    assert source.count("STATUS_STAGE = {") == 1
    plt.close(fig)


def test_unsaved_figures_get_a_name_paths_and_caption():
    S.load_style("04_non_academic_panels")
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    with patch.object(F, "savefig") as save, patch.object(S, "display_figure") as display:
        F.show_figures("example", caption="An explicit suggested caption.")
    save.assert_called_once_with(fig, "example", caption="An explicit suggested caption.")
    display.assert_called_once_with(fig)
    assert fig._ukb_caption == "An explicit suggested caption."


def test_notebook_does_not_repeat_uk_trial_plot_or_write_caption_files():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    by_id = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    assert "show_figures" in by_id["970365cb"]
    assert "show_figures" not in by_id["0d39d731"]
    assert 'selection[["figure", "panel"]].to_csv' in by_id["518abcd2"]
    assert "export_manifest()" in by_id["82cdc7fd"]


def test_trial_comparisons_preserve_rankings_and_use_readable_annotations():
    style = S.load_style("04_non_academic_01_clinical_trials")
    assert style["c_primary"] == S.palette("steel_blue")
    assert style["c_gold"] == S.palette("cream")
    names = [f"Category {i}" for i in range(15)]
    names[6] = "Health Disparities and Racial or Ethnic Minority Health Research"
    raw = pd.Series([47, 43, 33, 30, 30, 24, 24, 23, 22, 21, 21, 19, 19, 19, 19], index=names)
    fractional = (raw / 10).iloc[::-1]
    rows = [("MeSH categories", raw, fractional), ("RCDC categories", raw * 3, fractional * 2)]
    before = [series.copy() for _, r, f in rows for series in (r, f)]
    fig = F.trial_classification_figure(rows, style=style)
    # Export finalizes a second time; it must not shrink the requested annotations.
    F.finalize_figure(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert len(fig.axes) == 4
    assert [ax.get_title(loc="left") for ax in fig.axes] == list("ABCD")
    assert fig.get_layout_engine() is not None
    for i, (ax, expected) in enumerate(zip(fig.axes, before)):
        ranked = expected.sort_values(ascending=False, kind="stable")
        np.testing.assert_array_equal([bar.get_width() for bar in ax.patches], ranked.values)
        assert len(ax.patches) == len(ax.texts) == 15
        assert ax.yaxis_inverted()
        assert all(text.get_fontsize() == 12 for text in ax.texts)
        color = style["c_gold"] if i % 2 else style["c_primary"]
        assert {to_hex(bar.get_facecolor()).upper() for bar in ax.patches} == {color}
        boxes = [label.get_window_extent(renderer) for label in ax.get_yticklabels()]
        assert not any(a.overlaps(b) for a, b in zip(boxes, boxes[1:]))
        for text in ax.texts:
            box = text.get_window_extent(renderer)
            assert ax.bbox.x0 <= box.x0 < box.x1 <= ax.bbox.x1
    assert fig.axes[0].get_position().y0 == fig.axes[1].get_position().y0
    assert fig.axes[2].get_position().y0 == fig.axes[3].get_position().y0
    for actual, expected in zip([s for _, r, f in rows for s in (r, f)], before):
        pd.testing.assert_series_equal(actual, expected)
    plt.close(fig)


def test_trial_comparisons_replace_standalone_displays_without_losing_categories():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    by_id = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    for cell_id in ("98fc48f2", "1ff70dd7"):
        assert "savefig(" not in by_id[cell_id]
        assert "show_figures()" not in by_id[cell_id]
    for cell_id, name in (("9e2a4f36", "ct_mesh_rcdc_categories"),
                          ("f7eec542", "ct_disease_classifications"),
                          ("5a22f648", "ct_diseases_mesh_ancestors")):
        assert "trial_classification_figure(" in by_id[cell_id]
        assert f'savefig(fig, "{name}")' in by_id[cell_id]
        assert name in F.CAPTIONS
    code = "\n".join(by_id.values())
    for obsolete in ("ct_diseases_mesh", "ct_rcdc_all", "ct_rcdc_disease", "ct_diseases_mesh_leaf"):
        assert f'savefig(fig, "{obsolete}")' not in code


def test_trial_geography_maps_share_author_ramp_and_have_no_country_callouts():
    import geopandas as gpd
    from shapely.geometry import box
    world = gpd.GeoDataFrame({
        "ADMIN": ["United States", "United Kingdom", "No data", "Antarctica"],
        "ISO_A3": ["USA", "-99", "XXX", "ATA"],
        "ADM0_A3": ["USA", "GBR", "XXX", "ATA"],
    }, geometry=[box(-125, 25, -65, 50), box(-8, 50, 2, 59),
                 box(20, 20, 30, 30), box(-180, -90, 180, -70)], crs="EPSG:4326")
    trials = pd.Series({"USA": 49, "GBR": 19})
    papers = pd.Series({"USA": 80, "GBR": 110})
    fig = F.trial_geography_figure([
        (trials, "Number of trials"), (trials, "Number of trials"),
        (papers, "Number of papers"),
    ], style=S.load_style("04_non_academic_01_clinical_trials"), world=world)
    fig.canvas.draw()
    maps = [ax for ax in fig.axes if not hasattr(ax, "_colorbar")]
    assert len(maps) == 3
    assert [ax.get_title(loc="left") for ax in maps] == list("ABC")
    assert not any(ax.texts for ax in maps)
    assert not any(ax.lines for ax in maps)  # no callout leader lines
    assert all(len(ax.collections) == 2 for ax in maps)  # base + values, no highlighted borders
    for ax, expected in zip(maps, ([49, 19], [49, 19], [80, 110])):
        base, measured = ax.collections
        np.testing.assert_array_equal(base.get_facecolors(), [[1., 1., 1., 1.]])
        np.testing.assert_array_equal(measured.get_array(), expected)
        np.testing.assert_allclose(measured.cmap(np.linspace(0, 1, 256)),
                                   S.author_geography_colormap()(np.linspace(0, 1, 256)))
        assert measured.cmap.name == "Blues"
        assert ax.get_ylim() == (-58, 90)
    assert maps[0].get_position().y0 > maps[1].get_position().y1
    assert maps[1].get_position().y0 > maps[2].get_position().y1
    assert world.iloc[-1]["ADMIN"] == "Antarctica"  # caller data was not mutated
    plt.close(fig)


def test_geography_notebook_exports_only_the_combined_map():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    assert "annotate_top" not in cells["40571432"]
    assert "show_figures()" not in cells["40571432"]
    assert "trial_geography_figure([" in cells["8365c642"]
    assert cells["8365c642"].count('(iso_counts, "Number of trials")') == 2
    assert "same country counts" in cells["f92ef5ce"]
