"""Section 7 combines only analyses not retained elsewhere in the notebook."""
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest

from utils import data_analysis_04_non_academic_collab_helpers as H


def inputs():
    sectors = [
        {"University/HEI": 3, "UK company": 1},
        {"University/HEI": 1, "Hospital/Clinical": 2},
        {"Company (non-UK)": 2, "Hospital/Clinical": 2}, {},
        {"University/HEI": 2, "Hospital/Clinical": 1},
        {"University/HEI": 2, "UK company": 1},
        {"Government/Public": 1, "Research institute/Centre": 1}, {},
        {"Nonprofit/Charity": 1}, {"Other/Unknown": 1},
    ]
    frame = pd.DataFrame({
        "year": [2020] * 4 + [2021] * 4 + [2023] * 2,
        "times_cited": [0, 3, 8, np.nan, 15, 24, 99, 3, 8, 15],
        "collaborator_sector_counts": sectors,
        "journal.title": ["Journal A"] * 5 + ["Journal B"] * 3 + ["Journal C"] * 2,
    })
    for label in H.NON_ACADEMIC_SECTOR_LABELS:
        frame[H.sector_flag_col(label)] = [int(row.get(label, 0) > 0) for row in sectors]
        frame[H.sector_institutions_col(label)] = [
            [f"{label} organisation {j}" for j in range(row.get(label, 0))] for row in sectors]
    frame["uk_company_flag"] = frame[H.sector_flag_col("UK company")]
    frame["company_flag"] = frame["uk_company_flag"] | frame[H.sector_flag_col("Company (non-UK)")]
    frame["company_institutions_norm"] = [["Company A", "Company A", "Company B"] if flag else []
                                          for flag in frame["company_flag"]]
    journals = H.build_journal_company_table(frame, top_n=20, min_papers=1)
    return frame, journals.drop(columns=["share_uk_company"])


def test_four_chart_helpers_can_embed_without_extra_figures_or_exports():
    H.apply_project_plot_style()
    frame, journals = inputs()
    before, journal_before = frame.copy(deep=True), journals.copy(deep=True)
    fig, axes = plt.subplots(2, 2)
    numbers = plt.get_fignums()
    with patch.object(H, "show_figures") as show, patch.object(H, "save_figure_file") as save:
        results = [
            H.plot_collaboration_mix_share_stacked_area(frame, 2020, 2023, ax=axes[0, 0], legend=False),
            H.plot_yearly_median_log_citations_by_group(frame, start_year=2020, end_year=2023,
                                                       ax=axes[0, 1], legend=False),
            H.plot_collaborator_concentration_curves(frame, ax=axes[1, 0], legend=False),
            H.plot_top_journal_company_share(journals, ax=axes[1, 1], legend=False),
        ]
    assert plt.get_fignums() == numbers
    assert all(result[0] is fig and result[1] is ax for result, ax in zip(results, axes.flat))
    assert all(ax.get_legend() is None for ax in axes.flat)
    show.assert_not_called()
    save.assert_not_called()
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(journals, journal_before)
    plt.close(fig)


def test_combined_figure_keeps_percentages_medians_contributions_and_journal_denominators():
    frame, journals = inputs()
    before, journal_before = frame.copy(deep=True), journals.copy(deep=True)
    with patch.object(H, "show_figures") as show, patch.object(H, "save_figure_file") as save:
        fig = H.plot_additional_publication_figure(frame, journals, start_year=2020,
                                                  end_year=2023, min_papers_per_point=2)
    show.assert_not_called()
    save.assert_not_called()
    for _ in range(2):
        H.finalize_figure(fig)
        fig.canvas.draw()
    a, b, c, d = fig.axes
    assert [ax.get_title(loc="left") for ax in fig.axes] == list("ABCD")
    assert [(ax.get_subplotspec().rowspan.start, ax.get_subplotspec().colspan.start)
            for ax in fig.axes] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    masks = H._collaboration_mix_masks(frame)
    years = [2020, 2021, 2022, 2023]
    totals = frame.groupby("year").size().reindex(years, fill_value=0)
    shares = []
    for area in a.collections:
        label = area.get_label()
        expected = frame.loc[masks[label]].groupby("year").size().reindex(years, fill_value=0)
        expected = expected.div(totals.replace(0, 1)).to_numpy() * 100
        vertices = area.get_paths()[0].vertices
        actual = [np.ptp(vertices[vertices[:, 0] == year, 1]) for year in years]
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(area.get_facecolor()[0],
                                   to_rgba(H.SECTOR_COLORS.get(label, H.palette("cream"))))
        shares.append(actual)
    np.testing.assert_allclose(np.sum(shares, axis=0), [100, 100, 0, 100])
    assert a.collections[0].get_hatch() == "///"
    assert a.get_legend().legend_handles[0].get_hatch() == "///"

    citations = np.log10(frame.times_cited.fillna(0) + 1)
    for line in b.lines:
        label = line.get_label()
        mask = (~H._any_sector_collab_mask(frame) if label == "No taxonomy collaborator"
                else frame[H.sector_flag_col(label)].eq(1))
        expected = [citations[mask & frame.year.eq(year)].median()
                    if (mask & frame.year.eq(year)).sum() >= 2 else np.nan for year in years]
        np.testing.assert_allclose(line.get_ydata(), expected, equal_nan=True)
    for line in c.lines[1:]:
        label = line.get_label().removesuffix(" collaborators")
        column = ("company_institutions_norm" if label == "Company (any)"
                  else H.sector_institutions_col(label))
        counts = Counter(key for entries in frame[column] for key in {H.org_key(v) for v in entries})
        values = np.asarray(sorted(counts.values(), reverse=True))
        np.testing.assert_allclose(line.get_xdata(), np.arange(1, len(values) + 1) / len(values))
        np.testing.assert_allclose(line.get_ydata(), np.cumsum(values) / values.sum())
    ranked = journals.sort_values("papers", ascending=False).head(15).sort_values("non_academic_share")
    np.testing.assert_allclose([bar.get_width() for bar in d.patches], ranked.non_academic_share * 100)
    for marks in d.collections:
        np.testing.assert_allclose(marks.get_offsets()[:, 0],
                                   ranked[H.journal_share_col(marks.get_label())] * 100)
    assert [text.get_text() for text in d.texts] == [f"n={n}" for n in ranked.papers]
    assert "UK company" not in [marks.get_label() for marks in d.collections]

    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        assert ax._left_title.get_size() == 26
        assert ax.xaxis.label.get_size() == ax.yaxis.label.get_size() == 17
        assert {side: spine.get_visible() for side, spine in ax.spines.items()} == {
            "left": True, "bottom": True, "top": False, "right": False}
        legend = ax.get_legend()
        assert all(text.get_size() == 12 for text in legend.get_texts())
        box = ax.get_tightbbox(renderer)
        assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1)
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
    assert boxes[0].x1 < boxes[1].x0 and boxes[2].x1 < boxes[3].x0
    assert max(boxes[2].y1, boxes[3].y1) < min(boxes[0].y0, boxes[1].y0)
    assert "not unique-paper coverage" in fig._ukb_caption
    assert "at least 2 publications" in fig._ukb_caption
    assert "unfilled bars" in fig._ukb_caption.lower()
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(journals, journal_before)
    plt.close(fig)


def test_unique_no_taxonomy_citation_baseline_is_preserved_in_section_12(tmp_path):
    frame, _ = inputs()
    original = frame.copy(deep=True)
    with patch.object(H, "show_figures"), patch.object(H, "save_figure_file"):
        H.plot_publication_figure(frame, save_path=tmp_path / "citations")
    fig = plt.gcf()
    ax, = fig.axes
    assert "No taxonomy collaborator" in ax.get_yticklabels()[-1].get_text()
    assert "(n=2)" in ax.get_yticklabels()[-1].get_text()
    medians = [line for line in ax.lines if line.get_linewidth() == 1.5]
    expected = np.log10(frame.loc[~H._any_sector_collab_mask(frame), "times_cited"].fillna(0) + 1)
    np.testing.assert_allclose(medians[-1].get_xdata(), expected.median())
    assert ax.patches[-1].get_facecolor() == to_rgba(H.palette("cream"))
    pd.testing.assert_frame_equal(frame, original)
    plt.close(fig)


def test_no_eligible_journals_or_group_years_does_not_remove_other_panels():
    frame, _ = inputs()
    fig = H.plot_additional_publication_figure(frame, pd.DataFrame(), start_year=2020,
                                               end_year=2023, min_papers_per_point=100)
    assert len(fig.axes) == 4
    assert "No group-years meet the reporting threshold" in [t.get_text() for t in fig.axes[1].texts]
    assert "No eligible journals" in [t.get_text() for t in fig.axes[3].texts]
    assert fig.axes[0].collections and fig.axes[2].lines
    plt.close(fig)
    with pytest.raises(ValueError, match="start year"):
        H.plot_additional_publication_figure(frame, pd.DataFrame(), start_year=2023, end_year=2020)


def test_notebook_publishes_one_section_7_figure_without_duplicated_panels():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    source = cells["800e8726"] + cells["a671a27d"]
    assert source.count("h.plot_additional_publication_figure(") == 1
    assert source.count("show_figures(") == 1
    for duplicate in ("plot_flag_overlap_heatmap", "plot_citation_distribution_by_group",
                      "plot_collaboration_mix_share_stacked_area", "plot_top_journal_company_share",
                      "plot_yearly_median_log_citations_by_group", "plot_collaborator_concentration_curves"):
        assert duplicate not in source
    assert "build_journal_company_table(df, top_n=20, min_papers=25)" in source
    assert 'drop(columns=["share_uk_company"])' in source
    assert "show_figures(h.ADDITIONAL_PUBLICATION_EXPORT, caption=fig._ukb_caption)" in source
    assert "h.plot_publication_figure(" in cells["34dea78e"]
    assert "no-taxonomy baseline" in cells["f3ac1d62"]
