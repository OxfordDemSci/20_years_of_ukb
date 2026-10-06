"""Every eligible patent must appear in the main legal-status chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import pandas as pd

from utils import data_analysis_04_non_academic_panels as N
from utils import data_analysis_04_non_academic_figures as F
from utils import shared_style as S


ANNUAL_TOTALS = {2018: 2, 2019: 16, 2020: 40, 2021: 78,
                 2022: 93, 2023: 137, 2024: 159, 2025: 171}


def status_records():
    rows = []
    missing = {2021: [None], 2022: [pd.NA, "N/A"], 2024: [" "]}
    for year, total in ANNUAL_TOTALS.items():
        statuses = missing.get(year, [])
        if year == 2025:
            statuses = ["Granted Patent Expired"] * 7
        statuses = statuses + ["Active"] * (total - len(statuses))
        rows.extend({"publication_year": year, "legal_status_replaced": status}
                    for status in statuses)
    return pd.DataFrame(rows)


def test_unknown_statuses_are_retained_without_changing_source_records():
    records = status_records()
    original = records.copy(deep=True)
    counts = N.patent_legal_status_by_year(records)
    assert counts.sum(axis=1).to_dict() == ANNUAL_TOTALS
    assert counts.to_numpy().sum() == 696
    assert counts.Unknown[counts.Unknown > 0].to_dict() == {2021: 1, 2022: 2, 2024: 1}
    assert counts["Granted Patent Expired"].sum() == 7
    pd.testing.assert_frame_equal(records, original)


def test_figure_totals_legend_hatching_and_caption_include_unknown():
    S.load_style("04_non_academic_panels")
    counts = N.patent_legal_status_by_year(status_records())
    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        N.draw_patent_legal_status(ax, {"patents": {"legal_status_by_year": counts}})
        F.finalize_figure(fig)
        fig.canvas.draw()
        assert [text.get_text() for text in ax.texts] == [str(n) for n in ANNUAL_TOTALS.values()]
        assert sum(bar.get_height() for bar in ax.patches) == 696
        unknown = next(c for c in ax.containers if c.get_label() == "Unknown")
        assert [bar.get_height() for bar in unknown] == [0, 0, 0, 1, 2, 0, 1, 0]
        assert all(bar.get_hatch() == "///" and to_hex(bar.get_facecolor()) == "#ffffff"
                   and to_hex(bar.get_edgecolor()) == "#000000" for bar in unknown)
        legend = ax.get_legend()
        label_index = [text.get_text() for text in legend.get_texts()].index("Unknown")
        assert legend.legend_handles[label_index].get_hatch() == "///"
        assert "Unknown" in N.MAIN_CAPTION["B"]
        assert "omitted" not in N.MAIN_CAPTION["B"]
        assert "696 patent records, including 4" in " ".join(fig._ukb_caption_notes)
    finally:
        plt.close(fig)
