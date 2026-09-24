"""Collaboration SI panels share company identities without changing global roles."""
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd

from utils import data_analysis_04_non_academic_panels as N
from utils import data_analysis_04_non_academic_figures as F
from utils import shared_style as S


def test_collaboration_company_palette_is_consistent_across_panels_and_legends():
    S.load_style("04_non_academic_panels")
    original = N._sector_colors().copy()
    sectors = ["Company (non-UK)", "UK company", "Government/Public"]
    data = {"counts": {"corpus": 100}, "collaboration": {
        "sector_summary": pd.DataFrame({"mentions": [40, 20, 10], "papers": [30, 15, 8]}, index=sectors),
        "sector_share_by_year": pd.DataFrame([[20, 10, 5], [30, 15, 8]], index=[2020, 2021], columns=sectors),
        "company_by_year": pd.DataFrame([[20, 10], [30, 15]], index=[2020, 2021], columns=sectors[:2]),
        "top_orgs": {"Company (non-UK)": pd.Series([12], index=["Company A"]),
                     "UK company": pd.Series([5], index=["Company B"])},
        "division_company_share": pd.DataFrame({"pct_company": [14.5, 8.8], "size": [100, 200]},
                                                index=["Biological Sciences", "Health Sciences"]),
    }}
    try:
        with patch.object(N, "savefig"):
            fig = N.figure_si_collaboration(data)
        for _ in range(2):
            F.finalize_figure(fig)
            fig.canvas.draw()
        a, b, c, d, e, f = fig.axes
        blue, red, navy, yellow = S.palette("steel_blue", "red", "navy", "cream")
        for ax in (a, b):
            assert [to_hex(bar.get_facecolor()).upper() for bar in ax.patches] == [navy, red, blue]
        assert [to_hex(line.get_color()).upper() for line in c.lines] == [blue, red, navy]
        assert [to_hex(bar.get_facecolor()).upper() for bar in d.patches] == [blue, blue, red, red]
        assert {to_hex(bar.get_facecolor()).upper() for bar in e.patches} == {blue}
        assert {to_hex(bar.get_facecolor()).upper() for bar in f.patches} == {yellow}
        assert [to_hex(h.get_facecolor()).upper() for h in d.get_legend().legend_handles] == [blue, red]
        assert [to_hex(h.get_color()).upper() for h in c.get_legend().legend_handles] == [blue, red, navy]
        np.testing.assert_array_equal([bar.get_height() for bar in d.patches], [20, 30, 10, 15])
        np.testing.assert_allclose([bar.get_width() for bar in f.patches], [8.8, 14.5])
        assert N._sector_colors() == original
    finally:
        plt.close("all")
