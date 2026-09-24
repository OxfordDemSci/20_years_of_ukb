"""Policy panels retain their values while using readable manuscript styling."""
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import pytest

from utils import data_analysis_04_non_academic_panels as N
from utils import data_analysis_04_non_academic_figures as F
from utils import shared_style as S


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_policy_palette_grids_and_labels_survive_export_finalization():
    S.load_style("04_non_academic_panels")
    data = {"policy": {
        "by_year_origin": pd.DataFrame({"United Kingdom": [2, 4],
                                         "Rest of world": [3, 8]}, index=[2020, 2021]),
        "publishers": pd.Series([81, 39], index=["World Health Organization", "Scottish Government"]),
        "divisions": pd.Series([195, 104], index=["Health Sciences", "Human Society"]),
    }}

    def map_stub(ax, _):
        ax.set_axis_off()
        ax.figure.colorbar(plt.cm.ScalarMappable(), ax=ax).set_label("Policy documents (log)")

    with patch.object(N, "draw_policy_country_map", side_effect=map_stub), patch.object(N, "savefig") as save:
        fig = N.figure_si_policy(data)
    save.assert_called_once_with(fig, "04_04_supplementary_figure_03_policy")
    for _ in range(2):
        F.finalize_figure(fig)
        fig.canvas.draw()
    a, b, c, d, colorbar = fig.axes
    renderer = fig.canvas.get_renderer()
    assert c.get_tightbbox(renderer).x1 < d.get_tightbbox(renderer).x0
    blue, red = S.palette("steel_blue", "red")
    assert [to_hex(bar.get_facecolor()).upper() for bar in a.patches] == [blue, blue, red, red]
    assert {to_hex(bar.get_facecolor()).upper() for bar in c.patches} == {blue}
    assert {to_hex(bar.get_facecolor()).upper() for bar in d.patches} == {red}
    np.testing.assert_array_equal([bar.get_height() for bar in a.patches], [2, 4, 3, 8])
    np.testing.assert_array_equal([bar.get_width() for bar in c.patches], [39, 81])
    np.testing.assert_array_equal([bar.get_width() for bar in d.patches], [104, 195])
    for ax in (a, c, d):
        assert ax.xaxis.label.get_size() == ax.yaxis.label.get_size() == 16
        assert ax.get_axisbelow() is True
        for axis in (ax.xaxis, ax.yaxis):
            assert all(tick.gridline.get_visible() for tick in axis.get_major_ticks())
            assert not any(tick.gridline.get_visible() for tick in axis.get_minor_ticks())
        assert all(text.get_size() == 12 for text in ax.get_xticklabels() + ax.get_yticklabels())
    assert not b.axison
    assert colorbar.yaxis.label.get_size() == 16
    assert N.POLICY_MAP_RAMP == ["#FFFFFF", "blue", "steel_blue", "navy"]
    plt.close(fig)
