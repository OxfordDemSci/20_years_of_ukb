"""Main-figure grids stay consistent through finalisation and export."""
from contextlib import ExitStack
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from utils import data_analysis_04_non_academic_panels as N
from utils import data_analysis_04_non_academic_figures as F
from utils import shared_style as S


def draw_linear(ax, data):
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.minorticks_on()
    ax.grid(True, which="both")


def draw_log(ax, data):
    ax.plot([1, 10, 100], [1, 100, 10000])
    ax.set(xscale="log", yscale="log")
    ax.grid(False, which="both")


def draw_heatmap(ax, data):
    image = ax.imshow([[25, 50], [75, 100]])
    ax.figure.colorbar(image, ax=ax)


def assert_grid_consistency(fig):
    F.finalize_figure(fig)
    fig.canvas.draw()
    style = S.load_style("04_non_academic_panels", activate=False)
    for ax in fig.axes[:5]:
        assert ax.get_axisbelow() is True
        for axis in (ax.xaxis, ax.yaxis):
            assert axis.get_major_ticks()
            for tick in axis.get_major_ticks():
                line = tick.gridline
                assert line.get_visible()
                assert line.get_linestyle() == style["grid_linestyle"]
                assert line.get_color() == style["grid_color"]
                assert line.get_linewidth() == style["grid_linewidth"]
                assert line.get_alpha() == style["grid_alpha"]
            assert not any(tick.gridline.get_visible() for tick in axis.get_minor_ticks())
    for ax in fig.axes[5:]:
        for axis in (ax.xaxis, ax.yaxis):
            assert not any(tick.gridline.get_visible()
                           for tick in axis.get_major_ticks() + axis.get_minor_ticks())
    assert fig.axes[5]._ukb_cell_edges


@pytest.mark.parametrize("save", [False, True])
def test_main_figure_grids_are_applied_before_export_and_survive_finalisation(save):
    S.load_style("04_non_academic_panels")
    draws = {
        "draw_reach_by_year": draw_log,
        "draw_patent_legal_status": draw_linear,
        "draw_trials_by_year": draw_linear,
        "draw_collab_sector_share": draw_linear,
        "draw_altmetric_scatter": draw_log,
        "draw_collab_flag_overlap": draw_heatmap,
    }
    with ExitStack() as stack:
        for name, draw in draws.items():
            stack.enter_context(patch.object(N, name, draw))
        export = stack.enter_context(patch.object(
            N, "savefig", side_effect=lambda fig, name: assert_grid_consistency(fig)))
        fig = N.figure_main({}, save=save)
        try:
            assert len(fig.axes) == 7
            assert_grid_consistency(fig)
            assert_grid_consistency(fig)
            assert export.call_count == int(save)
        finally:
            plt.close(fig)
