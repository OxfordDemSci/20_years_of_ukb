"""The early attention figure retains only trends, not the later main-panel scatter."""

import ast
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import pytest

from utils import data_analysis_04_non_academic_figures as F
from utils import shared_style as S


ROOT = Path(__file__).resolve().parents[1]
CELL_ID = "fb7eb8b5-5f5f-4480-999c-223252b2c797"


def notebook_source():
    notebook = json.loads((ROOT / "src/data_analysis/04_non_academic.ipynb").read_text())
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


@pytest.mark.parametrize("has_news", [True, False])
def test_attention_figure_contains_only_the_unchanged_annual_series(has_news):
    S.load_style("04_non_academic_03_altmetric")
    source = notebook_source()[CELL_ID]
    plot_code = source[source.index("# Annual trajectories only;"):]
    yearly = pd.DataFrame({"News mentions": [2, 5, 9], "Policy mentions": [1, 0, 3]},
                          index=[2013, 2014, 2015])
    original = yearly.copy(deep=True)
    namespace = dict(plt=plt, yearly=yearly, HAS_NEWS=has_news,
                     ANALYSIS_START_YEAR=2013, ANALYSIS_END_YEAR=2015,
                     df_plot=pd.DataFrame(index=range(12)), stats_text="Summary retained")
    with patch.object(F, "savefig") as save, patch.object(F, "show_figures") as show:
        namespace["show_figures"] = show
        exec(compile(plot_code, "attention_trends", "exec"), namespace)
    save.assert_called_once()
    show.assert_called_once_with()
    fig, name = save.call_args.args
    assert name == "impact"
    caption = save.call_args.kwargs["caption"]
    assert "12 publication records" in caption
    assert "source-snapshot totals" in caption
    assert "all matched records" in caption
    assert "scatter" not in caption and "(B)" not in caption
    for _ in range(2):
        F.finalize_figure(fig)
        fig.canvas.draw()
    assert len(fig.axes) == (2 if has_news else 1)
    trend, policy = fig.axes[0], fig.axes[-1]
    assert trend.get_title(loc="left") == "A"
    assert not any(ax.collections for ax in fig.axes)
    np.testing.assert_array_equal(policy.lines[0].get_ydata(), yearly["Policy mentions"])
    assert to_hex(policy.lines[0].get_color()) == S.palette("red").lower()
    assert policy.get_ylabel() == "Policy mentions"
    if has_news:
        assert policy.get_title(loc="left") == ""
        np.testing.assert_allclose(trend.get_position().bounds, policy.get_position().bounds)
        np.testing.assert_array_equal(trend.lines[0].get_ydata(), yearly["News mentions"])
        assert to_hex(trend.lines[0].get_color()) == S.palette("steel_blue").lower()
        assert trend.get_ylabel() == "News mentions"
        assert "left axis" in caption and "right axis" in caption
    else:
        assert len(trend.lines) == 1
        assert "news-mention data are unavailable and are not plotted" in caption
    for ax in fig.axes:
        assert ax.xaxis.label.get_size() == ax.yaxis.label.get_size() == 17
        assert all(spine.get_visible() for spine in ax.spines.values())
        box = ax.get_tightbbox(fig.canvas.get_renderer())
        assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1)
    assert all(text.get_size() == 13 for text in trend.get_legend().get_texts())
    pd.testing.assert_frame_equal(yearly, original)
    plt.close(fig)


def test_duplicate_scatter_removed_here_and_retained_in_later_main_figure():
    source = notebook_source()
    early = source[CELL_ID]
    assert ".scatter(" not in early
    assert "scatter_callouts" not in early
    assert "build_annotation_text" not in early
    assert "NP.figure_main(D)" in "\n".join(source.values())
    panels = ast.parse((ROOT / "src/utils/data_analysis_04_non_academic_panels.py").read_text())
    main = next(node for node in panels.body
                if isinstance(node, ast.FunctionDef) and node.name == "figure_main")
    assert any(isinstance(node, ast.Name) and node.id == "draw_altmetric_scatter"
               for node in ast.walk(main))


def test_annual_totals_still_include_records_without_positive_attention_scores():
    source = notebook_source()[CELL_ID]
    aggregation = source[source.index("yearly ="):source.index("# 4. PRINTED SUMMARY")]
    namespace = dict(ANALYSIS_START_YEAR=2013, ANALYSIS_END_YEAR=2015,
                     df_plot=pd.DataFrame({"Year": [2013, 2013, 2015],
                                           "Altmetric Attention Score": [0, 2, 0],
                                           "News mentions": [5, 3, 0],
                                           "Policy mentions": [2, 1, 7]}))
    exec(compile(aggregation, "attention_annual_totals", "exec"), namespace)
    yearly = namespace["yearly"]
    assert yearly.index.tolist() == [2013, 2014, 2015]
    assert yearly["News mentions"].tolist() == [8, 0, 0]
    assert yearly["Policy mentions"].tolist() == [3, 0, 7]
