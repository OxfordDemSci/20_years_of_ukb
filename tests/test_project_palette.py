"""All active analysis styles use the author notebook's canonical palette."""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd

from utils import shared_style as S
from utils import data_analysis_04_non_academic_figures as F


def test_every_notebook_style_uses_the_same_palette_and_semantic_roles():
    settings = S.load_settings()["style"]
    assert settings["base"]["colors"] == S.PALETTE
    for name in settings["notebooks"]:
        style = S.load_style(name, activate=False)
        expected = S.NON_ACADEMIC_PALETTE if name.startswith("04_non_academic") else S.PALETTE
        assert style["colors"] == expected, name
        if "c_green" in style:
            assert style["c_green"] == S.palette("green"), name
        if "c_gold" in style:
            assert style["c_gold"] == S.palette("cream"), name
        if "c_accent" in style:
            assert style["c_accent"] == S.palette("red"), name


def test_non_academic_sources_and_colour_cycles_cannot_reintroduce_green():
    from IPython.core.inputtransformer2 import TransformerManager
    from matplotlib.colors import to_rgb
    green = S.palette("green")
    for name in S.load_settings()["style"]["notebooks"]:
        if not name.startswith("04_non_academic"):
            continue
        style = S.load_style(name)
        assert green not in style["colors"]
        assert "c_green" not in style
        assert green not in plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for key, value in style.items():
            if key.endswith("_colors"):
                assert green not in value.values(), (name, key)
    np.testing.assert_allclose(to_rgb(S.NON_ACADEMIC_PALETTE[6]),
                               (np.asarray(to_rgb(S.palette("red"))) + 1) / 2,
                               atol=1 / 255)
    for mapping in (F.SECTOR_COLORS, F.PATENT_STATUS_COLORS, F.TRIAL_SECTOR_COLORS):
        assert green not in mapping.values()
        assert len(set(mapping.values())) == len(mapping)
    style = S.load_style("04_non_academic_panels")
    assert style["sector_colors"] == F.SECTOR_COLORS

    root = Path(__file__).resolve().parents[1] / "src"
    sources = {str(path): path.read_text() for path in root.glob("utils/data_analysis_04_*.py")}
    sources["shared_patent_utils"] = (root / "utils/shared_patent_utils.py").read_text()
    nb = json.loads((root / "data_analysis/04_non_academic.ipynb").read_text())
    transform = TransformerManager().transform_cell
    sources.update({cell["id"]: transform("".join(cell["source"])) for cell in nb["cells"]
                    if cell["cell_type"] == "code"})
    for name, source in sources.items():
        literals = {node.value.lower() for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert not {"green", "c_green", green.lower()} & literals, name


def test_active_plotting_sources_do_not_restore_legacy_palettes():
    root = Path(__file__).resolve().parents[1] / "src"
    obsolete = {"#b80c09", "#d4af37", "#6e8b3d", "#345995", "#8c5a9e",
                "#008c95", "#b56a3b", "#4f6d7a", "set3", "tab20", "ocean_r", "ylgnbu"}
    paths = [*root.glob("utils/data_analysis_*.py"), root / "utils/shared_patent_utils.py"]
    for path in paths:
        tree = ast.parse(path.read_text())
        literals = {node.value.lower() for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        colormaps = {node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert not obsolete & (literals | colormaps), path.name
        for node in ast.walk(tree):
            if (isinstance(node, ast.keyword) and node.arg in {"color", "colors", "cmap"}
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                assert node.value.value.lower() not in {
                    "red", "orange", "green", "blue", "yellow", "purple", "cyan", "magenta",
                    "r", "g", "b", "c", "m", "y", "tab:blue", "tab:orange",
                }, (path.name, node.lineno)
    for path in root.glob("data_analysis/*.ipynb"):
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"]).lower()
                assert not any(color in source for color in obsolete if color.startswith("#")), path.name


def test_icd_annotations_are_bar_centred_with_constant_padding_and_matching_legend():
    labels = [f"Chapter {n}" for n in range(17)]
    labels[0] = "IV Endocrine, nutritional & metabolic (E00-E90)"
    counts = pd.Series([67, 62, 35, 27, 23, 22, 15, 13, 11, 9, 8, 7, 5, 4, 3, 2], index=labels[:-1])
    mesh = pd.Series([48, 44, 25, 31, 21, 22, 18, 29, 4, 11, 7, 12, 3, 0, 13, 2, 6], index=labels)
    comparison = pd.concat([mesh.rename("MeSH"), counts.rename("RCDC")], axis=1).fillna(0)
    before = comparison.copy()
    counts_before = counts.copy()
    fig = F.trial_icd_comparison_figure(counts, comparison, total_trials=168,
                                      style=S.load_style("04_non_academic_01_clinical_trials"))
    for _ in range(2):
        F.finalize_figure(fig)
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["A", "B"]
    left, right = fig.axes
    assert left.get_xlim() == right.get_xlim()
    assert left.get_ylim() == right.get_ylim()
    assert left.yaxis_inverted()
    assert not right.get_yticklabels()
    ordered = comparison.sort_values("RCDC", ascending=False, kind="stable")
    assert [" ".join(label.get_text().split()) for label in left.get_yticklabels()] == ordered.index.tolist()
    assert "\n" in left.get_yticklabels()[0].get_text()
    np.testing.assert_array_equal([bar.get_width() for bar in left.patches], ordered.RCDC)
    np.testing.assert_array_equal([bar.get_width() for bar in right.containers[0]], ordered.MeSH)
    np.testing.assert_array_equal([bar.get_width() for bar in right.containers[1]], ordered.RCDC)
    for a, m, r in zip(left.patches, *right.containers):
        centre = lambda bar: bar.get_y() + bar.get_height() / 2
        np.testing.assert_allclose(centre(a), (centre(m) + centre(r)) / 2, atol=1e-12)
    assert left.texts[-1].get_text() == "0 (0%)"
    for ax in fig.axes:
        assert ax._left_title.get_size() == 24
        assert ax.xaxis.label.get_size() == ax.yaxis.label.get_size() == 17
        assert all(t.get_size() == 14 for t in ax.get_xticklabels() + ax.get_yticklabels())
        assert ax.xaxis.label.get_fontfamily() == ["Helvetica"]
        assert len(ax.patches) == len(ax.texts)
        visible = []
        for bar, annotation in zip(ax.patches, ax.texts):
            np.testing.assert_allclose(annotation.xy, [bar.get_width(), bar.get_y() + bar.get_height() / 2])
            assert annotation.get_position() == (5, 0)
            assert annotation.get_fontsize() == 13
            if not annotation.get_text():
                assert bar.get_width() == 0
                continue
            box = annotation.get_window_extent(renderer)
            centre = ax.transData.transform(annotation.xy)
            assert abs((box.y0 + box.y1) / 2 - centre[1]) < 1e-6
            assert box.x1 < ax.bbox.x1
            visible.append(box)
        assert not any(a.overlaps(b) for i, a in enumerate(visible) for b in visible[i + 1:])
        box = ax.get_tightbbox(renderer)
        assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1)
    assert {to_hex(bar.get_facecolor()).upper() for bar in fig.axes[0].patches} == {S.palette("red")}
    legend = fig.axes[1].get_legend()
    assert all(t.get_size() == 14 for t in legend.get_texts())
    for container, handle, label in zip(fig.axes[1].containers, legend.legend_handles, legend.get_texts()):
        color = S.palette("steel_blue" if label.get_text() == "MeSH" else "red")
        assert to_hex(handle.get_facecolor()).upper() == color
        assert all(to_hex(bar.get_facecolor()).upper() == color for bar in container)
    pd.testing.assert_frame_equal(comparison, before)
    pd.testing.assert_series_equal(counts, counts_before)
    caption = F._caption(fig, "ct_rcdc_icd")
    assert "168 included trials" in caption
    assert "share chapter order and count scale" in caption
    assert "MeSH terms (blue)" in caption and "RCDC tags (red)" in caption
    plt.close(fig)


def test_three_route_icd_comparison_uses_blue_and_gold_with_matching_legend():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "b8cd5be3")["source"])
    code = source[source.index("ax = axes[1]"):source.index("fig.tight_layout(w_pad=4)")]
    style = S.load_style("04_non_academic_01_clinical_trials")
    comparison = pd.DataFrame({"MeSH (shipped)": [48, 44, 0],
                               "MeSH (leaf)": [52, 40, 5], "RCDC": [67, 62, 0]},
                              index=["Chapter A", "Chapter B", "Chapter C"])
    original = comparison.copy(deep=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    namespace = dict(axes=axes, np=np, comp3=comparison, STYLE=style, set_title=S.set_title)
    exec(compile(code, "three_route_icd", "exec"), namespace)
    for _ in range(2):
        F.finalize_figure(fig)
        fig.canvas.draw()
    ax = axes[1]
    expected = S.palette("blue", "steel_blue", "cream")
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == list(comparison.columns)
    for container, handle, column, color in zip(ax.containers, legend.legend_handles,
                                               comparison.columns, expected):
        assert {to_hex(bar.get_facecolor()).upper() for bar in container} == {color}
        assert to_hex(handle.get_facecolor()).upper() == color
        np.testing.assert_array_equal([bar.get_width() for bar in container], comparison[column])
    assert "c_green" not in code
    assert "RCDC uses gold" in F._caption(fig, "ct_icd_three_axes")
    pd.testing.assert_frame_equal(comparison, original)
    plt.close(fig)


def test_trial_starts_panel_uses_yellow_and_preserves_study_type_colours():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "src/data_analysis/04_non_academic.ipynb").read_text())
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "7d4d906b")["source"])
    code = source[source.index("def plot_grouped_status"):source.index("# optional: save")]
    style = S.load_style("04_non_academic_01_clinical_trials")
    trials = pd.DataFrame({"status_stage": ["Planned", "Ongoing", "Ongoing", "Completed"],
                           "study_type": ["Interventional", "Interventional",
                                          "Observational", "Observational"],
                           "start_year": [2022, 2023, 2023, 2025]})
    original = trials.copy(deep=True)
    namespace = dict(plt=plt, np=np, STYLE=style, df_ct=trials,
                     STAGE_ORDER=["Planned", "Ongoing", "Completed"],
                     ANALYSIS_START_YEAR=2022, ANALYSIS_END_YEAR=2025,
                     set_title=S.set_title)
    exec(compile(code, "trial_starts_panel", "exec"), namespace)
    fig = namespace["fig"]
    for _ in range(2):
        F.finalize_figure(fig)
        fig.canvas.draw()
    left, right = fig.axes
    assert {to_hex(bar.get_facecolor()).upper() for bar in right.patches} == {S.palette("cream")}
    np.testing.assert_array_equal([bar.get_height() for bar in right.patches], [1, 2, 0, 1])
    legend = left.get_legend()
    for container, handle, label in zip(left.containers, legend.legend_handles, legend.get_texts()):
        expected = style["trial_type_colors"][label.get_text()]
        assert all(to_hex(bar.get_facecolor()).upper() == expected for bar in container)
        assert to_hex(handle.get_facecolor()).upper() == expected
    assert "c_green" not in code
    pd.testing.assert_frame_equal(trials, original)
    plt.close(fig)


def test_interactive_topic_plots_restyle_embedded_trace_colours_without_changing_values():
    import plotly.graph_objects as go
    from utils.data_analysis_02_content_bert_topic import style_plotly_figure
    fig = go.Figure([
        go.Scatter(x=[2013, 2025], y=[1, 2], name="Topic one", legendgroup="one",
                   mode="lines+markers", line_color="purple", marker_color="purple"),
        go.Scatter(x=[2013, 2025], y=[2, 3], name="Topic two", legendgroup="two",
                   line_color="orange"),
        go.Scatter(x=[2013, 2025], y=[3, 4], name="Topic one", legendgroup="one",
                   line_color="purple"),
    ])
    before = [(list(t.x), list(t.y)) for t in fig.data]
    style_plotly_figure(fig)
    assert list(fig.layout.colorway) == S.PALETTE
    assert [(list(t.x), list(t.y)) for t in fig.data] == before
    assert [t.line.color.upper() for t in fig.data] == S.palette("red", "cream", "red")
    assert [t.marker.color.upper() for t in fig.data] == S.palette("red", "cream", "red")
    heat = go.Figure(go.Heatmap(z=[[0, 2], [3, 4]], colorscale="Viridis"))
    style_plotly_figure(heat)
    np.testing.assert_array_equal(heat.data[0].z, [[0, 2], [3, 4]])
    assert [c for _, c in heat.data[0].colorscale] == S.palette("light_blue", "cream", "red")
    assert heat.layout.coloraxis.colorscale == heat.data[0].colorscale
