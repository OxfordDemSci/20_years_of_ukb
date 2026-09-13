import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from utils import shared_style


def test_marker_helpers_apply_configured_scale():
    style = {"marker_size": 9.5, "dot_marker_area": 92}

    assert shared_style.marker_size(style) == 9.5
    assert shared_style.marker_size(style, scale=0.8) == pytest.approx(7.6)
    assert shared_style.marker_area(style) == 92
    assert shared_style.marker_area(style, scale=0.25) == 23


def test_apply_style_sets_matplotlib_default_marker_size():
    style = shared_style.load_style("05_author_characteristics", activate=False)

    with plt.rc_context():
        shared_style.apply_style(style)
        assert plt.rcParams["lines.markersize"] == style["marker_size"]


def test_academic_impact_colormap_uses_project_palette_endpoints():
    cmap = shared_style.academic_impact_colormap()

    assert to_hex(cmap(0.0)) == shared_style.PALETTE_COLORS["cream"].lower()
    assert to_hex(cmap(1.0)) == shared_style.PALETTE_COLORS["navy"].lower()


def test_svg_export_has_no_trailing_whitespace(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    style = {
        "save": True,
        "dpi": 72,
        "formats": ["svg"],
        "savedir": tmp_path,
    }

    try:
        [saved] = shared_style.savefig(figure, "test_figure", style)
    finally:
        plt.close(figure)

    assert not any(line.endswith((" ", "\t")) for line in saved.read_text().splitlines())
