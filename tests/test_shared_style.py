import matplotlib.pyplot as plt
import pytest

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
