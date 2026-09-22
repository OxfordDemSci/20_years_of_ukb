import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from utils import shared_style


@pytest.mark.parametrize('text,expected', [
    ('A', 'A'), ('b.', 'B'), ('(c)', 'C'),
    ('A  Classification coverage', 'A'), ('d. Clinical trials', 'D'),
    ('B: Category diversity', 'B'), ('A title', 'A'),
    ('Annual growth', ''), ('RCDC categories', ''), ('', ''),
])
def test_panel_titles_contain_only_explicit_letters(text, expected):
    assert shared_style.panel_title_letter(text) == expected


def test_facet_ylabel_wraps_without_dropping_field_names():
    fig, ax = plt.subplots()
    try:
        label = 'Cardiovascular Medicine and Haematology'
        shared_style.facet_ylabel(ax, label)
        assert '\n' in ax.get_ylabel()
        assert ax.get_ylabel().replace('\n', ' ') == label
    finally:
        plt.close(fig)


def test_title_helpers_keep_panel_letters_and_drop_prose():
    fig, ax = plt.subplots()
    try:
        shared_style.set_title(ax, 'B  Research categories', fontsize=20)
        assert ax.get_title(loc='left') == 'B'
        shared_style.set_title(ax, 'Legacy helper description', fontsize=9)
        assert ax.get_title(loc='left') == 'B'
        assert ax._left_title.get_fontsize() == 20
        shared_style.set_figure_title(fig, 'Overall descriptive heading')
        assert fig._suptitle.get_text() == ''
        shared_style.set_title(ax, '')
        assert ax.get_title(loc='left') == ''
    finally:
        plt.close(fig)


def test_finalization_catches_native_and_inset_titles_without_changing_labels():
    fig, ax = plt.subplots()
    try:
        ax.set_title('A  Coverage', loc='left')
        ax.set_title('Legacy centre title', loc='center')
        ax.set_title('Legacy right title', loc='right')
        inset = ax.inset_axes([.2, .2, .3, .3])
        inset.set_title('Inset description')
        fig.suptitle('Figure-wide heading')
        ax.set_ylabel('Publications classified (%)')
        note = ax.text(.5, .5, 'Keep data annotations')
        shared_style.finalize_figure(fig)
        shared_style.finalize_figure(fig)
        assert ax.get_title(loc='left') == 'A'
        assert not ax.get_title(loc='center')
        assert not ax.get_title(loc='right')
        assert not inset.get_title(loc='left')
        assert not fig._suptitle.get_text()
        assert ax.get_ylabel() == 'Publications classified (%)'
        assert note.get_text() == 'Keep data annotations'
    finally:
        plt.close(fig)


def test_display_figure_posts_relative_paths_then_caption_below_figure(monkeypatch, capsys):
    import IPython.display
    from utils import shared_paths as P
    events = []
    monkeypatch.setattr(IPython.display, "display",
                        lambda obj: events.append((obj, capsys.readouterr().out)))
    fig = plt.figure()
    try:
        paths = [P.ROOT / "output/figures/example.png", P.ROOT / "output/figures/example.pdf"]
        shared_style.display_figure(fig, paths, "A, Example caption.")
        assert events[0] == (fig, "")
        assert events[1][0].data == "**Suggested caption:** A, Example caption."
        assert "output/figures/example.png" in events[1][1]
        assert "output/figures/example.pdf" in events[1][1]
        assert str(P.ROOT) not in events[1][1]
    finally:
        plt.close(fig)


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


def test_legacy_svg_and_json_requests_export_pdf_only(tmp_path):
    figure, axis = plt.subplots(figsize=(1, 1))
    axis.plot([0, 1], [0, 1])
    style = {
        "save": True,
        "dpi": 72,
        "formats": ["svg", "json"],
        "savedir": tmp_path,
    }

    try:
        [saved] = shared_style.savefig(figure, "test_figure", style)
    finally:
        plt.close(figure)

    assert saved.suffix == ".pdf"
    assert list(tmp_path.iterdir()) == [saved]


def _grid_visible(ax, axis):
    """(major, minor) gridline visibility on one axis of `ax`."""
    target = getattr(ax, f"{axis}axis")
    return (
        any(tick.gridline.get_visible() for tick in target.get_major_ticks()),
        any(tick.gridline.get_visible() for tick in target.get_minor_ticks()),
    )


def test_grid_on_leaves_log_axes_bare():
    _, ax = plt.subplots()
    ax.plot([1, 10], [1, 100])
    ax.set_xscale("log")

    shared_style.grid_on(ax)

    assert _grid_visible(ax, "x") == (False, False)
    assert _grid_visible(ax, "y")[0] is True
    plt.close("all")


def test_grid_on_clears_a_grid_drawn_before_the_log_scale():
    _, ax = plt.subplots()
    ax.plot([1, 10], [1, 100])

    shared_style.grid_on(ax)
    ax.set_yscale("log")
    shared_style.grid_on(ax)

    assert _grid_visible(ax, "y") == (False, False)
    assert _grid_visible(ax, "x")[0] is True
    plt.close("all")


def test_grid_on_log_flag_forces_the_grid_back_on():
    _, ax = plt.subplots()
    ax.plot([1, 10], [1, 100])
    ax.set_xscale("log")

    shared_style.grid_on(ax, log=True)

    assert _grid_visible(ax, "x") == (True, True)
    plt.close("all")


def test_style_axis_inherits_the_log_grid_rule():
    style = shared_style.load_style("05_author_characteristics", activate=False)
    _, ax = plt.subplots()
    ax.plot([1, 10], [1, 100])
    ax.set_xscale("log")
    ax.set_yscale("log")

    shared_style.style_axis(ax, style)

    assert _grid_visible(ax, "x") == (False, False)
    assert _grid_visible(ax, "y") == (False, False)
    plt.close("all")
