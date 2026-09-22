"""Figure captions are notebook output, never artwork or sidecar text files."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from utils import shared_paths as P
from utils import shared_style as S
from utils.shared_figure_captions import caption_for_name, growth_captions, suggest_caption


@pytest.mark.parametrize('name', [
    P.MAIN_FIGURE_STEMS[3], P.MAIN_FIGURE_STEMS[4],
    P.MAIN_FIGURE_STEMS[5], P.MAIN_FIGURE_STEMS[6],
    '02_04_supplementary_figure_03_topic_robustness',
    '03_03_supplementary_figure_02_top_decile_share',
    '04_03_supplementary_figure_02_clinical_trials',
    '05_07_supplementary_figure_06_networks',
])
def test_registered_figures_have_detailed_captions(name):
    assert len(caption_for_name(name)) > 100


def test_growth_captions_retain_all_three_figures_and_snapshot():
    captions = growth_captions(2013, 2025, pd.Timestamp('2026-01-15'))
    assert set(captions) == {'main', 'overlap', 'annual'}
    assert '15 January 2026' in captions['overlap']


def test_export_then_display_orders_image_paths_and_caption(monkeypatch, tmp_path, capsys):
    import IPython.display
    events = []
    monkeypatch.setattr(IPython.display, 'display',
                        lambda obj: events.append((obj, capsys.readouterr().out)))
    fig, ax = plt.subplots()
    try:
        ax.set_xlabel('Publication year')
        ax.set_ylabel('Publications')
        paths = S.savefig(fig, 'example', {'save': True, 'savedir': tmp_path,
                                         'formats': ['pdf'], 'dpi': 72})
        S.display_figure(fig)
        assert events[0] == (fig, '')
        assert events[1][0].data.startswith('**Suggested caption:**')
        assert 'Publications by Publication year' in events[1][0].data
        assert 'Saved: example.pdf' in events[1][1]
        assert str(tmp_path) not in events[1][1]
        assert list(tmp_path.iterdir()) == paths
        assert not fig.texts
    finally:
        plt.close(fig)


def test_caption_is_available_even_when_export_is_disabled():
    fig, _ = plt.subplots()
    try:
        assert S.savefig(fig, P.MAIN_FIGURE_STEMS[5], {'save': False}) == []
        assert suggest_caption(fig) == caption_for_name(P.MAIN_FIGURE_STEMS[5])
    finally:
        plt.close(fig)


@pytest.mark.parametrize('letter', ['a.', '(B)', 'C:', 'D'])
def test_fallback_caption_does_not_repeat_panel_letters_as_descriptions(letter):
    fig, ax = plt.subplots()
    try:
        S.set_title(ax, letter)
        ax.set_xlabel('Publication year')
        ax.set_ylabel('Publications')
        assert 'Publications by Publication year' in suggest_caption(fig)
    finally:
        plt.close(fig)


def test_legacy_show_uses_shared_display_and_closes_figures(monkeypatch):
    plt.close('all')
    figures = [plt.figure(), plt.figure()]
    shown = []
    monkeypatch.setattr(S, 'display_figure', lambda fig: shown.append(fig))
    S.show_figures()
    assert shown == figures
    assert not plt.get_fignums()


def test_embedded_patent_heatmap_does_not_display_or_close_its_parent(monkeypatch):
    from utils import shared_patent_utils as patents
    fig, ax = plt.subplots()
    shown = []
    monkeypatch.setattr(patents, 'show_figures', lambda: shown.append(True))
    try:
        patents.plot_country_topic_heatmap(pd.DataFrame({'GB': [1, 2]}, index=['X', 'Y']), ax=ax)
        assert not shown
        assert plt.fignum_exists(fig.number)
    finally:
        plt.close(fig)
