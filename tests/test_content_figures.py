"""Content denominators, missing years and grouped layout, without topic fitting."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from utils import data_analysis_02_content_panels as panels
from utils.data_analysis_02_content_window import write_topic_window_provenance
from utils.shared_style import finalize_figure, load_style


class ContentFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_style('02_content')

    def tearDown(self):
        plt.close('all')

    def fixture(self):
        corpus = pd.DataFrame({'id': ['a', 'b', 'c'], 'year': [2024, 2024, 2025]})
        long = pd.DataFrame({'id': ['a', 'a', 'b', 'c'],
                             'year': [2024, 2024, 2024, 2025],
                             'category': ['X', 'Y', 'Y', 'X']})
        return {'corpus': corpus, 'years': panels.FLOW_YEARS,
                'for': panels._flow_block(long, 'category', 'for'),
                'rcdc': panels._flow_block(long, 'category', 'rcdc'),
                'topics': {'available': False, 'note': 'Missing cache', 'source': None}}

    def test_fractional_denominator_and_unclassified_year(self):
        D = self.fixture()
        self.assertAlmostEqual(D['for']['share'].loc[2024, 'X'], 100 / 3)
        self.assertAlmostEqual(D['rcdc']['share'].loc[2024, 'X'], 25)
        self.assertTrue(D['for']['share'].loc[2013].isna().all())
        self.assertTrue(D['for']['band'].loc[2013].isna().all())
        self.assertEqual(D['for']['n_papers'].loc[2013], 0)
        metrics = panels.annual_vocabulary_metrics(D)
        first = metrics.query('year == 2013 and vocabulary == "for"').iloc[0]
        self.assertTrue(np.isnan(first.effective_categories))
        self.assertTrue(np.isnan(first.coverage_pct))
        self.assertEqual(first.cumulative_categories, 0)
        last = metrics.query('year == 2025 and vocabulary == "for"').iloc[0]
        self.assertEqual(last.effective_categories, 1)
        self.assertEqual(last.coverage_pct, 100)

    def test_leading_change_table_has_ten_fields_independent_of_plot_selection(self):
        fields = [f'Field {i}' for i in range(12)]
        weights = pd.DataFrame([range(1, 13)] * len(panels.FLOW_YEARS),
                               index=panels.FLOW_YEARS, columns=fields)
        share = weights.div(weights.sum(axis=1), axis=0) * 100
        block = {'weights': weights, 'share': share, 'keep': fields[-8:]}
        table = panels.leading_category_change_table(block)
        self.assertEqual(len(table), 10)
        self.assertEqual(set(table.index), set(fields[2:]))
        self.assertEqual(table.shape[1], 6)
        self.assertEqual(table.loc['Field 11', 'rank_2025'], 1)
        self.assertEqual(block['keep'], fields[-8:])
        self.assertEqual(len(panels.leading_category_change_table(block, n=20)), 12)

    def test_unknown_early_rank_is_not_filled_with_zero(self):
        D = self.fixture()
        change = panels.category_change_table(D)
        self.assertTrue(change['share_2013_%'].isna().all())
        self.assertTrue(change['rank_2013'].isna().all())
        self.assertTrue(change['delta_pp'].isna().all())

    def test_topic_utility_schema_and_distinct_label_ids(self):
        corpus = pd.DataFrame({'id': ['a', 'b'], 'year': [2013, 2025]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bertopic_document_topic_assignments.csv'
            pd.DataFrame({'showcase_plus_id': ['a', 'b'], 'analysis_year': [2013, 2025],
                          'topic': [1, 2]}).to_csv(path, index=False)
            write_topic_window_provenance(path, [2013, 2025])
            with patch.object(panels, 'TOPIC_SOURCES', (path,)):
                long, actual, _ = panels.load_topic_year_matrix(corpus)
            self.assertEqual(actual, path)
            self.assertEqual(long.topic_label.tolist(), ['Topic 1', 'Topic 2'])
        self.assertNotEqual(panels._topic_label('T1: same / words'),
                            panels._topic_label('T2: same / words'))
        displayed = panels._topic_legend_labels(['T1: same / words', 'T2: same / words',
                                                 'T3: different / terms'])
        self.assertEqual(displayed['T1: same / words'], 'same / words [T1]')
        self.assertEqual(displayed['T2: same / words'], 'same / words [T2]')
        self.assertEqual(displayed['T3: different / terms'], 'different / terms')

    def test_selected_topic_matrix_cannot_supply_the_production_denominator(self):
        corpus = pd.DataFrame({'id': ['a'], 'year': [2025]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bertopic_topic_year_proportions_selected.csv'
            pd.DataFrame({'year': [2025], 'Topic 1': [.2]}).to_csv(path, index=False)
            write_topic_window_provenance(path, [2025])
            with patch.object(panels, 'TOPIC_SOURCES', (path,)):
                with self.assertRaisesRegex(ValueError, 'per-paper topic-results'):
                    panels.load_topic_year_matrix(corpus)

    def test_plots_read_the_exact_cache_selected_by_the_model_stage(self):
        corpus = pd.DataFrame({'id': ['a'], 'year': [2025]})
        with tempfile.TemporaryDirectory() as folder:
            selected = Path(folder) / 'showcase_plus_id_topics.csv'
            stale = Path(folder) / 'bertopic_document_topic_assignments.csv'
            pd.DataFrame({'id': ['a'], 'topics': ['T0: selected']}).to_csv(selected, index=False)
            write_topic_window_provenance(selected, [2025])
            stale.write_text('unverified,stale\na,b\n')
            with patch.object(panels, 'TOPIC_SOURCES', (stale, selected)):
                actual, source, _ = panels.load_topic_year_matrix(corpus, topic_results=selected)
            self.assertEqual(source, selected)
            self.assertEqual(actual.topic_label.tolist(), ['T0: selected'])

    def test_grouped_layout_and_selected_topics_keep_full_denominator(self):
        D = self.fixture()
        topic_long = pd.DataFrame({'id': ['a', 'b', 'c'], 'year': [2024, 2024, 2025],
                                   'topic_label': ['T1: X', 'T2: Y', 'T1: X']})
        block = panels._flow_block(topic_long, 'topic_label', 'topics')
        block['from_assignments'] = True
        # Deliberately draw one of two topics: 2024 must retain 50%, not become 100%.
        block['keep'] = ['T1: X']
        D['topics'] = block
        fig = panels.figure_main(D, save=False)
        A, B, C = fig.axes
        self.assertAlmostEqual(A.get_position().y0, B.get_position().y0)
        self.assertLess(C.get_position().y1, A.get_position().y0)
        self.assertGreater(C.get_position().width, A.get_position().width)
        self.assertEqual(block['share'].loc[2024, 'T1: X'], 50)
        self.assertIn('75.0%', C.texts[0].get_text())
        self.assertEqual([ax.get_title(loc='left') for ax in (A, B, C)], list('ABC'))
        self.assertEqual(A.get_legend()._ncols, 3)
        self.assertEqual(B.get_legend()._ncols, 3)
        self.assertEqual(panels.OTHER_COLOR, '#FFFFFF')
        self.assertIn('Fields of Research', A.get_ylabel())
        self.assertIn('RCDC', B.get_ylabel())
        finalize_figure(fig)
        fig.canvas.draw()
        self.assertFalse(C._left_title.get_window_extent().overlaps(
            C.texts[0].get_window_extent()))

    def test_category_heatmap_headings_are_letters_only(self):
        fig = panels.figure_si_category_changes(self.fixture(), save=False)
        finalize_figure(fig)
        fig.canvas.draw()
        self.assertEqual([ax.get_title(loc='left') for ax in fig.axes], ['A', 'B'])
        self.assertTrue(all(not ax.get_title() and not ax.get_title(loc='right')
                            for ax in fig.axes))
        self.assertIn('Fields of Research', fig.axes[0].get_ylabel())
        self.assertIn('RCDC', fig.axes[1].get_ylabel())

    def test_coverage_uses_letter_headings_and_large_shared_legend(self):
        fig = panels.figure_si_breadth_coverage(self.fixture(), save=False)
        finalize_figure(fig)
        fig.canvas.draw()
        self.assertEqual([ax.get_title(loc='left') for ax in fig.axes], list('ABCD'))
        legend = fig.legends[0]
        self.assertEqual(legend._ncols, len(legend.get_texts()))
        self.assertGreaterEqual(legend.get_texts()[0].get_fontsize(), 14)
        self.assertEqual(legend.get_frame().get_edgecolor(), (0, 0, 0, 1))
        self.assertTrue(all(not legend.get_window_extent().overlaps(ax.get_tightbbox())
                            for ax in fig.axes))

    def test_supplement_tables_preserve_complete_distributions(self):
        D = self.fixture()
        table = panels.category_share_table(D)
        self.assertEqual(set(table.vocabulary), {'for', 'rcdc'})
        valid = table.query('year == 2024').groupby('vocabulary').share_pct.sum()
        self.assertTrue(np.allclose(valid, 100))
        for figure in (panels.figure_si_category_changes(D, save=False),
                       panels.figure_si_breadth_coverage(D, save=False),
                       panels.figure_si_rank_flow(D, save=False)):
            figure.canvas.draw()

    def test_unverified_diagnostics_are_omitted_and_verified_metrics_render(self):
        D = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            D['topics'] = {'available': True, 'source': str(folder / 'assignments.csv')}
            frames = {
                'seed_grid_runs.csv': pd.DataFrame({
                    'parameter_index': [0, 0, 1, 1], 'seed': [1, 2, 1, 2],
                    'n_neighbors': [15, 15, 25, 25], 'min_cluster_size': [25, 25, 35, 35],
                    'min_samples': [5, 5, 10, 10], 'status': ['ok'] * 4,
                    'coherence_cv': [.5, .6, .65, .7], 'n_topics': [8, 10, 9, 10]}),
                'bertopic_seed_robustness_summary.csv': pd.DataFrame({'parameter_index': [1, 0]}),
                'bertopic_seed_pair_stability.csv': pd.DataFrame({
                    'parameter_index': [0, 1], 'ari': [.5, .7], 'nmi': [.6, .8]}),
                'bertopic_final_cluster_persistence.csv': pd.DataFrame({
                    'raw_cluster_size': [20, 40, 80], 'cluster_persistence': [.2, .3, .4]}),
            }
            for name, frame in frames.items():
                frame.to_csv(folder / name, index=False)
            self.assertIsNone(panels.load_topic_diagnostics(D))
            self.assertIsNone(panels.figure_si_topic_robustness(D, save=False))
            for name in frames:
                write_topic_window_provenance(folder / name, [2013, 2025])
            figure = panels.figure_si_topic_robustness(D, save=False)
            self.assertEqual(len(figure.axes), 4)
            figure.canvas.draw()


if __name__ == '__main__':
    unittest.main()
