"""Manuscript main-figure numbers remain distinct from analysis and SI order."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import shared_paths as P


class MainFigureNumberingTests(unittest.TestCase):
    def test_manuscript_stems_match_the_draft_order(self):
        self.assertEqual(
            P.MAIN_FIGURE_STEMS,
            {
                1: "figure_01_evolution_and_research_impact",
                2: "01_01_figure_02_growth_and_reach",
                3: "05_01_figure_03_author_characteristics",
                4: "02_01_figure_04_content_composition",
                5: "03_01_figure_05_academic_impact",
                6: "04_01_figure_06_non_academic_reach",
            },
        )

    def test_each_active_generator_uses_the_shared_number(self):
        sources = {
            2: ROOT / "src/data_analysis/01_growth.ipynb",
            3: ROOT / "src/utils/data_analysis_05_author_plots.py",
            4: ROOT / "src/utils/data_analysis_02_content_panels.py",
            5: ROOT / "src/utils/data_analysis_03_academic_impact_panels.py",
            6: ROOT / "src/utils/data_analysis_04_non_academic_panels.py",
        }
        for number, path in sources.items():
            with self.subTest(number=number, path=path.name):
                self.assertIn(
                    f"P.MAIN_FIGURE_STEMS[{number}]",
                    path.read_text(encoding="utf-8"),
                )

    def test_previous_local_main_figure_numbers_are_not_active(self):
        retired = {
            "01_01_figure_01_growth_and_reach",
            "02_01_figure_01_content_composition",
            "03_01_figure_01_academic_impact",
            "04_01_figure_01_non_academic_reach",
            "05_01_figure_01_author_characteristics",
        }
        active = [
            ROOT / "src/data_analysis/01_growth.ipynb",
            ROOT / "src/data_analysis/02_content.ipynb",
            ROOT / "src/data_analysis/03_academic_impact.ipynb",
            ROOT / "src/data_analysis/04_non_academic.ipynb",
            ROOT / "src/data_analysis/05_author_characteristics.ipynb",
            *sorted((ROOT / "src/utils").glob("data_analysis_0[1-5]*.py")),
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in active)
        for stem in retired:
            with self.subTest(stem=stem):
                self.assertNotIn(stem, combined)

    def test_supplementary_stems_remain_analysis_local(self):
        expected = {
            "01_02_supplementary_figure_01_indicator_overlap",
            "02_02_supplementary_figure_01_category_composition",
            "03_02_supplementary_figure_01_impact_map_full",
            "04_02_supplementary_figure_01_patents",
            "05_02_supplementary_figure_01_author_metrics",
        }
        active_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in [
                ROOT / "src/data_analysis/01_growth.ipynb",
                ROOT / "src/data_analysis/02_content.ipynb",
                ROOT / "src/data_analysis/03_academic_impact.ipynb",
                ROOT / "src/utils/data_analysis_04_non_academic_panels.py",
                ROOT / "src/utils/data_analysis_05_author_plots.py",
            ]
        )
        for stem in expected:
            with self.subTest(stem=stem):
                self.assertIn(stem, active_text)


if __name__ == "__main__":
    unittest.main()
