"""The consolidated academic-impact notebook retains every analysis phase."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "src" / "data_analysis"
NOTEBOOK = NOTEBOOK_DIR / "03_academic_impact.ipynb"
ARCHIVE = NOTEBOOK_DIR / "_archived"

SOURCE_CELL_COUNTS = {
    "03_academic_impact_01_for_analysis.ipynb": 40,
    "03_academic_impact_02_citation.ipynb": 10,
    "03_academic_impact_02_citation_extra.ipynb": 23,
    "03_academic_impact_99_all.ipynb": 21,
}

# These cells were deliberately rewritten only to turn notebook-to-notebook language
# into part-to-part language, correct stale paths/windows, and rename the panel style.
# Five small-multiple cells also move field names from titles to y-axis labels.
EXPECTED_REWRITTEN_CELL_IDS = {
    "d7b747de",
    "9817cf4f",
    "610b0ead",
    "a686192b",
    "35f087c3",
    "879TU0cIRHhD",
    "a23ccc79",
    "ccbdaad6",
    "72d9d8e6",
    "6f31bc80",
    "2e4dd06b",
    "f8c73484",
    "28def57b",
    "0888db1e",
    "553e3e0e",
    "97df02fc",
    "5fcf165b",
    "30993265",  # Import the shared wrapped facet-label helper.
}


class AcademicImpactConsolidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_only_one_academic_impact_notebook_is_active(self):
        active = sorted(path.name for path in NOTEBOOK_DIR.glob("03*.ipynb"))
        self.assertEqual(active, [NOTEBOOK.name])

    def test_all_source_notebooks_are_archived_and_accounted_for(self):
        provenance = self.notebook["metadata"]["ukb_consolidation"]
        self.assertEqual(provenance["source_cell_counts"], SOURCE_CELL_COUNTS)
        self.assertEqual(provenance["source_cells_total"], sum(SOURCE_CELL_COUNTS.values()))
        for source, count in SOURCE_CELL_COUNTS.items():
            archived = ARCHIVE / source.replace(".ipynb", "_pre_consolidation.ipynb")
            self.assertTrue(archived.is_file(), archived)
            payload = json.loads(archived.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["cells"]), count)

    def test_every_archived_cell_is_retained_in_source_order(self):
        combined_by_id = {cell["id"]: cell for cell in self.notebook["cells"]}
        combined_positions = {
            cell["id"]: position
            for position, cell in enumerate(self.notebook["cells"])
        }
        previous_position = -1
        rewritten_ids = set()

        for source in SOURCE_CELL_COUNTS:
            archived_path = ARCHIVE / source.replace(
                ".ipynb", "_pre_consolidation.ipynb"
            )
            archived = json.loads(archived_path.read_text(encoding="utf-8"))
            for archived_cell in archived["cells"]:
                cell_id = archived_cell["id"]
                self.assertIn(cell_id, combined_by_id)
                combined_cell = combined_by_id[cell_id]
                self.assertEqual(combined_cell["cell_type"], archived_cell["cell_type"])
                self.assertGreater(combined_positions[cell_id], previous_position)
                previous_position = combined_positions[cell_id]
                if combined_cell.get("source") != archived_cell.get("source"):
                    rewritten_ids.add(cell_id)

        self.assertEqual(rewritten_ids, EXPECTED_REWRITTEN_CELL_IDS)

    def test_combined_notebook_is_clean_and_has_isolated_parts(self):
        cells = self.notebook["cells"]
        self.assertEqual(len(cells), sum(SOURCE_CELL_COUNTS.values()) + 7)
        self.assertEqual(len({cell["id"] for cell in cells}), len(cells))
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        self.assertTrue(all(cell.get("execution_count") is None
                            or isinstance(cell["execution_count"], int) for cell in code_cells))
        self.assertFalse(any(output.get("output_type") == "error"
                             for cell in code_cells for output in cell.get("outputs", [])))
        self.assertEqual(self.source.count('run_line_magic("reset", "-f")'), 3)
        headings = [
            "# Part I: Fields of Research",
            "# Part II: Author-level citation impact",
            "# Part III: Showcase+ citation and disciplinary figures",
            "# Part IV: Assembled publication panels",
        ]
        positions = [self.source.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))

    def test_figure_and_table_contracts_remain_present(self):
        required_markers = [
            '"01_whole_db_trends_', '"02_ukbb_contribution_',
            '"03_growth_speed_', '"04_activity_index_', '"05_impact_map_',
            '"06_citation_impact_', '"07_citation_share_',
            '"09_footprint_quality_median_', '"09a_footprint_quality_cutoffs_',
            '"10_top_decile_pool_', '"11_fastest_growing_',
            '"fig3a_author_entry_cohort_academic_impact_profile"',
            '"fig_author_influence_fingerprint_heatmap"',
            '"author_summary_with_impact_metrics.csv"',
            '"fig01_disciplinary_composition"',
            '"fig02_cumulative_output_and_citation_stock"',
            '"fig04_field_citation_impact"', '"fig05_citation_concentration"',
            '"fig06_age_adjusted_citation_cohorts"',
            '"fig07_normalised_citation_performance"',
            "NP.figure_main(D)", "NP.figure_si_impact_map(D)",
            "NP.figure_si_top_decile(D)", "NP.figure_si_growth(D)",
            '"panel_manifest.csv"',
        ]
        for marker in required_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.source)


if __name__ == "__main__":
    unittest.main()
