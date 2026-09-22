"""The consolidated content notebook retains analysis and panel assembly."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "src" / "data_analysis"
NOTEBOOK = NOTEBOOK_DIR / "02_content.ipynb"
ARCHIVE = NOTEBOOK_DIR / "_archived"

SOURCE_CELL_COUNTS = {
    "02_content.ipynb": 18,
    "02_content_99_all.ipynb": 17,
}
ARCHIVE_NAMES = {
    "02_content.ipynb": "02_content_pre_panel_consolidation.ipynb",
    "02_content_99_all.ipynb": "02_content_99_all_pre_consolidation.ipynb",
}
EXPECTED_REWRITTEN_CELL_IDS = {
    "8e9e35a7",
    "27693594",
    "31762a1f",
    "40d763b2",
    "cef02f3e",
    "92913072",
    "c4404f53",
    "3af229bb",
    "c98ee4e2",
    "08705997",
}


class ContentConsolidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_only_one_content_notebook_is_active(self):
        active = sorted(path.name for path in NOTEBOOK_DIR.glob("02_content*.ipynb"))
        self.assertEqual(active, [NOTEBOOK.name])

    def test_both_source_notebooks_are_archived_and_accounted_for(self):
        provenance = self.notebook["metadata"]["ukb_consolidation"]
        self.assertEqual(provenance["source_cell_counts"], SOURCE_CELL_COUNTS)
        self.assertEqual(provenance["source_cells_total"], sum(SOURCE_CELL_COUNTS.values()))
        self.assertEqual(provenance["orchestration_cells"], 2)
        for source_name, count in SOURCE_CELL_COUNTS.items():
            archived = ARCHIVE / ARCHIVE_NAMES[source_name]
            self.assertTrue(archived.is_file(), archived)
            payload = json.loads(archived.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["cells"]), count)

    def test_every_archived_cell_is_retained_in_source_order(self):
        combined_by_id = {cell["id"]: cell for cell in self.notebook["cells"]}
        positions = {
            cell["id"]: position for position, cell in enumerate(self.notebook["cells"])
        }
        previous_position = -1
        rewritten_ids = set()

        for source_name in SOURCE_CELL_COUNTS:
            archived = json.loads(
                (ARCHIVE / ARCHIVE_NAMES[source_name]).read_text(encoding="utf-8")
            )
            for archived_cell in archived["cells"]:
                cell_id = archived_cell["id"]
                self.assertIn(cell_id, combined_by_id)
                combined_cell = combined_by_id[cell_id]
                self.assertEqual(combined_cell["cell_type"], archived_cell["cell_type"])
                self.assertGreater(positions[cell_id], previous_position)
                previous_position = positions[cell_id]
                if combined_cell.get("source") != archived_cell.get("source"):
                    rewritten_ids.add(cell_id)

        self.assertEqual(rewritten_ids, EXPECTED_REWRITTEN_CELL_IDS)

    def test_notebook_is_clean_and_parts_are_isolated(self):
        cells = self.notebook["cells"]
        self.assertEqual(len(cells), sum(SOURCE_CELL_COUNTS.values()) + 2)
        self.assertEqual(len({cell["id"] for cell in cells}), len(cells))
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(cell.get("outputs") == [] for cell in code_cells))
        self.assertEqual(self.source.count('run_line_magic("reset", "-f")'), 1)
        headings = [
            "# Part I: UK Biobank research content, 2013–2025",
            "# Part II: Publication-panel selection and assembly",
        ]
        heading_positions = [self.source.index(heading) for heading in headings]
        self.assertEqual(heading_positions, sorted(heading_positions))

    def test_figure_and_table_contracts_are_retained(self):
        required_markers = [
            "T.ensure_topic_results(",
            "G.export_growth_tables(",
            "C.figure_main(D, save=False)",
            "C.figure_si_category_changes(D, save=False)",
            "C.figure_si_breadth_coverage(D, save=False)",
            "C.figure_si_topic_robustness(D, save=False)",
            "E.export_content_tables(D, ARTIFACTS)",
            '"02_01_figure_01_content_composition"',
            '"02_02_supplementary_figure_01_category_composition"',
            '"02_03_supplementary_figure_02_coverage_and_breadth"',
            '"02_04_supplementary_figure_03_topic_robustness"',
            "CP.build_panel_data()",
            "CP.band_summary(D)",
            "CP.MAIN_CAPTION.items()",
            "CP.SI_CAPTIONS.items()",
            "CP.figure_main(D)",
            "CP.figure_si_rank_flow(D)",
            '"panel_band_rules.csv"',
            '"panel_selection.csv"',
            '("topics", "topics")',
            '("for", "for_l4")',
            '("rcdc", "rcdc")',
            'f"panel_{stem}_year_share.csv"',
            'f"panel_{stem}_drawn_bands.csv"',
            'f"panel_{stem}_start_vs_end.csv"',
            '"panel_manifest.csv"',
            'ARTIFACTS.save_manifest("02_content_manifest.csv")',
        ]
        for marker in required_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.source)

    def test_part_two_uses_shared_paths_and_style_without_notebook_magics(self):
        self.assertNotIn("%load_ext", self.source)
        self.assertNotIn("%autoreload", self.source)
        self.assertIn('STYLE = load_style("02_content")', self.source)
        self.assertIn("P.TABLE_CONTENT", self.source)


if __name__ == "__main__":
    unittest.main()
