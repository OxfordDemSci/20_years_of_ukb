"""The content notebook preserves distinct analyses without duplicate execution."""

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "src" / "data_analysis"
NOTEBOOK = NOTEBOOK_DIR / "02_content.ipynb"
ARCHIVE = NOTEBOOK_DIR / "_archived"
ARCHIVE_NAMES = {
    "02_content.ipynb": "02_content_pre_panel_consolidation.ipynb",
    "02_content_99_all.ipynb": "02_content_99_all_pre_consolidation.ipynb",
}
SOURCE_CELL_COUNTS = {
    "02_content.ipynb": 18,
    "02_content_99_all.ipynb": 17,
}


class ContentConsolidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join("".join(cell["source"])
                             for cell in cls.notebook["cells"]
                             if cell["cell_type"] == "code")
        cls.source = "\n".join("".join(cell["source"])
                               for cell in cls.notebook["cells"])
        cls.calls = [node.func.attr for node in ast.walk(ast.parse(cls.code))
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]

    def test_only_one_content_notebook_is_active(self):
        self.assertEqual(sorted(path.name for path in NOTEBOOK_DIR.glob("02_content*.ipynb")),
                         [NOTEBOOK.name])

    def test_archived_sources_and_removed_cells_are_accounted_for(self):
        provenance = self.notebook["metadata"]["ukb_consolidation"]
        self.assertEqual(provenance["source_cell_counts"], SOURCE_CELL_COUNTS)
        self.assertEqual(provenance["source_cells_total"], sum(SOURCE_CELL_COUNTS.values()))
        self.assertEqual(provenance["orchestration_cells"], 0)
        retained = {cell["id"] for cell in self.notebook["cells"]}
        removed = set(provenance["deduplicated_cell_ids"])
        self.assertFalse(retained & removed)
        archived_ids = set()
        for source, count in SOURCE_CELL_COUNTS.items():
            payload = json.loads((ARCHIVE / ARCHIVE_NAMES[source]).read_text())
            self.assertEqual(len(payload["cells"]), count)
            archived_ids.update(cell["id"] for cell in payload["cells"])
        self.assertEqual(archived_ids, retained | (removed - {
            "content-panel-boundary", "content-panel-reset"}))
        self.assertNotIn("6d55a725", retained)  # Duplicate main figure.
        self.assertIn("d4556617", retained)  # Distinct rank-flow supplement.
        self.assertIn("a04e0276", retained)  # Ten-field endpoint table.

    def test_figures_and_shared_aggregates_are_created_once(self):
        for function in (
            "ensure_topic_results", "build_panel_data", "band_summary",
            "export_growth_tables", "figure_main", "figure_si_category_changes",
            "figure_si_breadth_coverage", "figure_si_topic_robustness",
            "figure_si_rank_flow", "leading_category_change_table",
            "export_content_tables", "export_panel_manifest", "save_manifest",
        ):
            with self.subTest(function=function):
                self.assertEqual(self.calls.count(function), 1)
        self.assertEqual(self.calls.count("record_figures"), 5)
        self.assertNotIn('run_line_magic("reset"', self.code)
        self.assertNotIn("CP.", self.code)
        self.assertNotIn("Part II", self.source)

    def test_notebook_is_valid_and_sections_are_sequential(self):
        cells = self.notebook["cells"]
        self.assertEqual(len(cells), 22)
        self.assertEqual(len({cell["id"] for cell in cells}), len(cells))
        headings = ["".join(cell["source"]).splitlines()[0]
                    for cell in cells if cell["cell_type"] == "markdown"
                    and "".join(cell["source"]).startswith("## ")]
        self.assertEqual([int(heading.split()[1].rstrip(".")) for heading in headings],
                         list(range(1, 11)))
        for cell in cells:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertFalse(any(output.get("output_type") == "error"
                                     for output in cell.get("outputs", [])))

    def test_exports_and_missing_topic_preview_are_retained(self):
        for marker in (
            'P.MAIN_FIGURE_STEMS[4]', 'main_stem += "_incomplete"',
            '"02_02_supplementary_figure_01_category_composition"',
            '"02_03_supplementary_figure_02_coverage_and_breadth"',
            '"02_04_supplementary_figure_03_topic_robustness"',
            '"02_02_supplementary_figure_01_category_rank_flow"',
            'C.leading_category_change_table(D["for"], n=10)',
            'ARTIFACTS.save_manifest("02_content_manifest.csv")',
        ):
            self.assertIn(marker, self.code)
        self.assertGreater(self.code.index("E.export_content_tables"),
                           self.code.index("C.figure_si_rank_flow"))
        self.assertIn('STYLE = load_style("02_content")', self.code)
        self.assertEqual(self.calls.count("bootstrap"), 1)


if __name__ == "__main__":
    unittest.main()
