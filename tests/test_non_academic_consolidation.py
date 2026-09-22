"""The consolidated non-academic notebook retains every analysis phase."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "src" / "data_analysis"
NOTEBOOK = NOTEBOOK_DIR / "04_non_academic.ipynb"
ARCHIVE = NOTEBOOK_DIR / "_archived"

SOURCE_CELL_COUNTS = {
    "04_non_academic_01_clinical_trials.ipynb": 82,
    "04_non_academic_02_patents.ipynb": 20,
    "04_non_academic_03_altmetric.ipynb": 2,
    "04_non_academic_04_collaboration.ipynb": 28,
    "04_non_academic_99_all.ipynb": 23,
}

# These are consolidation-only prose changes plus the renamed assembly style key.
EXPECTED_REWRITTEN_CELL_IDS = {
    "0e9c60f1",
    "83ad9d7a",
    "eaa9fa86",
    "6f8aef66",
    "8e6e07a8",
    "d2c81ad4",
    "b8cd5be3",
    "701c21a5",
    "9b203fdb",
    "1ebdc673",
    "51bdeb20",
    "7782a022",
    "1c155f72",
    "f894ef46",
    "25400c97",
    "df82691a",
}


# Shared figure display now posts relative paths and suggested captions.
EXPECTED_REWRITTEN_CELL_IDS.update({
    "98fc48f2",
    "d2c81ad4",
    "c83c8411",
    "9e2a4f36",
    "1ff70dd7",
    "1a06eb29",
    "f7eec542",
    "5a22f648",
    "b8cd5be3",
    "701c21a5",
    "36b5d875",
    "7d4d906b",
    "40571432",
    "8365c642",
    "e3820dfe",
    "af8d5afa",
    "f3a1dc2e",
    "e32778ed",
    "21d3e3d0",
    "24a133cf",
    "9049bb49",
    "f1bd2b23",
    "fb7eb8b5-5f5f-4480-999c-223252b2c797",
    "902d9104",
    "c9e56414",
    "d6a3c010",
    "169681e0",
    "234ce457",
    "0affb6ac",
    "970365cb",
    "0d39d731",
    "7bae36f8",
    "ea4a8f46",
    "29fe3eea",
    "8664da3d",
    "44aa8574",
    "a979d05e",
    "073c11ce",
    "f894ef46",
    "2d712122",
    "14baeb66",
    "627a224c",
    "800e8726",
    "a671a27d",
    "f401016e",
    "937b241c",
    "64a4eb1a",
    "34dea78e",
})


class NonAcademicConsolidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_only_one_non_academic_notebook_is_active(self):
        active = sorted(path.name for path in NOTEBOOK_DIR.glob("04_non_academic*.ipynb"))
        self.assertEqual(active, [NOTEBOOK.name])

    def test_all_source_notebooks_are_archived_and_accounted_for(self):
        provenance = self.notebook["metadata"]["ukb_consolidation"]
        self.assertEqual(provenance["source_cell_counts"], SOURCE_CELL_COUNTS)
        self.assertEqual(provenance["source_cells_total"], sum(SOURCE_CELL_COUNTS.values()))
        self.assertEqual(provenance["orchestration_cells"], 10)
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
        self.assertEqual(len(cells), sum(SOURCE_CELL_COUNTS.values()) + 10)
        self.assertEqual(len({cell["id"] for cell in cells}), len(cells))
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        self.assertTrue(all(cell.get("execution_count") is None
                            or isinstance(cell["execution_count"], int) for cell in code_cells))
        self.assertFalse(any(output.get("output_type") == "error"
                             for cell in code_cells for output in cell.get("outputs", [])))
        self.assertEqual(self.source.count('run_line_magic("reset", "-f")'), 4)
        headings = [
            "# Part I: Clinical-trial impact",
            "# Part II: Patent impact",
            "# Part III: Policy and Altmetric attention",
            "# Part IV: Non-academic collaboration",
            "# Part V: Assembled publication panels",
        ]
        positions = [self.source.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))

    def test_figure_data_and_table_contracts_remain_present(self):
        required_markers = [
            'savefig(fig, "ct_diseases_mesh")',
            'savefig(fig, "ct_combined_figure", STYLE)',
            'savefig(fig, "ct_country_maps_trials_vs_papers")',
            "P.CT_UKBB_PAPERS",
            "'patents_modularized_export.csv'",
            'f"impact.{_ext}"',
            "h.plot_collaboration_mix_stacked_area",
            "h.plot_publication_figure",
            "NP.figure_main(D)",
            "NP.figure_si_patents(D)",
            "NP.figure_si_trials(D)",
            "NP.figure_si_policy(D)",
            "NP.figure_si_altmetric(D)",
            "NP.figure_si_collaboration(D)",
            '"panel_selection.csv"',
            '"panel_manifest.csv"',
        ]
        for marker in required_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.source)


if __name__ == "__main__":
    unittest.main()
