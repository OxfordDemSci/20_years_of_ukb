"""The refined academic-impact workflow retains every distinct analysis."""

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


class AcademicImpactConsolidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join("".join(c.get("source", [])) for c in cls.notebook["cells"])
        cls.cells = {c["id"]: c for c in cls.notebook["cells"]}

    def test_only_one_academic_impact_notebook_is_active(self):
        self.assertEqual(sorted(p.name for p in NOTEBOOK_DIR.glob("03*.ipynb")), [NOTEBOOK.name])

    def test_archived_sources_and_deduplication_are_accounted_for(self):
        provenance = self.notebook["metadata"]["ukb_consolidation"]
        self.assertEqual(provenance["source_cell_counts"], SOURCE_CELL_COUNTS)
        self.assertEqual(provenance["source_cells_total"], sum(SOURCE_CELL_COUNTS.values()))
        refinement = provenance["refinement"]
        removed = set(refinement["removed_cell_ids"])
        previous = -1
        positions = {c["id"]: i for i, c in enumerate(self.notebook["cells"])}
        for source, count in SOURCE_CELL_COUNTS.items():
            archived = ARCHIVE / source.replace(".ipynb", "_pre_consolidation.ipynb")
            payload = json.loads(archived.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["cells"]), count)
            for cell in payload["cells"]:
                if cell["id"] in removed:
                    self.assertNotIn(cell["id"], self.cells)
                    continue
                self.assertIn(cell["id"], self.cells)
                self.assertEqual(cell["cell_type"], self.cells[cell["id"]]["cell_type"])
                self.assertGreater(positions[cell["id"]], previous)
                previous = positions[cell["id"]]
        for duplicate, retained in refinement["duplicate_figures_replaced_by"].items():
            self.assertIn(duplicate, removed)
            self.assertIn(retained, self.cells)

    def test_single_build_and_no_namespace_resets(self):
        self.assertEqual(self.source.count("AI.build("), 1)
        self.assertIn("NP.build_panel_data(context=CTX)", self.source)
        self.assertNotIn('run_line_magic("reset"', self.source)
        self.assertEqual(len(self.cells), len(self.notebook["cells"]))
        self.assertFalse(any(o.get("output_type") == "error"
                             for c in self.cells.values() for o in c.get("outputs", [])))

    def test_all_fifteen_distinct_figures_use_one_publishing_path(self):
        self.assertEqual(self.source.count("Q.publish("), 15)
        self.assertEqual(self.notebook["metadata"]["ukb_consolidation"]["refinement"]
                         ["retained_figure_count"], 15)
        for duplicate in ("05_impact_map_", "07_citation_share_",
                          "09_footprint_quality_median_", "11_fastest_growing_"):
            self.assertNotIn(duplicate, self.source)
        required = [
            "Q.whole_database_trends", "Q.publication_contribution",
            "Q.annual_growth_speed", "Q.activity_over_time",
            "Q.citation_measure_comparison", "Q.citation_cutoffs",
            "Q.top_decile_pool", "Q.author_cohort_profile", "Q.author_fingerprint",
            "Q.citation_overview", "Q.citation_concentration_and_cohorts",
            "NP.figure_main(D, save=False)",
            "NP.figure_si_impact_map(D, save=False)",
            "NP.figure_si_top_decile(D, save=False)", "NP.figure_si_growth(D, save=False)",
        ]
        for marker in required:
            self.assertIn(marker, self.source)

    def test_citation_concentration_and_cohorts_are_published_once(self):
        self.assertEqual(self.source.count("Q.citation_concentration_and_cohorts("), 1)
        self.assertNotIn("Q.citation_cohorts(", self.source)
        self.assertNotIn('"fig06_age_adjusted_citation_cohorts"', self.source)
        merged = self.notebook["metadata"]["ukb_consolidation"]["refinement"]
        self.assertEqual(merged["combined_figures_replaced_by"]["_wMswquSRHhM"], "sgMydPt1RHhM")
        self.assertIn("Panel B shows age-adjusted citation impact", self.source)

    def test_four_citation_diagnostics_are_published_as_one_overview(self):
        self.assertEqual(self.source.count("Q.citation_overview("), 1)
        for function in ("disciplinary_composition", "cumulative_citation_stock",
                         "field_citation_stock", "citation_survival"):
            self.assertNotIn(f"Q.{function}(", self.source)
        merged = self.notebook["metadata"]["ukb_consolidation"]["refinement"]
        for old in ("JCUZAwHSRHhK", "yahZopFZRHhL", "88CbgM16RHhN"):
            self.assertEqual(merged["combined_figures_replaced_by"][old], "AYh5B5D4RHhK")
        for letter in "ABCD":
            self.assertIn(f"**{letter}.", self.source)

    def test_tables_and_window_guards_are_preserved(self):
        for marker in (
            "author_summary_with_impact_metrics.csv", "author_paper_fractional_credit_table.csv",
            "input_manifest_author_impact_notebook.csv", "panel_d_citation_cutoffs.csv",
            "si1_impact_table_all_weights.csv", "si2_fastest_growing_fields.csv",
            "NP.assert_parameters_match(D)", "ANALYSIS_MIN = 2015",
            'ARTIFACTS.save_manifest("panel_manifest.csv")',
        ):
            self.assertIn(marker, self.source)
        self.assertNotIn('selection.to_csv', self.source)
        self.assertNotIn('FIG_DIR.glob', self.source)

    def test_figure_scaffolding_is_not_duplicated_in_notebook(self):
        code = "\n".join("".join(c["source"]) for c in self.cells.values()
                         if c["cell_type"] == "code")
        for marker in ("plt.subplots(", "plt.rcParams", "COLOUR =", "set_title(",
                       "show_figures()", 'FIGURE_DIR = OUTPUT_DIR / "figures"'):
            self.assertNotIn(marker, code)
        for cell in self.cells.values():
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(NOTEBOOK), "exec")


if __name__ == "__main__":
    unittest.main()
