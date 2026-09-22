"""The consolidated dataset notebook retains both validation workflows."""

import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "src" / "data_analysis"
NOTEBOOK = NOTEBOOK_DIR / "00_dataset.ipynb"
ARCHIVE = NOTEBOOK_DIR / "_archived"
TABLE_DIR = ROOT / "output/validation/ukb_prompt_validation_heldout_v3/tables"

PROMPTS = (
    "p1_conservative", "p2_balanced", "p3_evidence_cues",
    "p4_context_no_shot", "p5_real_one_shot", "p6_real_five_shot",
)
MODELS = (
    "qwen2_5_7b", "llama3_8b", "mistral_7b", "zephyr_7b",
    "scibert_sim", "sbert_minilm_sim",
)

SOURCE_CELL_COUNTS = {
    "00_dataset.ipynb": 27,
    "00_dataset_2_validation_analysis_heldout.ipynb": 32,
}
ARCHIVE_NAMES = {
    "00_dataset.ipynb": "00_dataset_pre_heldout_consolidation.ipynb",
    "00_dataset_2_validation_analysis_heldout.ipynb": (
        "00_dataset_2_validation_analysis_heldout_pre_consolidation.ipynb"
    ),
}
EXPECTED_REWRITTEN_CELL_IDS = {
    "a892f85b",
    "6545cdf3",
    "991165bb",
    "f48f1104",
    "e17b334c",
    "b53550aa",
    "a0d600e3",
    "1944400f",
    "f8cc825a",
    "e8365236",
    "b58c3de4",
    "790fb147",
    "5c0253cb",
    "6e1c92cc",
    "c459fcdf",
    "b6ed2374",
    "281fb309",
    "7eba14c7",
    "3cbeffa1",  # Figure display helper moved into utils.
    "fb4b07e5",  # Relative paths in the final figure inventory.
}


class DatasetHeldoutConsolidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_only_one_dataset_notebook_is_active(self):
        active = sorted(path.name for path in NOTEBOOK_DIR.glob("00_dataset*.ipynb"))
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

    def test_notebook_is_valid_and_parts_are_isolated(self):
        import nbformat
        nbformat.validate(nbformat.from_dict(self.notebook))
        cells = self.notebook["cells"]
        self.assertEqual(len(cells), sum(SOURCE_CELL_COUNTS.values()) + 2)
        self.assertEqual(len({cell["id"] for cell in cells}), len(cells))
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        for cell in code_cells:
            compile("".join(cell["source"]), f"00_dataset:{cell['id']}", "exec")
        self.assertEqual(self.source.count('run_line_magic("reset", "-f")'), 1)
        headings = [
            "# Part I: Dataset agreement and original validation, 2013–2025",
            "# Part II: Locked development/held-out classifier validation",
        ]
        heading_positions = [self.source.index(heading) for heading in headings]
        self.assertEqual(heading_positions, sorted(heading_positions))

    def test_default_execution_is_local_and_reuses_saved_results(self):
        self.assertNotIn("%pip", self.source)
        self.assertNotIn("google.colab", self.source)
        self.assertNotIn("/content/drive", self.source)
        self.assertIn('UKB_RERUN_HELDOUT_MODELS", "0"', self.source)
        self.assertIn("P.VALIDATION_HELDOUT_TABLES", self.source)
        self.assertIn("sha256_file(path)", self.source)
        self.assertIn('EXPECTED_SELECTED_PROMPT = "p2_balanced"', self.source)
        self.assertIn("recomputed_heldout_metrics", self.source)
        self.assertIn("expected_class_counts", self.source)

    def test_figure_and_table_contracts_remain_present(self):
        required_markers = [
            '"00_01_figure_candidate_agreement"',
            '"00_02_figure_candidate_text"',
            '"00_03_figure_validation_performance"',
            '"00_04_figure_validation_agreement"',
            '"00_05_figure_semantic_diagnostics"',
            '"00_06_figure_consensus_validation"',
            '"00_07_figure_consensus_groups"',
            '"pairwise_agreement_all_six_prompts.png"',
            '"heldout_confusion_matrices_6_prompts_6_models.png"',
            '"development_metrics_all_prompts_models.csv"',
            '"heldout_metrics_all_prompts_models.csv"',
            '"heldout_metrics_selected_prompt_primary.csv"',
            '"heldout_confusion_counts_all_prompts_models.csv"',
        ]
        for marker in required_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.source)

        plot_source = (ROOT / "src/utils/data_analysis_00_heldout_figures.py").read_text()
        for stem in ("pairwise_agreement_{prompt}", "pairwise_agreement_all_six_prompts",
                     "heldout_precision_recall_f1_heatmaps", "heldout_selected_prompt_performance",
                     "heldout_confusion_matrices_6_prompts_6_models", "heldout_confusion_matrices_{prompt}"):
            self.assertIn(stem, plot_source)
        for renderer in ("render_heldout_agreement", "render_heldout_performance", "render_heldout_confusion"):
            self.assertIn(f"H.{renderer}(", self.source)

    def test_supplied_data_and_results_are_in_canonical_locations(self):
        self.assertFalse((ROOT / "required_data_for_rerun_the_models").exists())
        self.assertFalse(
            (ROOT / "result_files_for_regenerate_results_figures_without_rerun_model").exists()
        )
        self.assertTrue((ROOT / "data/validation/ukb_ground_truth_positive_labelled.csv").is_file())
        self.assertTrue((ROOT / "data/validation/ukb_negative_pre2013_labelled_final.csv").is_file())
        expected_results = {
            "benchmark_development.csv",
            "benchmark_heldout_test.csv",
            "development_metrics_all_prompts_models.csv",
            "development_prompt_selection.csv",
            "encoder_thresholds_development.csv",
            "few_shot_examples.csv",
            "heldout_confusion_counts_all_prompts_models.csv",
            "heldout_metrics_all_prompts_models.csv",
            "heldout_metrics_selected_prompt_primary.csv",
            "input_manifest.csv",
            "prompt_manifest.csv",
            "resolved_model_revisions.csv",
            "run_configuration.json",
            "selected_prompt_before_heldout.json",
        }
        expected_results.update(
            f"predictions_heldout_{prompt}.csv"
            for prompt in PROMPTS
        )
        expected_results.update(
            f"pairwise_agreement_{kind}_{prompt}.csv"
            for prompt in PROMPTS for kind in ("percent", "n")
        )
        self.assertEqual(len(expected_results), 32)
        self.assertTrue(all((TABLE_DIR / name).is_file() for name in expected_results))

    def test_frozen_bundle_matches_the_locked_design_and_heldout_metrics(self):
        development = pd.read_csv(TABLE_DIR / "benchmark_development.csv")
        heldout = pd.read_csv(TABLE_DIR / "benchmark_heldout_test.csv")
        demonstrations = pd.read_csv(TABLE_DIR / "few_shot_examples.csv")

        self.assertEqual(development["label"].value_counts().to_dict(), {0: 3200, 1: 3200})
        self.assertEqual(heldout["label"].value_counts().to_dict(), {0: 800, 1: 800})
        self.assertEqual(demonstrations["label"].value_counts().to_dict(), {1: 3, 0: 2})
        self.assertTrue(development["id"].is_unique)
        self.assertTrue(heldout["id"].is_unique)
        self.assertTrue(demonstrations["id"].is_unique)
        self.assertFalse(set(development["id"]) & set(heldout["id"]))
        self.assertFalse(
            set(demonstrations["id"])
            & (set(development["id"]) | set(heldout["id"]))
        )

        configuration = json.loads((TABLE_DIR / "run_configuration.json").read_text())
        self.assertEqual(configuration["sampling_seed"], 42)
        self.assertEqual(configuration["n_positive"], 4000)
        self.assertEqual(configuration["n_negative"], 4000)
        self.assertEqual(configuration["heldout_fraction"], 0.2)

        development_metrics = pd.read_csv(
            TABLE_DIR / "development_metrics_all_prompts_models.csv"
        )
        ranking = (
            development_metrics[development_metrics["model_type"].eq("LLM")]
            .groupby("prompt", as_index=False)
            .agg(
                mean_llm_f1=("f1", "mean"),
                mean_llm_precision=("precision", "mean"),
                mean_llm_recall=("recall", "mean"),
            )
            .sort_values(
                ["mean_llm_f1", "mean_llm_precision", "mean_llm_recall", "prompt"],
                ascending=[False, False, False, True],
            )
        )
        self.assertEqual(ranking.iloc[0]["prompt"], "p2_balanced")
        prompt_lock = json.loads(
            (TABLE_DIR / "selected_prompt_before_heldout.json").read_text()
        )
        self.assertEqual(prompt_lock["selected_prompt"], "p2_balanced")
        self.assertEqual(prompt_lock["sampling_seed"], 42)

        saved_metrics = pd.read_csv(TABLE_DIR / "heldout_metrics_all_prompts_models.csv")
        calculated_confusions = []
        for prompt in PROMPTS:
            predictions = pd.read_csv(TABLE_DIR / f"predictions_heldout_{prompt}.csv")
            self.assertEqual(predictions["id"].astype(str).tolist(), heldout["id"].astype(str).tolist())
            self.assertEqual(predictions["label"].astype(int).tolist(), heldout["label"].astype(int).tolist())
            truth = predictions["label"].astype(int).eq(1)
            for model in MODELS:
                predicted = (
                    predictions[model].astype("string").str.lower().eq("true").fillna(False)
                )
                parsed = (
                    predictions[f"{model}_parse_ok"]
                    .astype("string").str.lower().eq("true").fillna(False)
                )
                tn = int((~truth & ~predicted).sum())
                fp = int((~truth & predicted).sum())
                fn = int((truth & ~predicted).sum())
                tp = int((truth & predicted).sum())
                calculated_confusions.append({
                    "prompt": prompt, "model": model,
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                })
                row = saved_metrics[
                    saved_metrics["prompt"].eq(prompt)
                    & saved_metrics["model"].eq(model)
                ].squeeze()
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                self.assertEqual(int(row["n"]), len(heldout))
                self.assertEqual(int(row["n_parsed"]), int(parsed.sum()))
                self.assertAlmostEqual(float(row["parse_rate"]), float(parsed.mean()), places=12)
                self.assertAlmostEqual(float(row["accuracy"]), (tp + tn) / len(heldout), places=12)
                self.assertAlmostEqual(float(row["precision"]), precision, places=12)
                self.assertAlmostEqual(float(row["recall"]), recall, places=12)
                self.assertAlmostEqual(float(row["f1"]), f1, places=12)
                self.assertEqual(
                    tuple(int(row[name]) for name in ("tn", "fp", "fn", "tp")),
                    (tn, fp, fn, tp),
                )

        saved_confusions = pd.read_csv(
            TABLE_DIR / "heldout_confusion_counts_all_prompts_models.csv"
        )
        pd.testing.assert_frame_equal(
            saved_confusions.sort_values(["prompt", "model"]).reset_index(drop=True),
            pd.DataFrame(calculated_confusions)
            .sort_values(["prompt", "model"])
            .reset_index(drop=True),
            check_dtype=False,
        )
        primary = pd.read_csv(TABLE_DIR / "heldout_metrics_selected_prompt_primary.csv")
        expected_primary = saved_metrics[saved_metrics["prompt"].eq("p2_balanced")]
        pd.testing.assert_frame_equal(
            primary.drop(columns=["model_display"], errors="ignore")
            .sort_values("model").reset_index(drop=True),
            expected_primary.drop(columns=["model_display"], errors="ignore")
            .sort_values("model").reset_index(drop=True),
            check_dtype=False,
        )


if __name__ == "__main__":
    unittest.main()
