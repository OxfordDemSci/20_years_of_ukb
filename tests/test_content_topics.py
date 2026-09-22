"""The merged content workflow reuses results and preserves scoped modelling."""

import builtins
import contextlib
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import data_analysis_02_content_topics as topics
from utils import shared_paths as P
from utils.data_analysis_02_content_window import write_topic_window_provenance


def fake_parallel_seed(value, *, offset):
    return os.getpid(), value + offset


def fake_deferred_seed(parameter_index, params, seed, number, **kwargs):
    return ({
        "parameter_index": parameter_index, "seed": seed, "status": "unscored", "n_topics": 2,
        "outlier_rate": 0.0, "topic_diversity": 1.0, "topic_count_penalty": 0.3,
        "weighted_cluster_persistence": 0.2, "worker_pid": os.getpid(),
        "_topic_words": {0: ["gene", "expression"], 1: ["brain", "volume"]},
    }, np.asarray([0, 1]))


class ContentTopicWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.folder = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for name, value in {
            "OUTPUT": self.folder / "output", "CONTENT": self.folder / "content",
            "TOPIC_ASSIGNMENTS": self.folder / "content" / "showcase_plus_id_topics.csv",
            "ACADEMIC_IMPACT": self.folder / "academic_impact",
        }.items():
            self.stack.enter_context(patch.object(P, name, value))
        self.output = P.OUTPUT / "bertopic"
        self.cache = self.folder / "cache"
        self.cache.mkdir()

    def completed_results(self):
        self.output.mkdir(parents=True)
        path = self.output / "showcase_plus_id_topics.csv"
        pd.DataFrame({"id": ["first", "last"], "topics": ["T0: genetics", "T1: cognition"]}).to_csv(path, index=False)
        write_topic_window_provenance(path, [2013, 2025])
        return path

    def block_ml_imports(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "bertopic", "sentence_transformers", "gensim", "umap", "hdbscan", "sklearn"}:
                raise AssertionError(f"Unexpected optional model import: {name}")
            return original(name, *args, **kwargs)

        return patch.object(builtins, "__import__", guarded)

    def test_valid_results_return_before_import_loading_or_training(self):
        path = self.completed_results()
        before = {p: p.read_bytes() for p in self.output.iterdir()}
        with self.block_ml_imports(), patch.object(topics, "load_publications") as load:
            with patch.object(topics, "_fit_topic_results") as fit, contextlib.redirect_stdout(io.StringIO()):
                result = topics.ensure_topic_results(input_parquet=self.folder / "absent.parquet")
        self.assertEqual(result, path)
        load.assert_not_called()
        fit.assert_not_called()
        self.assertEqual({p: p.read_bytes() for p in self.output.iterdir()}, before)

    def test_verified_detailed_results_take_precedence_over_stale_compact_copy(self):
        self.output.mkdir(parents=True)
        detailed = self.output / "bertopic_document_topic_assignments.csv"
        pd.DataFrame({
            "id": ["first", "last"],
            "year": [2013, 2025],
            "topic": [0, 1],
            "topics": ["T0: genetics", "T1: cognition"],
        }).to_csv(detailed, index=False)
        write_topic_window_provenance(detailed, [2013, 2025])

        compact = self.output / "showcase_plus_id_topics.csv"
        pd.DataFrame({
            "id": ["first", "last"],
            "topics": ["stale topic", "stale topic"],
        }).to_csv(compact, index=False)

        with self.block_ml_imports(), patch.object(topics, "_fit_topic_results") as fit:
            with contextlib.redirect_stdout(io.StringIO()):
                result = topics.ensure_topic_results(input_parquet=self.folder / "absent.parquet")

        self.assertEqual(result, detailed)
        fit.assert_not_called()

    def test_missing_results_can_skip_training_without_imports(self):
        with self.block_ml_imports(), patch.object(topics, "load_publications") as load:
            with contextlib.redirect_stdout(io.StringIO()):
                result = topics.ensure_topic_results(train_if_missing=False)
        self.assertIsNone(result)
        load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_results_fail_without_refit(self):
        path = self.completed_results()
        path.with_suffix(".analysis_window.json").unlink()
        with self.block_ml_imports(), patch.object(topics, "_fit_topic_results") as fit:
            with self.assertRaisesRegex(FileNotFoundError, "Outdated or unverified"):
                topics.ensure_topic_results(train_if_missing=False)
        fit.assert_not_called()

    def test_force_bypasses_valid_result_but_still_checks_input(self):
        self.completed_results()
        with self.block_ml_imports(), self.assertRaisesRegex(FileNotFoundError, "Full-endpoint parquet"):
            topics.ensure_topic_results(input_parquet=self.folder / "absent.parquet", force=True)

    def test_preparation_uses_inclusive_bounds_before_deduplication(self):
        frame = pd.DataFrame({
            "id": ["first", "first", "last", "future_date", "past", "no_date", "short"],
            "year": [2012, 2013, 2025, 2025, 2012, None, 2020],
            "date": ["2012-12-31", "2013-01-01", "2025-12-31", "2026-01-01", "2012-12-31", None, None],
            "title": ["A useful scientific title that is long enough"] * 6 + ["Short"],
            "abstract": ["Background: <b>Some text</b>. Copyright 2020 publisher."] * 6 + [""],
        })
        actual = topics.prepare_publications(frame)
        self.assertEqual(actual["id"].tolist(), ["first", "last"])
        self.assertEqual(actual["year"].tolist(), [2013, 2025])
        self.assertNotIn("Copyright", actual.iloc[0]["topic_text"])
        self.assertNotIn("<b>", actual.iloc[0]["topic_text"])

    def test_grid_cache_identity_tracks_complete_text_year_and_order(self):
        frame = pd.DataFrame({"id": ["a", "b"], "topic_text": ["x" * 1100, "y" * 1100], "year": [2013, 2025]})
        base = topics.grid_cache_paths(frame, self.cache)
        for changed in [frame.iloc[::-1], frame.assign(year=[2014, 2025]), frame.assign(topic_text=["x" * 1100 + "z", "y" * 1100])]:
            self.assertNotEqual(base, topics.grid_cache_paths(changed, self.cache))
        with self.assertRaisesRegex(ValueError, "Topic training requires"):
            topics.grid_cache_paths(frame.assign(year=[2013, 2026]), self.cache)
        with patch.object(topics, "PIPELINE_VERSION", "changed-score-definition"):
            self.assertNotEqual(base, topics.grid_cache_paths(frame, self.cache))
        self.assertEqual(len(topics.PARAM_GRID) * len(topics.SEEDS), 25)

    def test_legacy_embedding_rows_reused_only_with_verified_order_and_text(self):
        raw = pd.DataFrame({
            "id": ["old", "new", "future"], "year": [2014, 2013, 2026],
            "title": ["Scientific text about cognition", "Scientific text about genetics", "Scientific text about future work"],
        })
        legacy = topics._prepare_publications(raw, legacy=True)
        digest = hashlib.sha1()
        for paper_id, document in zip(legacy["id"], legacy["topic_text"]):
            digest.update(paper_id.encode() + b"\t" + document[:1000].encode() + b"\n")
        path = self.cache / f"embeddings_allenai-specter_2docs_{digest.hexdigest()[:12]}.npy"
        np.save(path, np.asarray([[1., 2.], [3., 4.]]))
        publications = topics.prepare_publications(raw)
        with patch.object(pd, "read_parquet", return_value=raw), self.block_ml_imports():
            with contextlib.redirect_stdout(io.StringIO()):
                values, missing = topics._legacy_embeddings(publications, "source.parquet", self.cache)
        np.testing.assert_array_equal(values[0], [1., 2.])
        np.testing.assert_array_equal(missing, [1])
        # The key is not enough: all current text must match its reconstructed row.
        changed = publications.copy()
        changed.loc[0, "topic_text"] += " Changed content."
        with patch.object(pd, "read_parquet", return_value=raw), contextlib.redirect_stdout(io.StringIO()):
            _, missing = topics._legacy_embeddings(changed, "source.parquet", self.cache)
        np.testing.assert_array_equal(missing, [0, 1])

    def test_persistence_uses_hdbscan_labels_before_topic_renumbering(self):
        model = SimpleNamespace(cluster_persistence_=np.asarray([0.2, 0.8]), labels_=np.asarray([0, 0, 0, 1, -1]))
        metrics, table = topics.cluster_persistence_summary(model, {0: 1, 1: 0})
        self.assertAlmostEqual(metrics["weighted_cluster_persistence"], 0.35)
        self.assertEqual(table["topic"].tolist(), [1, 0])
        self.assertEqual(table["raw_cluster_size"].tolist(), [3, 1])

    def complete_seed_grid(self):
        records = []
        for parameter_index in range(len(topics.PARAM_GRID)):
            for seed in topics.SEEDS:
                values = [0, 0, 1, 1, 2, 2] if seed != 42 else [0, 0, 1, 1, 1, 2]
                np.save(self.cache / f"params_{parameter_index}_seed_{seed}.npy", values)
                records.append({
                    "parameter_index": parameter_index, "seed": seed, "status": "ok", "n_topics": 3,
                    "quality_score": 0.5, "coherence_cv": 0.4, "topic_diversity": 0.8,
                    "outlier_rate": 0.0, "raw_outlier_rate": 0.1,
                    "mean_cluster_persistence": 0.3, "median_cluster_persistence": 0.3,
                    "weighted_cluster_persistence": 0.3, "low_persistence_fraction": 0.1,
                })
        return pd.DataFrame(records)

    def test_robust_selection_uses_pair_agreement_and_medoid(self):
        stability, pairs = topics.select_robust_model(self.complete_seed_grid(), self.cache)
        self.assertEqual(stability.loc[0, "representative_seed"], 11)
        self.assertEqual(len(pairs), 50)
        self.assertEqual(stability["n_seeds"].tolist(), [5] * 5)
        self.assertAlmostEqual(
            stability.loc[0, "robust_score"], 0.5 + 0.25 * pairs["ari"].mean() + 0.1 * pairs["nmi"].mean(),
        )

    def test_selection_refuses_incomplete_seed_sets_before_reading_assignments(self):
        incomplete = self.complete_seed_grid().iloc[:-1]
        with patch.object(np, "load", side_effect=AssertionError("Selection must stop before reading labels")):
            with self.assertRaisesRegex(RuntimeError, "all 5 parameter sets × 5 seeds.*parameters 4/seed 101"):
                topics.select_robust_model(incomplete, self.cache)

    def test_coherence_vocabulary_contains_the_same_ngrams_as_topic_keywords(self):
        documents = [
            "Hippocampal brain volume and gene expression were measured.",
            "Gene expression correlates with hippocampal brain volume.",
        ]
        tokenised = topics.tokenise_documents(documents)
        vocabulary = {word for document in tokenised for word in document}
        self.assertIn("gene expression", vocabulary)
        self.assertIn("hippocampal brain volume", vocabulary)
        vectorizer = topics.build_vectorizer().set_params(min_df=1, max_df=1.0)
        vectorizer.fit(documents)
        self.assertTrue(set(vectorizer.get_feature_names_out()).issubset(vocabulary))

    def test_outlier_update_retains_original_ctfidf_model(self):
        ctfidf = object()
        model = SimpleNamespace(
            ctfidf_model=ctfidf, reduce_outliers=Mock(return_value=[0, 0]), update_topics=Mock(),
        )
        self.assertEqual(topics.reduce_outliers(model, ["one", "two"], [0, -1], np.ones((2, 3))), [0, 0])
        self.assertIs(model.update_topics.call_args.kwargs["ctfidf_model"], ctfidf)

    def test_dependency_check_rejects_known_incompatible_hdbscan_before_model_import(self):
        with self.block_ml_imports(), patch("importlib.metadata.version", return_value="0.8.40"):
            with self.assertRaisesRegex(RuntimeError, "hdbscan 0.8.40.*requirements-analysis.txt"):
                topics._check_model_dependencies()

    def test_parallel_dispatch_runs_independent_processes_with_unchanged_jobs(self):
        jobs = [(index,) for index in range(4)]
        results = list(topics._dispatch_seed_jobs(
            jobs, workers=2, fit_kwargs={"offset": 7}, fit_function=fake_parallel_seed,
        ))
        self.assertEqual(sorted(value for _, value in results), [7, 8, 9, 10])
        self.assertTrue(all(pid != os.getpid() for pid, _ in results))
        self.assertLessEqual(len({pid for pid, _ in results}), 2)

    def test_coordinator_commits_unordered_results_and_resumes_without_fits(self):
        publications = pd.DataFrame({"id": ["a", "b"], "topic_text": ["gene expression", "brain volume"], "year": [2013, 2025]})

        def fake_dispatch(jobs, *, workers, fit_kwargs):
            self.assertEqual(workers, 2)
            self.assertTrue(fit_kwargs["defer_scoring"])
            self.assertNotIn("dictionary", fit_kwargs)
            for index, params, seed, number in reversed(jobs):
                yield ({"parameter_index": index, "seed": seed, "status": "ok", "n_topics": 2,
                        "coherence_cv": 0.5, "outlier_rate": 0.0, "scoring_seconds": 0.1,
                        "weighted_cluster_persistence": 0.2}, np.asarray([0, 1]))

        with patch.object(topics, "PARAM_GRID", [topics.PARAM_GRID[0]]), patch.object(topics, "SEEDS", [11, 21]):
            with patch.object(topics, "_dispatch_seed_jobs", side_effect=fake_dispatch), contextlib.redirect_stdout(io.StringIO()):
                result, assignment_dir = topics.run_seed_grid(publications, np.ones((2, 3)), self.cache, workers=2)
            self.assertEqual(result["seed"].tolist(), [11, 21])
            np.testing.assert_array_equal(np.load(assignment_dir / "params_0_seed_11.npy"), [0, 1])
            self.assertFalse(list(self.cache.rglob("*.tmp.*")))
            with patch.object(topics, "_dispatch_seed_jobs", side_effect=AssertionError("Completed fits must be skipped")):
                with contextlib.redirect_stdout(io.StringIO()):
                    resumed, _ = topics.run_seed_grid(publications, np.ones((2, 3)), self.cache, workers=2)
            pd.testing.assert_frame_equal(result, resumed)

    def test_parallel_fit_diagnostics_are_scored_only_in_coordinator(self):
        publications = pd.DataFrame({"id": ["a", "b"], "topic_text": ["gene expression", "brain volume"], "year": [2013, 2025]})
        row = {"parameter_index": 0, "seed": 11, "status": "unscored", "n_topics": 2,
               "outlier_rate": 0.0, "weighted_cluster_persistence": 0.2,
               "_topic_words": {0: ["gene", "expression"], 1: ["brain", "volume"]}}

        def score_in_parent(metrics, words, tokens, dictionary):
            self.assertEqual(os.getpid(), coordinator_pid)
            self.assertEqual(words[0], ["gene", "expression"])
            return {"coherence_cv": 0.5, "quality_score": 0.6}

        coordinator_pid = os.getpid()
        with patch.object(topics, "PARAM_GRID", [topics.PARAM_GRID[0]]), patch.object(topics, "SEEDS", [11]):
            with patch.object(topics, "_dispatch_seed_jobs", return_value=iter([(row, np.asarray([0, 1]))])):
                with patch.object(topics, "_score_quality", side_effect=score_in_parent) as scoring:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result, _ = topics.run_seed_grid(publications, np.ones((2, 3)), self.cache, workers=2)
        scoring.assert_called_once()
        self.assertEqual(result.loc[0, "status"], "ok")
        self.assertEqual(result.loc[0, "quality_score"], 0.6)
        self.assertNotIn("_topic_words", result.columns)

    def test_worker_count_has_explicit_safe_limit(self):
        with patch.dict(os.environ, {"UKB_TOPIC_WORKERS": "2"}):
            self.assertEqual(topics._worker_count(), 2)
            self.assertEqual(topics._worker_count(1), 1)
        for value in (0, 3, -1, "auto"):
            with self.assertRaisesRegex(ValueError, "workers must be 1 or 2"):
                topics._worker_count(value)

    def test_loky_fits_and_coordinator_multiprocess_coherence_work_together(self):
        publications = pd.DataFrame({"id": ["a", "b"], "topic_text": ["gene expression", "brain volume"], "year": [2013, 2025]})
        with patch.object(topics, "PARAM_GRID", [topics.PARAM_GRID[0]]), patch.object(topics, "SEEDS", [11, 21]):
            with patch.object(topics, "_fit_one_seed", fake_deferred_seed), contextlib.redirect_stdout(io.StringIO()):
                results, _ = topics.run_seed_grid(publications, np.ones((2, 3)), self.cache, workers=2)
        self.assertEqual(results["status"].tolist(), ["ok", "ok"])
        self.assertTrue(results["worker_pid"].ne(os.getpid()).all())
        np.testing.assert_allclose(results["coherence_cv"], [1.0, 1.0], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
