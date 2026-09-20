"""Topic training boundaries and cache reuse, without downloading or fitting models."""

import ast
import contextlib
import io
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.shared_analysis_window import (
    ANALYSIS_START_DATE, ANALYSIS_START_YEAR, ANALYSIS_END_DATE, ANALYSIS_END_YEAR,
    filter_analysis_window,
)
from utils import data_analysis_02_content_window as topic_window
from utils.data_analysis_02_content_window import (
    require_topic_window_provenance, topic_corpus_hash, write_topic_window_provenance,
)
from utils import data_analysis_02_content_panels as panels


def boundary_corpus():
    return pd.DataFrame({
        "id": ["end", "future_date", "future_year", "unknown", "start", "year_only", "date_only",
               "past_date", "past_year", "before"],
        "year": [2025, 2025, 2026, None, 2013, 2025, None, 2013, 2012, 2012],
        "date": ["2025-12-31T23:59:59Z", "2026-01-01", "2025-10-01", None,
                 "2013-01-01", None, "2025-06-30", "2012-12-31T23:59:59Z",
                 "2013-01-01", "2012-06-01"],
        "title": ["A scientific paper with enough usable text"] * 10,
        "abstract": ["An abstract describing a cohort study."] * 10,
    })


class ContentCutoffTests(unittest.TestCase):
    def test_script_preparation_filters_before_any_model_work(self):
        path = ROOT / "src/utils/data_analysis_02_content_bert_topic.py"
        tree = ast.parse(path.read_text())
        names = {"norm_col", "infer_col", "clean_str", "normalise_title", "make_short_hash",
                 "get_year_from_row", "clean_topic_text", "make_showcase_id", "prepare_input"}
        # Execute the production preprocessing functions without optional ML imports.
        namespace = dict(pd=pd, np=np, re=re, hashlib=hashlib, Path=Path,
                         SimpleNamespace=SimpleNamespace,
                         ANALYSIS_START_DATE=ANALYSIS_START_DATE, ANALYSIS_START_YEAR=ANALYSIS_START_YEAR,
                         ANALYSIS_END_DATE=ANALYSIS_END_DATE, ANALYSIS_END_YEAR=ANALYSIS_END_YEAR,
                         filter_analysis_window=filter_analysis_window)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and (target.id.endswith("_CANDIDATES") or target.id == "config")
                for target in node.targets
            ):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        definitions = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        ], type_ignores=[])
        exec(compile(ast.fix_missing_locations(definitions), str(path), "exec"), namespace)
        with tempfile.TemporaryDirectory() as folder, patch.object(pd, "read_parquet", return_value=boundary_corpus()):
            with contextlib.redirect_stdout(io.StringIO()):
                actual = namespace["prepare_input"](Path("fixture.parquet"), Path(folder))
        self.assertEqual(actual["showcase_plus_id"].tolist(), ["end", "start", "year_only", "date_only"])
        year_only = boundary_corpus().drop(columns="date").assign(date_inserted="2026-09-20")
        with tempfile.TemporaryDirectory() as folder, patch.object(pd, "read_parquet", return_value=year_only):
            with contextlib.redirect_stdout(io.StringIO()):
                actual = namespace["prepare_input"](Path("fixture.parquet"), Path(folder))
        self.assertEqual(actual["showcase_plus_id"].tolist(),
                         ["end", "future_date", "start", "year_only", "past_date"])

    def test_notebook_filters_before_embeddings_and_includes_2013(self):
        path = ROOT / "src/data_analysis/02_content_1_bert_topic.ipynb"
        notebook = json.loads(path.read_text())
        helper_cell = next("".join(c["source"]) for c in notebook["cells"]
                           if "def extract_year(" in "".join(c["source"]))
        names = {"clean_value", "clean_topic_text", "infer_column", "extract_year"}
        definitions = ast.Module(
            body=[node for node in ast.parse(helper_cell).body
                  if isinstance(node, ast.FunctionDef) and node.name in names],
            type_ignores=[],
        )
        namespace = dict(pd=pd, np=np, re=re, PARQUET_PATH="fixture.parquet",
                         ANALYSIS_START_DATE=ANALYSIS_START_DATE, ANALYSIS_START_YEAR=ANALYSIS_START_YEAR,
                         ANALYSIS_END_DATE=ANALYSIS_END_DATE, ANALYSIS_END_YEAR=ANALYSIS_END_YEAR,
                         filter_analysis_window=filter_analysis_window,
                         load_showcase=lambda **kwargs: boundary_corpus(), display=lambda *args: None)
        configuration = next("".join(c["source"]) for c in notebook["cells"]
                             if "".join(c["source"]).startswith("MIN_YEAR ="))
        exec(compile(configuration, str(path), "exec"), namespace)
        exec(compile(definitions, str(path), "exec"), namespace)
        preparation = next("".join(c["source"]) for c in notebook["cells"]
                           if "raw = load_showcase(" in "".join(c["source"]))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(preparation, str(path), "exec"), namespace)
        self.assertEqual(namespace["publications"]["id"].tolist(),
                         ["end", "start", "year_only", "date_only"])

    def test_category_loader_filters_before_rankings(self):
        with patch.object(panels, "load_showcase", return_value=boundary_corpus()):
            actual = panels.load_corpus()
        self.assertEqual(actual["id"].tolist(), ["end", "start", "year_only", "date_only"])
        self.assertEqual(panels.FLOW_YEARS, list(range(2013, 2026)))

    def test_category_notebook_flow_and_reconciliation_include_2013(self):
        path = ROOT / "src/data_analysis/02_content_2_other_category_flow.ipynb"
        notebook = json.loads(path.read_text())
        configuration = next("".join(c["source"]) for c in notebook["cells"]
                             if "FLOW_MIN, FLOW_MAX =" in "".join(c["source"]))
        namespace = dict(ANALYSIS_START_YEAR=ANALYSIS_START_YEAR,
                         ANALYSIS_END_YEAR=ANALYSIS_END_YEAR)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(configuration, str(path), "exec"), namespace)
        self.assertEqual(namespace["FLOW_YEARS"], list(range(2013, 2026)))
        self.assertEqual((namespace["API_MIN"], namespace["API_MAX"]), (2013, 2025))

    def test_notebook_does_not_reuse_unscoped_grid_scores(self):
        path = ROOT / "src/data_analysis/02_content_1_bert_topic.ipynb"
        notebook = json.loads(path.read_text())
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if "RUNS_PATH =" in "".join(c["source"]))
        nodes = []
        for node in ast.parse(source).body:
            if isinstance(node, ast.For):
                break
            nodes.append(node)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "seed_grid_runs.csv").write_text(
                "parameter_index,seed,status,weighted_cluster_persistence\n0,42,ok,1\n"
            )
            namespace = dict(pd=pd, hashlib=hashlib, json=json, topic_corpus_hash=topic_corpus_hash,
                             paper_ids=["end"], grid_docs=["eligible document"], grid_indices=[0],
                             publications=pd.DataFrame({"year": [2025]}), PARAM_GRID=[{"n_neighbors": 15}],
                             SEEDS=[42], EMBEDDING_MODEL_NAME="allenai-specter", OUTPUT_DIR=folder,
                             CACHE_DIR=folder / "cache")
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
            self.assertEqual(namespace["completed"], set())
            self.assertNotEqual(namespace["RUNS_PATH"], folder / "seed_grid_runs.csv")

    def test_cache_key_rejects_future_inputs_and_tracks_complete_text(self):
        with self.assertRaisesRegex(ValueError, "2025-12-31"):
            topic_corpus_hash(["a", "b"], ["first", "second"], [2025, 2026])
        first = topic_corpus_hash(["a"], ["x" * 1000 + "first"], [2025])
        second = topic_corpus_hash(["a"], ["x" * 1000 + "second"], [2025])
        self.assertNotEqual(first, second)

    def test_cache_key_rejects_pre_2013_inputs_and_changes_with_window(self):
        with self.assertRaisesRegex(ValueError, "2013-01-01"):
            topic_corpus_hash(["a", "b"], ["first", "second"], [2012, 2013])
        current = topic_corpus_hash(["a"], ["eligible document"], [2025])
        with patch.object(topic_window, "ANALYSIS_START_DATE", pd.Timestamp("2014-01-01")):
            old_window = topic_corpus_hash(["a"], ["eligible document"], [2025])
        self.assertNotEqual(current, old_window)
        self.assertTrue(topic_corpus_hash(["a"], ["eligible document"], [2013]))

    def test_topic_tables_require_matching_training_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "topics.csv"
            path.write_text("id,year,topics\na,2025,Topic\n")
            with self.assertRaisesRegex(FileNotFoundError, "Outdated or unverified"):
                require_topic_window_provenance(path)
            write_topic_window_provenance(path, [2025])
            require_topic_window_provenance(path)
            path.write_text("id,year,topics\na,2026,Topic\n")
            with self.assertRaisesRegex(ValueError, "Outdated or unverified"):
                require_topic_window_provenance(path)

    def test_topic_tables_trained_on_future_papers_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "topics.csv"
            path.write_text("id,year,topics\na,2025,Topic\n")
            write_topic_window_provenance(path, [2025])
            sidecar = path.with_suffix(".analysis_window.json")
            metadata = json.loads(sidecar.read_text())
            metadata["training_max_year"] = 2026
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Outdated or unverified"):
                require_topic_window_provenance(path)

    def test_topic_tables_require_exact_start_and_end_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "topics.csv"
            path.write_text("id,year,topics\na,2013,Topic\n")
            write_topic_window_provenance(path, [2013, 2025])
            sidecar = path.with_suffix(".analysis_window.json")
            original = json.loads(sidecar.read_text())
            self.assertEqual(original["analysis_start_date"], "2013-01-01")
            require_topic_window_provenance(path)
            for key, value in (("analysis_start_date", None),
                               ("analysis_start_date", "2014-01-01"),
                               ("analysis_end_date", "2024-12-31"),
                               ("training_min_year", 2012)):
                with self.subTest(key=key, value=value):
                    metadata = original.copy()
                    if value is None:
                        metadata.pop(key)
                    else:
                        metadata[key] = value
                    sidecar.write_text(json.dumps(metadata))
                    with self.assertRaisesRegex(ValueError, "2013-01-01 through 2025-12-31"):
                        require_topic_window_provenance(path)

    def test_topic_assignments_are_joined_to_filtered_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "topics.csv"
            # All file years claim 2025; corpus dates must still enforce both boundaries.
            ids = ["end", "start", "future_date", "future_year", "past_date", "past_year", "before"]
            pd.DataFrame({"id": ids, "year": [2025] * len(ids),
                          "topics": ["T0: trait / cohort"] * len(ids)}).to_csv(path, index=False)
            write_topic_window_provenance(path, [2025])
            with patch.object(panels, "TOPIC_SOURCES", (path,)):
                actual, _, _ = panels.load_topic_year_matrix(boundary_corpus())
            self.assertEqual(actual["id"].tolist(), ["end", "start"])


if __name__ == "__main__":
    unittest.main()
