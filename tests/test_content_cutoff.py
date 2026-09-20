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
from utils.shared_analysis_window import ANALYSIS_END_DATE, ANALYSIS_END_YEAR, filter_analysis_window
from utils.data_analysis_02_content_window import (
    require_topic_window_provenance, topic_corpus_hash, write_topic_window_provenance,
)
from utils import data_analysis_02_content_panels as panels


def boundary_corpus():
    return pd.DataFrame({
        "id": ["end", "future_date", "future_year", "unknown", "early", "year_only", "date_only"],
        "year": [2025, 2025, 2026, None, 2013, 2025, None],
        "date": ["2025-12-31T23:59:59Z", "2026-01-01", "2025-10-01", None,
                 "2013-06-01", None, "2025-06-30"],
        "title": ["A scientific paper with enough usable text"] * 7,
        "abstract": ["An abstract describing a cohort study."] * 7,
    })


class ContentCutoffTests(unittest.TestCase):
    def test_script_preparation_filters_before_any_model_work(self):
        path = ROOT / "src/utils/data_analysis_02_content_bert_topic.py"
        tree = ast.parse(path.read_text())
        names = {"norm_col", "infer_col", "clean_str", "normalise_title", "make_short_hash",
                 "get_year_from_row", "clean_topic_text", "make_showcase_id", "prepare_input"}
        # Execute the production preprocessing functions without optional ML imports.
        namespace = dict(pd=pd, np=np, re=re, hashlib=hashlib, Path=Path,
                         ANALYSIS_END_DATE=ANALYSIS_END_DATE,
                         filter_analysis_window=filter_analysis_window,
                         config=SimpleNamespace(ID_COL=None, TITLE_COL=None, ABSTRACT_COL=None,
                                                YEAR_COL=None, DATE_COL=None, MIN_YEAR=2014))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id.endswith("_CANDIDATES")
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
        self.assertEqual(actual["showcase_plus_id"].tolist(), ["end", "year_only", "date_only"])
        year_only = boundary_corpus().drop(columns="date").assign(date_inserted="2026-09-20")
        with tempfile.TemporaryDirectory() as folder, patch.object(pd, "read_parquet", return_value=year_only):
            with contextlib.redirect_stdout(io.StringIO()):
                actual = namespace["prepare_input"](Path("fixture.parquet"), Path(folder))
        self.assertEqual(actual["showcase_plus_id"].tolist(), ["end", "future_date", "year_only"])

    def test_notebook_filters_before_embeddings_and_preserves_lower_bound(self):
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
        namespace = dict(pd=pd, np=np, re=re, MIN_YEAR=2014, PARQUET_PATH="fixture.parquet",
                         ANALYSIS_END_DATE=ANALYSIS_END_DATE, ANALYSIS_END_YEAR=ANALYSIS_END_YEAR,
                         filter_analysis_window=filter_analysis_window,
                         load_showcase=lambda **kwargs: boundary_corpus(), display=lambda *args: None)
        exec(compile(definitions, str(path), "exec"), namespace)
        preparation = next("".join(c["source"]) for c in notebook["cells"]
                           if "raw = load_showcase(" in "".join(c["source"]))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(preparation, str(path), "exec"), namespace)
        self.assertEqual(namespace["publications"]["id"].tolist(), ["end", "year_only", "date_only"])

    def test_category_loader_filters_before_rankings(self):
        with patch.object(panels, "load_showcase", return_value=boundary_corpus()):
            actual = panels.load_corpus()
        self.assertEqual(actual["id"].tolist(), ["end", "early", "year_only", "date_only"])

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

    def test_topic_assignments_are_joined_to_filtered_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "topics.csv"
            # All file years claim 2025; corpus dates must still exclude future papers.
            pd.DataFrame({"id": ["end", "future_date", "future_year"], "year": [2025] * 3,
                          "topics": ["T0: trait / cohort"] * 3}).to_csv(path, index=False)
            write_topic_window_provenance(path, [2025])
            with patch.object(panels, "TOPIC_SOURCES", (path,)):
                actual, _, _ = panels.load_topic_year_matrix(boundary_corpus())
            self.assertEqual(actual["id"].tolist(), ["end"])


if __name__ == "__main__":
    unittest.main()
