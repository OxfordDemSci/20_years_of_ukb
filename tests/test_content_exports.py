"""The deduplicated workflow retains legacy tables and tracks only current figures."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import data_analysis_02_content_exports as exports
from utils import data_analysis_02_content_panels as panels
from utils.shared_paths import ArtifactRegistry, raw_path


class ContentExportTests(unittest.TestCase):
    def fixture(self):
        long = pd.DataFrame([
            {"id": f"{year}-{i}", "year": year, "category": f"Field {i}"}
            for year in (2013, 2025) for i in range(12)
        ])
        data = {key: panels._flow_block(long, "category", key)
                for key in ("for", "rcdc", "topics")}
        data["topics"]["from_assignments"] = True
        data["corpus"] = long[["id", "year"]]
        data["counts"] = {}
        return data

    def test_all_panel_tables_are_exported_and_registered(self):
        data = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            registry = ArtifactRegistry(Path(directory))
            with patch.object(panels, "load_topic_diagnostics", return_value=None):
                tables = exports.export_content_tables(data, registry)
            required = {"panel_band_rules.csv", "panel_selection.csv"}
            for stem in ("for_l4", "rcdc", "topics"):
                required.update(f"panel_{stem}_{suffix}.csv"
                                for suffix in ("year_share", "drawn_bands", "start_vs_end"))
            self.assertTrue(required <= tables.keys())
            self.assertTrue(required <= {path.name for path in registry.table_paths})
            self.assertTrue(all(path.exists() for path in registry.table_paths))
            pd.testing.assert_frame_equal(
                tables["panel_for_l4_start_vs_end.csv"],
                panels.leading_category_change_table(data["for"], n=10).reset_index())
            self.assertEqual(len(tables["panel_for_l4_start_vs_end.csv"]), 10)
            selection = tables["panel_selection.csv"]
            self.assertFalse(selection.duplicated(["figure", "panel"]).any())
            self.assertIn("si_rank_flow", set(selection.figure))
            self.assertFalse(any("caption" in path.name for path in registry.table_paths))

    def test_absent_topics_do_not_create_topic_matrices(self):
        data = self.fixture()
        data["topics"] = {"available": False}
        tables = exports.panel_source_tables(data)
        self.assertFalse(any(name.startswith("panel_topics_") for name in tables))
        self.assertIn("panel_for_l4_start_vs_end.csv", tables)

    def test_manifest_excludes_stale_files_and_handles_no_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ArtifactRegistry(root)
            empty = exports.export_panel_manifest(registry)
            self.assertEqual(list(empty.columns), ["file", "kb", "modified"])
            self.assertTrue(empty.empty)
            current = root / "current.pdf"
            current.touch()
            (root / "stale.pdf").touch()
            registry.record_figures([current, current])
            manifest = exports.export_panel_manifest(registry)
            self.assertEqual(manifest.file.tolist(), [raw_path(current)])
            self.assertIn(root / "panel_manifest.csv", registry.table_paths)


if __name__ == "__main__":
    unittest.main()
