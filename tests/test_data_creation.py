"""The data-creation runner archives before it rebuilds, and its stages keep the corpus aligned."""

import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import shared_paths as P
from utils import data_creation_clinical_trials_fetch_ctgov as ctgov
from utils import data_creation_runtime as runtime
from utils.shared_showcase import endpoint_records, write_corpus_columns


def _load(relative: str, name: str):
    """Stage scripts live in numbered directories, which are not importable packages."""
    spec = importlib.util.spec_from_file_location(name, P.DATA_CREATION / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


runner = _load("run_data_creation.py", "run_data_creation")
mesh_stage = _load("06_clinical_trials_mesh/clinical_trials_mesh.py", "clinical_trials_mesh")
citations = _load("07_field_thresholds/field_thresholds.py", "field_thresholds")
altmetric = _load("05_altmetric_scraper/altmetric_scraper.py", "altmetric_scraper")


def _wide(path: Path) -> Path:
    """Two publications, one with two linked trials and one with none — the pipeline's shape."""
    pd.DataFrame({
        "id": ["pub.1", "pub.2"],
        "doi": ["10.1/a", None],
        "pmid": ["11", "22"],
        "times_cited": [5, 7],
        "field_citation_ratio": [1.5, None],
        "recent_citations": [1, 2],
        "relative_citation_ratio": [0.5, None],
        "supporting_grant_ids": ["[]", "[]"],
        "source_title": ['{"id": "jour.1"}', '{"id": "jour.2"}'],
        "clinical_trials__linked_ids": ['["NCT1","NCT2"]', ""],
        "clinical_trials__n_links": [2, 0],
        "clinical_trials__n_records": [2, 0],
        "clinical_trials__id": ['["NCT1","NCT2"]', ""],
        "clinical_trials__mesh_terms": ['[["Diabetes"],["Asthma"]]', ""],
    }).to_parquet(path, index=False)
    return path


class StageSelectionTests(unittest.TestCase):
    def test_order_from_and_only(self):
        self.assertEqual([s.id for s in runner.select_stages()], ["01", "05", "06", "07", "08"])
        self.assertEqual([s.id for s in runner.select_stages(start="06")], ["06", "07", "08"])
        self.assertEqual([s.id for s in runner.select_stages(only=["07", "05"])], ["05", "07"])

    def test_upstream_and_unknown_stages_are_refused(self):
        for upstream in ("00", "02", "03", "04"):
            with self.assertRaises(SystemExit):
                runner.select_stages(only=[upstream])

    def test_the_paid_stage_runs_last(self):
        self.assertEqual(runner.STAGES[-1].id, "08")


class CredentialTests(unittest.TestCase):
    def test_only_selected_stages_need_keys(self):
        stages = runner.select_stages(only=["06"])
        self.assertEqual(runner.missing_credentials(stages, False, environ={}), [])
        stages = runner.select_stages(only=["07", "08"])
        self.assertEqual(runner.missing_credentials(stages, False, environ={}), ["DIMENSION_KEY"])
        self.assertEqual(runner.missing_credentials(stages, True, environ={}),
                         ["DIMENSION_KEY", "ANTHROPIC_API_KEY"])

    def test_secret_name_is_mapped_for_the_unmodified_pipeline(self):
        env = runner.child_env(False, environ={"DIMENSION_KEY": "k"})
        self.assertEqual(env["DIMENSIONS_API_KEY"], "k")

    def test_offline_children_carry_no_keys_and_the_guard(self):
        env = runner.child_env(True, environ={"DIMENSION_KEY": "k", "ANTHROPIC_API_KEY": "a",
                                              "UKB_ALLOW_CLASSIFICATION": "1"})
        for name in ("DIMENSION_KEY", "DIMENSIONS_API_KEY", "ANTHROPIC_API_KEY",
                     "UKB_ALLOW_CLASSIFICATION"):
            self.assertNotIn(name, env)
        self.assertEqual(env[runtime.GUARD_ENV], "1")

    def test_missing_altmetric_key_is_named_not_echoed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(altmetric.AltmetricAuthorizationError) as caught:
                altmetric.resolve_credential(None, "ALTMETRIC_API_KEY", "API key")
        self.assertIn("ALTMETRIC_API_KEY", str(caught.exception))


class KeyedApiGuardTests(unittest.TestCase):
    def test_keyed_hosts_are_blocked_and_keyless_ones_are_not(self):
        original = socket.getaddrinfo
        try:
            runtime.install_keyed_api_guard(force=True)
            for host in ("app.dimensions.ai", "www.altmetric.com", "api.anthropic.com"):
                with self.assertRaises(runtime.KeyedApiBlocked):
                    socket.getaddrinfo(host, 443)
            self.assertFalse(runtime._is_keyed("clinicaltrials.gov"))
            self.assertFalse(runtime._is_keyed("id.nlm.nih.gov"))
        finally:
            socket.getaddrinfo = original


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name) / "data"
        self.owned = self.data / "showcase" / "corpus.parquet"
        self.owned_dir = self.data / "altmetric"
        self.not_owned = self.data / "analysis" / "for_counts_out" / "field_counts.parquet"
        for path, text in ((self.owned, "corpus"), (self.owned_dir / "a.csv", "alt"),
                           (self.not_owned, "vm")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.patch = patch.object(P, "DATA", self.data)
        self.patch.start()
        self.outputs = (self.owned, self.owned_dir, self.data / "never_existed.csv")

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_moves_owned_outputs_with_hashes_and_leaves_the_rest(self):
        snapshot = self.data / "archive" / "s1"
        manifest = runner.archive_outputs(snapshot, self.outputs)
        self.assertFalse(self.owned.exists())
        self.assertEqual((snapshot / "showcase" / "corpus.parquet").read_text(), "corpus")
        self.assertEqual((snapshot / "altmetric" / "a.csv").read_text(), "alt")
        self.assertTrue(self.not_owned.exists())
        recorded = {f["path"]: f["sha256"] for f in manifest["files"]}
        self.assertEqual(recorded["showcase/corpus.parquet"],
                         runner.file_sha256(snapshot / "showcase" / "corpus.parquet"))
        self.assertEqual(json.loads((snapshot / "archive_manifest.json").read_text())["snapshot"], "s1")

    def test_an_existing_snapshot_is_refused(self):
        (self.data / "archive" / "s1").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            runner.archive_outputs(self.data / "archive" / "s1", self.outputs)
        self.assertTrue(self.owned.exists())

    def test_offline_seeding_copies_files_and_whole_directories_back(self):
        snapshot = self.data / "archive" / "s1"
        runner.archive_outputs(snapshot, self.outputs)
        ctx = runner.Context(run_dir=self.data, snapshot=snapshot, offline=True,
                             publications=None, allow_classification=False)
        runner.seed_from_archive(ctx, (self.owned, self.owned_dir))
        self.assertEqual((self.owned_dir / "a.csv").read_text(), "alt")
        self.assertTrue((snapshot / "altmetric" / "a.csv").exists())   # copied, not moved

    def test_restore_round_trips_and_sets_the_newer_copy_aside(self):
        snapshot = self.data / "archive" / "s1"
        runner.archive_outputs(snapshot, self.outputs)
        self.owned.write_text("fresh")
        runner.restore_snapshot(snapshot)
        self.assertEqual(self.owned.read_text(), "corpus")
        aside = [p for p in (self.data / "archive").iterdir() if ".replaced-" in p.name]
        self.assertEqual((aside[0] / "showcase" / "corpus.parquet").read_text(), "fresh")


class DevelopmentInputTests(unittest.TestCase):
    def test_keeps_every_id_and_drops_every_endpoint_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _wide(Path(tmp) / "wide.parquet")
            out = runner.development_input(source, Path(tmp) / "pubs.parquet")
            frame = pd.read_parquet(out)
        self.assertEqual(frame["id"].tolist(), ["pub.1", "pub.2"])
        self.assertFalse([c for c in frame.columns if "__" in c])
        self.assertIn("supporting_grant_ids", frame.columns)


class MeshWriteBackTests(unittest.TestCase):
    def test_columns_align_with_the_trial_ids_and_read_back_through_the_loader(self):
        browse = pd.DataFrame([
            dict(id="NCT2", fetched=True, cond_leaf=["Asthma"], cond_leaf_ids=["D001249"],
                 cond_ancestors=["Lung Diseases"], intervention=[], error=""),
            dict(id="NCT1", fetched=True, cond_leaf=["Diabetes Mellitus"],
                 cond_leaf_ids=["D003920"], cond_ancestors=[], intervention=["Insulin"],
                 error=""),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = _wide(Path(tmp) / "wide.parquet")
            trials = endpoint_records("clinical_trials", path=path)
            enriched = ctgov.mesh_columns(trials, browse)
            corpus = pd.read_parquet(path, columns=["id", "clinical_trials__id"])
            block = mesh_stage.mesh_block(corpus["clinical_trials__id"], enriched)
            self.assertEqual(block["clinical_trials__mesh_leaf_ids"][1], "")
            write_corpus_columns(block, path)
            back = endpoint_records("clinical_trials", path=path).set_index("id")
            ids = pd.read_parquet(path, columns=["id"])["id"].tolist()
        self.assertEqual(back.at["NCT1", "mesh_leaf_ids"], ["D003920"])
        self.assertEqual(back.at["NCT2", "mesh_leaf"], ["Asthma"])
        self.assertEqual(back.at["NCT1", "mesh_intervention"], ["Insulin"])
        self.assertEqual(ids, ["pub.1", "pub.2"])

    def test_only_trials_that_cite_the_paper_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _wide(Path(tmp) / "wide.parquet")
            frame = pd.read_parquet(path)
            # NCT1 cites pub.1; NCT2 is only named by pub.1 (its publication_ids omit it)
            frame["clinical_trials__publication_ids"] = ['[["pub.1"],["pub.9"]]', ""]
            frame.to_parquet(path, index=False)
            corpus = pd.read_parquet(path, columns=["id", *[c for c in frame.columns
                                                            if c.startswith("clinical_trials__")]])
            block, tally = mesh_stage.citing_only_block(corpus)
            write_corpus_columns(block, path)
            again, _ = mesh_stage.citing_only_block(pd.read_parquet(path, columns=list(corpus.columns)))
            back = endpoint_records("clinical_trials", path=path)
        self.assertEqual(tally, {"links_before": 2, "links_after": 1, "misaligned_cells": 0})
        self.assertEqual(back["id"].tolist(), ["NCT1"])
        self.assertEqual(back["mesh_terms"].tolist(), [["Diabetes"]])
        self.assertEqual(block["clinical_trials__n_links"], [1, 0])
        self.assertEqual(again["clinical_trials__linked_ids"], block["clinical_trials__linked_ids"])

    def test_the_corpus_writer_refuses_misaligned_columns_and_the_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _wide(Path(tmp) / "wide.parquet")
            with self.assertRaises(ValueError):
                write_corpus_columns({"times_cited": [1]}, path)
            with self.assertRaises(ValueError):
                write_corpus_columns({"id": ["a", "b"]}, path)


class CtgovRetryTests(unittest.TestCase):
    def test_a_transient_failure_is_retried_and_a_cached_one_is_asked_again(self):
        calls = []

        def flaky(tid, timeout=30):
            calls.append(tid)
            if calls.count(tid) == 1:
                raise TimeoutError("timed out")
            return dict(id=tid, fetched=True, cond_leaf=["X"], cond_leaf_ids=["D1"],
                        cond_ancestors=[], intervention=[], error="")

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "browse.pkl"
            pd.DataFrame([dict(id="NCT9", fetched=False, cond_leaf=[], cond_leaf_ids=[],
                               cond_ancestors=[], intervention=[],
                               error="transient: reset")]).to_pickle(cache)
            with patch.object(ctgov, "_fetch_one", side_effect=flaky), \
                    patch.object(ctgov.time, "sleep"):
                out = ctgov.fetch_ctgov_browse(["NCT1", "NCT9", "ISRCTN5"], cache=cache,
                                               pause=0, backoff=0)
        by_id = out.set_index("id")
        self.assertTrue(by_id.at["NCT1", "fetched"])
        self.assertTrue(by_id.at["NCT9", "fetched"])       # the cached transient was re-asked
        self.assertFalse(by_id.at["ISRCTN5", "fetched"])   # non-NCT: final, never requested
        self.assertNotIn("ISRCTN5", calls)


class _FakeExplorer:
    """Recognises DOI 10.1/a and 10.1/d, and PMIDs 22 and 33 — never DOI 10.1/x."""

    RECORDS = {
        "10.1/a": {"id": "alt-a", "dois": ["10.1/a"], "pubmed": ["11"]},
        "22": {"id": "alt-b", "dois": ["10.1/b-published"], "pubmed": ["22"]},
        "33": {"id": "alt-c", "dois": [], "pubmed": ["33"]},
        "10.1/d": {"id": "alt-d", "dois": ["10.1/d"], "pubmed": []},
    }

    def __init__(self, *args, **kwargs):
        self.lists = {}

    def create_identifier_list(self, identifiers):
        list_id = f"list{len(self.lists)}"
        self.lists[list_id] = list(identifiers)
        return {"id": list_id, "counts": {}}

    def research_outputs_page(self, list_id, *, page_number, page_size):
        data = []
        for pid in self.lists[list_id]:
            spec = self.RECORDS.get(pid)
            if spec:
                data.append({"id": spec["id"], "attributes": {
                    "identifiers": {"dois": spec["dois"], "pubmed-ids": spec["pubmed"]},
                    "altmetric-score": 10, "mentions": {"msm": 2, "policy": 1},
                    "publication-date": "2020-01-01"}})
        return {"data": data, "links": {}, "meta": {"response": {"total-pages": 1}}}


class AltmetricRoundTests(unittest.TestCase):
    def _run(self, *extra):
        tmp = Path(self._tmp.name)
        source = tmp / "corpus.parquet"
        pd.DataFrame({
            "id": ["pub.a", "pub.b", "pub.c", "pub.d1", "pub.d2"],
            "doi": ["10.1/a", "10.1/x", None, "10.1/d", "10.1/d"],
            "pmid": ["11", "22", "33", None, None],
            "year": [2020] * 5,
        }).to_parquet(source, index=False)
        with patch.object(altmetric, "ExplorerClient", _FakeExplorer), \
                patch.dict(os.environ, {"ALTMETRIC_API_KEY": "k", "ALTMETRIC_API_SECRET": "s"}):
            altmetric.main(["--input", str(source), "--output", str(tmp / "out.csv"),
                            "--audit-output", str(tmp / "audit.csv"),
                            "--cache", str(tmp / "cache.jsonl"), "--progress-every", "0",
                            *extra])
        return pd.read_csv(tmp / "out.csv"), pd.read_csv(tmp / "audit.csv")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_doi_miss_is_recovered_by_pmid_in_round_two(self):
        out, audit = self._run("--allow-missing-doi")
        audit = audit.set_index("input_id")
        self.assertEqual(audit.at["pub.b", "lookup_status"], "found")
        self.assertEqual(audit.at["pub.b", "lookup_round"], 2)
        self.assertEqual(audit.at["pub.b", "matched_pid_type"], "pmid")
        self.assertEqual(audit.at["pub.a", "lookup_round"], 1)
        self.assertIn("pub.b", set(out["id"]))

    def test_one_round_reproduces_the_single_round_submissions(self):
        _, audit = self._run("--allow-missing-doi", "--max-pid-rounds", "1")
        self.assertEqual(audit.set_index("input_id").at["pub.b", "lookup_status"], "not_found")

    def test_doi_less_match_is_kept_with_its_id_and_shared_records_survive(self):
        out, _ = self._run("--allow-missing-doi")
        self.assertIn("pub.c", set(out["id"]))
        self.assertTrue({"pub.d1", "pub.d2"} <= set(out["id"]))
        self.assertEqual(list(out.columns)[:5], list(altmetric.OUTPUT_COLUMNS[:5]))
        out, _ = self._run()
        self.assertNotIn("pub.c", set(out["id"]))


class DimensionsRetryTests(unittest.TestCase):
    def test_a_server_side_400_is_retried_and_a_bad_query_is_not(self):
        from utils.data_analysis_03_academic_impact_dimensions_api import _transient_query_error

        class R:
            def __init__(self, code, text):
                self.status_code, self.text = code, text

        self.assertTrue(_transient_query_error(R(400, '{"errors": "Query failed due to internal '
                                                       'server error. Please try again later"}')))
        self.assertFalse(_transient_query_error(R(400, '{"errors": "Semantic Error: unknown field"}')))
        self.assertFalse(_transient_query_error(R(200, "internal server error")))


class CitationWriteBackTests(unittest.TestCase):
    def test_missing_ids_fall_back_by_doi_then_pmid(self):
        class FakeApi:
            def query(self, dsl):
                if "where doi in" in dsl:
                    return {"publications": [{"id": "pub.new", "doi": "10.1/a",
                                              "times_cited": 50}]}
                if "where pmid in" in dsl:
                    return {"publications": [{"id": "pub.new2", "pmid": "22",
                                              "times_cited": 9}]}
                return {}

        missing = pd.DataFrame({"id": ["pub.1", "pub.2"], "doi": ["10.1/A", None],
                                "pmid": ["11", "22"]})
        found = citations.fallback_records(FakeApi(), missing, ["id", "times_cited"])
        self.assertEqual(found["pub.1"]["_route"], "doi")
        self.assertEqual(found["pub.2"]["_route"], "pmid")

    def test_write_back_keeps_unresolved_values_and_integer_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _wide(Path(tmp) / "wide.parquet")
            corpus = pd.read_parquet(path, columns=["id", "doi", "pmid",
                                                    *citations.CITATION_COLUMNS])
            records = {"pub.1": {"id": "pub.1", "times_cited": 8, "field_citation_ratio": None,
                                 "recent_citations": None}}
            values, tally = citations.citation_values(corpus, records, {})
            write_corpus_columns(values, path)
            back = pd.read_parquet(path)
        self.assertEqual(tally, {"live": 1, "fallback": 0, "unresolved": 1})
        self.assertEqual(back["times_cited"].tolist(), [8, 7])
        self.assertEqual(back["recent_citations"].tolist(), [1, 2])     # null kept the old int
        self.assertTrue(pd.isna(back.at[0, "field_citation_ratio"]))     # a real absence
        self.assertEqual(back.at[0, "relative_citation_ratio"], 0.5)    # field not returned
        self.assertEqual(str(back["times_cited"].dtype), "int64")


if __name__ == "__main__":
    unittest.main()
