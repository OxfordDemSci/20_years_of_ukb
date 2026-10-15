#!/usr/bin/env python3
"""Rebuild every analysis input in one run: archive, then pull a fresh copy of each.

WHY
---
The analysis notebooks read eight data-creation outputs, and each was made by hand, at a
different time, by a different script — the corpus on one date, Altmetric on another, the
trials' MeSH terms on a third — with three of the steps living as helpers in `src/utils/`
that a notebook cell was expected to have run. This runner makes them one ordered job:

    A   archive    move every output in P.DATA_CREATION_OUTPUTS to data/archive/<snapshot>/
    01  endpoints  the all-endpoints wide corpus (dimensions_endpoint_pipeline.py, unmodified)
                   + the trials/policy side CSVs regenerated from it
    05  altmetric  Altmetric Explorer, with PID fallback rounds, keyed on the corpus id
    06  mesh       ClinicalTrials.gov MeSH terms, written into the corpus's trials block
    07  citations  field citation thresholds + UK Biobank citation counts into the corpus
    08  collab     Claude collaborator classification — paid, opt-in, always LAST
    R   report     changes.csv: every headline count, new against the archive

Everything BEFORE the endpoint pipeline is upstream and is not run here: the showcase
scraper (00), deduplication (02), candidate string matching (03) and GPU tagging (04) —
the last three work on the BMRC copy of the whole database. Their product is the
publication parquet this runner takes with `--publications`. Without it (development),
the input is the archived corpus's own publication columns, i.e. the same 26,109 ids.

THE ARCHIVE IS WHAT MAKES THE COPY FRESH. Moving the old outputs away, caches included,
means no stage can quietly resume from a months-old cache: every endpoint, trial, threshold
cell and Altmetric record is fetched again. The archive is the exact previous state, with
sha256s in `archive_manifest.json`; `--restore` moves a snapshot back, and nothing is ever
deleted. `--seed-collab-cache` is the one deliberate exception: it copies the archived
classifier cache back so only new institution sets are paid for.

KEYED APIS ARE CALLED ONCE. `--offline` is the development mode: keyed services
(Dimensions, Altmetric, Anthropic) are blocked in every stage, step 01 copies the archived
corpus instead of pulling, and the caches are seeded back from the archive so 05/07/08 run
from them. Pair it with `UKB_DATA_ROOT=<scratch copy of data/>` so development never moves
the real `data/`. The live run reads its keys from the environment, named after the
repository secrets: DIMENSION_KEY, ALTMETRIC_API_KEY, ALTMETRIC_API_SECRET,
ANTHROPIC_API_KEY — or, for any not set there, from the gitignored config/dsl.ini,
config/anthropic.ini and config/altmetric.ini (shapes in the *.ini.example files). Only the
selected stages' keys are required, and a missing one is named, never echoed.

USAGE
-----
    python src/data_creation/run_data_creation.py --list
    python src/data_creation/run_data_creation.py --dry-run
    UKB_DATA_ROOT=/tmp/ukb_sandbox/data python src/data_creation/run_data_creation.py --offline
    python src/data_creation/run_data_creation.py --allow-classification       # the live run
    python src/data_creation/run_data_creation.py --publications <00-04 output>.parquet ...
    python src/data_creation/run_data_creation.py --resume <run id>
    python src/data_creation/run_data_creation.py --restore <snapshot>
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import shared_paths as P  # noqa: E402
from utils.data_creation_runtime import GUARD_ENV, install_keyed_api_guard  # noqa: E402

PIPELINE = P.DATA_CREATION / "01_scraper_full_endpoint_collection_and_merge" / "dimensions_endpoint_pipeline.py"
ALTMETRIC = P.DATA_CREATION / "05_altmetric_scraper" / "altmetric_scraper.py"
MESH = P.DATA_CREATION / "06_clinical_trials_mesh" / "clinical_trials_mesh.py"
CITATIONS = P.DATA_CREATION / "07_field_thresholds" / "field_thresholds.py"
COLLAB = P.DATA_CREATION / "08_collaborator_tagging" / "collaborator_tagging.py"

#: Where the corpus is written. P.SHOWCASE_PLUS resolves to whichever candidate EXISTS at
#: import time, which after the archive step is none; the canonical location is the first.
CORPUS = P.SHOWCASE_PLUS_CANDIDATES[0]

#: The environment names of the repository secrets.
DIMENSION_KEY = "DIMENSION_KEY"
ALTMETRIC_KEYS = ("ALTMETRIC_API_KEY", "ALTMETRIC_API_SECRET")
ANTHROPIC_KEY = "ANTHROPIC_API_KEY"

#: What an offline run seeds back from the archive, so the stages run without a keyed call.
#: The whole FOR_COUNTS_API directory, not just its record cache: offline, stage 07 cannot
#: re-measure the thresholds, and analysis 03 reads every file in it.
OFFLINE_SEEDS = (
    P.CTGOV_CACHE,
    P.MESH_TREE_CACHE,
    P.ALTMETRIC_DIR / "altmetric_explorer_cache.jsonl",
    P.FOR_COUNTS_API,
    P.COLLAB_CACHE,
)

UPSTREAM = (
    ("00", "showcase scraper (00_scraper_showcase.py)"),
    ("02", "deduplication of the BMRC whole-database copy"),
    ("03", "candidate string matching on the BMRC copy"),
    ("04", "three-model LLM tagging (GPU)"),
)


# =============================================================================
# RUN CONTEXT
# =============================================================================
@dataclass
class Context:
    run_dir: Path
    snapshot: Path | None
    offline: bool
    publications: Path | None
    allow_classification: bool
    log: Path | None = None

    def say(self, message: str = "") -> None:
        print(message, flush=True)
        if self.log is not None:
            with self.log.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    def run(self, command: list[str], env: dict[str, str]) -> None:
        """Run one stage process, teeing its output to the console and the stage log."""
        shown = " ".join(P.raw_path(Path(c)) if os.sep in c else c for c in command[1:])
        self.say(f"$ python {shown}")
        process = subprocess.Popen(command, cwd=P.ROOT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        for line in process.stdout:
            self.say(line.rstrip("\n"))
        if process.wait() != 0:
            raise RuntimeError(f"{P.raw_path(Path(command[1]))} exited with {process.returncode}")

    def archived(self, path: Path) -> Path:
        """Where `path` lives inside this run's archive snapshot."""
        if self.snapshot is None:
            raise RuntimeError("this run has no archive snapshot (was it started with --no-archive?)")
        return self.snapshot / Path(path).resolve().relative_to(P.DATA.resolve())

    def archived_corpus(self) -> Path:
        """The archived corpus, from whichever candidate location it was moved out of."""
        for candidate in P.SHOWCASE_PLUS_CANDIDATES:
            if self.archived(candidate).exists():
                return self.archived(candidate)
        raise FileNotFoundError(f"no corpus in the archive snapshot {P.raw_path(self.snapshot)}")


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    needs: tuple[str, ...]
    run: Callable[[Context, dict], None]


# =============================================================================
# HASHING, ARCHIVE, RESTORE
# =============================================================================
def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())


def tree_size(path: Path) -> int:
    return sum(p.stat().st_size for p in tree_files(path))


def snapshot_name(corpus: Path | None) -> str:
    """`<UTC timestamp>_corpus-<newest Dimensions date_inserted>`: when, and of what."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    inserted = "none"
    if corpus is not None and corpus.exists():
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        try:
            column = pq.read_table(corpus, columns=["date_inserted"]).column("date_inserted")
            newest = pc.max(column).as_py()
            inserted = str(newest)[:10] if newest else "none"
        except (KeyError, ValueError, OSError):
            pass
    return f"{stamp}_corpus-{inserted}"


def archive_plan(outputs=P.DATA_CREATION_OUTPUTS) -> list[tuple[Path, int]]:
    return [(path, tree_size(path)) for path in outputs if path.exists()]


def archive_outputs(snapshot: Path, outputs=P.DATA_CREATION_OUTPUTS) -> dict:
    """MOVE every existing output into `snapshot`, mirroring its path under DATA.

    Hashes are taken before the move and written to `archive_manifest.json`. Refuses an
    existing snapshot directory and a cross-device move that the free space cannot hold.
    Nothing is deleted.
    """
    data = P.DATA.resolve()
    if snapshot.exists():
        raise FileExistsError(f"archive snapshot already exists: {P.raw_path(snapshot)}")
    plan = archive_plan(outputs)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    cross_device = [p for p, _ in plan if os.stat(p).st_dev != os.stat(snapshot.parent).st_dev]
    needed = sum(size for p, size in plan if p in cross_device)
    if needed and shutil.disk_usage(snapshot.parent).free < needed * 1.1:
        raise OSError(f"archive needs {needed / 1e9:.1f} GB on another device; not enough free")

    entries, files = [], []
    for path, size in plan:
        rel = path.resolve().relative_to(data)
        for file in tree_files(path):
            files.append({"path": file.resolve().relative_to(data).as_posix(),
                          "bytes": file.stat().st_size, "sha256": file_sha256(file)})
        entries.append({"path": rel.as_posix(), "bytes": size, "is_dir": path.is_dir()})
    snapshot.mkdir(parents=True)
    for entry in entries:
        source, dest = data / entry["path"], snapshot / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
    manifest = {
        "snapshot": snapshot.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": P.raw_path(data),
        "entries": entries,
        "files": files,
    }
    (snapshot / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def restore_snapshot(snapshot: Path) -> None:
    """Move a snapshot's entries back into place. Whatever is there now is moved aside to
    `<snapshot>.replaced-<timestamp>/`, never deleted."""
    data = P.DATA.resolve()
    manifest = json.loads((snapshot / "archive_manifest.json").read_text())
    aside = snapshot.with_name(
        f"{snapshot.name}.replaced-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    for entry in manifest["entries"]:
        current, archived = data / entry["path"], snapshot / entry["path"]
        if not archived.exists():
            print(f"  ! {entry['path']} is no longer in the snapshot; skipped")
            continue
        if current.exists():
            (aside / entry["path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(current), str(aside / entry["path"]))
        current.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(archived), str(current))
        print(f"  restored {entry['path']}")
    if aside.exists():
        print(f"  the outputs it replaced are in {P.raw_path(aside)}")


def seed_from_archive(ctx: Context, paths) -> None:
    """COPY archived caches back into place (never move: the archive stays complete)."""
    for path in paths:
        source = ctx.archived(path)
        if path.exists() or not source.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, path)
        else:
            shutil.copy2(source, path)
        ctx.say(f"  seeded {P.raw_path(path)} from the archive")


# =============================================================================
# CREDENTIALS
# =============================================================================
def stage_needs(stage: Stage, allow_classification: bool) -> tuple[str, ...]:
    if stage.id == "08" and not allow_classification:
        return ()
    return stage.needs


def _usable(value: str | None) -> bool:
    text = (value or "").strip()
    return bool(text) and not any(m in text.upper() for m in ("REPLACE", "YOUR_API_KEY", "YOUR KEY"))


def ini_credentials() -> dict[str, str]:
    """Keys from the gitignored config files, for any not set in the environment.

    config/dsl.ini       [instance.live] key        -> DIMENSION_KEY
    config/anthropic.ini [anthropic] api_key        -> ANTHROPIC_API_KEY
    config/altmetric.ini ALTMETRIC_API_KEY=… lines  -> ALTMETRIC_API_KEY, ALTMETRIC_API_SECRET

    Values are returned, never printed; a placeholder copied from the .example counts as absent.
    """
    import configparser

    found: dict[str, str] = {}
    for path, section, key, name in ((P.DSL_INI, "instance.live", "key", DIMENSION_KEY),
                                     (P.ANTHROPIC_INI, "anthropic", "api_key", ANTHROPIC_KEY)):
        if path.is_file():
            parser = configparser.ConfigParser()
            parser.read(path)
            value = parser.get(section, key, fallback="")
            if _usable(value):
                found[name] = value.strip()
    if P.ALTMETRIC_INI.is_file():
        for line in P.ALTMETRIC_INI.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            name, value = name.strip(), value.strip().strip("\"'")
            if sep and name in ALTMETRIC_KEYS and _usable(value):
                found[name] = value
    return found


def credential_environ(environ=os.environ) -> dict[str, str]:
    """The environment, with ini-file keys filled in for whatever it does not set."""
    env = dict(environ)
    for name, value in ini_credentials().items():
        if not _usable(env.get(name)):
            env[name] = value
    return env


def missing_credentials(stages, allow_classification: bool, environ=os.environ) -> list[str]:
    needed = dict.fromkeys(n for s in stages for n in stage_needs(s, allow_classification))
    return [name for name in needed if not environ.get(name, "").strip()]


def child_env(offline: bool, environ=os.environ) -> dict[str, str]:
    """The environment every stage process gets.

    Live: DIMENSION_KEY is copied to DIMENSIONS_API_KEY, the name the unmodified endpoint
    pipeline and `dimensions_api.load_config` read. Offline: every key is REMOVED, the
    keyed-API guard is switched on, and the Altmetric client gets placeholders so it can be
    constructed — the guard stops any request it would make, and its cache answers the rest.
    """
    env = dict(environ)
    env["PYTHONUNBUFFERED"] = "1"
    if offline:
        for name in (DIMENSION_KEY, "DIMENSIONS_API_KEY", ANTHROPIC_KEY, *ALTMETRIC_KEYS):
            env.pop(name, None)
        for name in ALTMETRIC_KEYS:
            env[name] = "offline-placeholder"
        env[GUARD_ENV] = "1"
        env.pop("UKB_ALLOW_CLASSIFICATION", None)
    elif env.get(DIMENSION_KEY):
        env["DIMENSIONS_API_KEY"] = env[DIMENSION_KEY]
    return env


# =============================================================================
# STAGES
# =============================================================================
def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("dimensions_endpoint_pipeline", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)        # type: ignore[union-attr]
    return module


def development_input(source: Path, destination: Path) -> Path:
    """The archived corpus's publication columns — every column without `__` — as the
    endpoint pipeline's `--publications` input. Same ids, same order."""
    import pandas as pd
    import pyarrow.parquet as pq

    columns = [c for c in pq.read_schema(source).names if "__" not in c]
    publications = pd.read_parquet(source, columns=columns)
    load_pipeline_module().validate_publications(publications)
    destination.parent.mkdir(parents=True, exist_ok=True)
    publications.to_parquet(destination, index=False, engine="pyarrow", compression="zstd")
    return destination


def regenerate_side_csvs(ctx: Context) -> None:
    from utils.data_analysis_04_non_academic_sources import ensure_endpoint_csv
    for endpoint, path in (("clinical_trials", P.CT_CSV), ("policy_documents", P.POLICY_CSV)):
        ensure_endpoint_csv(endpoint, path, source=CORPUS)
        ctx.say(f"  side CSV {P.raw_path(path)}")


def run_endpoints(ctx: Context, env: dict) -> None:
    if ctx.offline:
        source = ctx.archived_corpus()
        ctx.say(f"offline: copying the archived corpus instead of pulling ({P.raw_path(source)})")
        CORPUS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, CORPUS)
    else:
        if ctx.publications is not None:
            publications = ctx.publications
            ctx.say(f"production input: {P.raw_path(publications)}")
        else:
            publications = development_input(ctx.archived_corpus(),
                                             ctx.run_dir / "publications_input.parquet")
            ctx.say(f"development input: the archived corpus's publication columns "
                    f"-> {P.raw_path(publications)}")
        ctx.run([sys.executable, str(PIPELINE), "--publications", str(publications),
                 "--output", str(CORPUS), "--data-dir", str(P.ENDPOINT_TABLES)], env)
    regenerate_side_csvs(ctx)


def run_altmetric(ctx: Context, env: dict) -> None:
    command = [sys.executable, str(ALTMETRIC), "--input", str(CORPUS), "--allow-missing-doi"]
    if ctx.offline:
        command += ["--max-pid-rounds", "1"]     # later rounds are new lists: not in the cache
    ctx.run(command, env)


def run_mesh(ctx: Context, env: dict) -> None:
    ctx.run([sys.executable, str(MESH), "--corpus", str(CORPUS)], env)


def run_citations(ctx: Context, env: dict) -> None:
    command = [sys.executable, str(CITATIONS), "--corpus", str(CORPUS)]
    # No --fresh: the archive step already emptied FOR_COUNTS_API, so every cell is
    # measured anew anyway, and without it a --resume keeps the cells already measured
    # (the module checkpoints every 25) instead of paying for them twice.
    if ctx.offline:
        command.append("--skip-api")
    ctx.run(command, env)


def run_collab(ctx: Context, env: dict) -> None:
    if not (ctx.offline or ctx.allow_classification):
        raise SkipStage("needs --allow-classification (the paid Claude pass)")
    command = [sys.executable, str(COLLAB), "--corpus", str(CORPUS)]
    if ctx.allow_classification and not ctx.offline:
        command.append("--allow-api")
    ctx.run(command, env)


class SkipStage(Exception):
    """A stage that is deliberately not run this time."""


STAGES = (
    Stage("01", "all-endpoints wide corpus + side CSVs", (DIMENSION_KEY,), run_endpoints),
    Stage("05", "Altmetric Explorer (PID fallback rounds)", ALTMETRIC_KEYS, run_altmetric),
    Stage("06", "clinical-trial MeSH into the corpus", (), run_mesh),
    Stage("07", "field citation thresholds + UKB citation counts", (DIMENSION_KEY,), run_citations),
    Stage("08", "collaborator classification (Claude, paid)", (ANTHROPIC_KEY,), run_collab),
)


def select_stages(start: str | None = None, only=None) -> list[Stage]:
    ids = [s.id for s in STAGES]
    for wanted in [start, *(only or [])]:
        if wanted is not None and wanted not in ids:
            raise SystemExit(f"unknown stage {wanted!r}; stages are {', '.join(ids)}")
    stages = list(STAGES)
    if start is not None:
        stages = stages[ids.index(start):]
    if only:
        stages = [s for s in stages if s.id in set(only)]
    return stages


# =============================================================================
# CHANGE REPORT
# =============================================================================
ENDPOINTS = ("patents", "clinical_trials", "policy_documents", "grants", "datasets")


def headline_counts(corpus: Path, altmetric: Path, thresholds: Path, collab: Path) -> dict:
    """The counts STATE.md's ledger is built on, read off one generation of outputs."""
    import pandas as pd
    import pyarrow.parquet as pq
    from utils.shared_showcase import parse_listcol

    out: dict[str, float] = {}
    if corpus.exists():
        names = pq.read_schema(corpus).names
        out["corpus rows"] = pq.read_metadata(corpus).num_rows
        out["corpus columns"] = len(names)
        columns = ["times_cited"] + [c for e in ENDPOINTS for c in (f"{e}__id", f"{e}__n_links")
                                     if c in names]
        if "clinical_trials__mesh_leaf" in names:
            columns.append("clinical_trials__mesh_leaf")
        frame = pd.read_parquet(corpus, columns=columns)
        out["times_cited total"] = int(frame["times_cited"].sum())
        for endpoint in ENDPOINTS:
            if f"{endpoint}__id" not in frame:
                continue
            ids = {i for cell in frame[f"{endpoint}__id"] if cell for i in parse_listcol(cell)}
            out[f"{endpoint} records"] = len(ids)
            out[f"papers linked to {endpoint}"] = int((frame[f"{endpoint}__n_links"] > 0).sum())
        if "clinical_trials__mesh_leaf" in frame:
            leaf = {}
            for ids, cell in zip(frame["clinical_trials__id"], frame["clinical_trials__mesh_leaf"]):
                for tid, terms in zip(parse_listcol(ids) if ids else [],
                                      parse_listcol(cell) if cell else []):
                    terms = parse_listcol(terms) if isinstance(terms, str) else terms
                    leaf.setdefault(tid, bool(terms))
            out["trials with a MeSH leaf term"] = sum(leaf.values())
    if altmetric.exists():
        alt = pd.read_csv(altmetric)
        out["Altmetric rows"] = len(alt)
        if "id" in alt:
            out["Altmetric distinct corpus ids"] = int(alt["id"].dropna().nunique())
    if thresholds.exists():
        out["measured threshold cells"] = len(pd.read_csv(thresholds))
    if collab.exists():
        out["classified publications"] = len(pd.read_csv(collab, usecols=["id"]))
    return out


def change_report(ctx: Context) -> Path:
    import pandas as pd

    thresholds = P.FOR_COUNTS_API / "field_thresholds.for.csv"
    before: dict = {}
    if ctx.snapshot is not None:
        try:
            corpus = ctx.archived_corpus()
        except FileNotFoundError:
            corpus = ctx.archived(CORPUS)
        before = headline_counts(corpus, ctx.archived(P.ALTMETRIC_CSV),
                                 ctx.archived(thresholds), ctx.archived(P.COLLAB_FLAGGED))
    after = headline_counts(CORPUS, P.ALTMETRIC_CSV, thresholds, P.COLLAB_FLAGGED)
    rows = [{"metric": k, "archive": before.get(k), "current": after.get(k),
             "change": (after[k] - before[k]) if k in before and k in after else None}
            for k in dict.fromkeys([*before, *after])]
    path = ctx.run_dir / "changes.csv"
    pd.DataFrame(rows, dtype=object).to_csv(path, index=False)
    ctx.say("\nchanges against the archive")
    for row in rows:
        ctx.say(f"  {row['metric']:<40} {row['archive']!s:>14} -> {row['current']!s:>14}")
    return path


# =============================================================================
# MAIN
# =============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show the stages and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the archive plan, stages and credentials; change nothing")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--from", dest="start", metavar="STAGE")
    selection.add_argument("--only", nargs="+", metavar="STAGE")
    parser.add_argument("--publications", type=Path,
                        help="production input: the publication parquet stages 00-04 produce")
    parser.add_argument("--offline", action="store_true",
                        help="development: no keyed API call anywhere; caches seeded from the archive")
    parser.add_argument("--allow-classification", action="store_true",
                        help="run stage 08 against the Claude API (paid)")
    parser.add_argument("--seed-collab-cache", action="store_true",
                        help="copy the archived classifier cache back, so only new sets are paid for")
    parser.add_argument("--resume", metavar="RUN_ID",
                        help="continue an interrupted run: no archive, finished stages skipped")
    parser.add_argument("--no-archive", action="store_true",
                        help="do not archive first (outputs are overwritten in place)")
    parser.add_argument("--restore", metavar="SNAPSHOT",
                        help="move an archive snapshot back into place and exit")
    return parser.parse_args(argv)


def print_list() -> None:
    print("upstream, not run here:")
    for sid, name in UPSTREAM:
        print(f"  {sid}  {name}")
    print("run, in order:")
    print("  A   archive the current outputs to " + P.raw_path(P.DATA_ARCHIVE) + "/<snapshot>/")
    for stage in STAGES:
        needs = ", ".join(stage.needs) or "no key"
        print(f"  {stage.id}  {stage.name}  [{needs}]")
    print("  R   change report against the archive")


def write_summary(ctx: Context, rows: list[dict]) -> None:
    with (ctx.run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "name", "status", "seconds", "note"])
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(ctx: Context, run_id: str, rows: list[dict], started: str) -> None:
    manifest = {
        "run_id": run_id,
        "mode": "offline" if ctx.offline else "live",
        "input": P.raw_path(ctx.publications) if ctx.publications else "development: archived corpus",
        "snapshot": ctx.snapshot.name if ctx.snapshot else None,
        "started_utc": started,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": file_sha256(CORPUS) if CORPUS.exists() else None,
        "stages": rows,
    }
    citation = P.FOR_COUNTS_API / "citation_refresh.json"
    if citation.exists():
        manifest["citations"] = json.loads(citation.read_text())
    text = json.dumps(manifest, indent=2) + "\n"
    (ctx.run_dir / "run.json").write_text(text)
    P.DATA_CREATION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    P.DATA_CREATION_MANIFEST.write_text(text)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        print_list()
        return 0
    if args.restore:
        restore_snapshot(P.DATA_ARCHIVE / args.restore)
        return 0
    if args.publications is not None and not args.publications.is_file():
        raise SystemExit(f"--publications: no such file {args.publications}")

    stages = select_stages(args.start, args.only)
    environ = os.environ if args.offline else credential_environ()
    if args.offline:
        install_keyed_api_guard(force=True)
    else:
        missing = missing_credentials(stages, args.allow_classification, environ)
        if missing and not args.dry_run:
            raise SystemExit("missing credentials, set in the environment: " + ", ".join(missing))

    if args.dry_run:
        print(f"data root: {P.raw_path(P.DATA)}")
        print("archive plan (moved, not copied):" if not (args.resume or args.no_archive)
              else "no archive step")
        for path, size in archive_plan():
            print(f"  {P.raw_path(path):<80} {size / 1e6:10.1f} MB")
        print("step-01 input:", P.raw_path(args.publications) if args.publications
              else ("offline: archived corpus copied" if args.offline
                    else "development: archived corpus's publication columns"))
        print("stages:", ", ".join(f"{s.id} {s.name}" for s in stages))
        if not args.offline:
            missing = missing_credentials(stages, args.allow_classification, environ)
            print("credentials:", "all present" if not missing else "MISSING " + ", ".join(missing))
            for name in (DIMENSION_KEY, *ALTMETRIC_KEYS, ANTHROPIC_KEY):
                value = environ.get(name, "")
                source = "environment" if _usable(os.environ.get(name)) else "config/*.ini"
                print(f"  {name:<22} {'present, ' + str(len(value.strip())) + ' chars, from ' + source if _usable(value) else 'absent'}")
        return 0

    started = datetime.now(timezone.utc).isoformat()
    previous: dict[str, str] = {}
    if args.resume:
        run_id = args.resume
        run_dir = P.DATA_CREATION_RUNS / run_id
        state = json.loads((run_dir / "run.json").read_text())
        snapshot = P.DATA_ARCHIVE / state["snapshot"] if state.get("snapshot") else None
        previous = {row["stage"]: row["status"] for row in state.get("stages", [])}
        if state["mode"] != ("offline" if args.offline else "live"):
            raise SystemExit(f"run {run_id} was a {state['mode']} run; resume it the same way")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_dir = P.DATA_CREATION_RUNS / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        snapshot = None
        if not args.no_archive:
            snapshot = P.DATA_ARCHIVE / snapshot_name(P.SHOWCASE_PLUS if P.SHOWCASE_PLUS.exists() else None)
            print(f"archiving to {P.raw_path(snapshot)} ...", flush=True)
            manifest = archive_outputs(snapshot)
            print(f"  moved {len(manifest['entries'])} outputs ({len(manifest['files'])} files)")

    ctx = Context(run_dir=run_dir, snapshot=snapshot, offline=args.offline,
                  publications=args.publications, allow_classification=args.allow_classification)
    if snapshot is not None and not args.resume:
        if args.offline:
            seed_from_archive(ctx, OFFLINE_SEEDS)
        elif args.seed_collab_cache:
            seed_from_archive(ctx, (P.COLLAB_CACHE,))
    env = child_env(args.offline, environ)

    rows: list[dict] = []
    status = 0
    for stage in stages:
        if previous.get(stage.id) == "ok":
            rows.append({"stage": stage.id, "name": stage.name, "status": "ok",
                         "seconds": 0, "note": "done in the resumed run"})
            continue
        ctx.log = run_dir / f"{stage.id}.log"
        ctx.say(f"\n=== {stage.id}  {stage.name} ===")
        t0 = time.time()
        try:
            stage.run(ctx, env)
            row = {"status": "ok", "note": ""}
        except SkipStage as skip:
            ctx.say(f"skipped: {skip}")
            row = {"status": "skipped", "note": str(skip)}
        except Exception as exc:          # noqa: BLE001 - record, then stop the run
            ctx.say(f"FAILED: {exc}")
            row = {"status": "failed", "note": str(exc)[:300]}
            status = 1
        rows.append({"stage": stage.id, "name": stage.name,
                     "seconds": round(time.time() - t0, 1), **row})
        write_summary(ctx, rows)
        write_manifest(ctx, run_id, rows, started)
        if status:
            ctx.say(f"stopping: later stages depend on {stage.id}. "
                    f"Fix it, then --resume {run_id}")
            break

    ctx.log = run_dir / "report.log"
    try:
        change_report(ctx)
    except Exception as exc:              # noqa: BLE001 - the report never fails the run
        ctx.say(f"change report unavailable: {exc}")
    write_summary(ctx, rows)
    write_manifest(ctx, run_id, rows, started)
    ctx.say(f"\nrun {run_id}: {P.raw_path(run_dir)}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
