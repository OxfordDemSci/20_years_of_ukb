#!/usr/bin/env python3
"""Stage 07 — field citation thresholds, and UK Biobank's own citation counts.

WHY
---
Analysis 03 normalises every UK Biobank paper against its (year, FOR field) cell of the
whole database: the measured top-decile and median cut-offs in `field_thresholds.for.csv`
and the whole-database and UK Biobank partials beside it, all built by
`src/utils/data_analysis_03_academic_impact_dimensions_api.py` (`SOURCE = "api"` in the
notebook). That module was run by hand; this stage runs it on the corpus's ids.

Its `ukbb` step fetches every corpus paper live, citation counts included. Those counts
are written back into the corpus here, so the parquet's `times_cited`,
`field_citation_ratio`, `recent_citations` and `relative_citation_ratio` sit on the same
index date as the thresholds they are compared with — closing the "two citation
snapshots" split between 03's Part III (the parquet) and Parts I/II/IV (the API).

IDS THE LIVE INDEX NO LONGER RETURNS. Dimensions retracts, merges and re-identifies
records between releases; 29 of 26,109 were missing in the last pull. Each one is looked
up again by its DOI, then by its PMID. A hit supplies that paper's citation columns and is
logged in `id_remap.csv`; the corpus `id` itself is never rewritten, because it is the
join key everywhere. A paper found by neither keeps its previous values, and is counted.

USAGE
-----
    python src/data_creation/07_field_thresholds/field_thresholds.py --fresh
    python src/data_creation/07_field_thresholds/field_thresholds.py --skip-api   # write-back
                                                        # from the cached records only
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import shared_paths as P  # noqa: E402
from utils.data_creation_runtime import install_keyed_api_guard  # noqa: E402
from utils.shared_showcase import write_corpus_columns  # noqa: E402

API_MODULE = P.UTILS / "data_analysis_03_academic_impact_dimensions_api.py"
RECORDS_JSON = "api_ukbb_records.json"

#: The corpus columns refreshed from the live records. Integer counts keep their old value
#: when the API omits them (a null would turn the int64 column into floats); the two ratios
#: are genuinely absent for young or non-PubMed papers, so a null there is written as null.
CITATION_COLUMNS = ("times_cited", "field_citation_ratio", "recent_citations",
                    "relative_citation_ratio")
INTEGER_COLUMNS = {"times_cited", "recent_citations"}

#: `where doi in [...]` is capped at 512 entries; stay well inside it.
FALLBACK_BATCH = 400


def write_ids(corpus: Path, out: Path) -> Path:
    ids = pd.read_parquet(corpus, columns=["id"])["id"].astype(str)
    path = out / "ukbb_ids.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"{len(ids):,} corpus ids -> {P.raw_path(path)}")
    return path


def run_api(ids: Path, out: Path, category: str, min_ukbb: int, fresh: bool) -> None:
    """Mode `all`: whole -> ukbb -> percentiles -> ukbb -> background, in that order."""
    cmd = [sys.executable, str(API_MODULE), "all", "--category", category,
           "--ids", str(ids), "--out", str(out), "--min-ukbb", str(min_ukbb)]
    if fresh:
        cmd += ["--refresh", "--remeasure"]
    print("running:", " ".join(P.raw_path(Path(c)) if "/" in c else c for c in cmd[1:]),
          flush=True)
    subprocess.run(cmd, cwd=P.ROOT, check=True)


def _clean_pid(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().lower()


def fallback_records(api, missing: pd.DataFrame, fields) -> dict[str, dict]:
    """{old corpus id -> live record} for ids the live index did not return.

    Asked by DOI first, then by PMID for whatever is still missing. A record is accepted
    only when exactly one live publication carries that DOI/PMID, so an ambiguous match
    never silently supplies another paper's citations.
    """
    found: dict[str, dict] = {}
    returned = "+".join(dict.fromkeys([*fields, "doi", "pmid"]))
    for pid in ("doi", "pmid"):
        todo = missing[~missing["id"].isin(found)]
        todo = todo.assign(pid_value=todo[pid].map(_clean_pid))
        todo = todo[todo["pid_value"] != ""]
        values = sorted(set(todo["pid_value"]))
        hits: dict[str, list[dict]] = {}
        for start in range(0, len(values), FALLBACK_BATCH):
            chunk = values[start:start + FALLBACK_BATCH]
            listed = ",".join(json.dumps(v) for v in chunk)
            res = api.query(f"search publications where {pid} in [{listed}] "
                            f"return publications[{returned}] limit 1000")
            for record in res.get("publications", []):
                key = _clean_pid(record.get(pid))
                if key:
                    hits.setdefault(key, []).append(record)
        for row in todo.itertuples(index=False):
            candidates = hits.get(row.pid_value, [])
            if len(candidates) == 1:
                found[row.id] = dict(candidates[0], _route=pid)
    return found


def citation_values(corpus: pd.DataFrame, records: dict[str, dict],
                    fallback: dict[str, dict]) -> tuple[dict[str, list], dict[str, int]]:
    """New values for CITATION_COLUMNS, aligned to `corpus` row order, plus counts."""
    out: dict[str, list] = {c: [] for c in CITATION_COLUMNS}
    tally = {"live": 0, "fallback": 0, "unresolved": 0}
    for row in corpus.itertuples(index=False):
        record = records.get(row.id) or fallback.get(row.id)
        tally["live" if row.id in records else "fallback" if record else "unresolved"] += 1
        for column in CITATION_COLUMNS:
            old = getattr(row, column)
            if record is None or column not in record:
                out[column].append(old)
                continue
            new = record[column]
            if new is None and column in INTEGER_COLUMNS:
                new = old
            out[column].append(new)
    return out, tally


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--corpus", type=Path, default=P.SHOWCASE_PLUS)
    parser.add_argument("--out", type=Path, default=P.FOR_COUNTS_API)
    parser.add_argument("--category", default="for")
    parser.add_argument("--min-ukbb", type=int, default=10,
                        help="must match API_MIN_UKBB in 03_academic_impact.ipynb")
    parser.add_argument("--fresh", action="store_true",
                        help="re-fetch the records and re-measure every threshold cell")
    parser.add_argument("--skip-api", action="store_true",
                        help="make no Dimensions call: write back from the cached records "
                             "and skip the DOI/PMID fallback")
    parser.add_argument("--no-write", action="store_true",
                        help="leave the corpus's citation columns unchanged")
    args = parser.parse_args(argv)
    install_keyed_api_guard()

    ids = write_ids(args.corpus, args.out)
    if not args.skip_api:
        run_api(ids, args.out, args.category, args.min_ukbb, args.fresh)

    cache = args.out / RECORDS_JSON
    if not cache.exists():
        print(f"no {P.raw_path(cache)} — run without --skip-api first", file=sys.stderr)
        return 1
    records = {str(r["id"]): r for r in json.loads(cache.read_text()) if r.get("id")}
    corpus = pd.read_parquet(args.corpus, columns=["id", "doi", "pmid", *CITATION_COLUMNS])
    missing = corpus[~corpus["id"].isin(records)]
    print(f"live records {len(records):,}; corpus ids the index did not return: {len(missing):,}")

    fallback: dict[str, dict] = {}
    if len(missing) and not args.skip_api:
        from utils.data_analysis_03_academic_impact_dimensions_api import (
            DEFAULT_CONFIG, RECORD_FIELDS, Dimensions, load_config)
        api = Dimensions(load_config(P.ROOT / DEFAULT_CONFIG))
        fallback = fallback_records(api, missing, RECORD_FIELDS)
        remap = pd.DataFrame([{"corpus_id": old, "live_id": r.get("id"), "route": r["_route"]}
                              for old, r in sorted(fallback.items())],
                             columns=["corpus_id", "live_id", "route"])
        remap.to_csv(args.out / "id_remap.csv", index=False)
        print(f"recovered by DOI/PMID: {len(fallback):,} -> {P.raw_path(args.out / 'id_remap.csv')}")

    values, tally = citation_values(corpus, records, fallback)
    print(f"citation columns: {tally['live']:,} live, {tally['fallback']:,} by fallback, "
          f"{tally['unresolved']:,} kept at their previous values")
    summary = {
        "records_written_utc": datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc).isoformat(),
        "records_file": P.raw_path(cache),
        **tally,
        "times_cited_total_before": int(corpus["times_cited"].sum()),
        "times_cited_total_after": int(pd.Series(values["times_cited"]).sum()),
    }
    (args.out / "citation_refresh.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.no_write:
        print("--no-write: corpus citation columns left unchanged")
        return 0
    write_corpus_columns(values, args.corpus)
    print(f"wrote {', '.join(CITATION_COLUMNS)} -> {P.raw_path(args.corpus)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
