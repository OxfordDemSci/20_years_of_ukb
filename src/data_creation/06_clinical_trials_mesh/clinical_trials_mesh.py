#!/usr/bin/env python3
"""Stage 06 — keep the trials that cite a UK Biobank paper, and refresh their MeSH terms.

WHICH TRIALS
------------
The endpoint pipeline links a paper to a trial in two directions and merges them into one
`clinical_trials__linked_ids`: the TRIAL cites the paper (`clinical_trials.publication_ids`)
or the PAPER names the trial in its own record (`publications.clinical_trial_ids` — usually
a trial whose data the paper reports). Analysis 04's "reach into clinical trials" means the
first only. In the 2026-10-13 pull the second direction was 1,154 of 1,421 links and
quadrupled the trial count (195 -> 975), so this stage first narrows the block to the links
where the paper's id is in that trial's own `publication_ids` — the authors' decision, D54.
`--all-trial-links` keeps both directions.

WHY THE MESH STEP
-----------------
Dimensions ships a trial's `mesh_terms` as one flattened list in which ClinicalTrials.gov's
three buckets — the matched LEAF diseases, the MeSH-tree ANCESTORS, and terms from the
INTERVENTION text — are mixed, so tree scaffolding ("Pathologic Processes") outranks every
real disease. `src/utils/data_creation_clinical_trials_fetch_ctgov.py` re-fetches the
buckets separately; analysis 04 turns the leaf descriptor ids into ICD-10 chapters through
`data_analysis_04_non_academic_clinical_trials_mesh_tree.py`.

Until now those columns lived only in the side CSV `P.CT_CSV`, so the corpus block and the
CSV were two retrievals on two dates, and `mesh_leaf_ids` could be had from nowhere but the
CSV (`LOCAL_ONLY_FIELDS` in `data_analysis_04_non_academic_sources.py`). This stage writes
all four columns INTO the corpus's `clinical_trials__` block, aligned to
`clinical_trials__id` exactly as the endpoint pipeline aligns every other field, so every
trial field is read through `shared_showcase.endpoint_records` (D44) from one file.

WHAT IT DOES
------------
0. narrows the corpus's `clinical_trials__` block to trials that cite the paper (above);
1. reads every remaining trial from the corpus block;
2. fetches the ClinicalTrials.gov browse modules (keyless; cached in `P.CTGOV_CACHE`;
   transient failures retried);
3. writes `clinical_trials__mesh_leaf`, `__mesh_leaf_ids`, `__mesh_ancestors`,
   `__mesh_intervention` into the corpus, atomically;
4. resolves every leaf descriptor to MeSH tree numbers (keyless NLM SPARQL; cached in
   `P.MESH_TREE_CACHE`), so analysis 04's cache-only call finds all of them;
5. rewrites `P.CT_CSV` from the enriched records.

USAGE
-----
    python src/data_creation/06_clinical_trials_mesh/clinical_trials_mesh.py
    python src/data_creation/06_clinical_trials_mesh/clinical_trials_mesh.py --no-write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import shared_paths as P  # noqa: E402
from utils.data_creation_clinical_trials_fetch_ctgov import (  # noqa: E402
    NEW_COLS, fetch_ctgov_browse, mesh_columns, report)
from utils.data_analysis_04_non_academic_clinical_trials_mesh_tree import (  # noqa: E402
    fetch_tree_numbers)
from utils.data_creation_runtime import install_keyed_api_guard, json_array  # noqa: E402
from utils.shared_showcase import endpoint_records, parse_listcol, write_corpus_columns  # noqa: E402

PREFIX = "clinical_trials"
#: Pipeline bookkeeping columns of the block, recomputed rather than filtered.
LINK_META = (f"{PREFIX}__linked_ids", f"{PREFIX}__n_links", f"{PREFIX}__n_records")


def _cell_list(cell) -> list:
    """An endpoint array element: a list, or a JSON/repr string of one (double encoding)."""
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str) and cell.strip()[:1] == "[":
        return parse_listcol(cell)
    return []


def citing_only_block(corpus: pd.DataFrame) -> tuple[dict[str, list], dict[str, int]]:
    """The `clinical_trials__` block with only the trials that cite each row's paper.

    `corpus` holds `id` and every `clinical_trials__*` column, in file order. A trial is kept
    on a paper's row when the paper's id is in that trial's `publication_ids`; every field
    array is filtered at the same positions, and the three link-bookkeeping columns are
    recomputed. Idempotent: on an already-narrowed block it changes nothing.
    """
    id_col, pubs_col = f"{PREFIX}__id", f"{PREFIX}__publication_ids"
    fields = [c for c in corpus.columns if c.startswith(f"{PREFIX}__") and c not in LINK_META]
    out: dict[str, list] = {c: [] for c in [*fields, *LINK_META]}
    tally = {"links_before": 0, "links_after": 0, "misaligned_cells": 0}
    for row in corpus.itertuples(index=False):
        values = dict(zip(corpus.columns, row))
        ids = parse_listcol(values[id_col]) if isinstance(values[id_col], str) and values[id_col] else []
        pubs = parse_listcol(values[pubs_col]) if isinstance(values[pubs_col], str) and values[pubs_col] else []
        keep = [k for k in range(len(ids))
                if len(pubs) == len(ids) and values["id"] in _cell_list(pubs[k])]
        tally["links_before"] += len(ids)
        tally["links_after"] += len(keep)
        for column in fields:
            cell = values[column]
            array = parse_listcol(cell) if isinstance(cell, str) and cell else []
            if not keep:
                out[column].append("")
            elif len(array) != len(ids):
                tally["misaligned_cells"] += 1
                out[column].append("")
            else:
                out[column].append(json_array([array[k] for k in keep]))
        kept_ids = [ids[k] for k in keep]
        out[f"{PREFIX}__linked_ids"].append(json_array(kept_ids) if kept_ids else "")
        out[f"{PREFIX}__n_links"].append(len(kept_ids))
        out[f"{PREFIX}__n_records"].append(len(kept_ids))
    return out, tally


def mesh_block(record_ids: pd.Series, enriched: pd.DataFrame) -> dict[str, list]:
    """The four `clinical_trials__mesh_*` columns, one cell per publication row.

    `record_ids` is the corpus's `clinical_trials__id` column, in file order. Each cell is
    a JSON array whose k-th element belongs to the k-th trial id — the alignment every
    endpoint field has — and a row with no linked trial gets "", as the pipeline writes.
    The array follows `__id` (the records found), NOT `__linked_ids`: the pipeline drops a
    linked id it found no record for, so the two can differ in length.
    """
    by_id = enriched.drop_duplicates("id").set_index("id")
    out: dict[str, list] = {f"{PREFIX}__{c}": [] for c in NEW_COLS}
    for cell in record_ids:
        ids = parse_listcol(cell) if isinstance(cell, str) and cell else []
        for column in NEW_COLS:
            if not ids:
                out[f"{PREFIX}__{column}"].append("")
                continue
            values = [by_id.at[i, column] if i in by_id.index else [] for i in ids]
            out[f"{PREFIX}__{column}"].append(json_array(values))
    return out


def side_csv(enriched: pd.DataFrame, path: Path) -> None:
    """`P.CT_CSV`, list and dict cells JSON-encoded as `ensure_endpoint_csv` writes them."""
    serialized = enriched.map(
        lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized.to_csv(path, index=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--corpus", type=Path, default=P.SHOWCASE_PLUS)
    parser.add_argument("--csv", type=Path, default=P.CT_CSV)
    parser.add_argument("--refresh", action="store_true",
                        help="ignore both caches and fetch every trial and descriptor again")
    parser.add_argument("--no-write", action="store_true",
                        help="fetch and report only; leave the corpus and the CSV alone")
    parser.add_argument("--all-trial-links", action="store_true",
                        help="keep trials a paper merely names, not only trials citing it")
    args = parser.parse_args(argv)
    install_keyed_api_guard()

    if not args.all_trial_links:
        import pyarrow.parquet as pq
        columns = ["id", *[c for c in pq.read_schema(args.corpus).names
                           if c.startswith(f"{PREFIX}__")]]
        block, tally = citing_only_block(pd.read_parquet(args.corpus, columns=columns))
        print(f"trial links where the trial cites the paper: {tally['links_after']:,} "
              f"of {tally['links_before']:,}")
        if tally["misaligned_cells"]:
            print(f"!! {tally['misaligned_cells']} misaligned cells blanked")
        if not args.no_write and tally["links_after"] != tally["links_before"]:
            write_corpus_columns(block, args.corpus)
            print(f"narrowed the {PREFIX}__ block of {P.raw_path(args.corpus)}")

    trials = endpoint_records(PREFIX, path=args.corpus)
    trials = trials.drop(columns=[c for c in NEW_COLS if c in trials.columns])
    print(f"{len(trials):,} trials in the corpus block of {P.raw_path(args.corpus)}")

    browse = fetch_ctgov_browse(trials["id"].tolist(), cache=P.CTGOV_CACHE,
                                refresh=args.refresh)
    enriched = mesh_columns(trials, browse)
    if "mesh_terms" in enriched.columns:
        report(enriched)

    descriptors = {d for ids in enriched["mesh_leaf_ids"] for d in ids}
    trees = fetch_tree_numbers(descriptors, cache=P.MESH_TREE_CACHE, refresh=args.refresh)
    unresolved = sorted(d for d in descriptors if d not in trees)
    print(f"MeSH leaf descriptors: {len(descriptors):,}, resolved {len(descriptors) - len(unresolved):,}")
    with_leaf = int(enriched["mesh_leaf"].str.len().gt(0).sum())
    print(f"trials with >=1 leaf term: {with_leaf:,}/{len(enriched):,}")

    if args.no_write:
        print("--no-write: corpus and CSV left unchanged")
        return 0
    corpus = pd.read_parquet(args.corpus, columns=["id", f"{PREFIX}__id"])
    write_corpus_columns(mesh_block(corpus[f"{PREFIX}__id"], enriched), args.corpus)
    print(f"wrote {len(NEW_COLS)} columns into the {PREFIX}__ block of {P.raw_path(args.corpus)}")
    side_csv(enriched, args.csv)
    print(f"wrote {len(enriched):,} trials -> {P.raw_path(args.csv)}")
    if unresolved:
        print(f"!! {len(unresolved)} descriptors have no tree numbers cached; analysis 04's "
              f"cache-only lookup will raise until a re-run resolves them")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
