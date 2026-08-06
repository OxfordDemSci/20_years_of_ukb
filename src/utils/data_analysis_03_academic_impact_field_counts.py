#!/usr/bin/env python3
"""
Category counts over a large parquet corpus (local file or BMRC VM)
==================================================================

WHAT THIS ANSWERS
-----------------
"How many papers per year fall in each research category, with vs without the UKBB
dataset?" — for any of the Dimensions classification systems, not just FOR.

The UKBB papers are a *subset* of the full Dimensions corpus that lives on the VM as
~5,000 parquet files (~230 GB). So the comparison is:

    UKBB arm        = the corpus rows whose id IS in the UKBB id list
    background arm  = the corpus rows whose id is NOT in that list

Both arms are produced by the same code path, flipped with `--id-filter`.

CATEGORY SYSTEMS (`--category`)
-------------------------------
Every `category_*` column in the corpus is a multi-label set stored the same way. They
differ only in (a) which on-disk column holds them and (b) whether the codes form a
hierarchy. This script handles both shapes through one registry (see CATEGORIES):

    for        HIERARCHICAL — ANZSRC 2020 Fields of Research. 2-digit codes are L2
               divisions, 4-digit codes are L4 fields (parent division = first two
               digits). Emits both levels.  cover ~99.9 %
    rcdc       flat — NIH Research/Condition/Disease topics ("Genetics", "Aging").
               No codes; the label IS the identifier.  cover ~98.6 %, most granular.
    uoa        flat — REF Units of Assessment ("A01 Clinical Medicine").  cover ~99.6 %
    hrcs_hc    flat — HRCS Health Categories ("Cardiovascular").  cover ~74 %
    hrcs_rac   flat — HRCS Research Activity Codes ("2.1 ...").    cover ~63 %
    bra / hra  flat — Broad / Health Research Areas (4 buckets).   coarse
    sdg        flat — UN SDGs. present ~51 % but ~97 % a single label — low information.

A flat system has one level, named "L1". A hierarchical system has several
("L2", "L4"). Everything downstream is written in terms of `level`, so the same code
serves every system.

WHY THIS IS CHEAP (READ THIS BEFORE SIZING THE SLURM JOB)
---------------------------------------------------------
230 GB is the size of the *files*, not the size of the *work*. Parquet stores each
column as a contiguous chunk, so a projected read seeks straight past everything else.
Measured on data/showcase/showcase_plus_all_endpoint.parquet (26,109 rows, 72 columns):
id+year+type+category_for_2020 = 214 KB = 0.16 % of the 130 MB file. RCDC is ~50 KB
more. Run `probe` FIRST to confirm that on the real data — it reads only the parquet
footer and prints the projected byte cost, so you size the array job from measurement.

THREE MODES
-----------
    probe   footer-only scan  -> manifest.csv (rows, projected bytes, category column)
    count   the worker        -> partial parquets for one shard of the file list
    merge   sum the partials  -> final tables

`count` is shard-aware (`--shard i --num-shards n`), which is the only thing the Slurm
array job needs; see the companion .sh.

COUNTING SEMANTICS
------------------
One paper carries several codes, e.g. FOR:

    [{"id":"80003","name":"32 Biomedical and Clinical Sciences"},
     {"id":"80137","name":"4202 Epidemiology"}, ...]

Codes are de-duplicated within a paper, within each level, before counting.

    n_papers   absolute: the paper adds 1.0 to every distinct code it carries. Column
               totals exceed the paper count (double counting is the point — "how many
               papers touch this category").

    n_frac     fractional: the paper's weight is split evenly across its codes, EACH
               LEVEL NORMALISED SEPARATELY. So for any (year, type) slice the fractional
               total over a level equals the number of papers carrying >=1 code at that
               level (recorded in the `coverage` table) — which is what makes the
               reconciliation check in `merge` exact.

DENOMINATORS ARE NOT OPTIONAL
-----------------------------
Every run writes `totals` (papers seen / filtered / with-no-category per year,type) and
`coverage` (papers carrying >=1 code at each level). Without them you cannot tell "this
category shrank" from "the classification got sparser in that snapshot" — real for the
lower-coverage systems (HRCS, SDG).

PARSING: REGEX, NOT ast.literal_eval
------------------------------------
The cell is JSON with a rigid shape, so a single regex over the raw string yields the
same (code, label) pairs ~50x faster than ast.literal_eval per row. `selfcheck` asserts
the FOR fast-path matches shared_for_utils on real rows. Some newer corpus exports store
the column as a nested struct instead of text; both shapes are handled.

READ-ONLY BY CONSTRUCTION
-------------------------
Source files are only ever opened through pyarrow for reading. All output goes to --out.

Author: Jiani Y
Date: 2026-08-05
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

ID_COL, YEAR_COL, TYPE_COL = "id", "year", "type"
MISSING_YEAR = -1
MISSING_TYPE = "unknown"

# '"name":"4202 Epidemiology"' -> ('4202', 'Epidemiology'). Anchored on the JSON key so a
# digit-leading label elsewhere in the row cannot match. FOR-only (numeric-coded).
FOR_PAT = re.compile(r'"name"\s*:\s*"(\d{2,4}) ([^"]*)"')
# Any classification label: '"name":"<whatever>"'. Used for the flat systems.
NAME_PAT = re.compile(r'"name"\s*:\s*"([^"]*)"')


# =============================================================================
# CATEGORY REGISTRY  — the whole generalisation lives here
# =============================================================================
class CategorySpec:
    """One classification system: where it lives on disk and how to read its levels.

    kind='hierarchical'  numeric codes split by length into several levels (FOR).
    kind='flat'          one level "L1"; each label is its own identifier. `code_re`,
                         when set, peels a leading code token off the label
                         ("A01 Clinical Medicine" -> code "A01", label "Clinical
                         Medicine"); when None the whole name is both code and label
                         (RCDC's "Genetics").
    """

    def __init__(self, name: str, columns: Tuple[str, ...], kind: str,
                 code_re: Optional[Pattern] = None) -> None:
        self.name = name
        self.columns = columns
        self.kind = kind
        self.code_re = code_re

    @property
    def hierarchical(self) -> bool:
        return self.kind == "hierarchical"

    def resolve_column(self, schema_names: Iterable[str]) -> Optional[str]:
        """Which of this system's columns the file carries, or None. Never guesses."""
        names = set(schema_names)
        for c in self.columns:
            if c in names:
                return c
        return None

    def levels(self, cell, fmt: str) -> Dict[str, Set[Tuple[str, str]]]:
        """Map level -> set of (code, label) for one cell, de-duplicated within level."""
        if self.hierarchical:
            l2: Set[Tuple[str, str]] = set()
            l4: Set[Tuple[str, str]] = set()
            for code, label in _for_pairs(cell, fmt):
                if len(code) == 2:
                    l2.add((code, label))
                elif len(code) == 4:
                    l4.add((code, label))
                # 3-digit codes are not part of ANZSRC 2020; ignore.
            return {"L2": l2, "L4": l4}
        items: Set[Tuple[str, str]] = set()
        for code, label in _flat_pairs(cell, fmt, self.code_re):
            items.add((code, label))
        return {"L1": items}


CATEGORIES: Dict[str, CategorySpec] = {
    "for":      CategorySpec("for", ("category_for_2020", "category_for"), "hierarchical"),
    "rcdc":     CategorySpec("rcdc", ("category_rcdc",), "flat"),
    "uoa":      CategorySpec("uoa", ("category_uoa",), "flat",
                            re.compile(r"^([A-Za-z]\d[\w]*) (.*)$")),
    "hrcs_hc":  CategorySpec("hrcs_hc", ("category_hrcs_hc",), "flat"),
    "hrcs_rac": CategorySpec("hrcs_rac", ("category_hrcs_rac",), "flat",
                            re.compile(r"^(\d[\d.]*) (.*)$")),
    "sdg":      CategorySpec("sdg", ("category_sdg",), "flat",
                            re.compile(r"^(\d+) (.*)$")),
    "bra":      CategorySpec("bra", ("category_bra",), "flat"),
    "hra":      CategorySpec("hra", ("category_hra",), "flat"),
}


# =============================================================================
# CELL PARSING — text (primary) and struct (some newer exports)
# =============================================================================
def column_format(schema: pa.Schema, col: str) -> str:
    """'text' (JSON-as-string, parsed by regex) unless the column is a struct on disk."""
    return "struct" if pa.types.is_struct(schema.field(col).type) else "text"


def _for_pairs(cell, fmt: str) -> Iterable[Tuple[str, str]]:
    """(code, label) pairs from a FOR cell, whichever shape it's stored in."""
    if fmt == "struct":
        if not cell:
            return
        for level in ("first_level", "second_level"):
            for entry in (cell.get(level) or {}).get("full") or []:
                code = entry.get("code")
                if code:
                    yield code, entry.get("name") or ""
    elif cell:
        yield from FOR_PAT.findall(cell)


def _iter_struct_entries(cell) -> Iterable[Tuple[Optional[str], str]]:
    """Best-effort (code, name) from a flat-category struct cell. Dimensions struct
    shapes vary, so we tolerate a plain list of dicts and a {'full': [...]} wrapper.
    Verify the actual shape on the VM with `probe`/a spot check before trusting counts."""
    if cell is None:
        return
    seq = cell.get("full") if isinstance(cell, dict) else cell
    for entry in seq or []:
        if isinstance(entry, dict):
            yield entry.get("code"), entry.get("name") or ""


def _flat_pairs(cell, fmt: str, code_re: Optional[Pattern]) -> Iterable[Tuple[str, str]]:
    """(code, label) pairs from a flat-category cell.

    Without codes, identifier == label == the full name (RCDC). With `code_re`, the
    leading code token is split off so the label reads cleanly and the code is a stable
    key ("A01 Clinical Medicine" -> ("A01", "Clinical Medicine"))."""
    if fmt == "struct":
        for code, name in _iter_struct_entries(cell):
            if code:
                # a struct-carried code wins; strip it off the name if it leads.
                label = name[len(code):].lstrip() if name.startswith(code) else name
                yield code, label or name
            else:
                yield name, name
        return
    if not cell:
        return
    for name in NAME_PAT.findall(cell):
        if code_re is not None:
            m = code_re.match(name)
            if m:
                yield m.group(1), m.group(2)
                continue
        yield name, name


# =============================================================================
# FILE LIST
# =============================================================================
def collect_files(args: argparse.Namespace) -> List[str]:
    """Resolve --files / --file-list / --glob into a sorted, deterministic list.

    Sorted because the shard assignment must be identical in every array task; an
    unordered glob would silently make tasks overlap or skip files.
    """
    files: List[str] = []
    if args.files:
        files.extend(args.files)
    if args.file_list:
        with open(args.file_list) as fh:
            files.extend(ln.strip() for ln in fh if ln.strip() and not ln.startswith("#"))
    if args.glob:
        for pattern in args.glob:
            root, _, pat = pattern.rpartition("/")
            files.extend(str(p) for p in Path(root or ".").glob(pat or pattern))
    files = sorted(set(files))
    if not files:
        sys.exit("No input files. Use --files, --file-list or --glob.")
    return files


def take_shard(files: Sequence[str], shard: int, num_shards: int) -> List[str]:
    """Round-robin shard, so a run of large files is spread across tasks, not piled on one."""
    if num_shards <= 1:
        return list(files)
    if not 0 <= shard < num_shards:
        sys.exit(f"--shard {shard} out of range for --num-shards {num_shards}")
    return [f for i, f in enumerate(files) if i % num_shards == shard]


def _shard_suffix(args: argparse.Namespace) -> str:
    return f".shard{args.shard:05d}" if args.num_shards > 1 else ""


# =============================================================================
# MODE: probe  — footer only, no data pages
# =============================================================================
def cmd_probe(args: argparse.Namespace) -> None:
    spec = CATEGORIES[args.category]
    files = take_shard(collect_files(args), args.shard, args.num_shards)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"manifest.{spec.name}{_shard_suffix(args)}.csv"

    tot_rows = tot_proj = tot_file = 0
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "rows", "row_groups", "file_bytes", "projected_bytes",
                    "projected_pct", "cat_column", "error"])
        for path in files:
            try:
                pf = pq.ParquetFile(path)
                md = pf.metadata
                cat_col = spec.resolve_column(pf.schema_arrow.names)
                wanted = {ID_COL, YEAR_COL, TYPE_COL} | ({cat_col} if cat_col else set())
                proj = 0
                for rg in range(md.num_row_groups):
                    g = md.row_group(rg)
                    for i in range(g.num_columns):
                        col = g.column(i)
                        if col.path_in_schema in wanted:
                            proj += col.total_compressed_size
                fsz = os.path.getsize(path)
                w.writerow([path, md.num_rows, md.num_row_groups, fsz, proj,
                            f"{100 * proj / fsz:.3f}" if fsz else "", cat_col or "", ""])
                tot_rows += md.num_rows
                tot_proj += proj
                tot_file += fsz
            except Exception as exc:  # a corrupt file must not kill the survey
                w.writerow([path, "", "", "", "", "", "", f"{type(exc).__name__}: {exc}"])

    print(f"probed {len(files)} files ({spec.name}) -> {dest}")
    print(f"  rows            {tot_rows:,}")
    print(f"  file bytes      {tot_file / 1e9:.2f} GB")
    if tot_file:
        print(f"  projected read  {tot_proj / 1e9:.4f} GB  "
              f"({100 * tot_proj / tot_file:.3f} % of the corpus)")


# =============================================================================
# MODE: count
# =============================================================================
def load_id_set(path: str, column: str = ID_COL) -> Set[str]:
    """UKBB ids as a set. Accepts .parquet/.csv (column `id`) or one-id-per-line text."""
    p = Path(path)
    if p.suffix == ".parquet":
        ids = pq.read_table(path, columns=[column]).column(0).to_pylist()
    elif p.suffix in (".csv", ".tsv"):
        import pandas as pd
        ids = pd.read_csv(path, sep="\t" if p.suffix == ".tsv" else ",")[column].tolist()
    else:
        with open(path) as fh:
            ids = [ln.strip() for ln in fh if ln.strip()]
    out = {i for i in ids if i}
    if not out:
        sys.exit(f"--exclude-ids {path} yielded no ids (column '{column}'?)")
    return out


class Tally:
    """Accumulates the count / totals / coverage tables. Plain dicts — no pandas hot loop."""

    def __init__(self) -> None:
        # (year, type, level, code) -> [absolute, fractional]
        self.codes: Dict[Tuple[int, str, str, str], List[float]] = defaultdict(
            lambda: [0.0, 0.0])
        # (year, type) -> [seen, filtered_out, no_category]
        self.totals: Dict[Tuple[int, str], List[int]] = defaultdict(lambda: [0, 0, 0])
        # (year, type, level) -> papers carrying >=1 code at that level (the denominator)
        self.cov: Dict[Tuple[int, str, str], int] = defaultdict(int)
        self.labels: Dict[str, str] = {}          # code -> label, harvested from the data
        self.n_rows = self.n_kept = self.n_dropped = 0

    def add_row(self, year: int, typ: str, cell, spec: CategorySpec, fmt: str) -> None:
        t = self.totals[(year, typ)]
        t[0] += 1
        self.n_kept += 1

        any_tag = False
        for level, items in spec.levels(cell, fmt).items():
            if not items:
                continue
            any_tag = True
            self.cov[(year, typ, level)] += 1
            # Each level normalised on its own denominator — see COUNTING SEMANTICS.
            w = 1.0 / len(items)
            for code, label in items:
                slot = self.codes[(year, typ, level, code)]
                slot[0] += 1.0
                slot[1] += w
                if code not in self.labels:
                    self.labels[code] = label
        if not any_tag:
            t[2] += 1

    def note_filtered(self, year: int, typ: str) -> None:
        self.totals[(year, typ)][1] += 1
        self.n_dropped += 1


def count_file(path: str, tally: Tally, spec: CategorySpec, id_set: Optional[Set[str]],
               mode: str, batch_size: int) -> Tuple[int, str]:
    """Stream one file into `tally`. Returns (rows_read, category_column_used).

    `mode`: 'exclude' drops ids in the set (background arm), 'include' keeps only them
    (UKBB arm), 'none' ignores the set entirely.
    """
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    cat_col = spec.resolve_column(names)
    if cat_col is None:
        raise KeyError(f"no {spec.name} column (looked for {list(spec.columns)})")
    for required in (ID_COL, YEAR_COL):
        if required not in names:
            raise KeyError(f"missing required column '{required}'")
    has_type = TYPE_COL in names
    fmt = column_format(pf.schema_arrow, cat_col)

    cols = [ID_COL, YEAR_COL, cat_col] + ([TYPE_COL] if has_type else [])
    rows = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        ids = batch.column(cols.index(ID_COL)).to_pylist()
        years = batch.column(cols.index(YEAR_COL)).to_pylist()
        cells = batch.column(cols.index(cat_col)).to_pylist()
        types = (batch.column(cols.index(TYPE_COL)).to_pylist()
                 if has_type else [MISSING_TYPE] * len(ids))
        rows += len(ids)
        tally.n_rows += len(ids)

        for pid, year, typ, cell in zip(ids, years, types, cells):
            try:
                y = int(year)
            except (TypeError, ValueError):
                y = MISSING_YEAR
            t = typ or MISSING_TYPE
            if id_set is not None and mode != "none":
                hit = pid in id_set
                if (mode == "exclude" and hit) or (mode == "include" and not hit):
                    tally.note_filtered(y, t)
                    continue
            tally.add_row(y, t, cell, spec, fmt)
    return rows, cat_col


def tally_to_tables(tally: Tally, label: str, spec: CategorySpec) -> Tuple[pa.Table, pa.Table, pa.Table]:
    """(counts, totals, coverage) as arrow tables. The counts schema keeps the historical
    column names `for_label` / `parent_l2` (generic label / parent) so existing readers
    keep working; `parent_l2` is the 2-digit division for FOR, empty for flat systems."""
    keys = sorted(tally.codes)
    counts = pa.table({
        "arm": pa.array([label] * len(keys), pa.string()),
        "category": pa.array([spec.name] * len(keys), pa.string()),
        "year": pa.array([k[0] for k in keys], pa.int32()),
        "type": pa.array([k[1] for k in keys], pa.string()),
        "level": pa.array([k[2] for k in keys], pa.string()),
        "code": pa.array([k[3] for k in keys], pa.string()),
        "for_label": pa.array([tally.labels.get(k[3], "") for k in keys], pa.string()),
        "parent_l2": pa.array([k[3][:2] if spec.hierarchical else "" for k in keys], pa.string()),
        "n_papers": pa.array([tally.codes[k][0] for k in keys], pa.float64()),
        "n_frac": pa.array([tally.codes[k][1] for k in keys], pa.float64()),
    })
    tkeys = sorted(tally.totals)
    totals = pa.table({
        "arm": pa.array([label] * len(tkeys), pa.string()),
        "category": pa.array([spec.name] * len(tkeys), pa.string()),
        "year": pa.array([k[0] for k in tkeys], pa.int32()),
        "type": pa.array([k[1] for k in tkeys], pa.string()),
        "n_papers": pa.array([tally.totals[k][0] for k in tkeys], pa.int64()),
        "n_filtered_out": pa.array([tally.totals[k][1] for k in tkeys], pa.int64()),
        "n_no_cat": pa.array([tally.totals[k][2] for k in tkeys], pa.int64()),
    })
    ckeys = sorted(tally.cov)
    coverage = pa.table({
        "arm": pa.array([label] * len(ckeys), pa.string()),
        "category": pa.array([spec.name] * len(ckeys), pa.string()),
        "year": pa.array([k[0] for k in ckeys], pa.int32()),
        "type": pa.array([k[1] for k in ckeys], pa.string()),
        "level": pa.array([k[2] for k in ckeys], pa.string()),
        "n_with_level": pa.array([tally.cov[k] for k in ckeys], pa.int64()),
    })
    return counts, totals, coverage


def cmd_count(args: argparse.Namespace) -> None:
    spec = CATEGORIES[args.category]
    files = take_shard(collect_files(args), args.shard, args.num_shards)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    id_set = load_id_set(args.exclude_ids, args.id_column) if args.exclude_ids else None
    if args.id_filter != "none" and id_set is None:
        sys.exit(f"--id-filter {args.id_filter} needs --exclude-ids")
    print(f"[{args.label}/{spec.name}] {len(files)} file(s), id-filter={args.id_filter}"
          f"{f', {len(id_set):,} ids' if id_set else ''}", flush=True)

    tally = Tally()
    stem = f"{spec.name}.{args.label}{_shard_suffix(args)}"
    log_path = out / f"filelog.{stem}.csv"
    t_all = time.time()
    n_ok = n_err = 0
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "rows", "cat_column", "seconds", "error"])
        for i, path in enumerate(files, 1):
            t0 = time.time()
            try:
                rows, cat_col = count_file(path, tally, spec, id_set, args.id_filter,
                                           args.batch_size)
                w.writerow([path, rows, cat_col, f"{time.time() - t0:.2f}", ""])
                n_ok += 1
            except Exception as exc:
                # One unreadable file among thousands must not lose the shard's work.
                w.writerow([path, "", "", f"{time.time() - t0:.2f}",
                            f"{type(exc).__name__}: {exc}"])
                n_err += 1
                print(f"  !! {path}: {type(exc).__name__}: {exc}", file=sys.stderr,
                      flush=True)
            if args.progress and i % args.progress == 0:
                print(f"  {i}/{len(files)} files, {tally.n_rows:,} rows, "
                      f"{time.time() - t_all:.0f}s", flush=True)
            fh.flush()

    counts, totals, coverage = tally_to_tables(tally, args.label, spec)
    pq.write_table(counts, out / f"counts.{stem}.parquet", compression="zstd")
    pq.write_table(totals, out / f"totals.{stem}.parquet", compression="zstd")
    pq.write_table(coverage, out / f"coverage.{stem}.parquet", compression="zstd")

    print(f"[{args.label}/{spec.name}] {n_ok} ok / {n_err} failed | rows {tally.n_rows:,} | "
          f"counted {tally.n_kept:,} | filtered {tally.n_dropped:,} | "
          f"{len(counts):,} count rows | {time.time() - t_all:.0f}s")
    if n_err:
        print(f"  see {log_path} for the {n_err} failure(s)", file=sys.stderr)


# =============================================================================
# MODE: merge
# =============================================================================
GROUP_KEYS = ["arm", "category", "year", "type", "level", "code", "parent_l2"]


def cmd_merge(args: argparse.Namespace) -> None:
    import pandas as pd

    out = Path(args.out)
    src = Path(args.partials) if args.partials else out

    def _read(prefix: str, required: bool = True):
        parts = sorted(src.glob(f"{prefix}.*.parquet"))
        if not parts:
            if required:
                sys.exit(f"no {prefix}.*.parquet under {src}")
            return None
        print(f"  {prefix}: {len(parts)} partial(s)")
        df = pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)
        if "category" not in df.columns:      # partials written before categories existed
            df["category"] = "for"
        return df

    counts = _read("counts")
    # for_label is carried, not summed: keep the first non-empty label per (category, code).
    labels = (counts.loc[counts.for_label != "", ["category", "code", "for_label"]]
              .drop_duplicates(["category", "code"])
              .set_index(["category", "code"])["for_label"])
    counts = counts.groupby(GROUP_KEYS, as_index=False)[["n_papers", "n_frac"]].sum()
    counts["for_label"] = (counts.set_index(["category", "code"]).index.map(labels)
                           .to_numpy())
    counts["for_label"] = counts["for_label"].astype("object").where(
        counts["for_label"].notna(), "")
    counts["n_papers"] = counts.n_papers.round().astype("int64")
    counts = counts[["arm", "category", "year", "type", "level", "code", "for_label",
                     "parent_l2", "n_papers", "n_frac"]].sort_values(
        ["category", "arm", "level", "year", "n_papers"],
        ascending=[True, True, True, True, False])

    totals = _read("totals").groupby(["arm", "category", "year", "type"],
                                     as_index=False).sum(numeric_only=True)
    coverage = _read("coverage", required=False)
    if coverage is not None:
        coverage = coverage.groupby(["arm", "category", "year", "type", "level"],
                                    as_index=False).sum(numeric_only=True)

    counts.to_parquet(out / "field_counts.parquet", index=False)
    totals.to_parquet(out / "field_totals.parquet", index=False)
    counts.to_csv(out / "field_counts.csv", index=False)
    totals.to_csv(out / "field_totals.csv", index=False)
    if coverage is not None:
        coverage.to_parquet(out / "field_coverage.parquet", index=False)
        coverage.to_csv(out / "field_coverage.csv", index=False)

    print(f"\nwrote {out}/field_counts.parquet  ({len(counts):,} rows)")
    print(f"wrote {out}/field_totals.parquet  ({len(totals):,} rows)")
    if coverage is not None:
        print(f"wrote {out}/field_coverage.parquet ({len(coverage):,} rows)")
    print()

    no_cat_col = "n_no_cat" if "n_no_cat" in totals.columns else "n_no_for"
    for (cat, arm), g in totals.groupby(["category", "arm"]):
        papers = g.n_papers.sum()
        no_cat = g[no_cat_col].sum() if no_cat_col in g.columns else 0
        print(f"[{cat}/{arm}] papers {papers:,} | filtered out {g.n_filtered_out.sum():,} "
              f"| no category {no_cat:,} ({100 * no_cat / max(papers, 1):.2f} %)")

    # Fractional totals must reconcile with the per-level coverage; if they don't, the
    # per-level normalisation is wrong and every downstream share is wrong with it.
    if coverage is not None:
        chk = counts.groupby(["category", "arm", "level"]).n_frac.sum()
        ref = coverage.groupby(["category", "arm", "level"]).n_with_level.sum()
        print("\nreconciliation (n_frac total must equal papers carrying that level):")
        for key in ref.index:
            got = float(chk.get(key, 0.0))
            want = float(ref.loc[key])
            flag = "ok" if abs(got - want) < 1.0 else "MISMATCH"
            print(f"  {key[0]:9s} {key[1]:12s} {key[2]:3s} {got:14,.1f}  vs  "
                  f"{want:14,.0f}   {flag}")
    else:
        print("\n(no coverage.*.parquet found — skipping reconciliation; these are "
              "pre-coverage partials.)")


# =============================================================================
# MODE: selfcheck  — the FOR regex must agree with shared_for_utils
# =============================================================================
def cmd_selfcheck(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils import shared_for_utils as for_utils  # noqa: E402

    path = args.files[0] if args.files else "data/showcase/showcase_plus_all_endpoint.parquet"
    pf = pq.ParquetFile(path)
    for_col = CATEGORIES["for"].resolve_column(pf.schema_arrow.names)
    cells = next(pf.iter_batches(batch_size=args.n, columns=[for_col])).column(0).to_pylist()

    bad = 0
    for cell in cells:
        fast2 = {c for c, _ in FOR_PAT.findall(cell or "") if len(c) == 2}
        fast4 = {c for c, _ in FOR_PAT.findall(cell or "") if len(c) == 4}
        slow2, slow4 = set(), set()
        for d in for_utils.parse_listcol(cell):
            if isinstance(d, dict):
                code, _, level = for_utils.split_for_name(d.get("name"))
                (slow2 if level == "L2" else slow4 if level == "L4" else set()).add(code)
        if fast2 != slow2 or fast4 != slow4:
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH\n    fast {sorted(fast2)} {sorted(fast4)}"
                      f"\n    slow {sorted(slow2)} {sorted(slow4)}\n    {cell!r:.200}")
    print(f"selfcheck {path}: {len(cells)} cells, {bad} mismatch(es) vs shared_for_utils")
    sys.exit(1 if bad else 0)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples
  # 0. does the projection assumption hold on the VM? (pick the system with --category)
  %(prog)s probe --category rcdc --glob '/well/.../corpus/*.parquet' --out out/

  # 1. UKBB arm, RCDC, from the local file
  %(prog)s count --category rcdc \\
      --files data/showcase/showcase_plus_all_endpoint.parquet \\
      --exclude-ids data/showcase/showcase_plus_all_endpoint.parquet \\
      --id-filter include --label ukbb --out out/

  # 2. background arm, one shard of the VM corpus (what Slurm runs)
  %(prog)s count --category rcdc --file-list files.txt --shard 3 --num-shards 50 \\
      --exclude-ids ukbb_ids.txt --id-filter exclude --label background --out out/

  # 3. combine everything (handles multiple categories in one dir)
  %(prog)s merge --out out/
""")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_inputs(sp):
        sp.add_argument("--category", choices=sorted(CATEGORIES), default="for",
                        help="classification system to count (default: for)")
        sp.add_argument("--files", nargs="+", help="explicit parquet paths")
        sp.add_argument("--file-list", help="text file, one parquet path per line")
        sp.add_argument("--glob", nargs="+", help="glob pattern(s), e.g. 'dir/*.parquet'")
        sp.add_argument("--shard", type=int, default=0)
        sp.add_argument("--num-shards", type=int, default=1)
        sp.add_argument("--out", default="out", help="output directory (never the source)")

    sp = sub.add_parser("probe", help="footer-only survey; sizes the job")
    add_inputs(sp)

    sp = sub.add_parser("count", help="tally one shard")
    add_inputs(sp)
    sp.add_argument("--exclude-ids", help="parquet/csv/txt of UKBB paper ids")
    sp.add_argument("--id-column", default=ID_COL)
    sp.add_argument("--id-filter", choices=["exclude", "include", "none"],
                    default="exclude",
                    help="exclude=background arm, include=UKBB arm, none=whole corpus")
    sp.add_argument("--label", default="background", help="arm name in the output")
    sp.add_argument("--batch-size", type=int, default=100_000)
    sp.add_argument("--progress", type=int, default=25, help="log every N files; 0=off")

    sp = sub.add_parser("merge", help="sum partials into the final tables")
    sp.add_argument("--out", default="out")
    sp.add_argument("--partials", help="where the partials are (default: --out)")

    sp = sub.add_parser("selfcheck", help="assert the FOR regex matches shared_for_utils")
    sp.add_argument("--files", nargs="+")
    sp.add_argument("--n", type=int, default=5000)

    args = p.parse_args()
    {"probe": cmd_probe, "count": cmd_count, "merge": cmd_merge,
     "selfcheck": cmd_selfcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
