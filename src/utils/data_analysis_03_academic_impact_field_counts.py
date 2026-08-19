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

CITATION WEIGHTING (`--weights`)
--------------------------------
By default a paper contributes 1.0 to each category it carries. `--weights` adds
PARALLEL columns in which it contributes a citation quantity instead, so "how many
papers" and "how much citation impact" come out of one pass and can be compared:

    cit    times_cited          raw citation mass. 100 % coverage. Accrues over time, so
                                it is only comparable WITHIN a year and WITHIN one
                                snapshot — see THE SNAPSHOT TRAP below.
    fcr    field_citation_ratio Dimensions' field- and year-normalised ratio; 1.0 = world
                                average for that field and year. Published only once a
                                paper is ~2 years old, so it is ABSENT for the two most
                                recent years however well indexed they are, and its
                                coverage differs sharply between arms (97 % vs 73 %).
    top10  derived              1.0 if the paper is in the top decile of its (year, type)
                                by times_cited — a decile of the WHOLE corpus, so a
                                heavily-cited field holds more than a tenth of it and a
                                slow-citing field fewer.
    top10f derived, per-field   the same indicator taken within (year, type, CATEGORY):
                                the top tenth of the field itself, which is what
                                PP(top 10 %) means in the bibliometric literature. The
                                only weight whose value depends on WHICH category it is
                                being counted into — a paper can be top-decile in
                                epidemiology and not in genetics.
    mncs   derived, per-field   times_cited / mean times_cited of the same (year, type,
                                category) — a mean-normalised citation score built here
                                rather than taken from the corpus. It is FCR's job done
                                with our own reference set, which buys two things FCR
                                cannot: it exists for the newest years, and it is
                                measured for every paper carrying a citation count
                                (~99 % of both arms). Validate it against fcr on the
                                years where both exist before trusting it.
    recent recent_citations     citations in the last two years. Same snapshot caveat.

top10f and mncs both need the per-category reference tables written by `thresholds`;
pass them with --top-thresholds and --expected. Thin (year, type, category) cells fall
back to the (year, type) figure, so a small field is never scored against twelve papers.

Each weight emits FOUR columns per (year, type, level, code) cell:

    n_<w>          the weight summed absolutely   (paper counts once per category)
    n_<w>_frac     the weight split across the paper's categories, per level
    n_<w>_docs     papers with a NON-NULL value for that weight (absolute)
    n_<w>_docs_frac    the same, fractionally

The doc counts are not optional either. A null weight contributes 0 to the sums, so
without them "few citations" and "no FCR value for this paper" are indistinguishable,
and any mean is computed over the wrong denominator. Intensity is n_<w> / n_<w>_docs;
mass is n_<w>. `field_totals` carries the same pair per (year, type) so a share or an
activity index has a denominator that is not itself a sum over double-counted rows.

THE SNAPSHOT TRAP
-----------------
Citations accrue continuously, so an arm read from a LATER snapshot looks more cited
for no scientific reason. Paper counts survive a snapshot mismatch between the arms;
citation weights do not. Count BOTH arms from the same file list — the UKBB arm with
`--id-filter include`, the background arm with `--id-filter exclude` — and the two
share one as-of date by construction. Also stop citation-weighted claims about two
years short of the snapshot: fresh papers have not been cited yet, and while FCR
normalises by year it is still unstable in its last two.

WHY THIS IS CHEAP (READ THIS BEFORE SIZING THE SLURM JOB)
---------------------------------------------------------
230 GB is the size of the *files*, not the size of the *work*. Parquet stores each
column as a contiguous chunk, so a projected read seeks straight past everything else.
Measured on data/showcase/showcase_plus_all_endpoint.parquet (26,109 rows, 72 columns):
id+year+type+category_for_2020 = 214 KB = 0.16 % of the 130 MB file. RCDC is ~50 KB
more. Run `probe` FIRST to confirm that on the real data — it reads only the parquet
footer and prints the projected byte cost, so you size the array job from measurement.

THE MODES
---------
    probe       footer-only scan  -> manifest.csv (rows, projected bytes, columns found)
    count       the worker        -> partial parquets for one shard of the file list
    merge       sum the partials  -> final tables
    citdist     citation histograms per (year, type)  [only for --weights top10]
    thresholds  citdist partials  -> cit_thresholds.parquet, the top-decile cut-offs

`count` and `citdist` are shard-aware (`--shard i --num-shards n`), which is the only
thing the Slurm array job needs; see the companion .sh.

With top10 the order is citdist -> thresholds -> count -> merge; without it, the two
extra passes are unnecessary and the order is the old count -> merge.

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
# WEIGHT REGISTRY  — what a paper contributes, besides 1.0
# =============================================================================
CIT_COL = "times_cited"


class Reference:
    """A per-(year, type, level, code) reference number with a (year, type) fallback.

    Both derived per-field weights need one of these — a decile cut-off for top10f, an
    expected citation count for mncs — and both need the same rule when a field-year is
    too thin to define its own: fall back to the figure for the whole corpus that year
    rather than score the paper against a dozen neighbours.
    """

    def __init__(self, per_code: Dict[Tuple[str, str, int, str], float],
                 global_: Dict[Tuple[int, str], float]) -> None:
        self.per_code = per_code
        self.global_ = global_

    def get(self, year: int, typ: str, level: str, code: str) -> Optional[float]:
        v = self.per_code.get((level, code, year, typ))
        return self.global_.get((year, typ)) if v is None else v

    def __bool__(self) -> bool:
        return bool(self.per_code or self.global_)


class WeightSpec:
    """One citation weight: the on-disk column it reads and how it turns into a number.

    kind='column'    the column value passes straight through (cit, fcr, recent).
    kind='paper'     derived, but one value per paper: the (year, type) decile (top10).
    kind='category'  derived AND dependent on which category it is being counted into
                     (top10f, mncs). These are the reason `value()` takes level/code:
                     one paper carries several of them, one per category it belongs to.
    """

    def __init__(self, name: str, column: str, doc: str, kind: str = "column") -> None:
        self.name = name
        self.column = column
        self.doc = doc
        self.kind = kind

    @property
    def per_category(self) -> bool:
        return self.kind == "category"

    def _raw(self, raw) -> Optional[float]:
        """None is NOT zero: it means unmeasured, and it keeps the paper out of the
        n_<w>_docs denominator instead of dragging the mean down."""
        if raw is None:
            return None
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        return None if v != v else v    # NaN is how parquet doubles carry "missing"

    def value(self, raw, year: int, typ: str, ref: Optional[Reference] = None,
              level: str = "", code: str = "") -> Optional[float]:
        v = self._raw(raw)
        if v is None or self.kind == "column":
            return v
        if not ref:
            return None
        cut = ref.get(year, typ, level, code)
        if cut is None:
            return None
        if self.name == "mncs":
            # An expected value of zero means nothing in that cell was cited at all;
            # the ratio is undefined rather than infinite.
            return None if cut <= 0 else v / cut
        return 1.0 if v >= cut else 0.0


WEIGHTS: Dict[str, WeightSpec] = {
    "cit":    WeightSpec("cit", CIT_COL, "raw citations (times_cited)"),
    "fcr":    WeightSpec("fcr", "field_citation_ratio",
                         "field- and year-normalised citation ratio (1.0 = world average)"),
    "recent": WeightSpec("recent", "recent_citations", "citations in the last two years"),
    "top10":  WeightSpec("top10", CIT_COL,
                         "1.0 when the paper is in the top decile of its (year, type)",
                         kind="paper"),
    "top10f": WeightSpec("top10f", CIT_COL,
                         "1.0 when the paper is in the top decile of its (year, type, "
                         "category) — PP(top 10%) proper", kind="category"),
    "mncs":   WeightSpec("mncs", CIT_COL,
                         "citations over the mean of its (year, type, category) — a "
                         "field-normalised score of our own", kind="category"),
}
# The four columns each weight contributes, in the order tally_to_tables writes them.
WEIGHT_SUFFIXES = ("", "_frac", "_docs", "_docs_frac")


def parse_weights(arg: Optional[str]) -> List[str]:
    """'cit,fcr,top10' -> ['cit', 'fcr', 'top10']. Unknown names are a hard error:
    a typo must not silently produce a table with a column missing."""
    if not arg:
        return []
    names = [w.strip() for w in arg.split(",") if w.strip()]
    unknown = [w for w in names if w not in WEIGHTS]
    if unknown:
        sys.exit(f"--weights: unknown {unknown}; choose from {sorted(WEIGHTS)}")
    seen: List[str] = []
    for w in names:                     # de-duplicate, order preserved
        if w not in seen:
            seen.append(w)
    return seen


def weight_columns(names: Sequence[str]) -> List[str]:
    """Column names for a set of weights: n_cit, n_cit_frac, n_cit_docs, ..."""
    return [f"n_{w}{sfx}" for w in names for sfx in WEIGHT_SUFFIXES]


def load_reference(path: str, column: str, what: str) -> Reference:
    """Read a reference table written by `thresholds` into a Reference.

    Rows with an empty `code` are the (year, type) fallback; the rest are per-category.
    Tables written before per-category references existed have no level/code columns at
    all, and load as fallback-only — which is exactly what they are."""
    tbl = pq.read_table(path).to_pydict()
    if column not in tbl:
        sys.exit(f"{path} has no '{column}' column (is it the right table?)")
    per_code: Dict[Tuple[str, str, int, str], float] = {}
    global_: Dict[Tuple[int, str], float] = {}
    levels = tbl.get("level") or [""] * len(tbl["year"])
    codes = tbl.get("code") or [""] * len(tbl["year"])
    for lvl, code, year, typ, val in zip(levels, codes, tbl["year"], tbl["type"],
                                         tbl[column]):
        if val is None:
            continue
        key_typ = typ or MISSING_TYPE
        if code:
            per_code[(lvl or "", code, int(year), key_typ)] = float(val)
        else:
            global_[(int(year), key_typ)] = float(val)
    if not per_code and not global_:
        sys.exit(f"{path} yielded no {what}")
    return Reference(per_code, global_)


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
            # RCDC's struct shape names the label "value", not "name" (id: '558',
            # value: 'Prevention') — confirmed against /well/.../000000000000.parquet.
            yield entry.get("code"), entry.get("name") or entry.get("value") or ""


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
    # Probe the weight columns too, so the projection printed here is the projection the
    # weighted `count` will actually pay — not the paper-count one.
    weights = parse_weights(getattr(args, "weights", None))
    wcols = {WEIGHTS[w].column for w in weights}

    tot_rows = tot_proj = tot_file = 0
    n_no_weight = 0
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "rows", "row_groups", "file_bytes", "projected_bytes",
                    "projected_pct", "cat_column", "missing_weight_cols", "error"])
        for path in files:
            try:
                pf = pq.ParquetFile(path)
                md = pf.metadata
                names = set(pf.schema_arrow.names)
                cat_col = spec.resolve_column(names)
                missing = sorted(wcols - names)
                n_no_weight += bool(missing)
                wanted = ({ID_COL, YEAR_COL, TYPE_COL} | (wcols & names)
                          | ({cat_col} if cat_col else set()))
                proj = 0
                for rg in range(md.num_row_groups):
                    g = md.row_group(rg)
                    for i in range(g.num_columns):
                        col = g.column(i)
                        if col.path_in_schema in wanted:
                            proj += col.total_compressed_size
                fsz = os.path.getsize(path)
                w.writerow([path, md.num_rows, md.num_row_groups, fsz, proj,
                            f"{100 * proj / fsz:.3f}" if fsz else "", cat_col or "",
                            " ".join(missing), ""])
                tot_rows += md.num_rows
                tot_proj += proj
                tot_file += fsz
            except Exception as exc:  # a corrupt file must not kill the survey
                w.writerow([path, "", "", "", "", "", "", "",
                            f"{type(exc).__name__}: {exc}"])

    print(f"probed {len(files)} files ({spec.name}"
          f"{', weights ' + ','.join(weights) if weights else ''}) -> {dest}")
    print(f"  rows            {tot_rows:,}")
    print(f"  file bytes      {tot_file / 1e9:.2f} GB")
    if tot_file:
        print(f"  projected read  {tot_proj / 1e9:.4f} GB  "
              f"({100 * tot_proj / tot_file:.3f} % of the corpus)")
    if n_no_weight:
        print(f"  !! {n_no_weight} file(s) lack a weight column — those papers would be "
              f"unmeasured, not zero; see missing_weight_cols in {dest}")


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
    """Accumulates the count / totals / coverage tables. Plain dicts — no pandas hot loop.

    Every per-cell slot is [absolute, fractional] for the paper count, followed by
    [absolute, fractional, docs, docs_frac] for each weight in `weights`, in order.
    """

    def __init__(self, weights: Sequence[str] = ()) -> None:
        self.weights = list(weights)
        width = 2 + 4 * len(self.weights)
        # (year, type, level, code) -> [absolute, fractional, <4 per weight>...]
        self.codes: Dict[Tuple[int, str, str, str], List[float]] = defaultdict(
            lambda: [0.0] * width)
        # (year, type) -> [seen, filtered_out, no_category]
        self.totals: Dict[Tuple[int, str], List[int]] = defaultdict(lambda: [0, 0, 0])
        # (year, type) -> [sum of weight, papers carrying it] per weight — the honest
        # denominators for a share, unpolluted by per-category double counting.
        self.wtotals: Dict[Tuple[int, str], List[float]] = defaultdict(
            lambda: [0.0] * (2 * len(self.weights)))
        # (year, type, level) -> [papers with >=1 code, then sum of weight per weight]
        self.cov: Dict[Tuple[int, str, str], List[float]] = defaultdict(
            lambda: [0.0] * (1 + len(self.weights)))
        self.labels: Dict[str, str] = {}          # code -> label, harvested from the data
        self.n_rows = self.n_kept = self.n_dropped = 0

    def add_row(self, year: int, typ: str, cell, spec: CategorySpec, fmt: str,
                wvals: Sequence[Optional[float]] = (),
                per_cat: Sequence[Optional[Tuple[WeightSpec, object, Reference]]] = ()
                ) -> None:
        """Count one paper.

        `wvals` holds the paper-level weights, one per weight, already resolved.
        `per_cat` holds (spec, raw value, reference) for the per-category weights, which
        cannot be resolved until we know which category we are counting into — that
        resolution happens inside the code loop below.
        """
        t = self.totals[(year, typ)]
        t[0] += 1
        self.n_kept += 1

        if self.weights:
            wt = self.wtotals[(year, typ)]
            for i, v in enumerate(wvals):
                if v is None:
                    continue
                if per_cat and per_cat[i] is not None:
                    # A per-category weight has no single paper-level value; only its
                    # "was this paper measurable" flag is meaningful here. NaN, not 0,
                    # so a mean built on it downstream is loudly wrong rather than quietly.
                    wt[2 * i] = float("nan")
                    wt[2 * i + 1] += 1.0
                else:
                    wt[2 * i] += v
                    wt[2 * i + 1] += 1.0

        any_tag = False
        for level, items in spec.levels(cell, fmt).items():
            if not items:
                continue
            any_tag = True
            cov = self.cov[(year, typ, level)]
            cov[0] += 1
            # Each level normalised on its own denominator — see COUNTING SEMANTICS.
            w = 1.0 / len(items)
            for code, label in items:
                slot = self.codes[(year, typ, level, code)]
                slot[0] += 1.0
                slot[1] += w
                for i, v in enumerate(wvals):
                    if per_cat and per_cat[i] is not None:
                        wspec, raw, ref = per_cat[i]
                        v = wspec.value(raw, year, typ, ref, level, code)
                    if v is None:       # unmeasured: no sum, and no place in the mean
                        continue
                    base = 2 + 4 * i
                    slot[base] += v
                    slot[base + 1] += v * w
                    slot[base + 2] += 1.0
                    slot[base + 3] += w
                    # Coverage accumulates the FRACTIONAL contribution, so the identity
                    # "sum over codes of n_<w>_frac == n_<w>_with_level" holds for a
                    # per-category weight exactly as it does for a paper-level one
                    # (where the terms are all equal and sum back to v).
                    cov[1 + i] += v * w
                if code not in self.labels:
                    self.labels[code] = label
        if not any_tag:
            t[2] += 1

    def note_filtered(self, year: int, typ: str) -> None:
        self.totals[(year, typ)][1] += 1
        self.n_dropped += 1


def count_file(path: str, tally: Tally, spec: CategorySpec, id_set: Optional[Set[str]],
               mode: str, batch_size: int,
               refs: Optional[Dict[str, Reference]] = None
               ) -> Tuple[int, str, List[str]]:
    """Stream one file into `tally`. Returns (rows_read, category column, weight columns).

    `mode`: 'exclude' drops ids in the set (background arm), 'include' keeps only them
    (UKBB arm), 'none' ignores the set entirely.

    A weight whose column this file does not carry is read as unmeasured for every row
    of it — never as zero — and the missing column is reported back for the file log.
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
    # Several weights can share one column (cit and top10 both read times_cited), so the
    # projection carries each column once and the specs index back into it.
    specs = [WEIGHTS[w] for w in tally.weights]
    wcols: List[str] = []
    for s in specs:
        if s.column in names and s.column not in wcols:
            wcols.append(s.column)
    cols += wcols
    missing = sorted({s.column for s in specs if s.column not in names})

    rows = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        ids = batch.column(cols.index(ID_COL)).to_pylist()
        years = batch.column(cols.index(YEAR_COL)).to_pylist()
        cells = batch.column(cols.index(cat_col)).to_pylist()
        types = (batch.column(cols.index(TYPE_COL)).to_pylist()
                 if has_type else [MISSING_TYPE] * len(ids))
        raws = {c: batch.column(cols.index(c)).to_pylist() for c in wcols}
        blank = [None] * len(ids)
        wraw = [raws.get(s.column, blank) for s in specs]
        rows += len(ids)
        tally.n_rows += len(ids)

        for i, (pid, year, typ, cell) in enumerate(zip(ids, years, types, cells)):
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
            # Paper-level weights resolve here; per-category ones can only be resolved
            # once we know which category we are counting into, so they travel into
            # add_row as (spec, raw, reference) and are resolved in its code loop.
            wvals = []
            per_cat = []
            for s, col in zip(specs, wraw):
                ref = (refs or {}).get(s.name)
                if s.per_category:
                    wvals.append(s._raw(col[i]))     # only "is it measurable" here
                    per_cat.append((s, col[i], ref))
                else:
                    wvals.append(s.value(col[i], y, t, ref))
                    per_cat.append(None)
            tally.add_row(y, t, cell, spec, fmt, wvals, per_cat)
    return rows, cat_col, missing


def tally_to_tables(tally: Tally, label: str, spec: CategorySpec) -> Tuple[pa.Table, pa.Table, pa.Table]:
    """(counts, totals, coverage) as arrow tables. The counts schema keeps the historical
    column names `for_label` / `parent_l2` (generic label / parent) so existing readers
    keep working; `parent_l2` is the 2-digit division for FOR, empty for flat systems.

    Weight columns are appended, four per weight, and are absent entirely when the run
    asked for no weights — so a weightless run writes exactly the old schema."""
    keys = sorted(tally.codes)
    counts = {
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
    }
    for i, w in enumerate(tally.weights):
        for j, sfx in enumerate(WEIGHT_SUFFIXES):
            counts[f"n_{w}{sfx}"] = pa.array(
                [tally.codes[k][2 + 4 * i + j] for k in keys], pa.float64())

    tkeys = sorted(tally.totals)
    totals = {
        "arm": pa.array([label] * len(tkeys), pa.string()),
        "category": pa.array([spec.name] * len(tkeys), pa.string()),
        "year": pa.array([k[0] for k in tkeys], pa.int32()),
        "type": pa.array([k[1] for k in tkeys], pa.string()),
        "n_papers": pa.array([tally.totals[k][0] for k in tkeys], pa.int64()),
        "n_filtered_out": pa.array([tally.totals[k][1] for k in tkeys], pa.int64()),
        "n_no_cat": pa.array([tally.totals[k][2] for k in tkeys], pa.int64()),
    }
    for i, w in enumerate(tally.weights):
        totals[f"n_{w}_total"] = pa.array(
            [tally.wtotals[k][2 * i] for k in tkeys], pa.float64())
        totals[f"n_{w}_docs_total"] = pa.array(
            [tally.wtotals[k][2 * i + 1] for k in tkeys], pa.float64())

    ckeys = sorted(tally.cov)
    coverage = {
        "arm": pa.array([label] * len(ckeys), pa.string()),
        "category": pa.array([spec.name] * len(ckeys), pa.string()),
        "year": pa.array([k[0] for k in ckeys], pa.int32()),
        "type": pa.array([k[1] for k in ckeys], pa.string()),
        "level": pa.array([k[2] for k in ckeys], pa.string()),
        "n_with_level": pa.array([int(tally.cov[k][0]) for k in ckeys], pa.int64()),
    }
    for i, w in enumerate(tally.weights):
        coverage[f"n_{w}_with_level"] = pa.array(
            [tally.cov[k][1 + i] for k in ckeys], pa.float64())

    return pa.table(counts), pa.table(totals), pa.table(coverage)


def cmd_count(args: argparse.Namespace) -> None:
    spec = CATEGORIES[args.category]
    files = take_shard(collect_files(args), args.shard, args.num_shards)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    id_set = load_id_set(args.exclude_ids, args.id_column) if args.exclude_ids else None
    if args.id_filter != "none" and id_set is None:
        sys.exit(f"--id-filter {args.id_filter} needs --exclude-ids")

    weights = parse_weights(args.weights)
    # Fail here rather than write a table of zeros: without its reference table a
    # derived weight is None for every paper, and the column would read as "nothing is
    # highly cited" instead of "this was never measured".
    refs: Dict[str, Reference] = {}
    if {"top10", "top10f"} & set(weights):
        if not args.top_thresholds:
            sys.exit(f"--weights {'/'.join({'top10', 'top10f'} & set(weights))} needs "
                     f"--top-thresholds (run the citdist and thresholds modes first)")
        thr = load_reference(args.top_thresholds, "threshold", "thresholds")
        for w in ("top10", "top10f"):
            if w in weights:
                refs[w] = thr
        if "top10f" in weights and not thr.per_code:
            sys.exit(f"--weights top10f needs PER-CATEGORY cut-offs, but "
                     f"{args.top_thresholds} holds only (year, type) rows. Re-run "
                     f"citdist with --category {args.category} and then thresholds.")
    if "mncs" in weights:
        if not args.expected:
            sys.exit("--weights mncs needs --expected (written by the thresholds mode)")
        refs["mncs"] = load_reference(args.expected, "mean_cit", "expected citations")
    wnote = ", weights=" + ",".join(weights) if weights else ""
    print(f"[{args.label}/{spec.name}] {len(files)} file(s), id-filter={args.id_filter}"
          f"{f', {len(id_set):,} ids' if id_set else ''}{wnote}", flush=True)

    tally = Tally(weights)
    stem = f"{spec.name}.{args.label}{_shard_suffix(args)}"
    log_path = out / f"filelog.{stem}.csv"
    t_all = time.time()
    n_ok = n_err = 0
    n_missing_w = 0
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "rows", "cat_column", "missing_weight_cols", "seconds",
                    "error"])
        for i, path in enumerate(files, 1):
            t0 = time.time()
            try:
                rows, cat_col, missing = count_file(path, tally, spec, id_set,
                                                    args.id_filter, args.batch_size,
                                                    refs)
                w.writerow([path, rows, cat_col, " ".join(missing),
                            f"{time.time() - t0:.2f}", ""])
                n_missing_w += bool(missing)
                n_ok += 1
            except Exception as exc:
                # One unreadable file among thousands must not lose the shard's work.
                w.writerow([path, "", "", "", f"{time.time() - t0:.2f}",
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
    for i, wname in enumerate(weights):
        tot = sum(v[2 * i] for v in tally.wtotals.values())
        docs = sum(v[2 * i + 1] for v in tally.wtotals.values())
        measured = f"({docs / max(tally.n_kept, 1):.1%} of those counted carried a value)"
        if WEIGHTS[wname].per_category:
            # Its value depends on the category, so there is no paper-level sum to show;
            # the per-category totals are in the counts table.
            print(f"  weight {wname:6s} per-category   over {int(docs):>12,} papers "
                  f"{measured}")
        else:
            print(f"  weight {wname:6s} sum {tot:>18,.1f} over {int(docs):>12,} papers "
                  f"{measured}")
    if n_missing_w:
        print(f"  !! {n_missing_w} file(s) lacked a weight column — see "
              f"missing_weight_cols in {log_path}", file=sys.stderr)
    if n_err:
        print(f"  see {log_path} for the {n_err} failure(s)", file=sys.stderr)


# =============================================================================
# MODE: citdist  — the citation distribution, for the cut-offs and the expected means
# =============================================================================
# times_cited is a small non-negative integer, so the EXACT distribution is affordable:
# no sampling, no sketch, and the 90th percentile that comes out of `thresholds` is the
# true one. This is a separate pass because the cut-off for a cell has to be known
# before the first paper of that cell is counted.
#
# Two populations are accumulated at once:
#
#   global      one entry per PAPER, keyed (year, type). The fallback, and what the
#               paper-level `top10` weight uses.
#   per-category one entry per (PAPER, CATEGORY), keyed (level, code, year, type). This
#               is what makes PP(top 10 %) mean "the top tenth of THIS field" — the
#               figure the bibliometric literature reports — and what gives `mncs` its
#               reference mean. A paper carrying three fields lands in three of them,
#               which is correct: it is competing in all three.
#
# MEMORY. The per-category histogram is a fixed-width int32 array per cell rather than a
# dict of counts, because a dict of (cell, citations) tuples over ~30k cells would cost
# gigabytes. Counts are capped at CIT_CAP: 99.1 % of the corpus sits below 200 citations
# and the decile cut-off is ~50, so the cap costs nothing at the percentile, and the
# exact sum is accumulated separately so the MEAN stays exact too. ~30k cells x 501
# int32 = ~60 MB.
CIT_CAP = 500


class CitAccum:
    """Exact per-cell citation distribution: capped histogram + exact count and sum."""

    def __init__(self) -> None:
        self.hist: Dict[tuple, "object"] = {}
        self.stats: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0])

    def add(self, key: tuple, cit: int) -> None:
        h = self.hist.get(key)
        if h is None:
            h = self.hist[key] = [0] * (CIT_CAP + 1)
        h[cit if cit < CIT_CAP else CIT_CAP] += 1
        s = self.stats[key]
        s[0] += 1
        s[1] += cit

    def to_tables(self, key_names: Sequence[str]) -> Tuple[pa.Table, pa.Table]:
        """(histogram, stats) in long form; only non-empty buckets are written."""
        hk, hc, hn = [], [], []
        for key, arr in self.hist.items():
            for cit, n in enumerate(arr):
                if n:
                    hk.append(key)
                    hc.append(cit)
                    hn.append(n)
        hist = {name: pa.array([k[i] for k in hk]) for i, name in enumerate(key_names)}
        hist["times_cited"] = pa.array(hc, pa.int64())
        hist["n"] = pa.array(hn, pa.int64())

        sk = sorted(self.stats)
        stats = {name: pa.array([k[i] for k in sk]) for i, name in enumerate(key_names)}
        stats["n_papers"] = pa.array([self.stats[k][0] for k in sk], pa.int64())
        stats["sum_cit"] = pa.array([self.stats[k][1] for k in sk], pa.int64())
        return pa.table(hist), pa.table(stats)


def cmd_citdist(args: argparse.Namespace) -> None:
    files = take_shard(collect_files(args), args.shard, args.num_shards)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # --category is optional here: without it only the (year, type) fallback is built,
    # which is all the paper-level top10 weight needs.
    spec = CATEGORIES[args.category] if args.category else None

    # The reference population is the WHOLE database, both arms together: "top 10 % of
    # the literature" is the point of the indicator. No id filtering happens here.
    glob_acc, cat_acc = CitAccum(), CitAccum()
    n_rows = n_ok = n_err = n_missing = 0
    t_all = time.time()
    log_path = out / f"filelog.citdist{_shard_suffix(args)}.csv"
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "rows", "seconds", "error"])
        for i, path in enumerate(files, 1):
            t0 = time.time()
            try:
                pf = pq.ParquetFile(path)
                names = set(pf.schema_arrow.names)
                if CIT_COL not in names:
                    raise KeyError(f"missing required column '{CIT_COL}'")
                has_type = TYPE_COL in names
                cat_col = spec.resolve_column(names) if spec else None
                if spec and cat_col is None:
                    raise KeyError(f"no {spec.name} column (looked for "
                                   f"{list(spec.columns)})")
                cols = [YEAR_COL, CIT_COL] + ([TYPE_COL] if has_type else []) \
                    + ([cat_col] if cat_col else [])
                fmt = column_format(pf.schema_arrow, cat_col) if cat_col else "text"
                rows = 0
                for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
                    years = batch.column(cols.index(YEAR_COL)).to_pylist()
                    cits = batch.column(cols.index(CIT_COL)).to_pylist()
                    types = (batch.column(cols.index(TYPE_COL)).to_pylist()
                             if has_type else [MISSING_TYPE] * len(years))
                    cells = (batch.column(cols.index(cat_col)).to_pylist()
                             if cat_col else [None] * len(years))
                    rows += len(years)
                    for year, typ, cit, cell in zip(years, types, cits, cells):
                        if cit is None:
                            n_missing += 1
                            continue
                        try:
                            y = int(year)
                        except (TypeError, ValueError):
                            y = MISSING_YEAR
                        t = typ or MISSING_TYPE
                        c = int(cit)
                        glob_acc.add((y, t), c)
                        if spec:
                            for level, items in spec.levels(cell, fmt).items():
                                for code, _ in items:
                                    cat_acc.add((level, code, y, t), c)
                w.writerow([path, rows, f"{time.time() - t0:.2f}", ""])
                n_rows += rows
                n_ok += 1
            except Exception as exc:
                w.writerow([path, "", f"{time.time() - t0:.2f}",
                            f"{type(exc).__name__}: {exc}"])
                n_err += 1
                print(f"  !! {path}: {type(exc).__name__}: {exc}", file=sys.stderr,
                      flush=True)
            if args.progress and i % args.progress == 0:
                print(f"  {i}/{len(files)} files, {n_rows:,} rows, "
                      f"{time.time() - t_all:.0f}s", flush=True)
            fh.flush()

    sfx = _shard_suffix(args)
    hist, stats = glob_acc.to_tables(("year", "type"))
    pq.write_table(hist, out / f"citdist{sfx}.parquet", compression="zstd")
    pq.write_table(stats, out / f"citstats{sfx}.parquet", compression="zstd")
    msg = f"{len(glob_acc.hist):,} (year, type) cells"
    if spec:
        chist, cstats = cat_acc.to_tables(("level", "code", "year", "type"))
        pq.write_table(chist, out / f"citdist.{spec.name}{sfx}.parquet",
                       compression="zstd")
        pq.write_table(cstats, out / f"citstats.{spec.name}{sfx}.parquet",
                       compression="zstd")
        msg += f" + {len(cat_acc.hist):,} ({spec.name}, year, type) cells"
    print(f"citdist {n_ok} ok / {n_err} failed | rows {n_rows:,} | {msg} | "
          f"{n_missing:,} rows with no {CIT_COL} | {time.time() - t_all:.0f}s")


# =============================================================================
# MODE: thresholds  — citdist partials -> the top-decile cut-off per (year, type)
# =============================================================================
def _cut_off(group, total: int, pct: float) -> Tuple[int, int, float]:
    """The lowest citation count whose "at least this many" tail stays within `pct` %.

    Citations are a tied, lumpy integer distribution (a fifth of all papers sit at 0),
    so an exact 10 % is usually unreachable: taking the largest tail that does not
    exceed it keeps the indicator conservative, and the achieved share is reported so
    the compromise is visible rather than assumed.
    """
    tail, achieved = 0, 0.0
    cut = int(group.times_cited.max()) + 1
    for cites, n in zip(group.times_cited, group.n):
        if (tail + n) / total > pct / 100:
            break
        tail += int(n)
        cut = int(cites)
        achieved = 100 * tail / total
    return cut, tail, achieved


def cmd_thresholds(args: argparse.Namespace) -> None:
    import pandas as pd

    out = Path(args.out)
    src = Path(args.partials) if args.partials else out
    pct = args.percentile

    def read(stem):
        """Partials for one table, sharded or not. The two spellings are listed
        explicitly because 'citdist.*.parquet' would also swallow 'citdist.for.parquet'
        and quietly mix a per-category table into the global one."""
        parts = sorted(src.glob(f"{stem}.parquet")) + sorted(src.glob(f"{stem}.shard*.parquet"))
        if not parts:
            return None
        return pd.concat([pq.read_table(p).to_pandas() for p in parts],
                         ignore_index=True)

    # The (year, type) fallback is always built; the per-category tables only exist if
    # citdist was given a --category.
    frames = {"": (read("citdist"), read("citstats"))}
    for cat in sorted(CATEGORIES):
        d, s = read(f"citdist.{cat}"), read(f"citstats.{cat}")
        if d is not None and s is not None:
            frames[cat] = (d, s)
    if frames[""][0] is None:
        sys.exit(f"no citdist parquet under {src} — run the citdist mode first")

    thr_rows, exp_rows = [], []
    for cat, (dist, stats) in frames.items():
        keys = ["year", "type"] if not cat else ["level", "code", "year", "type"]
        dist = dist.groupby(keys + ["times_cited"], as_index=False).n.sum()
        stats = stats.groupby(keys, as_index=False)[["n_papers", "sum_cit"]].sum()
        print(f"  {cat or '(year,type)':10s} {int(stats.n_papers.sum()):>14,} rows, "
              f"{len(stats):,} cells")

        for key, g in dist.sort_values("times_cited", ascending=False).groupby(keys):
            total = int(g.n.sum())
            if total < args.min_papers:
                continue     # a percentile of twelve papers is not a percentile
            cut, tail, achieved = _cut_off(g, total, pct)
            row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            row.update(category=cat, percentile=pct, threshold=cut, n_papers=total,
                       n_at_or_above=tail, achieved_pct=round(achieved, 3))
            thr_rows.append(row)

        for _, r in stats.iterrows():
            if r.n_papers < args.min_papers:
                continue
            row = {k: r[k] for k in keys}
            row.update(category=cat, n_papers=int(r.n_papers), sum_cit=int(r.sum_cit),
                       mean_cit=r.sum_cit / r.n_papers)
            exp_rows.append(row)

    cols = ["category", "level", "code", "year", "type"]

    def frame(rows, extra):
        df = pd.DataFrame(rows)
        for c in ("level", "code"):
            if c not in df.columns:
                df[c] = ""
        df[["level", "code"]] = df[["level", "code"]].fillna("")
        return df[cols + extra].sort_values(cols)

    thr = frame(thr_rows, ["percentile", "threshold", "n_papers", "n_at_or_above",
                           "achieved_pct"])
    exp = frame(exp_rows, ["n_papers", "sum_cit", "mean_cit"])
    for df, name in ((thr, "cit_thresholds"), (exp, "cit_expected")):
        df.to_parquet(out / f"{name}.parquet", index=False)
        df.to_csv(out / f"{name}.csv", index=False)
        per_cat = int((df.code != "").sum())
        print(f"wrote {out}/{name}.parquet  ({len(df):,} rows: {len(df) - per_cat:,} "
              f"(year, type) fallback + {per_cat:,} per-category)")

    fallback = thr[thr.code == ""]
    shown = fallback[fallback.type == "article"]
    print(f"\n(year, type) cut-offs for articles — citations needed to be top {pct}%:")
    for _, r in shown.tail(args.show).iterrows():
        print(f"  {int(r.year)}  >= {int(r.threshold):>5,} citations   "
              f"{int(r.n_at_or_above):>10,} / {int(r.n_papers):>12,} papers "
              f"= {r.achieved_pct:5.2f}%")

    # The whole point of the per-category cut-offs is that they differ from each other;
    # print the spread so a reader can see how much the global one was papering over.
    per_cat = thr[(thr.code != "") & (thr.type == "article")]
    if len(per_cat):
        latest = per_cat[per_cat.year == per_cat.year.max()]
        for cat, g in latest.groupby("category"):
            g = g.sort_values("threshold")
            glob = fallback[(fallback.type == "article")
                            & (fallback.year == latest.year.max())]
            gv = int(glob.threshold.iloc[0]) if len(glob) else -1
            print(f"\n{cat} article cut-offs in {int(latest.year.max())} "
                  f"(the whole-corpus figure is {gv}):")
            for _, r in pd.concat([g.head(3), g.tail(3)]).iterrows():
                print(f"  {r.level:3s} {str(r.code)[:34]:34s} >= {int(r.threshold):>4} "
                      f"citations   ({int(r.n_papers):>10,} papers, "
                      f"{r.achieved_pct:4.1f}%)")


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
        dfs = []
        for p in parts:
            d = pq.read_table(p).to_pandas()
            if "category" not in d.columns:   # partials written before categories existed
                d["category"] = "for"
            dfs.append(d)
        return pd.concat(dfs, ignore_index=True)

    counts = _read("counts")
    # Every measure column is summable and named n_*; weights therefore need no special
    # case here, and a mixed directory (one run weighted, one not) merges cleanly because
    # concat fills the absent columns with NaN, which sums as 0 below.
    value_cols = [c for c in counts.columns if c.startswith("n_")]
    counts[value_cols] = counts[value_cols].fillna(0.0)
    # for_label is carried, not summed: keep the first non-empty label per (category, code).
    labels = (counts.loc[counts.for_label != "", ["category", "code", "for_label"]]
              .drop_duplicates(["category", "code"])
              .set_index(["category", "code"])["for_label"])
    counts = counts.groupby(GROUP_KEYS, as_index=False)[value_cols].sum()
    counts["for_label"] = (counts.set_index(["category", "code"]).index.map(labels)
                           .to_numpy())
    counts["for_label"] = counts["for_label"].astype("object").where(
        counts["for_label"].notna(), "")
    counts["n_papers"] = counts.n_papers.round().astype("int64")
    counts = counts[["arm", "category", "year", "type", "level", "code", "for_label",
                     "parent_l2"] + value_cols].sort_values(
        ["category", "arm", "level", "year", "n_papers"],
        ascending=[True, True, True, True, False])
    WEIGHTS_FOUND = [w for w in WEIGHTS if f"n_{w}" in value_cols]

    totals = _read("totals").groupby(["arm", "category", "year", "type"],
                                     as_index=False).sum(numeric_only=True, min_count=1)
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
        for wname in WEIGHTS_FOUND:
            tot_col, doc_col = f"n_{wname}_total", f"n_{wname}_docs_total"
            if tot_col not in g.columns:
                continue
            docs = g[doc_col].sum()
            # min_count=1 again: a plain .sum() over an all-NaN column returns 0.0, and
            # "no paper-level total" would print as "zero citations".
            total = g[tot_col].sum(min_count=1)
            if docs and total != total:      # NaN total: a per-category weight
                print(f"    {wname:6s} measured on {int(docs):>12,} papers "
                      f"({docs / max(papers, 1):.1%}); no paper-level total — its value "
                      f"depends on which category it is counted into")
                continue
            if not docs:
                # Almost always a directory holding one weighted run and one older
                # unweighted one. The zeros are indistinguishable from "never cited"
                # downstream, so say it here rather than let a chart imply it.
                print(f"    {wname:6s} NOT COUNTED for this arm — its n_{wname}* columns "
                      f"are zero because the partials predate --weights, not because "
                      f"the papers went uncited. Re-run this arm before comparing.")
                continue
            print(f"    {wname:6s} {total:>18,.1f} over {int(docs):>12,} "
                  f"papers ({docs / max(papers, 1):.1%} measured, mean "
                  f"{total / max(docs, 1):.2f})")

    # Fractional totals must reconcile with the per-level coverage; if they don't, the
    # per-level normalisation is wrong and every downstream share is wrong with it. The
    # same identity has to hold for every weight, which is what catches a weight that was
    # summed but not normalised.
    if coverage is not None:
        print("\nreconciliation (fractional total must equal the per-level total):")
        pairs = [("n_frac", "n_with_level", "papers")]
        pairs += [(f"n_{w}_frac", f"n_{w}_with_level", w) for w in WEIGHTS_FOUND
                  if f"n_{w}_with_level" in coverage.columns]
        for frac_col, cov_col, what in pairs:
            chk = counts.groupby(["category", "arm", "level"])[frac_col].sum()
            ref = coverage.groupby(["category", "arm", "level"])[cov_col].sum()
            for key in ref.index:
                got = float(chk.get(key, 0.0))
                want = float(ref.loc[key])
                # Weights are sums of doubles over 10^8 rows, so compare relatively.
                flag = ("ok" if abs(got - want) <= max(1.0, 1e-6 * abs(want))
                        else "MISMATCH")
                print(f"  {what:7s} {key[0]:9s} {key[1]:12s} {key[2]:3s} {got:16,.1f} "
                      f" vs {want:16,.1f}   {flag}")
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
  %(prog)s probe --category rcdc --weights cit,fcr \\
      --glob '/well/.../corpus/*.parquet' --out out/

  # 1a. citation distribution, for the top-decile cut-offs (whole corpus, no id filter)
  %(prog)s citdist --file-list files.txt --shard 3 --num-shards 50 --out out/
  # 1b. ... and turn the histograms into thresholds
  %(prog)s thresholds --out out/

  # 2. the two arms, SAME file list so they share one citation snapshot
  %(prog)s count --category rcdc --file-list files.txt --shard 3 --num-shards 50 \\
      --weights cit,fcr,top10 --top-thresholds out/cit_thresholds.parquet \\
      --exclude-ids ukbb_ids.txt --id-filter exclude --label background --out out/
  %(prog)s count --category rcdc --file-list files.txt --shard 3 --num-shards 50 \\
      --weights cit,fcr,top10 --top-thresholds out/cit_thresholds.parquet \\
      --exclude-ids ukbb_ids.txt --id-filter include --label ukbb --out out/

  # 3. combine everything (handles multiple categories in one dir)
  %(prog)s merge --out out/
""")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_inputs(sp, category=True):
        if category:
            sp.add_argument("--category", choices=sorted(CATEGORIES), default="for",
                            help="classification system to count (default: for)")
        sp.add_argument("--files", nargs="+", help="explicit parquet paths")
        sp.add_argument("--file-list", help="text file, one parquet path per line")
        sp.add_argument("--glob", nargs="+", help="glob pattern(s), e.g. 'dir/*.parquet'")
        sp.add_argument("--shard", type=int, default=0)
        sp.add_argument("--num-shards", type=int, default=1)
        sp.add_argument("--out", default="out", help="output directory (never the source)")

    def add_weights(sp):
        sp.add_argument("--weights", default="",
                        help="comma-separated citation weights to add alongside the "
                             "paper counts: " + ", ".join(
                                 f"{n} ({s.doc})" for n, s in WEIGHTS.items()))

    sp = sub.add_parser("probe", help="footer-only survey; sizes the job")
    add_inputs(sp)
    add_weights(sp)

    sp = sub.add_parser("count", help="tally one shard")
    add_inputs(sp)
    add_weights(sp)
    sp.add_argument("--top-thresholds",
                    help="cit_thresholds.parquet from the `thresholds` mode; required "
                         "by --weights top10 / top10f")
    sp.add_argument("--expected",
                    help="cit_expected.parquet from the `thresholds` mode; required "
                         "by --weights mncs")
    sp.add_argument("--exclude-ids", help="parquet/csv/txt of UKBB paper ids")
    sp.add_argument("--id-column", default=ID_COL)
    sp.add_argument("--id-filter", choices=["exclude", "include", "none"],
                    default="exclude",
                    help="exclude=background arm, include=UKBB arm, none=whole corpus")
    sp.add_argument("--label", default="background", help="arm name in the output")
    sp.add_argument("--batch-size", type=int, default=100_000)
    sp.add_argument("--progress", type=int, default=25, help="log every N files; 0=off")

    sp = sub.add_parser("citdist",
                        help="citation distribution per (year, type) and per category")
    add_inputs(sp, category=False)
    sp.add_argument("--category", choices=sorted(CATEGORIES), default=None,
                    help="also build per-category references for this system "
                         "(needed by --weights top10f / mncs)")
    # --category is optional here, unlike everywhere else: without it citdist builds
    # only the (year, type) fallback, which is all the paper-level top10 weight needs.
    sp.add_argument("--batch-size", type=int, default=100_000)
    sp.add_argument("--progress", type=int, default=25, help="log every N files; 0=off")

    sp = sub.add_parser("thresholds", help="citdist partials -> top-decile cut-offs")
    sp.add_argument("--out", default="out")
    sp.add_argument("--partials", help="where the citdist partials are (default: --out)")
    sp.add_argument("--percentile", type=float, default=10.0,
                    help="top X%% by citations (default: 10)")
    sp.add_argument("--min-papers", type=int, default=100,
                    help="skip a (year, type) thinner than this — a percentile of 12 "
                         "papers is not a percentile (default: 100)")
    sp.add_argument("--show", type=int, default=25, help="cut-off rows to print")

    sp = sub.add_parser("merge", help="sum partials into the final tables")
    sp.add_argument("--out", default="out")
    sp.add_argument("--partials", help="where the partials are (default: --out)")

    sp = sub.add_parser("selfcheck", help="assert the FOR regex matches shared_for_utils")
    sp.add_argument("--files", nargs="+")
    sp.add_argument("--n", type=int, default=5000)

    args = p.parse_args()
    {"probe": cmd_probe, "count": cmd_count, "merge": cmd_merge,
     "citdist": cmd_citdist, "thresholds": cmd_thresholds,
     "selfcheck": cmd_selfcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
