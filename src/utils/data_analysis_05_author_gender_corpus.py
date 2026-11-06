#!/usr/bin/env python3
"""
Name-inferred gender by FoR field over a large parquet corpus (BMRC VM or a local file)
=====================================================================================

WHAT THIS ANSWERS
-----------------
"Within each Field of Research UK Biobank works in, is the female share of UK Biobank
authorships higher or lower than the female share of all authorships in that field?"

Both arms are counted off the same Dimensions copy, flipped with `--id-filter` exactly as
in the 03 field-count job (data_analysis_03_academic_impact_field_counts.py, "FC" below):

    ukbb        the corpus rows whose id IS in the UK Biobank id list
    background  the corpus rows whose id is NOT in it

SAME METHOD AS 05
-----------------
Gender is the 05 name category (shared_name_gender): the five-library offline ensemble on
each authorship's first name, then the researcher_id consensus that fills an Unknown from
the same researcher's other authorships and blanks every authorship of a researcher whose
names disagree. The unit is the author-paper pair, the field is FoR with every L4 code's
division folded into L2 (05's `add_for_columns(fold_parents=True)`), and the share is
Female / (Female + Male) with Unknown reported beside it (D42).

`names -> classify -> identity -> count -> merge` reproduces 05's `gender_by_field`
exactly when pointed at the local showcase parquet with `--id-filter none` and no
`--codes` (tests/test_05_author_gender_corpus.py holds that parity check).

Two scope rules differ from 05 on purpose, and the driver sets both:

    window      --first-year 2015 --last-year 2025 (05 itself runs 2013–2025)
    --codes     only UK Biobank's own FoR codes (L2 and L4, from `make-inputs`). A paper
                with none of them is not counted; a paper with some is counted only
                under those. Per-field shares are unaffected — a field's background is
                every authorship in that field — the restriction only drops the fields
                UK Biobank never publishes in.

The consensus evidence is every in-window paper in the files read, whatever its field
or arm. On the VM that is the whole literature; locally it is UK Biobank alone. A UK
Biobank author with an initials-only byline can therefore be resolved on the VM by a
paper outside UK Biobank — the same rule applied to both arms, recorded in D-entry of
doc/decisions_05_authors.md.

THE MODES  (array modes are shard-aware; see the companion .sh for the Slurm chain)
---------
    make-inputs  LOCAL. ukbb_ids.txt, ukbb_for_codes.csv, for_2020_codes.txt for BMRC.
    schema       footers (+ a few sampled rows) -> schema_files.csv / schema_groups.csv.
                 Run first: early files may store authors or FoR differently.
    names        corpus read 1: first-name keys and (researcher_id, key) pairs,
                 bucketed by a stable hash so the next two modes split cleanly.
    classify     one bucket of name keys -> 05's ensemble category per key.
    identity     one bucket of researchers -> consensus fills and conflicts.
    count        corpus read 2: authorships tallied by (year, level, code, role,
                 gender, reason) for one arm.
    merge        partials -> gender_field_counts / _papers / _totals / _coverage.
    status       which shards of which mode are done (filesystem only, no squeue).

MISSING DATA IS COUNTED, NEVER DROPPED
--------------------------------------
Every Unknown carries its reason: missing_name, initials_only, no_offline_prediction,
offline_library_conflict or identity_conflict; a consensus fill is marked
identity_consensus_fill. Paper-level gaps are in the totals and papers tables: papers
with no author list, and author lists Dimensions truncated (authors_count larger than
the list), with the number of missing slots.

READ-ONLY BY CONSTRUCTION
-------------------------
Source files are only opened through pyarrow for reading. All output goes to --out.

Author: Jiani Y
Date: 2026-11-06
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if __package__:
    from . import data_analysis_03_academic_impact_field_counts as FC
    from . import shared_name_gender as NG
    from .shared_analysis_window import ANALYSIS_END_YEAR, filter_analysis_window
else:
    import data_analysis_03_academic_impact_field_counts as FC
    import shared_name_gender as NG
    from shared_analysis_window import ANALYSIS_END_YEAR, filter_analysis_window

ID_COL, YEAR_COL, TYPE_COL, DATE_COL = "id", "year", "type", "date"
AUTHORS_COL, AUTHORS_COUNT_COL = "authors", "authors_count"
FIRST_NAME, RESEARCHER_ID = "first_name", "researcher_id"
FOR_SPEC = FC.CATEGORIES["for"]

FIRST_YEAR = 2015
LAST_YEAR = ANALYSIS_END_YEAR
N_BUCKETS = 32

# A name key is the given-name token 05 classifies, or one of two sentinels for the
# authorships it cannot classify by construction. Tokens hold only letters and -'’, so
# the angle brackets cannot collide with a real name.
MISSING_KEY = "<missing_name>"
INITIALS_KEY = "<initials_only>"

GENDERS = ("Female", "Male", "Unknown")
ROLES = ("Single author", "First author", "Middle author", "Last author")
FILL_REASON = "identity_consensus_fill"
CONFLICT_REASON = "identity_conflict"
CONFLICT = "Conflict"
LEVELS = ("L2", "L4")


# =============================================================================
# NAMES — the exact path 05 takes from a raw first_name to a category
# =============================================================================
def normalise_text(value) -> Optional[str]:
    """05's `_normalise_text`: collapse whitespace, strip, empty -> None."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def name_key(first_name) -> str:
    """The token 05's inference actually sees, or the sentinel for why there is none.

    `infer_name_gender(raw)` depends on `raw` only through `normalise_given_name(raw)`,
    and re-normalising the token returns it unchanged, so classifying the key gives
    the same answer as classifying the raw name. Keying on it rather than on the raw
    string folds "Anna", "Anna M." and "ANNA" style variants that 05 already treats
    as one name into one lookup row.
    """
    value = normalise_text(first_name) or ""
    token, _ = NG.normalise_given_name(value)
    if token:
        return token
    return MISSING_KEY if not value.strip() else INITIALS_KEY


def classify_key(key: str) -> NG.NameGenderInference:
    if key == MISSING_KEY:
        return NG.infer_name_gender("")
    if key == INITIALS_KEY:
        return NG.infer_name_gender("X.")
    return NG.infer_name_gender(key)


def stable_bucket(value: str, n_buckets: int) -> int:
    """Hash that is identical in every process (Python's hash() is salted per run)."""
    return zlib.crc32(value.encode("utf-8")) % n_buckets


class KeyCache(dict):
    """first_name -> name key, memoised: given names repeat far more than they vary."""

    def __missing__(self, first_name):
        value = self[first_name] = name_key(first_name)
        return value


# =============================================================================
# SCHEMA — which on-disk shapes this script knows how to read
# =============================================================================
def _leaf_paths(pf: pq.ParquetFile) -> List[str]:
    return [pf.schema.column(i).path for i in range(len(pf.schema))]


def authors_layout(pf: pq.ParquetFile) -> Tuple[Optional[str], List[str], str]:
    """('text' | 'struct' | None, columns to project, problem).

    text    the column is a JSON string per paper (the local showcase export).
    struct  list<struct<first_name, researcher_id, ...>>. Only the two leaves are read,
            so a file's affiliations — most of the author bytes — are never touched.
    None    any other shape. The file is failed loudly, never skipped silently.
    """
    schema = pf.schema_arrow
    if AUTHORS_COL not in schema.names:
        return None, [], "no authors column"
    typ = schema.field(AUTHORS_COL).type
    if pa.types.is_string(typ) or pa.types.is_large_string(typ):
        return "text", [AUTHORS_COL], ""
    if (pa.types.is_list(typ) or pa.types.is_large_list(typ)) and pa.types.is_struct(typ.value_type):
        children = {typ.value_type.field(i).name for i in range(typ.value_type.num_fields)}
        if FIRST_NAME not in children:
            return None, [], f"authors struct has no {FIRST_NAME} ({sorted(children)})"
        wanted = {FIRST_NAME, RESEARCHER_ID}
        leaves = [p for p in _leaf_paths(pf)
                  if p.split(".")[0] == AUTHORS_COL and len(p.split(".")) == 4
                  and p.split(".")[3] in wanted]
        return "struct", leaves, ""
    return None, [], f"unsupported authors type {typ}"


def for_layout(pf: pq.ParquetFile) -> Tuple[Optional[str], Optional[str], str]:
    col = FOR_SPEC.resolve_column(pf.schema_arrow.names)
    if col is None:
        return None, None, "no FoR column"
    typ = pf.schema_arrow.field(col).type
    if pa.types.is_string(typ) or pa.types.is_large_string(typ):
        return col, "text", ""
    if pa.types.is_struct(typ):
        return col, "struct", ""
    return None, None, f"unsupported FoR type {typ}"


def fingerprint(pf: pq.ParquetFile) -> str:
    """The parts of a file's schema this job depends on, as one comparable string."""
    s = pf.schema_arrow

    def typ(name):
        return str(s.field(name).type) if name in s.names else "-"

    for_col = FOR_SPEC.resolve_column(s.names) or "-"
    author_type = typ(AUTHORS_COL)
    if AUTHORS_COL in s.names and "struct" in author_type:
        # Leaf names only: the affiliations sub-schema is irrelevant here and noisy.
        value_type = s.field(AUTHORS_COL).type.value_type
        author_type = "list<struct<" + ",".join(
            value_type.field(i).name for i in range(value_type.num_fields)) + ">>"
    return " | ".join([
        f"year={typ(YEAR_COL)}", f"date={typ(DATE_COL)}", f"type={typ(TYPE_COL)}",
        f"for={for_col}:{typ(for_col) if for_col != '-' else '-'}",
        f"authors={author_type}", f"authors_count={typ(AUTHORS_COUNT_COL)}",
    ])


def _year_range(pf: pq.ParquetFile) -> Tuple[str, str]:
    """Min/max of `year` from row-group statistics — footer only, no data pages."""
    md = pf.metadata
    names = _leaf_paths(pf)
    if YEAR_COL not in names:
        return "", ""
    idx = names.index(YEAR_COL)
    lo = hi = None
    for rg in range(md.num_row_groups):
        st = md.row_group(rg).column(idx).statistics
        if st is None or not st.has_min_max:
            return "", ""
        lo = st.min if lo is None else min(lo, st.min)
        hi = st.max if hi is None else max(hi, st.max)
    return ("" if lo is None else str(lo)), ("" if hi is None else str(hi))


def cmd_schema(args: argparse.Namespace) -> None:
    files = FC.take_shard(FC.collect_files(args), args.shard, args.num_shards)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab = _read_vocab(args.for_vocab) if args.for_vocab else None
    sfx = FC._shard_suffix(args)
    rows_out: List[dict] = []
    samples: Dict[str, dict] = {}
    for path in files:
        row = {"path": path, "fingerprint": "", "rows": "", "row_groups": "",
               "file_bytes": "", "projected_bytes": "", "year_min": "", "year_max": "",
               "authors_format": "", "for_format": "", "has_researcher_id": "",
               "sampled_codes": "", "sampled_non2020_codes": "", "error": ""}
        try:
            pf = pq.ParquetFile(path)
            md = pf.metadata
            afmt, acols, aprob = authors_layout(pf)
            fcol, ffmt, fprob = for_layout(pf)
            row.update(fingerprint=fingerprint(pf), rows=md.num_rows,
                       row_groups=md.num_row_groups, file_bytes=Path(path).stat().st_size,
                       authors_format=afmt or "UNSUPPORTED", for_format=ffmt or "UNSUPPORTED")
            row["year_min"], row["year_max"] = _year_range(pf)
            row["has_researcher_id"] = (
                any(c.endswith("." + RESEARCHER_ID) for c in acols) if afmt == "struct"
                else afmt == "text")
            wanted = set(acols) | {ID_COL, YEAR_COL, DATE_COL, TYPE_COL, AUTHORS_COUNT_COL}
            if fcol:
                wanted.add(fcol)
            proj = 0
            for rg in range(md.num_row_groups):
                g = md.row_group(rg)
                for i in range(g.num_columns):
                    c = g.column(i)
                    if c.path_in_schema in wanted:
                        proj += c.total_compressed_size
            row["projected_bytes"] = proj
            problems = [p for p in (aprob, fprob) if p]
            if args.sample_rows and md.num_rows and not problems:
                batch = next(pf.iter_batches(batch_size=args.sample_rows,
                                             columns=[fcol] + acols))
                codes = [code for cell in batch.column(0).to_pylist()
                         for code, _ in FC._for_pairs(cell, ffmt)]
                row["sampled_codes"] = len(codes)
                if vocab is not None:
                    row["sampled_non2020_codes"] = sum(c not in vocab for c in codes)
                fp = row["fingerprint"]
                if fp not in samples:
                    auth = flatten_authors(batch.column(1), afmt,
                                           np.arange(min(3, batch.num_rows)))
                    samples[fp] = {"path": path, "authorships": auth.head(6).to_dict("records")}
            row["error"] = "; ".join(problems)
        except Exception as exc:  # one corrupt footer must not kill the survey
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows_out.append(row)

    files_df = pd.DataFrame(rows_out)
    files_df.to_csv(out / f"schema_files{sfx}.csv", index=False)
    with open(out / f"schema_samples{sfx}.json", "w") as fh:
        json.dump(samples, fh, indent=2, default=str)
    if args.num_shards <= 1:
        summarise_schema(out)


def summarise_schema(out: Path) -> pd.DataFrame:
    """Combine schema_files*.csv into schema_groups.csv and print what to look at."""
    parts = sorted(out.glob("schema_files*.csv"))
    files_df = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in parts],
                         ignore_index=True)
    num = lambda c: pd.to_numeric(files_df[c], errors="coerce")
    files_df["_rows"], files_df["_bytes"], files_df["_proj"] = num("rows"), num("file_bytes"), num("projected_bytes")
    files_df["_ymin"], files_df["_ymax"] = num("year_min"), num("year_max")
    files_df["_codes"], files_df["_non2020"] = num("sampled_codes"), num("sampled_non2020_codes")
    groups = files_df.groupby(["fingerprint", "authors_format", "for_format"], dropna=False).agg(
        n_files=("path", "size"), rows=("_rows", "sum"), year_min=("_ymin", "min"),
        year_max=("_ymax", "max"), sampled_codes=("_codes", "sum"),
        sampled_non2020_codes=("_non2020", "sum"),
        n_errors=("error", lambda s: int((s != "").sum())), sample_path=("path", "first"),
    ).reset_index()
    groups.to_csv(out / "schema_groups.csv", index=False)

    print(f"{len(files_df):,} files, {int(files_df._rows.sum()):,} rows, "
          f"{files_df._bytes.sum() / 1e9:.1f} GB on disk")
    if files_df._bytes.sum():
        print(f"projected read per corpus pass: {files_df._proj.sum() / 1e9:.2f} GB "
              f"({100 * files_df._proj.sum() / files_df._bytes.sum():.1f} %)")
    print(f"year range from footers: {files_df._ymin.min():.0f}–{files_df._ymax.max():.0f}; "
          f"files reaching {LAST_YEAR}: {int((files_df._ymax >= LAST_YEAR).sum()):,}")
    print(f"\n{len(groups)} schema group(s) -> {out / 'schema_groups.csv'}")
    for _, g in groups.iterrows():
        print(f"  {g.n_files:>6,} files  authors={g.authors_format:11s} for={g.for_format:11s} "
              f"years {g.year_min}–{g.year_max}  errors {g.n_errors}\n      {g.fingerprint}")
    bad = files_df[files_df.error != ""]
    if len(bad):
        print(f"\n!! {len(bad)} file(s) cannot be read by this job — see the error column. "
              f"`count` will fail them and `merge` will refuse until they are resolved.")
    non2020 = files_df._non2020.sum()
    if non2020:
        print(f"\n!! {int(non2020):,} of {int(files_df._codes.sum()):,} sampled FoR codes are "
              f"not ANZSRC 2020 codes — UK Biobank's codes will not match those files.")
    return groups


def _read_vocab(path: str) -> Set[str]:
    return {ln.strip() for ln in open(path) if ln.strip() and not ln.startswith("#")}


# =============================================================================
# READING — one batch of papers -> one flat frame of authorships
# =============================================================================
def flatten_authors(column, fmt: str, rows: np.ndarray) -> pd.DataFrame:
    """Authorships of the papers at `rows` of `column`: row, first_name, researcher_id.

    Entries that are not author records (a null struct, a non-dict JSON element) are
    dropped, as 05's `parse_listcol` + isinstance(dict) filter drops them, and position
    and team size are counted afterwards over what remains.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return pd.DataFrame({"row": pd.Series(dtype="int64"), FIRST_NAME: pd.Series(dtype=object),
                             RESEARCHER_ID: pd.Series(dtype=object)})
    taken = column.take(pa.array(rows))
    if fmt == "struct":
        parents = pc.list_parent_indices(taken).to_numpy()
        flat = taken.flatten()
        valid = flat.is_valid().to_numpy(zero_copy_only=False)
        names = flat.field(FIRST_NAME).to_pylist()
        rids = (flat.field(RESEARCHER_ID).to_pylist()
                if flat.type.get_field_index(RESEARCHER_ID) >= 0 else [None] * len(flat))
        frame = pd.DataFrame({"row": rows[parents], FIRST_NAME: names, RESEARCHER_ID: rids})
        frame = frame[valid]
    elif fmt == "text":
        out_rows, out_names, out_rids = [], [], []
        for row, cell in zip(rows, taken.to_pylist()):
            for author in _parse_json_list(cell):
                if isinstance(author, dict):
                    out_rows.append(row)
                    out_names.append(author.get(FIRST_NAME))
                    out_rids.append(author.get(RESEARCHER_ID))
        frame = pd.DataFrame({"row": np.asarray(out_rows, dtype=np.int64),
                              FIRST_NAME: out_names, RESEARCHER_ID: out_rids})
    else:
        raise ValueError(f"unsupported authors format {fmt!r}")
    frame[FIRST_NAME] = frame[FIRST_NAME].map(lambda v: normalise_text(v) or "")
    frame[RESEARCHER_ID] = frame[RESEARCHER_ID].map(normalise_text)
    return frame.reset_index(drop=True)


def _parse_json_list(cell) -> list:
    if not cell:
        return []
    try:
        value = json.loads(cell)
    except (TypeError, ValueError):
        # The rare cell json rejects (single quotes, Python literals): the shared parser
        # is the one 05 uses, so a cell parses here exactly when it parses there.
        if __package__:
            from .shared_for import parse_listcol
        else:
            from shared_for import parse_listcol
        value = parse_listcol(cell)
    return value if isinstance(value, list) else []


def add_roles(auth: pd.DataFrame) -> pd.DataFrame:
    """05's authorship_role: position within the parsed list and the list's length."""
    auth["position"] = auth.groupby("row").cumcount() + 1
    auth["team_size"] = auth.groupby("row")["row"].transform("size")
    auth["role"] = np.select(
        [auth.team_size.eq(1), auth.position.eq(1), auth.position.eq(auth.team_size)],
        ["Single author", "First author", "Last author"], default="Middle author")
    return auth


def paper_fields(cell, fmt: str, codes: Optional[Dict[str, Set[str]]],
                 labels: Dict[str, str]) -> Tuple[Dict[str, List[str]], bool]:
    """({'L2': [...], 'L4': [...]}, carried any FoR code) for one paper.

    L2 is folded: an L4 code's division is code[:2] whether or not the 2-digit code
    was tagged — 05's fold_parents=True. With `codes`, each level keeps only the
    in-set codes. Labels are harvested into `labels` as a side effect.
    """
    l2: Set[str] = set()
    l4: Set[str] = set()
    for code, label in FC._for_pairs(cell, fmt):
        if len(code) == 2:
            l2.add(code)
        elif len(code) == 4:
            l4.add(code)
            l2.add(code[:2])
        else:
            continue
        if label and code not in labels:
            labels[code] = label
    any_code = bool(l2 or l4)
    if codes is not None:
        l2 &= codes["L2"]
        l4 &= codes["L4"]
    return {"L2": sorted(l2), "L4": sorted(l4)}, any_code


class PaperBatch:
    """The in-window papers of one record batch, with everything both corpus passes need."""

    def __init__(self, batch: pa.RecordBatch, cols: List[str], window_cols: List[str],
                 first_year: int, last_year: int) -> None:
        frame = filter_analysis_window(batch.select(window_cols).to_pandas())
        years = pd.to_numeric(frame[YEAR_COL], errors="coerce")
        keep = years.between(first_year, last_year)
        self.rows = frame.index[keep].to_numpy(dtype=np.int64)
        self.years = years[keep].astype(int).to_numpy()
        self.batch = batch
        self.cols = cols

    def column(self, name: str):
        return self.batch.column(self.cols.index(name))


def open_corpus_file(path: str, need_for: bool, need_ids: bool):
    """(ParquetFile, columns, window columns, authors format, FoR column, FoR format).

    Raises on any shape this job does not know, so the file is logged as failed."""
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    if YEAR_COL not in names:
        raise KeyError(f"missing required column '{YEAR_COL}'")
    afmt, acols, problem = authors_layout(pf)
    if afmt is None:
        raise KeyError(problem)
    cols = [YEAR_COL] + ([DATE_COL] if DATE_COL in names else [])
    window_cols = list(cols)
    if need_ids:
        if ID_COL not in names:
            raise KeyError(f"missing required column '{ID_COL}'")
        cols.append(ID_COL)
    fcol = ffmt = None
    if need_for:
        fcol, ffmt, problem = for_layout(pf)
        if fcol is None:
            raise KeyError(problem)
        cols.append(fcol)
        if AUTHORS_COUNT_COL in names:
            cols.append(AUTHORS_COUNT_COL)
    cols += acols
    return pf, cols, window_cols, afmt, fcol, ffmt


def _arm_rows(pb: PaperBatch, id_set: Optional[Set[str]], mode: str
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, years, years of the in-window papers that belong to the other arm)."""
    if id_set is None or mode == "none":
        return pb.rows, pb.years, pb.years[:0]
    ids = pb.column(ID_COL).to_pylist()
    hit = np.fromiter((ids[r] in id_set for r in pb.rows), dtype=bool, count=len(pb.rows))
    keep = hit if mode == "include" else ~hit
    return pb.rows[keep], pb.years[keep], pb.years[~keep]


# =============================================================================
# SHARED BOOKKEEPING — file logs and done markers
# =============================================================================
def _done_path(out: Path, mode: str, tag: str) -> Path:
    return out / "done" / f"{mode}.{tag}.json"


def _write_done(out: Path, mode: str, tag: str, stats: dict) -> None:
    p = _done_path(out, mode, tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=2, default=str))
    tmp.replace(p)          # the marker appears only after every output is on disk


def _require_done(out: Path, mode: str, tags: Sequence[str], allow_failed: bool) -> None:
    missing, failed = [], []
    for tag in tags:
        p = _done_path(out, mode, tag)
        if not p.exists():
            missing.append(tag)
        elif json.loads(p.read_text()).get("n_failed", 0):
            failed.append(tag)
    if missing:
        sys.exit(f"{mode}: {len(missing)} unit(s) not done ({', '.join(missing[:8])}"
                 f"{' …' if len(missing) > 8 else ''}) — rerun them first")
    if failed and not allow_failed:
        sys.exit(f"{mode}: {len(failed)} unit(s) logged failed files ({', '.join(failed[:8])}). "
                 f"Read their filelog.*.csv; --allow-failed proceeds without those files.")


def _shard_tags(num_shards: int, prefix: str = "") -> List[str]:
    return [f"{prefix}shard{i:05d}" for i in range(max(num_shards, 1))]


def _bucket_tags(n_buckets: int) -> List[str]:
    return [f"b{i:03d}" for i in range(n_buckets)]


def _shard_tag(args) -> str:
    return f"shard{args.shard:05d}"


def _stream_files(files: Sequence[str], log_path: Path, worker, progress: int) -> Tuple[int, int]:
    """Run `worker(path) -> (rows, rows_used)` over files with a per-file log.
    One unreadable file among thousands must not lose the shard's work."""
    n_ok = n_err = 0
    t_all = time.time()
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "rows", "rows_used", "seconds", "error"])
        for i, path in enumerate(files, 1):
            t0 = time.time()
            try:
                rows, used = worker(path)
                w.writerow([path, rows, used, f"{time.time() - t0:.2f}", ""])
                n_ok += 1
            except Exception as exc:
                w.writerow([path, "", "", f"{time.time() - t0:.2f}",
                            f"{type(exc).__name__}: {exc}"])
                n_err += 1
                print(f"  !! {path}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            fh.flush()
            if progress and i % progress == 0:
                print(f"  {i}/{len(files)} files, {time.time() - t_all:.0f}s", flush=True)
    return n_ok, n_err


# =============================================================================
# MODE: names — corpus read 1
# =============================================================================
def cmd_names(args: argparse.Namespace) -> None:
    files = FC.take_shard(FC.collect_files(args), args.shard, args.num_shards)
    out = Path(args.out)
    tag = _shard_tag(args)
    for sub in ("names", "idpairs", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cache = KeyCache()
    key_counts: Counter = Counter()
    pairs: List[pd.DataFrame] = []
    n_pairs_rows = [0]

    def compact():
        if len(pairs) > 1:
            merged = pd.concat(pairs, ignore_index=True).drop_duplicates()
            pairs[:] = [merged]

    def worker(path):
        pf, cols, window_cols, afmt, _, _ = open_corpus_file(path, need_for=False, need_ids=False)
        rows = used = 0
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
            rows += batch.num_rows
            pb = PaperBatch(batch, cols, window_cols, args.first_year, args.last_year)
            used += len(pb.rows)
            auth = flatten_authors(_authors_column(batch), afmt, pb.rows)
            if auth.empty:
                continue
            keys = auth[FIRST_NAME].map(cache)
            key_counts.update(keys.value_counts().to_dict())
            resolved = auth[RESEARCHER_ID].notna()
            if resolved.any():
                frame = pd.DataFrame({"rid": auth.loc[resolved, RESEARCHER_ID].to_numpy(),
                                      "key": keys[resolved].to_numpy()}).drop_duplicates()
                pairs.append(frame)
                n_pairs_rows[0] += len(frame)
                if n_pairs_rows[0] > args.compact_rows:
                    compact()
                    n_pairs_rows[0] = len(pairs[0]) if pairs else 0
        return rows, used

    t0 = time.time()
    n_ok, n_err = _stream_files(files, out / "logs" / f"filelog.names.{tag}.csv", worker, args.progress)
    compact()

    names = pd.DataFrame({"key": list(key_counts), "n_authorships": list(key_counts.values())})
    names["bucket"] = names.key.map(lambda k: stable_bucket(k, args.buckets))
    idp = pairs[0] if pairs else pd.DataFrame({"rid": pd.Series(dtype=object), "key": pd.Series(dtype=object)})
    idp = idp.assign(bucket=idp.rid.map(lambda r: stable_bucket(r, args.buckets)))
    for b in range(args.buckets):
        nb = names[names.bucket == b].drop(columns="bucket")
        pb_ = idp[idp.bucket == b].drop(columns="bucket")
        nb.to_parquet(out / "names" / f"names.b{b:03d}.{tag}.parquet", index=False)
        pb_.to_parquet(out / "idpairs" / f"idpairs.b{b:03d}.{tag}.parquet", index=False)
    stats = {"files": len(files), "n_ok": n_ok, "n_failed": n_err, "buckets": args.buckets,
             "authorships": int(sum(key_counts.values())), "distinct_keys": len(names),
             "id_key_pairs": len(idp), "seconds": round(time.time() - t0, 1),
             "first_year": args.first_year, "last_year": args.last_year}
    _write_done(out, "names", tag, stats)
    print(f"[names {tag}] {n_ok} ok / {n_err} failed | {stats['authorships']:,} authorships | "
          f"{len(names):,} keys | {len(idp):,} researcher-key pairs | {stats['seconds']}s")


def _authors_column(batch: pa.RecordBatch):
    """The authors column of a batch. A struct projection of two leaves still comes back
    as one `authors` column, holding only those two fields."""
    return batch.column(batch.schema.names.index(AUTHORS_COL))


# =============================================================================
# MODE: classify — one bucket of name keys
# =============================================================================
LOOKUP_COLUMNS = ["key", "gender", "reason", "method", "vote_count", "vote_margin",
                  "source_count", "source_predictions", "n_authorships"]


def cmd_classify(args: argparse.Namespace) -> None:
    out = Path(args.out)
    _require_done(out, "names", _shard_tags(args.num_shards), args.allow_failed)
    (out / "lookup").mkdir(parents=True, exist_ok=True)
    buckets = [args.bucket] if args.bucket is not None else range(args.buckets)
    for b in buckets:
        t0 = time.time()
        parts = sorted((out / "names").glob(f"names.b{b:03d}.shard*.parquet"))
        names = (pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
                 .groupby("key", as_index=False)["n_authorships"].sum())
        rows = []
        for key in names.key:
            r = classify_key(key)
            rows.append((key, r.category, r.unknown_reason, r.method, r.vote_count,
                         r.vote_margin, r.source_count, r.source_predictions))
        lookup = pd.DataFrame(rows, columns=LOOKUP_COLUMNS[:-1]).merge(names, on="key")
        lookup.to_parquet(out / "lookup" / f"name_lookup.b{b:03d}.parquet", index=False)
        stats = {"keys": len(lookup), "n_failed": 0, "seconds": round(time.time() - t0, 1),
                 **{g: int(lookup.loc[lookup.gender == g, "n_authorships"].sum()) for g in GENDERS}}
        _write_done(out, "classify", f"b{b:03d}", stats)
        print(f"[classify b{b:03d}] {len(lookup):,} keys | authorships F {stats['Female']:,} "
              f"M {stats['Male']:,} U {stats['Unknown']:,} | {stats['seconds']}s")


def load_lookup(out: Path) -> pd.DataFrame:
    parts = sorted((out / "lookup").glob("name_lookup.b*.parquet"))
    if not parts:
        sys.exit(f"no name_lookup.b*.parquet under {out / 'lookup'} — run classify first")
    return pd.concat([pd.read_parquet(p, columns=["key", "gender", "reason"]) for p in parts],
                     ignore_index=True)


# =============================================================================
# MODE: identity — 05's researcher_id consensus, one bucket of researchers
# =============================================================================
def identity_overrides(pairs: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """researcher_id -> 'Female' | 'Male' | 'Conflict', for the researchers that change
    something. Same rule as shared_name_gender.classify_authorship_names:

      * the classified direct categories of a researcher's authorships are the evidence;
      * one category  -> it fills every Unknown authorship of that researcher;
      * two categories -> every authorship of that researcher becomes Unknown.

    A one-category researcher with no Unknown authorship changes nothing, so is not kept.
    """
    merged = pairs.merge(lookup[["key", "gender"]], on="key", how="left")
    if merged.gender.isna().any():
        missing = merged.loc[merged.gender.isna(), "key"].unique()[:5]
        raise KeyError(f"{merged.gender.isna().sum()} researcher-name pairs have no lookup "
                       f"row (e.g. {list(missing)}) — the names/classify outputs are incomplete")
    flags = pd.crosstab(merged.rid, merged.gender).reindex(columns=list(GENDERS), fill_value=0) > 0
    conflict = flags.Female & flags.Male
    single = flags.Female ^ flags.Male
    fill = single & flags.Unknown
    over = pd.Series(CONFLICT, index=flags.index[conflict])
    over = pd.concat([over, pd.Series(np.where(flags.Female[fill], "Female", "Male"),
                                      index=flags.index[fill])])
    return over.rename("override").rename_axis("rid").reset_index()


def cmd_identity(args: argparse.Namespace) -> None:
    out = Path(args.out)
    _require_done(out, "names", _shard_tags(args.num_shards), args.allow_failed)
    _require_done(out, "classify", _bucket_tags(args.buckets), False)
    (out / "identity").mkdir(parents=True, exist_ok=True)
    lookup = load_lookup(out)
    buckets = [args.bucket] if args.bucket is not None else range(args.buckets)
    for b in buckets:
        t0 = time.time()
        parts = sorted((out / "idpairs").glob(f"idpairs.b{b:03d}.shard*.parquet"))
        pairs = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).drop_duplicates()
        over = identity_overrides(pairs, lookup)
        over.to_parquet(out / "identity" / f"overrides.b{b:03d}.parquet", index=False)
        stats = {"researchers": int(pairs.rid.nunique()), "n_failed": 0,
                 "conflicts": int(over.override.eq(CONFLICT).sum()),
                 "fills": int(over.override.ne(CONFLICT).sum()),
                 "seconds": round(time.time() - t0, 1)}
        _write_done(out, "identity", f"b{b:03d}", stats)
        print(f"[identity b{b:03d}] {stats['researchers']:,} researchers | "
              f"{stats['fills']:,} fill | {stats['conflicts']:,} conflict | {stats['seconds']}s")


def load_overrides(out: Path) -> Dict[str, str]:
    parts = sorted((out / "identity").glob("overrides.b*.parquet"))
    if not parts:
        sys.exit(f"no overrides.b*.parquet under {out / 'identity'} — run identity first")
    over = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return dict(zip(over.rid, over.override))


def assign_gender(auth: pd.DataFrame, keys: pd.Series, lookup: Dict[str, Tuple[str, str]],
                  overrides: Dict[str, str]) -> pd.DataFrame:
    """Direct category from the lookup, then 05's identity fill and conflict, in 05's order."""
    looked = keys.map(lookup)
    if looked.isna().any():
        missing = keys[looked.isna()].unique()[:5]
        raise KeyError(f"{looked.isna().sum()} authorships have a name key with no lookup row "
                       f"(e.g. {list(missing)}) — names must cover every file count reads")
    gender = looked.map(lambda t: t[0])
    reason = looked.map(lambda t: t[1])
    over = auth[RESEARCHER_ID].map(overrides)
    fill = gender.eq("Unknown") & over.isin(["Female", "Male"])
    gender = gender.mask(fill, over)
    reason = reason.mask(fill, FILL_REASON)
    conflict = over.eq(CONFLICT)
    gender = gender.mask(conflict, "Unknown")
    reason = reason.mask(conflict, CONFLICT_REASON)
    auth["gender"] = gender.to_numpy()
    auth["reason"] = reason.fillna("").to_numpy()
    return auth


# =============================================================================
# MODE: count — corpus read 2, one arm
# =============================================================================
COUNT_KEYS = ["year", "level", "code", "role", "gender", "reason"]
PAPER_KEYS = ["year", "level", "code"]


def read_codes(path: str) -> Dict[str, Set[str]]:
    df = pd.read_csv(path, dtype=str)
    return {lvl: set(df.loc[df.level == lvl, "code"]) for lvl in LEVELS}


def count_batch(pb: PaperBatch, rows: np.ndarray, years: np.ndarray, afmt: str, ffmt: str,
                fcol: str, codes, labels, cache: KeyCache, lookup, overrides) -> dict:
    """One batch of one arm -> partial tallies (frames) and paper totals (dict by year)."""
    cells = pb.column(fcol).to_pylist()
    counts_col = (pb.column(AUTHORS_COUNT_COL).to_pylist()
                  if AUTHORS_COUNT_COL in pb.cols else None)
    totals = defaultdict(lambda: Counter())
    keep_rows, keep_years, field_rows = [], [], []
    for row, year in zip(rows, years):
        t = totals[int(year)]
        t["n_papers"] += 1
        fields, any_code = paper_fields(cells[row], ffmt, codes, labels)
        if not any_code:
            t["n_no_for"] += 1
            continue
        if not (fields["L2"] or fields["L4"]):
            t["n_no_inset_for"] += 1
            continue
        t["n_counted"] += 1
        keep_rows.append(row)
        keep_years.append(int(year))
        for level in LEVELS:
            for code in fields[level]:
                field_rows.append((row, level, code, len(fields[level])))
    result = {"totals": totals, "counts": None, "papers": None, "coverage": None}
    if not keep_rows:
        return result
    year_of = dict(zip(keep_rows, keep_years))
    fields_df = pd.DataFrame(field_rows, columns=["row", "level", "code", "n_level"])
    fields_df["year"] = fields_df.row.map(year_of)

    auth = flatten_authors(_authors_column(pb.batch), afmt, np.asarray(keep_rows))
    n_auth = auth.groupby("row").size() if len(auth) else pd.Series(dtype=int)
    paper = pd.DataFrame({"row": keep_rows, "year": keep_years})
    paper["n_authors"] = paper.row.map(n_auth).fillna(0).astype(int)
    paper["reported"] = ([counts_col[r] for r in keep_rows] if counts_col is not None
                         else [None] * len(keep_rows))
    reported = pd.to_numeric(paper.reported, errors="coerce")
    paper["no_authors"] = paper.n_authors.eq(0)
    paper["missing_slots"] = (reported - paper.n_authors).clip(lower=0).fillna(0).astype(int)
    paper["truncated"] = paper.missing_slots.gt(0)
    for y, g in paper.groupby("year"):
        t = totals[int(y)]
        t["n_no_authors"] += int(g.no_authors.sum())
        t["n_truncated"] += int(g.truncated.sum())
        t["n_missing_slots"] += int(g.missing_slots.sum())
        t["n_authorships"] += int(g.n_authors.sum())

    per_field = fields_df.merge(paper[["row", "no_authors", "truncated", "missing_slots"]], on="row")
    result["papers"] = per_field.groupby(PAPER_KEYS).agg(
        n_papers=("row", "size"), n_papers_no_authors=("no_authors", "sum"),
        n_papers_truncated=("truncated", "sum"), n_missing_slots=("missing_slots", "sum"),
    ).reset_index()
    if auth.empty:
        return result

    auth = add_roles(auth)
    keys = auth[FIRST_NAME].map(cache)
    auth = assign_gender(auth, keys, lookup, overrides)
    long = auth[["row", "role", "gender", "reason", "team_size"]].merge(
        fields_df[["row", "year", "level", "code", "n_level"]], on="row")
    long["n_credit"] = 1.0 / (long.team_size * long.n_level)
    result["counts"] = long.groupby(COUNT_KEYS).agg(
        n_authorships=("row", "size"), n_credit=("n_credit", "sum")).reset_index()
    # A paper's credit is 1/team_size per author, split over its codes at each level, so
    # per level the credit summed over codes must equal the papers that carry the level
    # and at least one author — the identity `merge` reconciles.
    with_level = fields_df.drop_duplicates(["row", "level"])[["row", "year", "level"]]
    cov = with_level[with_level.row.isin(auth.row.unique())]
    n_auth = auth.groupby("row").size()
    result["coverage"] = cov.assign(n_auth=cov.row.map(n_auth)).groupby(["year", "level"]).agg(
        n_papers_with_level=("row", "size"),
        n_authorships_with_level=("n_auth", "sum")).reset_index()
    return result


class Accumulator:
    """Concatenate partial group-by frames and re-reduce them every so often."""

    def __init__(self, keys: List[str], every: int = 40) -> None:
        self.keys, self.every, self.parts = keys, every, []

    def add(self, frame: Optional[pd.DataFrame]) -> None:
        if frame is None or frame.empty:
            return
        self.parts.append(frame)
        if len(self.parts) >= self.every:
            self.parts = [self.result()]

    def result(self) -> pd.DataFrame:
        if not self.parts:
            return pd.DataFrame(columns=self.keys)
        df = pd.concat(self.parts, ignore_index=True)
        return df.groupby(self.keys, as_index=False).sum(numeric_only=True)


def cmd_count(args: argparse.Namespace) -> None:
    files = FC.take_shard(FC.collect_files(args), args.shard, args.num_shards)
    out = Path(args.out)
    tag = f"{args.label}.{_shard_tag(args)}"
    for sub in ("counts", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    id_set = FC.load_id_set(args.ids, args.id_column) if args.ids else None
    if args.id_filter != "none" and id_set is None:
        sys.exit(f"--id-filter {args.id_filter} needs --ids")
    codes = read_codes(args.codes) if args.codes else None
    _require_done(out, "identity", _bucket_tags(args.buckets), False)
    lk = load_lookup(out)
    lookup = dict(zip(lk.key, zip(lk.gender, lk.reason.fillna(""))))
    del lk
    overrides = load_overrides(out)
    print(f"[count {tag}] {len(files)} file(s), id-filter={args.id_filter}, "
          f"{len(lookup):,} name keys, {len(overrides):,} identity overrides, "
          f"codes={'all' if codes is None else sum(map(len, codes.values()))}", flush=True)

    cache = KeyCache()
    labels: Dict[str, str] = {}
    counts = Accumulator(COUNT_KEYS)
    papers = Accumulator(PAPER_KEYS)
    coverage = Accumulator(["year", "level"])
    totals: Dict[int, Counter] = defaultdict(Counter)

    def worker(path):
        pf, cols, window_cols, afmt, fcol, ffmt = open_corpus_file(
            path, need_for=True, need_ids=id_set is not None)
        rows_read = used = 0
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
            rows_read += batch.num_rows
            pb = PaperBatch(batch, cols, window_cols, args.first_year, args.last_year)
            rows, years, other = _arm_rows(pb, id_set, args.id_filter)
            for y, k in zip(*np.unique(other, return_counts=True)):
                totals[int(y)]["n_filtered_arm"] += int(k)
            used += len(rows)
            part = count_batch(pb, rows, years, afmt, ffmt, fcol, codes, labels, cache,
                               lookup, overrides)
            for y, c in part["totals"].items():
                totals[y].update(c)
            counts.add(part["counts"])
            papers.add(part["papers"])
            coverage.add(part["coverage"])
        return rows_read, used

    t0 = time.time()
    n_ok, n_err = _stream_files(files, out / "logs" / f"filelog.count.{tag}.csv", worker, args.progress)
    stem = out / "counts"
    counts_df = counts.result().assign(arm=args.label)
    counts_df.to_parquet(stem / f"counts.{tag}.parquet", index=False)
    papers.result().assign(arm=args.label).to_parquet(stem / f"papers.{tag}.parquet", index=False)
    coverage.result().assign(arm=args.label).to_parquet(stem / f"coverage.{tag}.parquet", index=False)
    tot = pd.DataFrame([{"year": y, **c} for y, c in sorted(totals.items())]).fillna(0)
    tot.assign(arm=args.label).to_parquet(stem / f"totals.{tag}.parquet", index=False)
    pd.DataFrame({"code": list(labels), "label": list(labels.values())}).to_parquet(
        stem / f"labels.{tag}.parquet", index=False)
    stats = {"files": len(files), "n_ok": n_ok, "n_failed": n_err,
             "authorships": int(counts_df.n_authorships.sum()) if len(counts_df) else 0,
             "seconds": round(time.time() - t0, 1), "arm": args.label,
             "id_filter": args.id_filter, "first_year": args.first_year,
             "last_year": args.last_year, "codes": args.codes or ""}
    _write_done(out, "count", tag, stats)
    print(f"[count {tag}] {n_ok} ok / {n_err} failed | {stats['authorships']:,} "
          f"authorship-field rows | {stats['seconds']}s")


# =============================================================================
# MODE: merge
# =============================================================================
def cmd_merge(args: argparse.Namespace) -> None:
    out = Path(args.out)
    arms = [a for a in args.arms.split(",") if a]
    for arm in arms:
        _require_done(out, "count", _shard_tags(args.num_shards, f"{arm}."), args.allow_failed)

    def read(kind, keys):
        parts = [p for arm in arms for p in sorted((out / "counts").glob(f"{kind}.{arm}.shard*.parquet"))]
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df = df.groupby(["arm", *keys], as_index=False).sum(numeric_only=True)
        # A shard that met no paper writes an empty partial, which makes every count a
        # float after the concat; all counts but the fractional credit are integers.
        for c in df.columns:
            if c.startswith("n_") and c != "n_credit":
                df[c] = df[c].fillna(0).astype("int64")
        if "year" in df.columns:
            df["year"] = df["year"].astype("int64")
        return df

    counts = read("counts", COUNT_KEYS)
    papers = read("papers", PAPER_KEYS)
    coverage = read("coverage", ["year", "level"])
    totals = read("totals", ["year"])
    labels = (pd.concat([pd.read_parquet(p) for p in sorted((out / "counts").glob("labels.*.parquet"))])
              .drop_duplicates("code").set_index("code")["label"])
    for df in (counts, papers):
        df["label"] = df.code.map(labels).fillna("")
    int_cols = [c for c in totals.columns if c.startswith("n_")]
    totals[int_cols] = totals[int_cols].astype("int64")

    for name, df in (("gender_field_counts", counts), ("gender_field_papers", papers),
                     ("gender_field_coverage", coverage), ("gender_field_totals", totals)):
        df.to_parquet(out / f"{name}.parquet", index=False)
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {out / name}.parquet ({len(df):,} rows)")

    problems = reconcile(counts, coverage, totals, arms)
    for line in problems["lines"]:
        print(line)
    if problems["mismatch"]:
        sys.exit("merge: reconciliation failed — the tables above were written for inspection "
                 "but must not be used")


def reconcile(counts: pd.DataFrame, coverage: pd.DataFrame, totals: pd.DataFrame,
              arms: Sequence[str]) -> dict:
    """The identities that must hold if the tallies are complete and non-overlapping."""
    lines, mismatch = ["\nreconciliation:"], False
    chk = counts.groupby(["arm", "level"]).n_credit.sum()
    ref = coverage.groupby(["arm", "level"]).n_papers_with_level.sum()
    for key in ref.index:
        got, want = float(chk.get(key, 0.0)), float(ref.loc[key])
        ok = abs(got - want) <= max(1e-6 * want, 1e-6)
        mismatch |= not ok
        lines.append(f"  credit {key[0]:12s} {key[1]}  {got:16,.1f} vs {want:16,.1f}  "
                     f"{'ok' if ok else 'MISMATCH'}")
    by_arm = totals.set_index(["arm", "year"])
    if len(arms) == 2 and "n_filtered_arm" in totals.columns:
        # include + exclude off one file list: each arm's filtered papers are the other's
        # papers, year by year. Anything else means the arms overlap or one is short.
        a, b = arms
        ta, tb = by_arm.loc[a], by_arm.loc[b]
        years = ta.index.union(tb.index)
        ta, tb = ta.reindex(years, fill_value=0), tb.reindex(years, fill_value=0)
        bad = (ta.n_papers != tb.n_filtered_arm) | (tb.n_papers != ta.n_filtered_arm)
        mismatch |= bool(bad.any())
        lines.append(f"  arms   {a} / {b} partition the in-window papers: "
                     f"{'ok' if not bad.any() else 'MISMATCH in ' + ', '.join(map(str, years[bad]))}")
    for arm, g in totals.groupby("arm"):
        n = g.n_papers.sum()
        lines.append(f"  [{arm}] in-window papers {n:,} | counted {g.n_counted.sum():,} | "
                     f"no FoR {g.n_no_for.sum():,} | no in-set FoR {g.get('n_no_inset_for', pd.Series([0])).sum():,} | "
                     f"no authors {g.n_no_authors.sum():,} | truncated {g.n_truncated.sum():,} "
                     f"({g.n_missing_slots.sum():,} slots)")
    return {"lines": lines, "mismatch": mismatch}


# =============================================================================
# MODE: status — what is done, from the done markers alone
# =============================================================================
def cmd_status(args: argparse.Namespace) -> None:
    out = Path(args.out)
    plan = {"names": _shard_tags(args.num_shards),
            "classify": _bucket_tags(args.buckets),
            "identity": _bucket_tags(args.buckets)}
    for arm in [a for a in args.arms.split(",") if a]:
        plan[f"count.{arm}"] = _shard_tags(args.num_shards, f"{arm}.")
    if args.missing_only:
        # Machine-readable, for the driver's `resume`: the Slurm array indices of the
        # unit's unfinished tasks, comma-separated, and nothing else.
        if args.missing_only not in plan:
            sys.exit(f"--missing-only: unknown unit {args.missing_only!r}; one of {list(plan)}")
        mode = args.missing_only.split(".")[0]
        tags = plan[args.missing_only]
        print(",".join(str(i) for i, t in enumerate(tags) if not _done_path(out, mode, t).exists()))
        return
    for name, tags in plan.items():
        mode = name.split(".")[0]
        missing = [i for i, t in enumerate(tags) if not _done_path(out, mode, t).exists()]
        failed = [i for i, t in enumerate(tags) if _done_path(out, mode, t).exists()
                  and json.loads(_done_path(out, mode, t).read_text()).get("n_failed", 0)]
        print(f"{name:20s} done {len(tags) - len(missing):>5}/{len(tags):<5} "
              f"missing={','.join(map(str, missing)) or '-'}  "
              f"with_failed_files={','.join(map(str, failed)) or '-'}")


# =============================================================================
# MODE: make-inputs — LOCAL, the three small files BMRC needs
# =============================================================================
def cmd_make_inputs(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils import shared_paths as P  # noqa: E402

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.files[0]) if args.files else P.SHOWCASE_PLUS
    df = pq.read_table(src, columns=[ID_COL, YEAR_COL, DATE_COL, "category_for_2020"]).to_pandas()
    ids = sorted({i for i in df[ID_COL] if i})
    (out / "ukbb_ids.txt").write_text("\n".join(ids) + "\n")

    window = filter_analysis_window(df)
    window = window[pd.to_numeric(window[YEAR_COL], errors="coerce").between(args.first_year, args.last_year)]
    labels: Dict[str, str] = {}
    n = {lvl: Counter() for lvl in LEVELS}
    for cell in window["category_for_2020"]:
        fields, _ = paper_fields(cell, "text", None, labels)
        for lvl in LEVELS:
            n[lvl].update(fields[lvl])
    vocab = pd.read_csv(P.FOR_2020_CODES, dtype=str)
    names = dict(zip(vocab.code, vocab.name))
    rows = [{"level": lvl, "code": c, "label": names.get(c) or labels.get(c, ""),
             "n_ukbb_papers": k} for lvl in LEVELS for c, k in sorted(n[lvl].items())]
    pd.DataFrame(rows).to_csv(out / "ukbb_for_codes.csv", index=False)
    (out / "for_2020_codes.txt").write_text("\n".join(sorted(vocab.code)) + "\n")
    print(f"{len(ids):,} ids, {len(window):,} papers in {args.first_year}–{args.last_year}: "
          f"{len(n['L2'])} L2 + {len(n['L4'])} L4 codes -> {out}")


# =============================================================================
# COMPARISON — LOCAL, on the merged tables copied back from BMRC
# =============================================================================
def wilson_interval(successes, total, z: float = 1.959963984540054):
    """Vectorised Wilson 95 % interval in percent (05's `_wilson_interval`, element-wise)."""
    k = np.asarray(successes, dtype=float)
    n = np.asarray(total, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denominator = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denominator
    low, high = 100 * (centre - margin), 100 * (centre + margin)
    return np.where(n > 0, low, np.nan), np.where(n > 0, high, np.nan)


def gender_shares(counts: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Female / (Female + Male) per group, with the Unknown share and the two bounds an
    Unknown could move it to (all of them Female, all of them Male)."""
    wide = (counts.groupby([*keys, "gender"], observed=True).n_authorships.sum()
            .unstack(fill_value=0).reindex(columns=list(GENDERS), fill_value=0))
    wide["classified"] = wide.Female + wide.Male
    wide["total"] = wide.classified + wide.Unknown
    share = lambda num, den: 100 * num / den.replace(0, np.nan)
    wide["female_share"] = share(wide.Female, wide.classified)
    wide["ci_low"], wide["ci_high"] = wilson_interval(wide.Female, wide.classified)
    wide["unknown_share"] = share(wide.Unknown, wide.total)
    wide["female_share_if_unknown_male"] = share(wide.Female, wide.total)
    wide["female_share_if_unknown_female"] = share(wide.Female + wide.Unknown, wide.total)
    return wide.reset_index()


def _standardised(counts: pd.DataFrame, strata: Sequence[str], groups: Sequence[str],
                  ukbb: str, background: str) -> pd.DataFrame:
    """Background female share re-weighted to UK Biobank's mix over `strata`, per group.

    Within each group the background share in every stratum is weighted by that
    stratum's share of UK Biobank's classified authorships, so a gap that survives is
    not a mix effect. Strata where the background has no classified authorship are
    dropped and the weights renormalised.
    """
    keys = [*groups, *strata]
    s = gender_shares(counts[counts.arm.isin([ukbb, background])], ["arm", *keys])
    u = s.loc[s.arm == ukbb, [*keys, "classified"]]
    b = s.loc[s.arm == background, [*keys, "female_share"]]
    j = u.merge(b, on=keys).dropna(subset=["female_share"])
    j = j[j.classified > 0]
    if not groups:
        if not j.classified.sum():      # e.g. a role UK Biobank has no authorship in
            return float("nan")
        return float((j.classified * j.female_share).sum() / j.classified.sum())
    j["weighted"] = j.classified * j.female_share
    g = j.groupby(list(groups))
    return (g.weighted.sum() / g.classified.sum()).rename("background_standardised").reset_index()


def compare_fields(counts: pd.DataFrame, level: str = "L2", *, ukbb: str = "ukbb",
                   background: str = "background", role: Optional[str] = None,
                   min_classified: int = 0) -> pd.DataFrame:
    """One row per field: both arms' female share, the gap, and the year-standardised
    background share (the background re-weighted to UK Biobank's year mix in that field)."""
    c = counts[counts.level == level]
    if role is not None:
        c = c[c.role == role]
    shares = gender_shares(c, ["arm", "code", "label"])
    cols = ["Female", "Male", "Unknown", "classified", "female_share", "ci_low", "ci_high",
            "unknown_share", "female_share_if_unknown_male", "female_share_if_unknown_female"]
    u = shares[shares.arm == ukbb].set_index(["code", "label"])[cols].add_prefix("ukbb_")
    b = shares[shares.arm == background].set_index(["code", "label"])[cols].add_prefix("background_")
    out = u.join(b, how="left").reset_index()
    std = _standardised(c, ["year"], ["code"], ukbb, background)
    out = out.merge(std, on="code", how="left")
    out["gap_pp"] = out.ukbb_female_share - out.background_female_share
    out["gap_standardised_pp"] = out.ukbb_female_share - out.background_standardised
    out["level"] = level
    out = out[out.ukbb_classified >= min_classified]
    return out.sort_values("ukbb_female_share", ascending=False).reset_index(drop=True)


def compare_overall(counts: pd.DataFrame, level: str = "L2", *, ukbb: str = "ukbb",
                    background: str = "background") -> pd.DataFrame:
    """UK Biobank vs the background over all in-set fields of one level, per role and
    overall: the crude background share, and the background re-weighted to UK Biobank's
    field x year mix. The unit is the authorship-field pair, as in the field table."""
    c = counts[counts.level == level]
    rows = []
    for role in [None, *ROLES]:
        r = c if role is None else c[c.role == role]
        crude = gender_shares(r, ["arm"]).set_index("arm")
        std = _standardised(r, ["code", "year"], [], ukbb, background)
        rows.append({
            "level": level, "role": role or "All authorships",
            "ukbb_female_share": crude.female_share.get(ukbb, np.nan),
            "ukbb_ci_low": crude.ci_low.get(ukbb, np.nan),
            "ukbb_ci_high": crude.ci_high.get(ukbb, np.nan),
            "ukbb_classified": crude.classified.get(ukbb, 0),
            "background_female_share": crude.female_share.get(background, np.nan),
            "background_standardised": float(std),
            "ukbb_unknown_share": crude.unknown_share.get(ukbb, np.nan),
            "background_unknown_share": crude.unknown_share.get(background, np.nan),
        })
    out = pd.DataFrame(rows)
    out["gap_standardised_pp"] = out.ukbb_female_share - out.background_standardised
    return out


# Paper-level gaps, in the order a reader follows the funnel. n_filtered_arm (the other
# arm's papers) and n_authorships are bookkeeping, not missing data, so are left out.
PAPER_MEASURES = ("n_papers", "n_counted", "n_no_for", "n_no_inset_for", "n_no_authors",
                  "n_truncated", "n_missing_slots")


def missing_summary(counts: pd.DataFrame, totals: pd.DataFrame, level: str = "L2") -> pd.DataFrame:
    """Why authorships are Unknown, and what is missing at paper level, per arm.

    One long table: `unit` is 'authorship-field pair' (rows by gender and reason, share
    of the arm's pairs at `level`) or 'paper' (the totals table, share of the arm's
    in-window papers)."""
    c = counts[counts.level == level]
    pairs = (c.groupby(["arm", "gender", "reason"]).n_authorships.sum().rename("n").reset_index())
    pairs["share_pct"] = 100 * pairs.n / pairs.groupby("arm").n.transform("sum")
    pairs.insert(1, "unit", f"authorship-field pair ({level})")
    measures = [m for m in PAPER_MEASURES if m in totals.columns]
    paper = totals.groupby("arm")[measures].sum()
    paper = paper.reset_index().melt(id_vars="arm", var_name="reason", value_name="n")
    in_window = paper[paper.reason == "n_papers"].set_index("arm").n
    paper["share_pct"] = 100 * paper.n / paper.arm.map(in_window)
    # Missing author slots are slots, not papers: a share of papers would mislead.
    paper.loc[paper.reason == "n_missing_slots", "share_pct"] = np.nan
    paper.insert(1, "unit", "paper")
    paper["gender"] = ""
    return pd.concat([pairs, paper[pairs.columns]], ignore_index=True)


def load_results(out_dir) -> Dict[str, pd.DataFrame]:
    out = Path(out_dir)
    return {name: pd.read_parquet(out / f"gender_field_{name}.parquet")
            for name in ("counts", "papers", "totals", "coverage")}


# =============================================================================
# CLI
# =============================================================================
def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def inputs(sp):
        sp.add_argument("--files", nargs="+")
        sp.add_argument("--file-list")
        sp.add_argument("--glob", nargs="+")
        sp.add_argument("--shard", type=int, default=0)
        sp.add_argument("--num-shards", type=int, default=1)

    def common(sp):
        sp.add_argument("--out", required=True, help="output directory (never the source)")
        sp.add_argument("--buckets", type=int, default=N_BUCKETS)

    def window(sp):
        sp.add_argument("--first-year", type=int, default=FIRST_YEAR)
        sp.add_argument("--last-year", type=int, default=LAST_YEAR)

    def reading(sp):
        sp.add_argument("--batch-size", type=int, default=50_000)
        sp.add_argument("--progress", type=int, default=10)

    sp = sub.add_parser("schema", help="footer survey + sampled rows; run first")
    inputs(sp); sp.add_argument("--out", required=True)
    sp.add_argument("--for-vocab", help="for_2020_codes.txt from make-inputs")
    sp.add_argument("--sample-rows", type=int, default=200)

    sp = sub.add_parser("schema-summary", help="combine sharded schema_files*.csv")
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("names", help="corpus read 1: name keys and researcher pairs")
    inputs(sp); common(sp); window(sp); reading(sp)
    sp.add_argument("--compact-rows", type=int, default=5_000_000)

    for name in ("classify", "identity"):
        sp = sub.add_parser(name)
        common(sp)
        sp.add_argument("--bucket", type=int, help="one bucket (Slurm array index); default all")
        sp.add_argument("--num-shards", type=int, default=1, help="shards of the names pass")
        sp.add_argument("--allow-failed", action="store_true")

    sp = sub.add_parser("count", help="corpus read 2: one arm")
    inputs(sp); common(sp); window(sp); reading(sp)
    sp.add_argument("--ids", help="UK Biobank id list (txt/csv/parquet)")
    sp.add_argument("--id-column", default=ID_COL)
    sp.add_argument("--id-filter", choices=["include", "exclude", "none"], default="none")
    sp.add_argument("--label", default="all")
    sp.add_argument("--codes", help="ukbb_for_codes.csv; omit to count every FoR code")

    sp = sub.add_parser("merge")
    sp.add_argument("--out", required=True)
    sp.add_argument("--num-shards", type=int, default=1)
    sp.add_argument("--arms", default="background,ukbb")
    sp.add_argument("--allow-failed", action="store_true")

    sp = sub.add_parser("status")
    common(sp)
    sp.add_argument("--num-shards", type=int, default=1)
    sp.add_argument("--arms", default="background,ukbb")
    sp.add_argument("--missing-only", help="also print the missing indices of this unit "
                                           "on one line, e.g. names or count.ukbb")

    sp = sub.add_parser("make-inputs", help="LOCAL: ids, UK Biobank FoR codes, 2020 vocabulary")
    sp.add_argument("--files", nargs="+", help="showcase parquet (default P.SHOWCASE_PLUS)")
    sp.add_argument("--out", required=True)
    window(sp)

    args = p.parse_args(argv)
    {"schema": cmd_schema, "schema-summary": lambda a: summarise_schema(Path(a.out)),
     "names": cmd_names, "classify": cmd_classify, "identity": cmd_identity,
     "count": cmd_count, "merge": cmd_merge, "status": cmd_status,
     "make-inputs": cmd_make_inputs}[args.cmd](args)


if __name__ == "__main__":
    main()
