#!/usr/bin/env python3
"""
Whole-database FOR counts from the Dimensions Analytics API
===========================================================

WHAT THIS IS FOR
----------------
The background arm of the academic-impact analysis — "how many papers did the whole
world publish in this field this year" — has until now come from a 230 GB copy of the
Dimensions corpus on the BMRC VM, counted by the companion script
`data_analysis_03_academic_impact_field_counts.py`. The API can answer the same
question with about twenty queries, because Dimensions will facet a search by
`category_for_2020` and hand back the counts already aggregated.

    search publications where year=2014 return category_for_2020 limit 1000

One query per year gives the whole year x field matrix. Twenty-one years is twenty-one
queries, under a minute inside the API's 30-requests-per-minute limit, against the VM
job's array of Slurm tasks.

BOTH PATHWAYS ARE KEPT, DELIBERATELY
------------------------------------
This does not replace the VM job, and the two are not interchangeable:

    the VM       reads every row, so it can count anything a row contains —
                 FRACTIONAL counts, per-paper citation weights, the top-decile
                 indicator, per-(year, type, field) reference distributions. It is
                 also a FIXED SNAPSHOT, which is what makes the two arms comparable.
    the API      returns aggregates only. It cannot give a fractional count (that
                 needs to know how many categories each paper carried), and it is
                 LIVE — today's index, not the snapshot the UK Biobank arm came from.

So the API path is the fast way to a whole-database denominator, and the VM path is
the only way to the fractional and citation-weighted columns. `counts` below writes
its output in exactly the schema the VM job writes, so the notebooks can read either.

THE SNAPSHOT SEAM (read before trusting a number)
-------------------------------------------------
The background arm is built here as (API whole-database count) - (our UK Biobank
count). The subtrahend comes from the VM snapshot and the minuend from the live index,
so the two disagree by however much Dimensions has indexed since the snapshot was
taken. For older years that is nothing; for the most recent year it can be percent.
The subtraction reports every cell it had to clamp at zero — a cell going negative
means the UK Biobank arm is claiming papers the live index no longer has, which is a
sign the seam matters for that year, not a rounding error.

FRACTIONAL COUNTS ARE NOT AVAILABLE FOR THE BACKGROUND ARM
----------------------------------------------------------
`n_frac` is NaN on the background side, because splitting a paper's weight across its
categories needs the per-paper category list and a facet only returns totals. The UK
Biobank arm has it (it is fetched per paper), but a measure only one arm carries is
dropped by the analysis module, which then falls back to whole counting for the activity
index and says so. Citation weights are a different story and this docstring used to get
it wrong: `ukbb` fetches `times_cited`, `field_citation_ratio` and `recent_citations` per
paper, `whole` brings back `citations_avg` and `citations_median` per cell, and
`percentiles` measures the real per-field cut-offs by counting. What the API cannot give is a
per-paper FCR for the background arm — only `fcr_gavg`, a geometric mean that cannot be
summed into a comparable column — so `fcr` is the one weight that ends up one-armed.

MODES
-----
    check      authenticate, run one trivial query, print what came back
    codes      refresh the FOR code list (id <-> code <-> name) from the API
    counts     the year x field matrix -> partials in the VM schema
    whole      whole-database facets + the reference means and medians
    ukbb        UK Biobank's arm, fetched per paper and scored
    percentiles the MEASURED per-field cut-offs — top decile AND median in one pass,
                binary-searched by counting (`deciles` is kept as an alias)
    background  whole - ukbb, cell by cell
    all         the four above in the order they have to run
    calibrate  LEGACY: K for the top10p proxy, superseded by `deciles`
    selftest   exercise the table building on synthetic data; no network, no key

THE ORDER IS NOT NEGOTIABLE
---------------------------
    whole -> ukbb -> percentiles -> ukbb -> background

`whole` first, because its `citations_avg` is the reference every per-paper score is
normalised against. `percentiles` cannot run before `ukbb`, because it reads the UK Biobank
cells to decide which thresholds are worth paying for — and `ukbb` then has to run AGAIN
to score each paper against the cut-offs it just measured. The records are cached, so the
second `ukbb` costs nothing. `all` does all five steps in that order.

Author: Jiani Y
Date: 2026-08-16
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# The FOR code list shipped with the repo: level, id, code, name, publication_count.
# The API facets by entity id ("80003"), the corpus and the analysis key on the numeric
# code ("32"), so this file is the bridge and is loaded for every counts run.
DEFAULT_CODES_CSV = Path("doc/category_for_2020_codes.csv")
DEFAULT_CONFIG = Path("config/dsl.ini")

CATEGORY_FIELDS = {"for": "category_for_2020", "rcdc": "category_rcdc"}

# What one publication record carries. Both classification systems are pulled in the
# same fetch, so a single pass over the UK Biobank ids serves both notebooks.
RECORD_FIELDS = ("id", "year", "type", "times_cited", "field_citation_ratio",
                 "recent_citations", "category_for_2020", "category_rcdc")

# `where id in [...]` accepts 512 entries; 1000 is rejected outright.
ID_BATCH = 512

# The citation aggregates the facet carries. citations_avg is not decoration: it is the
# reference mean that turns a paper's citation count into a field-normalised score, and
# it comes from the same index as the papers being scored.
DEFAULT_AGGREGATES = ("citations_total", "citations_avg", "citations_median", "fcr_gavg")

# 30 requests per IP per minute is the documented ceiling, so 2.1 s between queries
# keeps a single process comfortably under it even with no jitter.
DEFAULT_SLEEP = 2.1


# =============================================================================
# CREDENTIALS
# =============================================================================
def load_config(path: Path, instance: str = "live") -> Dict[str, str]:
    """Read url + key (or login/password) from a dimcli-style dsl.ini.

    The file holds a secret, so it is never echoed back: errors name the FILE and the
    KEY THAT IS MISSING, never the value of anything.
    """
    env_key = os.environ.get("DIMENSIONS_API_KEY")
    if env_key:
        return {"url": os.environ.get("DIMENSIONS_API_URL", "https://app.dimensions.ai"),
                "key": env_key}
    if not path.exists():
        sys.exit(f"no {path} — copy config/dsl.ini.example to it and add your key "
                 f"(or set DIMENSIONS_API_KEY in the environment)")
    parser = configparser.ConfigParser()
    parser.read(path)
    section = f"instance.{instance}"
    if section not in parser:
        sys.exit(f"{path} has no [{section}] section (found: {parser.sections()})")
    conf = dict(parser[section])
    if not conf.get("url"):
        conf["url"] = "https://app.dimensions.ai"
    if not (conf.get("key") or (conf.get("login") and conf.get("password"))):
        sys.exit(f"[{section}] in {path} needs either 'key' or 'login'+'password'")
    return conf


# =============================================================================
# THE CLIENT
# =============================================================================
class Dimensions:
    """A thin DSL client: authenticate, run a query, retry politely.

    dimcli does all this and more, but it keeps a global logged-in session and prints
    as it goes; a job that has to be reproducible from a config path and quiet in a log
    is better served by the twenty lines it actually needs.
    """

    def __init__(self, conf: Dict[str, str], sleep: float = DEFAULT_SLEEP,
                 max_retries: int = 5, verbose: bool = True) -> None:
        import requests                      # imported here so selftest needs no network stack
        self._requests = requests
        self.url = conf["url"].rstrip("/")
        self._auth_payload = ({"key": conf["key"]} if conf.get("key")
                              else {"username": conf["login"], "password": conf["password"]})
        self.sleep = sleep
        self.max_retries = max_retries
        self.verbose = verbose
        self.token: Optional[str] = None
        self.n_queries = 0
        self._last_call = 0.0

    def login(self) -> None:
        r = self._requests.post(f"{self.url}/api/auth", json=self._auth_payload,
                                timeout=60)
        if r.status_code != 200:
            sys.exit(f"authentication failed ({r.status_code}) against {self.url}/api/auth "
                     f"— check the key in your dsl.ini")
        self.token = r.json()["token"]
        if self.verbose:
            print(f"authenticated against {self.url} (token valid ~2h)")

    def query(self, dsl: str) -> dict:
        """Run one DSL query, honouring the rate limit and retrying on 429/5xx."""
        if self.token is None:
            self.login()
        # Space the calls out rather than firing and apologising: the documented limit
        # is 30/minute/IP and a 429 costs more than the wait would have.
        wait = self.sleep - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        backoff = 5.0
        for attempt in range(1, self.max_retries + 1):
            self._last_call = time.time()
            try:
                r = self._requests.post(
                    f"{self.url}/api/dsl/v2", data=dsl.encode("utf-8"),
                    headers={"Authorization": f"JWT {self.token}",
                             "Content-Type": "application/json"}, timeout=300)
            except self._requests.exceptions.RequestException as exc:
                # A thousand-query run will meet a dropped connection sooner or later;
                # that is a reason to wait and retry, not to lose the run.
                if self.verbose:
                    print(f"  {type(exc).__name__}, waiting {backoff:.0f}s "
                          f"(attempt {attempt}/{self.max_retries})", flush=True)
                time.sleep(backoff)
                backoff *= 2
                continue
            self.n_queries += 1
            if r.status_code == 200:
                return r.json()
            if r.status_code == 401:          # the 2-hour token expired mid-run
                if self.verbose:
                    print("  token expired, re-authenticating", flush=True)
                self.login()
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = float(r.headers.get("Retry-After", backoff))
                if self.verbose:
                    print(f"  {r.status_code} from the API, waiting {retry_after:.0f}s "
                          f"(attempt {attempt}/{self.max_retries})", flush=True)
                time.sleep(retry_after)
                backoff *= 2
                continue
            # Anything else is a bad query, and retrying a bad query is just rude.
            sys.exit(f"query failed ({r.status_code}): {r.text[:500]}\n  DSL: {dsl}")
        sys.exit(f"gave up after {self.max_retries} attempts\n  DSL: {dsl}")


# =============================================================================
# THE FOR CODE LIST
# =============================================================================
def load_codes(path: Path):
    """id -> (level, code, name) from the shipped CSV.

    `level` is normalised to the L2/L4 spelling the rest of the analysis uses, so the
    output of this script keys identically to the VM job's.
    """
    import pandas as pd
    if not path.exists():
        sys.exit(f"no {path} — the FOR code list ships in doc/; pass --codes to point "
                 f"somewhere else, or run the `codes` mode to rebuild it from the API")
    df = pd.read_csv(path, dtype={"id": str, "code": str})
    df["level"] = df.level.map(lambda s: "L2" if str(s).startswith("L2") else "L4")
    return df.set_index("id")[["level", "code", "name"]]


def cmd_codes(args: argparse.Namespace) -> None:
    """Rebuild the id/code/name list from the API, so it cannot silently go stale."""
    import pandas as pd
    api = Dimensions(load_config(Path(args.config), args.instance), sleep=args.sleep)
    field = CATEGORY_FIELDS[args.category]
    res = api.query(f"search publications return {field} limit 1000")
    rows = res.get(field, [])
    if not rows:
        sys.exit(f"the API returned no {field} facet rows")
    df = pd.DataFrame(rows)
    # Dimensions names FOR entities either "32 Biomedical..." or "Biomedical..."
    # depending on the release; split a leading numeric code off when it is there so
    # the CSV always carries code and name separately.
    def split(name):
        head, _, tail = str(name).partition(" ")
        return (head, tail) if head.isdigit() else ("", str(name))
    df["code"], df["name_only"] = zip(*df.name.map(split))
    df["level"] = df.code.map(lambda c: "L2" if len(c) == 2 else ("L4" if len(c) == 4 else ""))
    out = df.rename(columns={"count": "publication_count"})[
        ["level", "id", "code", "name_only", "publication_count"]]
    out = out.rename(columns={"name_only": "name"}).sort_values(["level", "code"])
    dest = Path(args.out)
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  ({len(out):,} categories, {int(out.publication_count.sum()):,} "
          f"publication-category pairs)")
    if (out.code == "").any():
        print(f"  !! {(out.code == '').sum()} row(s) had no numeric code in the name — "
              f"the analysis keys on the code, so check those before using this file")


# =============================================================================
# PER-PAPER RECORDS — the UK Biobank arm
# =============================================================================
# The whole database has to be faceted, because nobody is fetching 155 million records
# through an API. UK Biobank's own papers are 26,109 of them, which is 51 queries at
# 512 ids apiece, and that changes what this pathway can do: with the records in hand
# the arm is built exactly as the VM job builds it — fractional counts, per-paper
# citation weights, both levels of the hierarchy — from the SAME index the denominator
# came from. That is the whole reason to do it this way rather than subtracting a
# snapshot from a live total.
def fetch_records(api: "Dimensions", ids: Sequence[str], batch: int = ID_BATCH,
                  verbose: bool = True) -> List[dict]:
    """Every publication in `ids`, with the fields RECORD_FIELDS names."""
    fields = "+".join(RECORD_FIELDS)
    out: List[dict] = []
    n_batches = math.ceil(len(ids) / batch)
    for i in range(n_batches):
        chunk = ids[i * batch:(i + 1) * batch]
        lst = ",".join(f'"{x}"' for x in chunk)
        res = api.query(f"search publications where id in [{lst}] "
                        f"return publications[{fields}] limit {len(chunk)}")
        got = res.get("publications", [])
        out.extend(got)
        if verbose and (i + 1) % 10 == 0 or i + 1 == n_batches:
            print(f"  {i + 1}/{n_batches} batches, {len(out):,} records", flush=True)
        if len(got) < len(chunk) and verbose:
            # Ids retract, merge or move between Dimensions releases; say how many
            # rather than let the arm quietly shrink.
            print(f"    batch {i + 1}: asked {len(chunk)}, got {len(got)}")
    return out


def records_to_counts(records: Iterable[dict], codes, category: str, arm: str,
                      year_min: int, year_max: int, expected=None, calibration=None,
                      field_thresholds=None, field_medians=None):
    """Per-paper records -> the counts and totals tables the VM job writes.

    This is the VM job's COUNTING SEMANTICS reimplemented on records rather than on
    parquet rows: codes de-duplicated within a paper within each level, fractional
    weights normalised per level, a weight's `docs` counter incremented only where the
    paper actually carried a value. If the two ever disagree, they disagree here.
    """
    import numpy as np
    import pandas as pd

    field = CATEGORY_FIELDS[category]
    # (year, level, code) -> measures
    cells: Dict[tuple, Dict[str, float]] = {}
    totals: Dict[int, Dict[str, float]] = {}
    weights = ["cit", "fcr", "recent", "mncs"]
    if calibration:
        weights.append("top10p")      # the proxy, only when a calibration exists
    if field_thresholds:
        weights.append("top10f")      # the measured decile, where a cell was measured
    if field_medians:
        weights.append("top50f")      # the field-year MEDIAN: the other half of the cut
    src = {"cit": "times_cited", "fcr": "field_citation_ratio",
           "recent": "recent_citations"}

    def blank():
        d = {"n_papers": 0.0, "n_frac": 0.0}
        for w in weights:
            d.update({f"n_{w}": 0.0, f"n_{w}_frac": 0.0,
                      f"n_{w}_docs": 0.0, f"n_{w}_docs_frac": 0.0})
        return d

    n_kept = n_out_of_range = 0
    for rec in records:
        try:
            year = int(rec.get("year"))
        except (TypeError, ValueError):
            continue
        if not year_min <= year <= year_max:
            n_out_of_range += 1
            continue
        n_kept += 1
        t = totals.setdefault(year, {"n_papers": 0.0, "n_no_cat": 0.0,
                                     **{f"n_{w}_total": 0.0 for w in weights},
                                     **{f"n_{w}_docs_total": 0.0 for w in weights}})
        t["n_papers"] += 1
        raw = {w: rec.get(src[w]) for w in weights if w in src}

        # levels: the same code list split the way the analysis reads it
        by_level: Dict[str, set] = {}
        for entry in rec.get(field) or []:
            cid = str(entry.get("id", ""))
            if cid not in codes.index:
                continue
            meta = codes.loc[cid]
            by_level.setdefault(meta.level, set()).add((meta.code, meta["name"]))
        if not by_level:
            t["n_no_cat"] += 1
            continue

        typ = rec.get("type") or ""
        k = (calibration.get((year, typ)) or calibration.get((year, "*"))
             if calibration else None)

        for w in weights:
            if w in ("mncs", "top10p"):
                continue                      # per-category, handled in the loop below
            v = raw.get(w)
            if v is not None and v == v:
                t[f"n_{w}_total"] += float(v)
                t[f"n_{w}_docs_total"] += 1.0
        if raw.get("cit") is not None:
            t["n_mncs_docs_total"] += 1.0
            t["n_mncs_total"] = float("nan")  # per-category: no paper-level total
            if calibration and k:
                t["n_top10p_docs_total"] += 1.0
                t["n_top10p_total"] = float("nan")
            if field_thresholds:
                t["n_top10f_docs_total"] += 1.0
                t["n_top10f_total"] = float("nan")
            if field_medians:
                t["n_top50f_docs_total"] += 1.0
                t["n_top50f_total"] = float("nan")

        for level, items in by_level.items():
            wgt = 1.0 / len(items)
            for code, name in items:
                cell = cells.setdefault((year, level, code, name), blank())
                cell["n_papers"] += 1.0
                cell["n_frac"] += wgt
                cit = raw.get("cit")
                ref = (expected.get((year, level, code)) if expected else None)
                mncs = (float(cit) / ref) if (cit is not None and ref) else None
                for w in weights:
                    if w == "mncs":
                        # times_cited over the mean of this (year, field): the same
                        # field-normalised score the VM job builds, with the reference
                        # taken from the API's own facet rather than a corpus pass.
                        v = mncs
                    elif w == "top10f":
                        # Measured, not assumed: this cell's own decile cut-off, found
                        # by counting. Cells never measured stay unmeasured (None), so
                        # they sit out of the mean rather than counting as "not top".
                        thr = (field_thresholds or {}).get((year, level, code))
                        v = ((1.0 if float(cit) >= thr[0] else 0.0)
                             if (thr and cit is not None) else None)
                    elif w == "top50f":
                        # Above the MEDIAN of the same (year, field): the top decile's
                        # coarser sibling, and the one that says how much of UK Biobank's
                        # output is in the better-cited half rather than the best tenth.
                        # Measured by `percentiles` where it could be (an integer cut-off
                        # counted against the live index), and only otherwise the facet's
                        # interpolated citations_median — see load_field_medians for what
                        # the difference costs.
                        med = (field_medians or {}).get((year, level, code))
                        mthr = med[0] if isinstance(med, tuple) else med
                        v = ((1.0 if float(cit) >= mthr else 0.0)
                             if (mthr is not None and mthr == mthr and cit is not None)
                             else None)
                    elif w == "top10p":
                        # the proxy: top-decile IF the distribution's shape is the same
                        # in this field as in the corpus at large. See cmd_calibrate.
                        v = (1.0 if mncs >= k else 0.0) if (mncs is not None and k) else None
                    else:
                        v = raw.get(w)
                        v = float(v) if (v is not None and v == v) else None
                    if v is None:
                        continue
                    cell[f"n_{w}"] += v
                    cell[f"n_{w}_frac"] += v * wgt
                    cell[f"n_{w}_docs"] += 1.0
                    cell[f"n_{w}_docs_frac"] += wgt

    rows = []
    for (year, level, code, name), m in sorted(cells.items()):
        rows.append({"arm": arm, "category": category, "year": year, "type": "all",
                     "level": level, "code": code, "for_label": name,
                     "parent_l2": code[:2] if level == "L4" else "", **m})
    counts = pd.DataFrame(rows)

    trows = []
    for year, m in sorted(totals.items()):
        trows.append({"arm": arm, "category": category, "year": year, "type": "all",
                      "n_filtered_out": 0, **m})
    tot = pd.DataFrame(trows)
    print(f"  {n_kept:,} records in {year_min}-{year_max} "
          f"({n_out_of_range:,} outside the window), {len(counts):,} cells")
    return counts, tot


# =============================================================================
# MODE: counts
# =============================================================================
def facet_counts(api: Dimensions, field: str, year: int, aggregate: Sequence[str],
                 extra_where: str = "") -> List[dict]:
    """One year's facet over the classification. `aggregate` adds citation metrics."""
    agg = f" aggregate {','.join(aggregate)}" if aggregate else ""
    where = f"where year={year}" + (f" and {extra_where}" if extra_where else "")
    dsl = f"search publications {where} return {field}{agg} limit 1000"
    return api.query(dsl).get(field, [])


def year_totals(api: Dimensions, years: Sequence[int], extra_where: str = "") -> Dict[int, int]:
    """Publications per year — one query for the whole range, the denominator behind
    every share the notebooks draw."""
    where = f"where year>={min(years)} and year<={max(years)}"
    if extra_where:
        where += f" and {extra_where}"
    rows = api.query(f"search publications {where} return year limit 1000").get("year", [])
    return {int(r["id"]): int(r["count"]) for r in rows}


def build_counts_table(per_year: Dict[int, List[dict]], codes, category: str,
                       arm: str, aggregate: Sequence[str] = ()):
    """Facet rows -> the counts schema the VM job writes.

    Kept free of network and of pandas-side effects so `selftest` can exercise it: this
    is where a silent mismatch with the VM schema would do the damage, not in the HTTP.
    """
    import numpy as np
    import pandas as pd

    rows = []
    unknown = set()
    for year, facet in sorted(per_year.items()):
        for r in facet:
            fid = str(r.get("id", ""))
            if fid not in codes.index:
                unknown.add(fid)
                continue
            meta = codes.loc[fid]
            row = {
                "arm": arm,
                "category": category,
                "year": int(year),
                "type": "all",          # the facet aggregates over publication types
                "level": meta.level,
                "code": meta.code,
                "for_label": meta["name"],
                "parent_l2": meta.code[:2] if meta.level == "L4" else "",
                "n_papers": float(r.get("count", 0)),
                # Not available from an aggregate endpoint — see the module docstring.
                "n_frac": float("nan"),
            }
            for a in aggregate:
                row[f"api_{a}"] = r.get(a, float("nan"))
            rows.append(row)
    df = pd.DataFrame(rows)
    if unknown:
        print(f"  !! {len(unknown)} category id(s) the code list does not know "
              f"({sorted(unknown)[:5]}...) — refresh doc/category_for_2020_codes.csv "
              f"with the `codes` mode")
    return df


def load_ukbb_counts(path: Path, category: str):
    """The UK Biobank arm we subtract, summed to (year, level, code).

    Read from whatever the VM job left behind, using the same filename patterns the
    analysis module uses, so 'the UK Biobank arm' means one thing in both places.
    """
    import pandas as pd
    patterns = [f"counts.{category}.ukbb*.parquet"]
    if category == "for":
        patterns.append("counts.ukbb*.parquet")     # pre-`--category` filenames
    files: List[Path] = []
    for pat in patterns:
        files = sorted(path.glob(pat))
        if files:
            break
    if not files:
        sys.exit(f"no UK Biobank partials in {path} (looked for {patterns}) — point "
                 f"--ukbb-counts at the directory the VM job wrote")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  UK Biobank arm: {len(files)} partial(s), {int(df.n_papers.sum()):,} "
          f"paper-category pairs")
    return df, files


def cmd_counts(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    years = list(range(args.year_min, args.year_max + 1))
    field = CATEGORY_FIELDS[args.category]
    aggregate = [a.strip() for a in args.aggregate.split(",") if a.strip()]
    codes = load_codes(Path(args.codes))
    out = Path(args.out)

    if args.dry_run:
        print("dry run — the queries that would be sent:\n")
        agg = f" aggregate {','.join(aggregate)}" if aggregate else ""
        print(f"  search publications where year>={years[0]} and year<={years[-1]} "
              f"return year limit 1000")
        for y in years[:3]:
            print(f"  search publications where year={y} return {field}{agg} limit 1000")
        print(f"  ... and {len(years) - 3} more, one per year "
              f"({len(years) + 1} queries in total, ~{(len(years) + 1) * args.sleep:.0f}s "
              f"at {args.sleep}s apart)")
        return

    out.mkdir(parents=True, exist_ok=True)
    api = Dimensions(load_config(Path(args.config), args.instance), sleep=args.sleep)
    t0 = time.time()

    totals = year_totals(api, years)
    print(f"whole-database publications {years[0]}-{years[-1]}: "
          f"{sum(totals.values()):,}")

    per_year = {}
    for i, y in enumerate(years, 1):
        per_year[y] = facet_counts(api, field, y, aggregate)
        print(f"  {y}: {len(per_year[y]):,} categories, "
              f"{sum(int(r['count']) for r in per_year[y]):,} paper-category pairs "
              f"({i}/{len(years)})", flush=True)

    whole = build_counts_table(per_year, codes, args.category, "whole", aggregate)
    whole_path = out / f"api_whole.{args.category}.parquet"
    whole.to_parquet(whole_path, index=False)
    print(f"\nwrote {whole_path}  ({len(whole):,} rows) — the API's answer, untouched")

    # -- the background arm = whole - UK Biobank ------------------------------------
    ukbb, ukbb_files = load_ukbb_counts(Path(args.ukbb_counts), args.category)
    keys = ["year", "level", "code"]
    u = (ukbb[ukbb.year.between(args.year_min, args.year_max)]
         .groupby(keys, as_index=False).n_papers.sum()
         .rename(columns={"n_papers": "ukbb_papers"}))
    bg = whole.merge(u, on=keys, how="left")
    bg["ukbb_papers"] = bg.ukbb_papers.fillna(0.0)
    bg["n_papers"] = bg.n_papers - bg.ukbb_papers

    # A negative cell means the UK Biobank snapshot holds papers the live index does
    # not report in that field-year: the seam between the two sources, not noise.
    neg = bg[bg.n_papers < 0]
    if len(neg):
        worst = neg.nsmallest(5, "n_papers")
        print(f"\n  !! {len(neg)} cell(s) went negative when the UK Biobank arm was "
              f"subtracted — the live index and the VM snapshot disagree there.\n"
              f"     Clamped to zero. Worst:")
        for _, r in worst.iterrows():
            print(f"       {r.year} {r.level} {str(r.code):5s} {str(r.for_label)[:34]:34s} "
                  f"api {r.n_papers + r.ukbb_papers:>12,.0f} - ukbb {r.ukbb_papers:>7,.0f}"
                  f" = {r.n_papers:>8,.0f}")
        bg["n_papers"] = bg.n_papers.clip(lower=0)
    bg["arm"] = "background"
    bg = bg.drop(columns=["ukbb_papers"])

    bg_path = out / f"counts.{args.category}.background.parquet"
    bg.to_parquet(bg_path, index=False)
    print(f"wrote {bg_path}  ({len(bg):,} rows)")

    # The UK Biobank partials are copied in so this directory is a complete, self-
    # contained input for the notebooks — point COUNTS_DIR here and nothing else moves.
    import shutil
    for f in ukbb_files:
        shutil.copy2(f, out / f.name)
    print(f"copied {len(ukbb_files)} UK Biobank partial(s) into {out}")

    pd.DataFrame({"year": list(totals), "n_publications": list(totals.values())}
                 ).sort_values("year").to_csv(out / f"api_year_totals.csv", index=False)
    print(f"\n{api.n_queries} queries in {time.time() - t0:.0f}s")
    print(f"point the notebook's COUNTS_DIR at {out} to use this as the background arm")


# =============================================================================
# MODE: calibrate  — the top-decile PROXY, and why it is only a proxy
# =============================================================================
# A percentile needs a distribution and a facet returns totals, so the API cannot give a
# real top-decile indicator. What it can give is the ratio
#
#     K(year, type) = (the top-decile citation threshold) / (the mean citation count)
#
# for the corpus as a whole, one binary search per cell. A paper is then called
# top-decile when its citations reach K times its FIELD's mean — which is exactly
# `mncs >= K`, since mncs is citations over that same field mean.
#
# THE ASSUMPTION, STATED PLAINLY: that the SHAPE of the citation distribution is the
# same in every field of a given year and type, so one K serves all of them. Measured
# against the live index for 2019 articles, K per field is
#
#     Clinical Sciences 2.38 · Pharmacology 2.40 · Epidemiology 2.42 ·
#     Biological Psychology 2.37 · Statistics 2.71 · a 717-paper cell 3.13
#
# — tight for large fields, drifting for small ones. And K itself drifts with year as
# counts thin out and go lumpy (2.47 in 2020, 2.92 in 2023 on the corpus).
#
# So the column is named n_top10p, never n_top10: aggregate rates from it are close,
# individual papers near the boundary are guesses, and anything quotable should use the
# VM pathway's measured n_top10f instead.
def cmd_calibrate(args: argparse.Namespace) -> None:
    import pandas as pd

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    api = Dimensions(load_config(Path(args.config), args.instance), sleep=args.sleep)
    years = list(range(args.year_min, args.year_max + 1))
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    pct = args.percentile

    def total(where: str) -> int:
        r = api.query(f"search publications where {where} return publications limit 1")
        return int(r.get("_stats", {}).get("total_count", 0))

    rows = []
    for typ in types:
        # one query gives every year's count and mean for this type
        facet = api.query(f'search publications where type="{typ}" and '
                          f"year>={years[0]} and year<={years[-1]} "
                          f"return year aggregate citations_avg limit 1000").get("year", [])
        stats = {int(r["id"]): (int(r["count"]), float(r.get("citations_avg") or 0))
                 for r in facet}
        for y in years:
            n, mean = stats.get(y, (0, 0.0))
            if n < args.min_papers or mean <= 0:
                continue
            target = n * pct / 100.0
            # The threshold is near 2.4x the mean, so bracket around that rather than
            # searching from one: it halves the queries per cell.
            lo, hi = 1, max(10, int(6 * mean))
            while lo < hi:
                mid = (lo + hi) // 2
                if total(f'type="{typ}" and year={y} and times_cited>={mid}') > target:
                    lo = mid + 1
                else:
                    hi = mid
            n_at = total(f'type="{typ}" and year={y} and times_cited>={lo}')
            rows.append({"year": y, "type": typ, "percentile": pct, "n_papers": n,
                         "mean_cit": round(mean, 4), "threshold": lo,
                         "k": round(lo / mean, 4), "n_at_or_above": n_at,
                         "achieved_pct": round(100 * n_at / n, 3)})
            print(f"  {typ:9s} {y}  n {n:>9,}  mean {mean:7.2f}  thr {lo:>4}  "
                  f"K {lo / mean:5.2f}  achieved {100 * n_at / n:5.2f}%", flush=True)

    df = pd.DataFrame(rows)
    dest = out / "top10_calibration.csv"
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest}  ({len(df)} cells, {api.n_queries} queries)")
    for typ, g in df.groupby("type"):
        print(f"  {typ:9s} K ranges {g.k.min():.2f}-{g.k.max():.2f} "
              f"(median {g.k.median():.2f})")


def load_calibration(out: Path) -> Dict[tuple, float]:
    """(year, type) -> K, with a per-year article fallback for the rarer types."""
    import pandas as pd
    path = out / "top10_calibration.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    k = {(int(r.year), r.type): float(r.k) for r in df.itertuples()}
    # 1.8% of UK Biobank's papers are chapters, proceedings and the like; scoring them
    # against the article ratio beats dropping them, and beats calibrating a cell of 30.
    for y, g in df[df.type == "article"].groupby("year"):
        k[(int(y), "*")] = float(g.k.iloc[0])
    return k


# =============================================================================
# MODE: percentiles  — the MEASURED per-field cut-offs, by counting
# =============================================================================
# There is no percentile aggregation in the DSL beyond citations_median, but a percentile
# does not need one: the count of papers at or above a citation count is a filter away, so
# the cut-off can be binary-searched exactly.
#
#     search publications where year=Y and category_for_2020.id="F"
#           and times_cited>=X return publications limit 1     ->  _stats.total_count
#
# That is the real PP(top 10%) — the same thing the VM job computes from its histograms —
# and it supersedes the top10p proxy wherever it is measured.
#
# WHY THE MEDIAN IS MEASURED HERE TOO, RATHER THAN TAKEN FROM THE FACET.
# `citations_median` does come back with every whole-database facet row, free, and this
# job used to cut the n_top50f band on it. It is an INTERPOLATED percentile over a
# distribution of small integers with enormous ties, so it comes back fractional — 1,873
# of 2,052 L4 cells in the last run — and `times_cited >= 2.02` is really `>= 3`, which
# throws away every paper sitting exactly ON the median. Worse, nothing measured what
# share of the field that cut actually kept, so the background arm had to IMPUTE it as
# half the papers. Measuring it removes both problems at once: an integer cut-off, and
# `n_at_or_above`, the field's real count above it — exactly what the decile already had.
#
# ONE PASS, MANY PERCENTILES. The per-cell queries — the cell's size, its mean, and every
# `times_cited>=k` probe — are shared across the percentiles being measured, and the
# percentiles are searched in ASCENDING order so each one's cut-off is a hard upper bound
# on the next one's bracket (the median cut can never exceed the decile cut). Measuring
# 10 and 50 together therefore costs about a third more than measuring 10 alone, not
# twice as much.
#
# WHICH CELLS. Every (year, field) UK Biobank publishes in is ~680 cells and five hours;
# --min-ukbb 10 measures the 254 of them that carry enough UK Biobank papers for a rate to
# mean anything, which is every cell the notebooks actually draw (they drop thinner ones
# themselves, at the same floor). Cells left unmeasured keep the proxy, and the analysis
# counts each paper only in the measure it actually has (that is what the _docs counters
# are for), so mixing them does not corrupt a mean — it narrows what the mean is over.
#
# ALREADY-MEASURED CELLS ARE SKIPPED. The destination is read first and a cell measured at
# every percentile asked for is not paid for again, so lowering --min-ukbb tops up rather
# than restarts, and a run killed halfway resumes from its last checkpoint. --remeasure
# overrides that for when the live index has moved far enough to matter.
#
# WHAT A CELL CAN FAIL TO SUPPORT. Citations are lumpy and a recent year is mostly zeros,
# so a percentile is often unreachable: the cut lands at >=1 citation and keeps 42% of the
# field, not 50%, because more than half of it is uncited. The row records what it
# achieved (`achieved_pct`) and what the next cut down would have kept (`prev_pct`), so
# the gap between them is the tie mass that could not be split. Nothing is rounded into
# looking exact.
#
# ACROSS TYPES, not per type. The VM job thresholds per (year, type, field); here the
# cell is (year, field) over all types. The field cell is dominated by articles, and
# splitting it would double the query cost while making the preprint sub-cells too thin
# to place a decile in at all (see the calibration in §4.3).
def parse_percentiles(raw) -> list:
    """"10,50" -> [10.0, 50.0], ascending, de-duplicated, validated."""
    if isinstance(raw, (int, float)):
        raw = str(raw)
    try:
        pcts = sorted({round(float(x), 3) for x in str(raw).split(",") if x.strip()})
    except ValueError:
        sys.exit(f"--percentiles wants a comma-separated list of numbers, got {raw!r}")
    if not pcts or any(not 0 < x < 100 for x in pcts):
        sys.exit(f"--percentiles must all be strictly between 0 and 100, got {pcts}")
    return pcts


def cmd_percentiles(args: argparse.Namespace) -> None:
    import pandas as pd

    out = Path(args.out)
    counts_path = out / f"counts.{args.category}.ukbb.parquet"
    if not counts_path.exists():
        sys.exit(f"no {counts_path} — run the `ukbb` mode first; its cells decide which "
                 f"thresholds are worth measuring")
    u = pd.read_parquet(counts_path)
    u = u[u.year.between(args.year_min, args.year_max) & (u.n_papers >= args.min_ukbb)]
    cells = sorted({(int(r.year), r.level, r.code) for r in u.itertuples()})
    pcts = parse_percentiles(getattr(args, "percentiles", None) or args.percentile)

    # RESUME. Every row in the destination cost real queries, so a cell already measured
    # at every percentile being asked for is skipped rather than paid for twice. That is
    # what makes a top-up cheap: dropping --min-ukbb from 20 to 10 measures the cells the
    # lower floor adds, not the ones the higher floor already covered. --remeasure forces
    # the whole set (the live index moves, so a cut-off does go stale eventually).
    dest_path = Path(args.out) / f"field_thresholds.{args.category}.csv"
    n_skipped = 0
    if dest_path.exists() and not getattr(args, "remeasure", False):
        done = pd.read_csv(dest_path, dtype={"code": str})
        if "percentile" not in done.columns:
            done["percentile"] = 10.0
        have = {}
        for r in done.itertuples():
            have.setdefault((int(r.year), r.level, str(r.code)), set()).add(
                round(float(r.percentile), 3))
        wanted = {round(x, 3) for x in pcts}
        keep = [c for c in cells if not wanted <= have.get(c, set())]
        n_skipped = len(cells) - len(keep)
        cells = keep
        if n_skipped:
            print(f"{n_skipped:,} cell(s) already measured at percentile(s) "
                  f"{', '.join(f'{x:g}' for x in pcts)} — skipping them "
                  f"(--remeasure to redo)")
    if not cells:
        print("nothing left to measure")
        return
    # One query for the cell (size + mean together), ~7 probes to bracket and bisect the
    # first percentile, ~4 for each one after it because the previous cut-off caps the
    # bracket and the probes already made are cached.
    per_cell = 1 + 8 + 4 * (len(pcts) - 1)
    print(f"{len(cells):,} (year, field) cell(s) to measure "
          f"(>= {args.min_ukbb} UK Biobank papers"
          + (f", {n_skipped:,} already done" if n_skipped else "") + ") "
          f"x percentile(s) {', '.join(f'{x:g}' for x in pcts)}\n"
          f"-> ~{len(cells) * per_cell:,} queries, "
          f"~{len(cells) * per_cell * args.sleep / 60:.0f} min")

    codes = load_codes(Path(args.codes))
    id_of = {(r.level, r.code): i for i, r in codes.iterrows()}
    field = CATEGORY_FIELDS[args.category]
    api = Dimensions(load_config(Path(args.config), args.instance), sleep=args.sleep)

    def total(year, fid, extra=""):
        r = api.query(f"search publications where year={year} and "
                      f'{field}.id="{fid}"{extra} return publications limit 1')
        return int(r.get("_stats", {}).get("total_count", 0))

    def cell_stats(year, fid):
        """The cell's size AND its mean citation count, in one query.

        The aggregate query carries `_stats.total_count` like any other search, so
        asking for the size separately would pay twice for the same filter.
        """
        r = api.query(f"search publications where year={year} and "
                      f'{field}.id="{fid}" return year aggregate citations_avg limit 5')
        row = next((x for x in r.get("year", []) if int(x["id"]) == year), None)
        n = int(r.get("_stats", {}).get("total_count", 0) or 0)
        if not n and row:
            n = int(row.get("count") or 0)
        return n, (float(row.get("citations_avg") or 0) if row else 0.0)

    dest = out / f"field_thresholds.{args.category}.csv"

    def merge_write(new_rows):
        """Fold `new_rows` into the destination CSV and return the whole table.

        MERGE, never overwrite. A second run is normally a top-up — a couple of early
        years, a lower --min-ukbb, another percentile — and writing only what this run
        measured would silently throw away every cell the last run paid for. Rows
        measured again win; the rest survive. The percentile is part of a row's identity,
        so the decile and the median of one cell do not overwrite each other.
        """
        df = pd.DataFrame(new_rows)
        if dest.exists():
            old_df = pd.read_csv(dest, dtype={"code": str})
            if "percentile" not in old_df.columns:
                old_df["percentile"] = 10.0   # a file from before this mode existed
            if len(df):
                keys = ["year", "level", "code", "percentile"]
                df = (pd.concat([old_df, df.astype({"code": str})], ignore_index=True)
                      .drop_duplicates(subset=keys, keep="last")
                      .sort_values(keys).reset_index(drop=True))
            else:
                df = old_df
        if len(df):
            df.to_csv(dest, index=False)
        return df

    rows, n_new, t0 = [], 0, time.time()
    for i, (year, level, code) in enumerate(cells, 1):
        fid = id_of.get((level, code))
        if fid is None:
            continue
        n, mean = cell_stats(year, fid)
        if n < args.min_papers:
            continue

        # Every probe this cell makes, kept: the percentiles walk overlapping ranges of k
        # and the bisections revisit the same values, so the cache is most of the saving
        # from measuring them together.
        seen: Dict[int, int] = {}

        def at(k, _year=year, _fid=fid, _n=n):
            """How many papers in this cell have at least k citations."""
            if k <= 0:
                return _n
            if k not in seen:
                seen[k] = total(_year, _fid, f" and times_cited>={k}")
            return seen[k]

        cap = None          # the cut-off of the previous (coarser) percentile
        for pct in pcts:
            target = n * pct / 100.0
            # Bracket. For the first percentile, from the cell's own mean: the tail
            # cut-off sits near a few times it, so start at 8x and widen if that is
            # still above the target. Afterwards the previous cut-off IS the bracket —
            # a cut that keeps 10% of the field cannot be below one that keeps 50%.
            if cap is not None:
                hi = cap
            else:
                hi = max(10, int(8 * mean) + 2)
                while at(hi) > target:
                    hi *= 2
            lo = 1
            while lo < hi:
                mid = (lo + hi) // 2
                if at(mid) > target:
                    lo = mid + 1
                else:
                    hi = mid
            n_at, n_prev = at(lo), at(lo - 1)
            cap = lo
            rows.append({"category": args.category, "year": year, "level": level,
                         "code": code, "n_papers": n, "mean_cit": round(mean, 4),
                         "percentile": pct, "threshold": lo, "n_at_or_above": n_at,
                         "achieved_pct": round(100 * n_at / n, 3),
                         # What the next cut down would have kept. The gap to
                         # achieved_pct is the tie mass at the cut-off — the reason a
                         # percentile is usually unreachable exactly, and the honest
                         # measure of how well this cell supports one at all.
                         "prev_pct": round(100 * n_prev / n, 3),
                         "k_vs_mean": round(lo / mean, 4) if mean else None})
        if i % 10 == 0 or i == len(cells):
            print(f"  {i}/{len(cells)} cells, {api.n_queries} queries, "
                  f"{time.time() - t0:.0f}s", flush=True)
        # Checkpoint. This pass is hours long and every row in it cost real queries, so
        # it is written down as it goes rather than held until the end, where one dropped
        # connection would take the whole run with it. A resumed run re-measures only the
        # cells the checkpoint never reached.
        if rows and i % max(int(getattr(args, "checkpoint", 25) or 25), 1) == 0:
            merge_write(rows)
            n_new += len(rows)
            rows = []

    n_new += len(rows)
    df = merge_write(rows)
    if df.empty:
        print("\nno cells measured — nothing to write")
        return
    print(f"\nwrote {dest}  ({n_new} row(s) measured this run, {len(df)} in the file, "
          f"{api.n_queries} queries, {time.time() - t0:.0f}s)")
    for pct, g in df.groupby("percentile"):
        print(f"\n  percentile {pct:g}  ({len(g):,} cell(s))")
        print(f"    cut-off:           {g.threshold.min():.0f}-{g.threshold.max():.0f} "
              f"citations (median {g.threshold.median():.0f})")
        print(f"    achieved:          {g.achieved_pct.min():.2f}-{g.achieved_pct.max():.2f}% "
              f"(median {g.achieved_pct.median():.2f}%)")
        if g.k_vs_mean.notna().any():
            print(f"    cut-off / mean:    {g.k_vs_mean.min():.2f}-{g.k_vs_mean.max():.2f} "
                  f"(median {g.k_vs_mean.median():.2f})")
        # A cell whose achieved share falls well short of the target could not be cut
        # there at all: the tie it had to split is bigger than the slice asked for,
        # which in a recent year means most of the field is uncited.
        short = g[g.achieved_pct < pct - 5]
        if len(short):
            print(f"    !! {len(short)} cell(s) fall >5 points short — the cut-off could "
                  f"not split the ties\n       (worst {short.achieved_pct.min():.1f}% at "
                  f"cut-off {int(short.loc[short.achieved_pct.idxmin(), 'threshold'])}; "
                  f"years {short.year.min()}-{short.year.max()}). The count above it is "
                  f"still exact —\n       what is approximate is calling it the "
                  f"{pct:g}th percentile.")
    print("\n  re-run `ukbb` (records are cached, so it costs nothing) then `background`")


def load_field_cuts(out: Path, category: str, percentile: float = 10.0):
    """(year, level, code) -> (cut-off, n_at_or_above, n_papers) for measured cells.

    `n_at_or_above` is the whole database's own count above that cut-off, which is what
    lets the background arm be measured rather than imputed from the percentile.
    """
    import pandas as pd
    path = out / f"field_thresholds.{category}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"code": str})
    if "percentile" in df.columns:
        df = df[df.percentile.astype(float).round(3) == round(float(percentile), 3)]
    elif round(float(percentile), 3) != 10.0:
        return {}       # a file from before the percentiles mode holds the decile only
    return {(int(r.year), r.level, str(r.code)):
            (float(r.threshold), float(r.n_at_or_above), float(r.n_papers))
            for r in df.itertuples()}


def load_field_thresholds(out: Path, category: str, percentile: float = 10.0):
    """The top-decile cut-offs — the name the rest of this file has always used."""
    return load_field_cuts(out, category, percentile)


# =============================================================================
# MODES: whole / ukbb / background  — the hybrid pathway
# =============================================================================
# Run in that order, or use `all`:
#
#   whole       21 facet queries -> the whole-database year x field matrix, carrying
#               citations_total and citations_avg. The averages become the reference
#               that field-normalises everything else, so this runs first.
#   ukbb        51 record queries -> UK Biobank's arm, built per paper against that
#               reference: fractional counts and citation weights included.
#   background  whole - ukbb, cell by cell. Both sides now come from the same index on
#               the same day, so this is arithmetic rather than an approximation.
def _write(df, path: Path, what: str) -> None:
    df.to_parquet(path, index=False)
    print(f"wrote {path}  ({len(df):,} {what})")


def cmd_whole(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    years = list(range(args.year_min, args.year_max + 1))
    field = CATEGORY_FIELDS[args.category]
    codes = load_codes(Path(args.codes))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    api = Dimensions(load_config(Path(args.config), args.instance), sleep=args.sleep)
    aggregate = list(DEFAULT_AGGREGATES)

    per_year = {}
    for i, y in enumerate(years, 1):
        per_year[y] = facet_counts(api, field, y, aggregate)
        print(f"  {y}: {len(per_year[y]):,} categories ({i}/{len(years)})", flush=True)
    whole = build_counts_table(per_year, codes, args.category, "whole", aggregate)

    # The facet's own averages, keyed the way the record builder needs them.
    whole["mean_cit"] = whole["api_citations_avg"]
    _write(whole[["arm", "category", "year", "type", "level", "code", "for_label",
                  "parent_l2", "n_papers", "mean_cit"]
                 + [f"api_{a}" for a in aggregate]],
           out / f"api_whole.{args.category}.parquet", "cells")

    # Year totals, aggregated on the year facet so a paper in three fields is counted
    # once — a sum over the field facet would count it three times.
    tot = api.query(f"search publications where year>={years[0]} and year<={years[-1]} "
                    f"return year aggregate citations_total limit 1000").get("year", [])
    ytot = pd.DataFrame([{"year": int(r["id"]), "n_papers": int(r["count"]),
                          "n_cit_total": float(r.get("citations_total") or 0)}
                         for r in tot]).sort_values("year")
    ytot.to_csv(out / f"api_year_totals.{args.category}.csv", index=False)
    print(f"wrote {out}/api_year_totals.{args.category}.csv "
          f"({int(ytot.n_papers.sum()):,} publications {years[0]}-{years[-1]})")
    print(f"\n{api.n_queries} queries")


def load_expected(out: Path, category: str) -> Dict[tuple, float]:
    """(year, level, code) -> mean citations, from the whole-database facet."""
    import pandas as pd
    path = out / f"api_whole.{category}.parquet"
    if not path.exists():
        sys.exit(f"no {path} — run the `whole` mode first; its citations_avg is the "
                 f"reference the per-paper score is normalised against")
    df = pd.read_parquet(path)
    return {(int(r.year), r.level, r.code): float(r.mean_cit)
            for r in df.itertuples() if r.mean_cit == r.mean_cit}


def load_field_medians(out: Path, category: str):
    """(year, level, code) -> (cut-off, n_at_or_above, n_papers) at the field-year median.

    TWO SOURCES, IN ORDER OF PREFERENCE.

    1. MEASURED, by `percentiles --percentiles 50`: an integer cut-off found by counting,
       plus the whole database's own count above it. Both arms then use the same rule and
       nothing is imputed.
    2. The facet's `citations_median`, which costs nothing but is an INTERPOLATED
       percentile over small integers with heavy ties — it comes back fractional (0.1,
       0.88, 2.02), so `>= median` silently means `>= ceil(median)` and no count of the
       field above it exists. Used only where nothing was measured, and the caller is
       told which it got.

    The tuple shape is shared with `load_field_cuts` so the two cut-offs are consumed
    identically; the fallback carries NaN where it has no measured count.
    """
    import pandas as pd
    measured = load_field_cuts(out, category, 50.0)
    if measured:
        return measured
    path = out / f"api_whole.{category}.parquet"
    if not path.exists() or "api_citations_median" not in pd.read_parquet(
            path, columns=None).columns:
        return {}
    df = pd.read_parquet(path)
    nan = float("nan")
    return {(int(r.year), r.level, str(r.code)): (float(r.api_citations_median), nan, nan)
            for r in df.itertuples()
            if r.api_citations_median == r.api_citations_median}


def cmd_ukbb(args: argparse.Namespace) -> None:
    import pandas as pd

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    codes = load_codes(Path(args.codes))
    ids = [l.strip() for l in open(args.ids) if l.strip()]
    print(f"{len(ids):,} UK Biobank ids from {args.ids}")

    cache = out / "api_ukbb_records.json"
    if cache.exists() and not args.refresh:
        records = json.loads(cache.read_text())
        print(f"reusing {cache} ({len(records):,} records; --refresh to re-fetch)")
    else:
        api = Dimensions(load_config(Path(args.config), args.instance), sleep=args.sleep)
        records = fetch_records(api, ids, args.batch)
        cache.write_text(json.dumps(records))
        print(f"cached {len(records):,} records in {cache} ({api.n_queries} queries)")
        missing = len(ids) - len({r.get("id") for r in records})
        if missing:
            print(f"  !! {missing:,} id(s) the live index did not return — retracted, "
                  f"merged or re-identified since the id list was made")

    expected = load_expected(out, args.category)
    calibration = load_calibration(out)
    field_thresholds = load_field_thresholds(out, args.category)
    if field_thresholds:
        print(f"  top-decile MEASURED in {len(field_thresholds)} (year, field) cells")
    field_medians = load_field_medians(out, args.category)
    if field_medians:
        _measured = any(v[1] == v[1] for v in field_medians.values()
                        if isinstance(v, tuple))
        print(f"  field-year MEDIAN available in {len(field_medians)} cells "
              + ("(MEASURED by `percentiles`, integer cut-off)" if _measured else
                 "(the facet's INTERPOLATED citations_median — run `percentiles "
                 "--percentiles 50`\n   for a measured cut-off and a counted "
                 "background arm)"))
    if calibration:
        print(f"  top-decile PROXY on: {len(calibration)} calibrated (year, type) cells")
    elif not field_thresholds:
        print("  no measured cut-offs and no calibration — run `deciles` (then `ukbb` "
              "again) for\n  the real per-field top decile. The legacy `calibrate` "
              "proxy is no longer read by the notebooks.")
    counts, totals = records_to_counts(records, codes, args.category, "ukbb",
                                       args.year_min, args.year_max, expected,
                                       calibration, field_thresholds, field_medians)
    _write(counts, out / f"counts.{args.category}.ukbb.parquet", "cells")
    _write(totals, out / f"totals.{args.category}.ukbb.parquet", "year rows")


def cmd_background(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    out = Path(args.out)
    whole = pd.read_parquet(out / f"api_whole.{args.category}.parquet")
    ukbb_path = out / f"counts.{args.category}.ukbb.parquet"
    if not ukbb_path.exists():
        sys.exit(f"no {ukbb_path} — run the `ukbb` mode first")
    ukbb = pd.read_parquet(ukbb_path)

    keys = ["year", "level", "code"]
    cols = ["n_papers", "n_cit", "n_cit_docs", "n_mncs", "n_mncs_docs"]
    cols += [c for c in ("n_top10p", "n_top10p_docs", "n_top10f", "n_top10f_docs",
                         "n_top50f", "n_top50f_docs") if c in ukbb.columns]
    u = ukbb.groupby(keys, as_index=False)[cols].sum()
    u = u.rename(columns={c: f"u_{c}" for c in u.columns if c not in keys})
    bg = whole.merge(u, on=keys, how="left").fillna(
        {c: 0.0 for c in u.columns if c not in keys})

    bg["n_cit"] = bg.api_citations_total - bg.u_n_cit
    bg["n_cit_docs"] = bg.n_papers - bg.u_n_cit_docs
    # By construction the whole database's mean score in a (year, field) is 1.0 — that
    # is what citations_avg IS — so its mncs sum is its paper count. Subtracting UK
    # Biobank's leaves the rest of the world's, and the ratio the notebook computes
    # from it reads directly as "UK Biobank's papers here are cited N times the field
    # average".
    bg["n_mncs"] = bg.n_papers - bg.u_n_mncs
    bg["n_mncs_docs"] = bg.n_papers - bg.u_n_mncs_docs
    # The whole database's top-decile RATE in a field is the percentile by definition —
    # that is what a decile is — so its proxy sum is 10% of its papers. Subtracting UK
    # Biobank's leaves the rest of the world's, and the ratio the notebook forms from it
    # reads as "N times as likely to be in the field's most-cited tenth".
    thr = load_field_thresholds(out, args.category)
    if thr and "u_n_top10f" in bg.columns:
        # n_at_or_above is the exact number of papers over the cut-off in that cell, so
        # the whole arm's figure is measured too — no 10% assumption anywhere here.
        key = list(zip(bg.year.astype(int), bg.level, bg.code.astype(str)))
        bg["w_top10f"] = [thr.get(k, (None, float("nan"), None))[1] for k in key]
        bg["n_top10f"] = bg.w_top10f - bg.u_n_top10f
        bg["n_top10f_docs"] = bg.n_papers.where(bg.w_top10f.notna()) - bg.u_n_top10f_docs
        bg = bg.drop(columns=["w_top10f"])
    if "u_n_top10p" in bg.columns:
        bg["n_top10p"] = bg.n_papers * (args.percentile / 100.0) - bg.u_n_top10p
        bg["n_top10p_docs"] = bg.n_papers - bg.u_n_top10p_docs
    # The median cut, measured the same way as the decile wherever `percentiles` ran at
    # 50: n_at_or_above is the field's own count above the cut-off, so this arm is counted
    # rather than assumed. Where it was not measured the old imputation stands — half a
    # field-year's papers are at or above its median, which is what a median IS — but the
    # facet's median is interpolated and its ties make the true share a few points off 50,
    # so the fallback is flagged rather than silent.
    med_cuts = load_field_cuts(out, args.category, 50.0)
    if med_cuts and "u_n_top50f" in bg.columns:
        key = list(zip(bg.year.astype(int), bg.level, bg.code.astype(str)))
        bg["w_top50f"] = [med_cuts.get(k, (None, float("nan"), None))[1] for k in key]
        bg["n_top50f"] = bg.w_top50f - bg.u_n_top50f
        bg["n_top50f_docs"] = bg.n_papers.where(bg.w_top50f.notna()) - bg.u_n_top50f_docs
        print(f"  median cut MEASURED in {int(bg.w_top50f.notna().sum()):,} cell(s) — the "
              f"above-median arm is counted, not imputed")
        bg = bg.drop(columns=["w_top50f"])
    elif "u_n_top50f" in bg.columns:
        print("  !! no measured median cut (run `percentiles --percentiles 50`) — the "
              "above-median\n     arm falls back to imputing half the field")
        _has_med = bg.api_citations_median.notna()
        bg["n_top50f"] = (bg.n_papers * 0.5).where(_has_med) - bg.u_n_top50f
        bg["n_top50f_docs"] = bg.n_papers.where(_has_med) - bg.u_n_top50f_docs
    bg["n_papers"] = bg.n_papers - bg.u_n_papers

    neg = bg[bg.n_papers < 0]
    if len(neg):
        print(f"  !! {len(neg)} cell(s) negative after subtraction — clamped. Both arms "
              f"come from the same index, so this should be empty; investigate if not.")
        for c in ("n_papers", "n_cit", "n_cit_docs", "n_mncs", "n_mncs_docs",
                  "n_top10p", "n_top10p_docs", "n_top10f", "n_top10f_docs",
                  "n_top50f", "n_top50f_docs"):
            if c in bg.columns:
                bg[c] = bg[c].clip(lower=0)
    bg["arm"] = "background"
    bg["n_frac"] = float("nan")     # aggregates cannot be split across categories
    out_cols = ["arm", "category", "year", "type", "level", "code", "for_label",
                "parent_l2", "n_papers", "n_frac", "n_cit", "n_cit_docs", "n_mncs",
                "n_mncs_docs", "api_fcr_gavg"]
    out_cols += [c for c in ("n_top10p", "n_top10p_docs", "n_top10f", "n_top10f_docs",
                             "n_top50f", "n_top50f_docs") if c in bg.columns]
    _write(bg[out_cols], out / f"counts.{args.category}.background.parquet", "cells")

    ytot = pd.read_csv(out / f"api_year_totals.{args.category}.csv")
    utot = pd.read_parquet(out / f"totals.{args.category}.ukbb.parquet")
    t = ytot.merge(utot[["year", "n_papers", "n_cit_total"]], on="year", how="left",
                   suffixes=("", "_ukbb")).fillna(0.0)
    t["n_papers"] = t.n_papers - t.n_papers_ukbb
    t["n_cit_total"] = t.n_cit_total - t.n_cit_total_ukbb
    t["n_cit_docs_total"] = t.n_papers
    t["n_mncs_total"], t["n_mncs_docs_total"] = t.n_papers, t.n_papers
    t["n_top10p_total"] = t.n_papers * (args.percentile / 100.0)
    t["n_top10p_docs_total"] = t.n_papers
    # Totals are needed by the notebook's citation-window guard. The field-level
    # measured counts live in the cell table; at year level the complete corpus
    # contributes the percentile denominator.
    t["n_top10f_total"] = t.n_papers * (args.percentile / 100.0)
    t["n_top10f_docs_total"] = t.n_papers
    t["n_top50f_total"] = t.n_papers * 0.5
    t["n_top50f_docs_total"] = t.n_papers
    t["arm"], t["category"], t["type"] = "background", args.category, "all"
    t["n_filtered_out"], t["n_no_cat"] = 0, 0
    _write(t[["arm", "category", "year", "type", "n_papers", "n_filtered_out",
              "n_no_cat", "n_cit_total", "n_cit_docs_total", "n_mncs_total",
              "n_mncs_docs_total", "n_top10p_total", "n_top10p_docs_total",
              "n_top10f_total", "n_top10f_docs_total",
              "n_top50f_total", "n_top50f_docs_total"]],
           out / f"totals.{args.category}.background.parquet", "year rows")
    print(f"\npoint the notebook at this directory (SOURCE = \"api\") — both arms now "
          f"come from one index on one day")


def cmd_all(args: argparse.Namespace) -> None:
    """whole -> [calibrate] -> ukbb -> percentiles -> ukbb -> background.

    The second `ukbb` is not a mistake and not a retry: `percentiles` measures the
    per-field cut-offs from the cells the first pass found, and only a second pass can
    score each paper against them. It re-reads the cached records, so it costs no queries.
    """
    cmd_whole(args)
    print()
    # The top10p proxy is legacy — `deciles` measures the real thing — so the calibration
    # is opt-in now rather than automatic. The analysis module no longer reads the column.
    if args.calibrate or args.recalibrate:
        if not (Path(args.out) / "top10_calibration.csv").exists() or args.recalibrate:
            cmd_calibrate(args)
            print()
    cmd_ukbb(args)
    print()
    if args.skip_deciles:
        print("skipping `percentiles` (--skip-percentiles): this run will have NO measured "
              "cut-offs,\nand charts 8-10 of the notebook will sit out.\n")
    else:
        # `--min-papers` means two different things to the two modes it reaches here:
        # calibrate's floor is per (year, type) over the whole corpus (1,000), the
        # percentile pass's is per (year, field) and has to be far lower (200) or most
        # fields never get a cut-off. One flag cannot be both, so it gets its own.
        import copy
        dargs = copy.copy(args)
        dargs.min_papers = args.deciles_min_papers
        cmd_percentiles(dargs)
        print()
        print("re-running `ukbb` to score each paper against the cut-offs just "
              "measured\n(the records are cached, so this costs no queries)\n")
        cmd_ukbb(args)
        print()
    cmd_background(args)


# =============================================================================
# MODE: check
# =============================================================================
def cmd_check(args: argparse.Namespace) -> None:
    api = Dimensions(load_config(Path(args.config), args.instance), sleep=args.sleep)
    field = CATEGORY_FIELDS[args.category]
    res = api.query(f"search publications where year=2020 return {field} limit 5")
    rows = res.get(field, [])
    print(f"\n{field} facet, 2020, first {len(rows)} rows:")
    for r in rows:
        print(f"  {r.get('id'):>8}  {str(r.get('name'))[:44]:44s} {int(r.get('count', 0)):>12,}")
    print(f"\n_stats: {json.dumps(res.get('_stats', {}))}")
    print("\nnow trying the citation aggregations the docs list for publications:")
    for agg in ("citations_total", "citations_avg", "fcr_gavg", "recent_citations_total"):
        try:
            r = api.query(f"search publications where year=2020 return {field} "
                          f"aggregate {agg} limit 3")
            got = r.get(field, [])
            sample = got[0].get(agg) if got else None
            print(f"  {agg:22s} OK   e.g. {sample}")
        except SystemExit as exc:
            print(f"  {agg:22s} FAILED  {str(exc)[:90]}")


# =============================================================================
# MODE: selftest  — the table building, without a key or a network
# =============================================================================
def cmd_selftest(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    codes = load_codes(Path(args.codes))
    sample = codes.index[:4].tolist()
    per_year = {2014: [{"id": sample[0], "name": "x", "count": 100},
                       {"id": sample[1], "name": "y", "count": 50}],
                2015: [{"id": sample[0], "name": "x", "count": 120},
                       {"id": "99999", "name": "unknown", "count": 7}]}
    df = build_counts_table(per_year, codes, "for", "whole")

    expected = ["arm", "category", "year", "type", "level", "code", "for_label",
                "parent_l2", "n_papers", "n_frac"]
    assert list(df.columns) == expected, f"schema drift: {list(df.columns)}"
    assert len(df) == 3, f"the unknown id should have been dropped, got {len(df)} rows"
    assert df.n_frac.isna().all(), "n_frac must be NaN on the API path"
    assert set(df.level) <= {"L2", "L4"}
    for _, r in df[df.level == "L4"].iterrows():
        assert r.parent_l2 == r.code[:2]

    # The VM job's schema is the contract; compare against a real partial if one is here.
    vm = sorted(Path(args.ukbb_counts).glob("counts.for.ukbb*.parquet"))
    if vm:
        vm_cols = list(pd.read_parquet(vm[0]).columns)
        missing = [c for c in expected if c not in vm_cols]
        assert not missing, f"columns the VM job does not have: {missing}"
        print(f"schema matches the VM partial {vm[0].name}")
        print(f"  VM extras not produced by the API path: "
              f"{[c for c in vm_cols if c not in expected]}")
    print(f"\nselftest OK — {len(df)} rows built from {len(per_year)} synthetic years")
    print(df.to_string(index=False))


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples
  # 0. is the key working, and which aggregations does this account get?
  %(prog)s check

  # 1. what would it ask for? (no key needed)
  %(prog)s counts --dry-run

  # 2. the whole-database year x field matrix, minus our UK Biobank papers
  %(prog)s counts --out data/analysis/academic_impact/for_counts_api \\
      --ukbb-counts data/analysis/academic_impact/for_counts_out

  # 3. the hybrid: both arms from the same index on the same day.
  #    whole -> ukbb -> percentiles -> ukbb -> background, in that order (see the module).
  #    ~73 queries for the counts, plus ~3,600 for the measured cut-offs (~3 h at
  #    --min-ukbb 10; a re-run skips what is already measured).
  %(prog)s all --category for
  %(prog)s all --category for --skip-percentiles   # counts only, ~3 minutes

  # 3b. the cut-offs on their own, once `ukbb` has run (both percentiles in one pass)
  %(prog)s percentiles --category for --percentiles 10,50

  # 4. the same records serve RCDC too — they were fetched with both classifications
  %(prog)s whole --category rcdc && %(prog)s ukbb --category rcdc \\
      && %(prog)s background --category rcdc
""")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help=f"dsl.ini holding the API key (default: {DEFAULT_CONFIG})")
        sp.add_argument("--instance", default="live", help="section of dsl.ini to use")
        sp.add_argument("--category", choices=sorted(CATEGORY_FIELDS), default="for")
        sp.add_argument("--codes", default=str(DEFAULT_CODES_CSV),
                        help="the id/code/name list for the classification")
        sp.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                        help="seconds between queries (the API allows 30/minute)")

    sp = sub.add_parser("check", help="authenticate and probe what the account can do")
    common(sp)

    sp = sub.add_parser("codes", help="refresh the id/code/name list from the API")
    common(sp)
    sp.add_argument("--out", default=str(DEFAULT_CODES_CSV))

    sp = sub.add_parser("counts", help="year x field counts -> partials in the VM schema")
    common(sp)
    sp.add_argument("--year-min", type=int, default=2004)
    sp.add_argument("--year-max", type=int, default=2024)
    sp.add_argument("--out", default="data/analysis/academic_impact/for_counts_api")
    sp.add_argument("--ukbb-counts",
                    default="data/analysis/academic_impact/for_counts_out",
                    help="directory holding the VM job's UK Biobank partials, which are "
                         "subtracted to make the background arm and copied into --out")
    sp.add_argument("--aggregate", default="",
                    help="citation metrics to carry alongside the counts, e.g. "
                         "citations_total,citations_avg,fcr_gavg")
    sp.add_argument("--dry-run", action="store_true",
                    help="print the queries and exit; needs no key")

    def window(sp):
        sp.add_argument("--year-min", type=int, default=2004)
        sp.add_argument("--year-max", type=int, default=2024)
        sp.add_argument("--out", default="data/analysis/academic_impact/for_counts_api")

    sp = sub.add_parser("calibrate",
                        help="the top-decile PROXY's K per (year, type); see the module")
    common(sp); window(sp)
    sp.add_argument("--types", default="article,preprint",
                    help="publication types to calibrate; others fall back to article")
    sp.add_argument("--percentile", type=float, default=10.0)
    sp.add_argument("--min-papers", type=int, default=1000)

    for _name in ("percentiles", "deciles"):     # `deciles` is the old name, kept working
        sp = sub.add_parser(_name,
                            help="MEASURED per-field cut-offs (decile AND median), by "
                                 "counting — see the module")
        common(sp); window(sp)
        sp.add_argument("--percentiles", default="10,50",
                        help="comma-separated percentiles to measure in one pass "
                             "(default: 10,50 — the top decile and the median). They "
                             "share every per-cell query, so the second costs about a "
                             "third of the first")
        sp.add_argument("--percentile", type=float, default=10.0,
                        help="LEGACY single percentile; --percentiles supersedes it")
        sp.add_argument("--min-ukbb", type=int, default=10,
                        help="only measure cells where UK Biobank has this many papers "
                             "(default 10: the floor the notebooks read at, so one pass "
                             "covers every cell they draw)")
        sp.add_argument("--remeasure", action="store_true",
                        help="re-measure cells already in the file instead of resuming")
        sp.add_argument("--min-papers", type=int, default=200,
                        help="skip a field-year thinner than this in the whole database")
        sp.add_argument("--checkpoint", type=int, default=25,
                        help="write the measured rows to disk every N cells, so a run "
                             "that dies keeps what it has already paid for")

    sp = sub.add_parser("whole", help="whole-database facets + the reference means")
    common(sp); window(sp)

    sp = sub.add_parser("ukbb", help="UK Biobank's arm, fetched per paper")
    common(sp); window(sp)
    sp.add_argument("--ids",
                    default="data/analysis/academic_impact/for_counts_out/ukbb_ids.txt")
    sp.add_argument("--batch", type=int, default=ID_BATCH,
                    help=f"ids per query; the API caps the list at {ID_BATCH}")
    sp.add_argument("--refresh", action="store_true",
                    help="re-fetch even if api_ukbb_records.json is already there")

    sp = sub.add_parser("background", help="whole - ukbb, cell by cell")
    common(sp); window(sp)
    sp.add_argument("--percentile", type=float, default=10.0)

    sp = sub.add_parser("all",
                        help="whole -> ukbb -> deciles -> ukbb -> background, in order")
    common(sp); window(sp)
    sp.add_argument("--ids",
                    default="data/analysis/academic_impact/for_counts_out/ukbb_ids.txt")
    sp.add_argument("--batch", type=int, default=ID_BATCH)
    sp.add_argument("--refresh", action="store_true")
    sp.add_argument("--skip-percentiles", "--skip-deciles", action="store_true",
                    dest="skip_deciles",
                    help="leave out the measured per-field cut-offs (~3,000 queries, "
                         "~110 min). The run then has no top-decile or median column")
    sp.add_argument("--percentiles", default="10,50",
                    help="percentiles the measuring pass should cover (default: 10,50)")
    sp.add_argument("--checkpoint", type=int, default=25,
                    help="percentiles: write measured rows to disk every N cells")
    sp.add_argument("--min-ukbb", type=int, default=10,
                    help="percentiles: only measure cells with this many UK Biobank "
                         "papers (default 10, matching the notebooks' floor)")
    sp.add_argument("--remeasure", action="store_true",
                    help="percentiles: re-measure cells already in the file")
    sp.add_argument("--deciles-min-papers", type=int, default=200,
                    help="deciles: skip a (year, field) thinner than this in the whole "
                         "database. Separate from --min-papers, which is calibrate's "
                         "per-(year, type) floor")
    sp.add_argument("--calibrate", action="store_true",
                    help="LEGACY: also compute K for the top10p proxy. `deciles` "
                         "measures the real cut-off, and the analysis no longer reads "
                         "the proxy column")
    sp.add_argument("--recalibrate", action="store_true",
                    help="redo the legacy calibration even if it is already there")
    sp.add_argument("--types", default="article,preprint")
    sp.add_argument("--percentile", type=float, default=10.0)
    sp.add_argument("--min-papers", type=int, default=1000)

    sp = sub.add_parser("selftest", help="check the table building offline")
    common(sp)
    sp.add_argument("--ukbb-counts",
                    default="data/analysis/academic_impact/for_counts_out")

    args = p.parse_args()
    {"check": cmd_check, "codes": cmd_codes, "counts": cmd_counts,
     "calibrate": cmd_calibrate, "percentiles": cmd_percentiles,
     "deciles": cmd_percentiles, "whole": cmd_whole,
     "ukbb": cmd_ukbb,
     "background": cmd_background, "all": cmd_all,
     "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
