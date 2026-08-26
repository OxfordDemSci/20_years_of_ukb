"""
ClinicalTrials.gov MeSH Re-fetch
================================

Rebuilds a clean disease axis for the clinical-trials dataset from the ClinicalTrials.gov
REST API (v2), which needs no API key.

WHY
---
Dimensions' `mesh_terms` column is NOT indexer-assigned MeSH. ClinicalTrials.gov *derives*
it from the trial's free-text conditions/interventions and returns it in three distinct
buckets, which Dimensions flattens into a single column:

  * conditionBrowseModule.meshes     -> LEAF terms actually matched  (Alzheimer Disease,
                                        Diabetes Mellitus, Prostatic Neoplasms) = the diseases
  * conditionBrowseModule.ancestors  -> MeSH-tree scaffolding       (Pathologic Processes,
                                        Signs and Symptoms, Nervous System Diseases)
  * interventionBrowseModule.*       -> terms from the intervention text (Magnetic Resonance
                                        Spectroscopy, Absorptiometry Photon, drug classes)

Flattened together, the tree scaffolding dominates any frequency ranking (`Pathologic
Processes` is the single most common "disease" in the shipped column), so the leaf terms are
what a disease analysis actually wants. Re-fetching also refreshes records the Dimensions
snapshot took before NLM had indexed them.

Non-NCT registries (ISRCTN, EU-CTR, ...) have no NLM browse module; they are recorded as
empty rather than guessed at.

WHAT IT DOES
------------
Fetches the browse module for every trial id, caches the raw result to a pickle (so reruns
are instant and offline), and writes three new columns back to the clinical-trials CSV:

  mesh_leaf | mesh_leaf_ids | mesh_ancestors | mesh_intervention   (stringified lists, like
                                                    the other list-columns in the file)

`mesh_leaf_ids` holds the MeSH descriptor D-numbers for the leaf terms; src/utils/data_analysis_04_non_academic_clinical_trials_mesh_tree.py
turns those into ICD-10 chapters via MeSH's own hierarchy (no keyword guessing).

The original `mesh_terms` column is left untouched, so the two can be compared.

USAGE
-----
    python src/utils/data_creation_data_creation_clinical_trials_fetch_ctgov.py                 # fetch (cached) + patch CSV
    python src/utils/data_creation_data_creation_clinical_trials_fetch_ctgov.py --refresh       # ignore cache, re-fetch all
    python src/utils/data_creation_data_creation_clinical_trials_fetch_ctgov.py --no-write      # report only, don't touch CSV

    from utils.data_creation_clinical_trials_fetch_ctgov import fetch_ctgov_browse, enrich_clinical_trials
    df = enrich_clinical_trials()          # returns the patched frame

Author: Jiani Y
Date: 2026-07-13
"""

import argparse
import ast
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# --- paths (anchored on this file: utils/ -> src/ -> repo root) ---------------
_ROOT = Path(__file__).resolve().parents[2]
CT_CSV = _ROOT / "data/analysis/non_academic/clinical_trials/clinical_trials.csv"
CTGOV_CACHE = _ROOT / "data/analysis/non_academic/clinical_trials/ctgov_browse.pkl"

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies/{nct}"
CTGOV_FIELDS = ("protocolSection.conditionsModule,"
                "derivedSection.conditionBrowseModule,"
                "derivedSection.interventionBrowseModule")

NEW_COLS = ["mesh_leaf", "mesh_leaf_ids", "mesh_ancestors", "mesh_intervention"]


# =============================================================================
# FETCH
# =============================================================================
def _terms(module: dict, key: str) -> List[str]:
    """Term names from one browse-module bucket ('meshes' / 'ancestors')."""
    return [m["term"] for m in (module.get(key) or []) if m.get("term")]


def _ids(module: dict, key: str) -> List[str]:
    """MeSH descriptor ids (D-numbers) from one bucket — e.g. 'D029424' for COPD.

    These are what make a principled ICD mapping possible: a descriptor id resolves to MeSH
    tree numbers (see src/utils/data_analysis_04_non_academic_clinical_trials_mesh_tree.py), whereas the term STRING only supports keyword
    guessing. CTgov returns them alongside every term, so keeping them costs nothing.
    """
    return [m["id"] for m in (module.get(key) or []) if m.get("id")]


def fetch_ctgov_browse(ids: Iterable[str], cache: Path = CTGOV_CACHE,
                       pause: float = 0.2, refresh: bool = False) -> pd.DataFrame:
    """NLM browse modules for `ids`, cached to disk.

    Returns one row per id: id | fetched | cond_leaf | cond_ancestors | intervention.
    Only ids missing from the cache are requested, so adding trials to the CSV later and
    re-running costs just the new ones. Failures (network, withdrawn record) are recorded
    as fetched=False with empty term lists — never silently mixed with genuine empties.
    """
    have = pd.read_pickle(cache) if (cache.exists() and not refresh) else pd.DataFrame()
    done = set(have["id"]) if len(have) else set()
    todo = [i for i in ids if i not in done]
    if not todo:
        print(f"cache hit: all {len(done)} trials already fetched ({cache})")
        return have

    print(f"fetching {len(todo)} trials from ClinicalTrials.gov ...")
    rows = []
    for k, tid in enumerate(todo, 1):
        if not str(tid).startswith("NCT"):      # ISRCTN / EU-CTR: no NLM browse module exists
            rows.append(dict(id=tid, fetched=False, cond_leaf=[], cond_leaf_ids=[],
                             cond_ancestors=[], intervention=[]))
            continue
        try:
            url = CTGOV_API.format(nct=tid) + f"?fields={CTGOV_FIELDS}"
            with urllib.request.urlopen(url, timeout=30) as r:
                derived = json.load(r).get("derivedSection", {})
            cond = derived.get("conditionBrowseModule", {})
            interv = derived.get("interventionBrowseModule", {})
            rows.append(dict(id=tid, fetched=True,
                             cond_leaf=_terms(cond, "meshes"),
                             cond_leaf_ids=_ids(cond, "meshes"),
                             cond_ancestors=_terms(cond, "ancestors"),
                             intervention=_terms(interv, "meshes") + _terms(interv, "ancestors")))
        except Exception as exc:                # noqa: BLE001 - report, don't abort the run
            print(f"  ! {tid}: {exc}")
            rows.append(dict(id=tid, fetched=False, cond_leaf=[], cond_leaf_ids=[],
                             cond_ancestors=[], intervention=[]))
        if k % 25 == 0:
            print(f"  {k}/{len(todo)}")
        time.sleep(pause)                       # be polite to the API

    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(cache)
    print(f"cached {len(out)} rows -> {cache}")
    return out


# =============================================================================
# PATCH THE CSV
# =============================================================================
def _names(cell) -> List[str]:
    """Names from a stringified list cell (list-of-dicts or list-of-strings)."""
    try:
        v = ast.literal_eval(cell) if isinstance(cell, str) else (cell or [])
    except (ValueError, SyntaxError):
        return []
    if not isinstance(v, list):
        return []
    return [e["name"] if isinstance(e, dict) else e for e in v
            if (isinstance(e, dict) and e.get("name")) or (isinstance(e, str) and e.strip())]


def enrich_clinical_trials(csv: Path = CT_CSV, cache: Path = CTGOV_CACHE,
                           refresh: bool = False, write: bool = True) -> pd.DataFrame:
    """Add mesh_leaf / mesh_ancestors / mesh_intervention to the clinical-trials CSV."""
    df = pd.read_csv(csv)
    browse = fetch_ctgov_browse(df["id"].tolist(), cache=cache, refresh=refresh)
    b = browse.drop_duplicates("id").set_index("id")

    def take(col):
        return df["id"].map(b[col]).apply(lambda x: x if isinstance(x, list) else [])

    df["mesh_leaf"] = take("cond_leaf")
    df["mesh_leaf_ids"] = take("cond_leaf_ids")      # D-numbers -> data_analysis_04_non_academic_clinical_trials_mesh_tree.py -> ICD chapter
    df["mesh_ancestors"] = take("cond_ancestors")
    df["mesh_intervention"] = take("intervention")

    _report(df)

    if write:
        # store as stringified lists, matching how the other list-columns round-trip
        out = df.copy()
        for c in NEW_COLS:
            out[c] = out[c].apply(repr)
        out.to_csv(csv, index=False)
        print(f"\nwrote {len(NEW_COLS)} new columns -> {csv}")
    return df


def _report(df: pd.DataFrame) -> None:
    """Before/after summary: coverage, terms per trial, and the two rankings side by side."""
    old = df["mesh_terms"].apply(_names)
    n = len(df)
    print(f"\ntrials with >=1 MeSH term — shipped: {int(old.str.len().gt(0).sum())}/{n}"
          f" | registry leaf: {int(df['mesh_leaf'].str.len().gt(0).sum())}/{n}")
    print(f"median terms per trial   — shipped: {old.str.len().median():.0f}"
          f" | registry leaf: {df['mesh_leaf'].str.len().median():.0f}")

    c_old, c_new = Counter(), Counter()
    for ts in old:
        c_old.update(set(ts))
    for ts in df["mesh_leaf"]:
        c_new.update(set(ts))
    print(f"\n{'SHIPPED mesh_terms (leaf+ancestors+interventions)':<52}   LEAF ONLY (diseases)")
    for (a, av), (bb, bv) in zip(c_old.most_common(10), c_new.most_common(10)):
        print(f"  {av:3d}  {a[:44]:<44}     {bv:3d}  {bb[:40]}")

    gained = df[old.str.len().eq(0) & df["mesh_leaf"].str.len().gt(0)]
    if len(gained):
        print(f"\nrecovered (no shipped MeSH, registry now has leaf terms): {len(gained)}")
        for _, r in gained.iterrows():
            print(f"  {r['id']}: {r['mesh_leaf']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--refresh", action="store_true", help="ignore the cache and re-fetch all")
    p.add_argument("--no-write", action="store_true", help="report only; leave the CSV alone")
    p.add_argument("--csv", type=Path, default=CT_CSV)
    a = p.parse_args()
    enrich_clinical_trials(csv=a.csv, refresh=a.refresh, write=not a.no_write)
