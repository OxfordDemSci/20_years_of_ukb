"""
MeSH descriptor -> tree number -> ICD-10 chapter
================================================

WHY
---
The first ICD mapping in the clinical-trials notebook was a hand-written **keyword** list
("heart" -> Circulatory). It is fragile in three ways:

  * substring matching fires on the wrong thing — "alcohol" matched "NON-alcoholic Fatty
    Liver Disease", so every NAFLD trial was counted as Mental & behavioural until patched;
  * the chapter order silently decides ties (Circulatory is tested before Nervous so that
    "stroke" lands in I60-I69);
  * it only knows the keywords someone thought to type — 74 of 220 leaf terms went unmapped,
    including real conditions (Wounds and Injuries, Premature Birth, Gingivitis).

MeSH already encodes what we are trying to guess. Every descriptor has **tree numbers**
(`Pulmonary Disease, Chronic Obstructive` = `D029424` -> `C08.381.495.389`), and the
top-level MeSH categories line up with the ICD-10 body-system chapters. ClinicalTrials.gov
hands us the descriptor **D-id** with every condition term, so we can map
descriptor -> tree -> chapter and stop guessing from strings.

Tree numbers come from NLM's public MeSH SPARQL endpoint (no key, no licence gate — unlike
the UMLS crosswalk), cached to disk after the first run.

WHAT IT IS NOT
--------------
Still an *approximation*: MeSH and ICD are different vocabularies built for different jobs,
so a chapter-level alignment is the honest granularity — this does not claim term-level
ICD codes. But it is derived from NLM's own hierarchy rather than from keywords, it is
auditable (`MESH_CAT_TO_ICD` below), and it maps ~everything.

Descriptors carrying only F01 (Behavior) or C23 (generic signs/symptoms) trees are treated
as **not a disease** rather than forced into a chapter — that is a feature: it is how
`Motor Activity` and `Inflammation` correctly stay unmapped.

Author: Jiani Y
Date: 2026-07-14
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

SPARQL = "https://id.nlm.nih.gov/mesh/sparql"

# Anchored on this file (utils/ -> src/ -> repo root) so the cache resolves the same
# whether this module is imported from a notebook or run as a script.
_ROOT = Path(__file__).resolve().parents[2]
CACHE = _ROOT / "data/analysis/non_academic/clinical_trials/mesh_tree_numbers.pkl"

# =============================================================================
# MeSH top-level category -> ICD-10 chapter
# =============================================================================
# Labels are IDENTICAL to the notebook's keyword-map chapters, so the two mappings stay
# directly comparable. Ordering inside a descriptor is resolved by PRIORITY below, not by
# dict order.
MESH_CAT_TO_ICD: Dict[str, str] = {
    "C01": "I   Infectious & parasitic (A00–B99)",
    "C04": "II  Neoplasms (C00–D48)",
    "C15": "III Blood & immune (D50–D89)",      # haemic & lymphatic
    "C20": "III Blood & immune (D50–D89)",      # immune system diseases
    "C18": "IV  Endocrine, nutritional & metabolic (E00–E90)",   # nutritional & metabolic
    "C19": "IV  Endocrine, nutritional & metabolic (E00–E90)",   # endocrine system
    "F03": "V   Mental & behavioural (F00–F99)",
    "C10": "VI  Nervous system (G00–G99)",
    "C11": "VII Eye & adnexa (H00–H59)",
    "C09": "VIII Ear & mastoid (H60–H95)",      # otorhinolaryngologic
    "C14": "IX  Circulatory system (I00–I99)",
    "C08": "X   Respiratory system (J00–J99)",
    "C06": "XI  Digestive system (K00–K93)",
    "C07": "XI  Digestive system (K00–K93)",    # stomatognathic (mouth/teeth = K00–K14)
    "C17": "XII Skin (L00–L99)",                # skin & connective tissue
    "C05": "XIII Musculoskeletal (M00–M99)",
    "C12": "XIV Genitourinary (N00–N99)",       # urogenital (incl. pregnancy sub-tree, see below)
    "C13": "XIV Genitourinary (N00–N99)",       # legacy female-urogenital branch
    "C16": "XVII Congenital malformations (Q00–Q99)",
    "C25": "XIX Injury & poisoning (S00–T98)",  # chemically-induced disorders
    "C26": "XIX Injury & poisoning (S00–T98)",  # wounds & injuries
    "C21": "XIX Injury & poisoning (S00–T98)",  # disorders of environmental origin
    "C23": "XVIII Symptoms & signs (R00–R99)",  # generic — last resort only (see PRIORITY)
}

# Sub-tree overrides, checked BEFORE the top-level category: MeSH puts pregnancy under the
# urogenital branch and neonatal disease under congenital, but ICD gives each its own chapter.
SUBTREE_OVERRIDES = [
    ("C13.703", "XV  Pregnancy & childbirth (O00–O99)"),   # pregnancy complications
    ("C12.050.703", "XV  Pregnancy & childbirth (O00–O99)"),
    ("C16.614", "XVI Perinatal conditions (P00–P96)"),     # infant, newborn, diseases
]

# Generic branches: real chapters, but only used if the descriptor has nothing more specific.
GENERIC_CATS = {"C23"}

# Trees that are NOT diseases at all — a descriptor with only these maps to None.
# F01 = Behavior and Behavior Mechanisms (Motor Activity, Health Behavior, Risk-Taking, ...)
NON_DISEASE_CATS = {"F01", "F02", "F04", "E", "G", "H", "I", "J", "L", "M", "N", "V", "Z"}


# =============================================================================
# FETCH tree numbers (NLM SPARQL, cached)
# =============================================================================
def _sparql_batch(dids: List[str], timeout: int = 40) -> Dict[str, List[str]]:
    values = " ".join(f"mesh:{d}" for d in dids)
    q = (
        "PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#> "
        "PREFIX mesh: <http://id.nlm.nih.gov/mesh/> "
        f"SELECT ?d ?tn WHERE {{ VALUES ?d {{ {values} }} ?d meshv:treeNumber ?tn }}"
    )
    url = SPARQL + "?" + urllib.parse.urlencode(
        {"query": q, "format": "JSON", "limit": 5000, "inference": "false"})
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = json.load(r)
    out: Dict[str, List[str]] = {d: [] for d in dids}
    for b in data["results"]["bindings"]:
        did = b["d"]["value"].rsplit("/", 1)[-1]
        tn = b["tn"]["value"].rsplit("/", 1)[-1]
        out.setdefault(did, []).append(tn)
    return out


def _sparql_mapped_to(scrs: List[str], timeout: int = 40) -> Dict[str, List[str]]:
    """Supplementary Concept Record -> the real descriptor(s) it maps to.

    MeSH SCRs (ids starting 'C', e.g. C537243 'Prostate cancer, familial') are specific
    disease concepts that carry NO tree numbers of their own; they declare a
    `preferredMappedTo` heading instead. Following that link is what lets them reach a
    chapter — otherwise every rare-disease trial silently drops out of the ICD counts.
    """
    values = " ".join(f"mesh:{d}" for d in scrs)
    q = (
        "PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#> "
        "PREFIX mesh: <http://id.nlm.nih.gov/mesh/> "
        f"SELECT ?s ?m WHERE {{ VALUES ?s {{ {values} }} ?s meshv:preferredMappedTo ?m }}"
    )
    url = SPARQL + "?" + urllib.parse.urlencode(
        {"query": q, "format": "JSON", "limit": 5000, "inference": "false"})
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = json.load(r)
    out: Dict[str, List[str]] = {}
    for b in data["results"]["bindings"]:
        scr = b["s"]["value"].rsplit("/", 1)[-1]
        tgt = b["m"]["value"].rsplit("/", 1)[-1].split("Q")[0]   # strip any qualifier suffix
        out.setdefault(scr, []).append(tgt)
    return out


def fetch_tree_numbers(dids: Iterable[str], cache: Path = CACHE,
                       batch: int = 40, pause: float = 0.3,
                       refresh: bool = False) -> Dict[str, List[str]]:
    """{descriptor id -> [tree numbers]} from NLM's MeSH SPARQL endpoint, cached to disk."""
    known: Dict[str, List[str]] = {}
    if cache.exists() and not refresh:
        known = pd.read_pickle(cache)
    todo = sorted({d for d in dids if d and d not in known})
    if todo:
        print(f"MeSH tree: fetching {len(todo)} descriptors from NLM ...")
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            try:
                known.update(_sparql_batch(chunk))
            except Exception as exc:                     # noqa: BLE001
                print(f"  ! batch {i // batch}: {exc}")
                for d in chunk:
                    known.setdefault(d, [])
            time.sleep(pause)

        # SCRs have no trees of their own — inherit them from the heading they map to
        scrs = [d for d in todo if not known.get(d)]
        if scrs:
            try:
                mapped = _sparql_mapped_to(scrs)
                targets = sorted({t for ts in mapped.values() for t in ts if t not in known})
                for i in range(0, len(targets), batch):
                    known.update(_sparql_batch(targets[i:i + batch]))
                    time.sleep(pause)
                for scr, tgts in mapped.items():
                    known[scr] = sorted({tn for t in tgts for tn in known.get(t, [])})
                print(f"MeSH tree: resolved {len(mapped)} supplementary concepts "
                      f"via preferredMappedTo")
            except Exception as exc:                 # noqa: BLE001
                print(f"  ! SCR resolution failed: {exc}")

        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(known, cache)
        print(f"MeSH tree: cached {len(known)} descriptors -> {cache}")
    return known


# =============================================================================
# MAP tree numbers -> ICD chapter
# =============================================================================
def chapters_from_trees(trees: Iterable[str]) -> Set[str]:
    """ICD-10 chapters for one descriptor's tree numbers.

    Specific body-system trees win; the generic 'Pathological Conditions, Signs and Symptoms'
    branch (C23) is used only when the descriptor has nothing else; behaviour-only descriptors
    (F01) map to nothing, which is how `Motor Activity` stays out of the disease counts.
    """
    specific, generic = set(), set()
    for tn in trees:
        for prefix, chap in SUBTREE_OVERRIDES:
            if tn.startswith(prefix):
                specific.add(chap)
                break
        else:
            cat = tn.split(".")[0]
            chap = MESH_CAT_TO_ICD.get(cat)
            if not chap:
                continue
            (generic if cat in GENERIC_CATS else specific).add(chap)
    return specific or generic


def descriptor_chapters(dids: Iterable[str],
                        tree_map: Dict[str, List[str]]) -> Set[str]:
    """ICD-10 chapters for a list of descriptor ids (e.g. one trial's MeSH leaves)."""
    out: Set[str] = set()
    for d in dids:
        out |= chapters_from_trees(tree_map.get(d, []))
    return out


def audit(dids: Iterable[str], tree_map: Dict[str, List[str]]) -> pd.DataFrame:
    """One row per descriptor: its trees and the chapter(s) they resolve to — for eyeballing."""
    rows = []
    for d in sorted(set(dids)):
        trees = tree_map.get(d, [])
        rows.append({"descriptor": d, "trees": ", ".join(trees) or "—",
                     "chapters": ", ".join(sorted(chapters_from_trees(trees))) or "— none —"})
    return pd.DataFrame(rows)
