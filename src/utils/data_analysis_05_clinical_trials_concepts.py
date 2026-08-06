"""
Trial -> cited-paper concept linkage (+ the concept stop-list)
=============================================================

Shared by TWO sections of `01_analysis_05_non_academic_impact_03_clinic_trials.ipynb`:

  * §3.6  mesh-leaf x concept heatmap  (needs the link early, before §8 exists)
  * §8.1  concept ranking of the cited UKB papers

Keeping the stop-list here means the two sections cannot drift apart.

WHY A STOP-LIST AT ALL
----------------------
`papers_20260713.pkl` ships `concepts` as a plain list of strings with NO relevance scores
(`concepts_scores` is gone from this export), and papers carry a **median of ~69 concepts**
each. Unfiltered, the ranking is pure corpus scaffolding — *study · risk · disease ·
association · participants · biobank · data*. The stop-list drops those generic phrases;
matching is EXACT on the lowercased phrase, so bare "disease" goes while "cardiovascular
disease" stays.

Author: Jiani Y
Date: 2026-07-14
"""

import ast
from collections import defaultdict
from typing import Dict, Iterable, List, Set

import pandas as pd

# =============================================================================
# STOP-LIST — three families, kept separate so they are easy to audit / extend
# =============================================================================
CONCEPT_STOP_STATS = {          # statistics / methods scaffolding
    "hazard ratio", "confidence interval", "cox proportional hazards model", "odds ratio",
    "standard deviation", "logistic regression", "study population", "effect size",
    "statistical significance", "sensitivity analysis", "proportional hazards model",
    "cox regression", "cox", "regression", "regression models", "meta-analysis",
    "confounding", "adjustment", "cox proportional hazards models",
    "mendelian randomization analysis", "model", "models", "analysis", "analyses",
    "method", "methods", "approach", "data", "dataset", "datasets", "results", "findings",
    "effect", "effects", "estimates", "estimation", "evidence", "measures", "measurements",
    "measurement", "scores", "score", "risk score", "risk scores", "threshold",
    "validation", "prediction", "predictors", "variables", "sample", "samples",
    "sample size", "subgroup", "subgroups", "ratio", "index", "interaction", "proportion",
    "number", "total", "values", "research", "quality", "use", "control", "controls",
    "correlation", "information",
}
CONCEPT_STOP_CORPUS = {         # every paper in this corpus IS a UK Biobank cohort study
    "uk biobank", "biobank", "uk biobank study", "uk biobank participants",
    "biobank participants", "uk biobank cohort", "biobank cohort", "uk biobank data",
    "biobank study", "participants", "individuals", "cohort", "cohort study", "study",
    "studies", "large cohort", "prospective cohort study", "observational study",
    "general population", "population", "populations", "people", "adults", "men", "women",
    "subjects", "patients", "group", "groups", "case", "cases",
}
CONCEPT_STOP_EPI = {            # generic epidemiology vocabulary
    "association", "associations", "association studies", "risk", "risks", "risk factor",
    "risk factors", "high risk", "health", "years", "age", "baseline", "follow-up",
    "follow up", "outcome", "outcomes", "main outcomes", "incidence", "sex", "prevalence",
    "european ancestry", "ethnicity", "disease", "diseases", "disorder", "disorders",
    "condition", "conditions", "factors", "mortality", "death", "deaths", "time", "levels",
    "level", "increase", "reduction", "rate", "rates", "events", "relationship", "type",
    "status", "change", "changes", "differences", "difference", "function", "activity",
    "body", "life",   # 'life' is the tail of 'quality of life' etc.; 'lifestyle' is kept
}
CONCEPT_STOP = CONCEPT_STOP_STATS | CONCEPT_STOP_CORPUS | CONCEPT_STOP_EPI


def _parse_listcol(x) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def clean_concepts(cell, stop: Set[str] = CONCEPT_STOP) -> List[str]:
    """Lowercased, stop-listed concept list for one paper."""
    return [c.lower().strip() for c in _parse_listcol(cell)
            if isinstance(c, str) and c.lower().strip() not in stop]


def paper_concept_map(papers: pd.DataFrame, id_col: str = "id",
                      concept_col: str = "concepts") -> Dict[str, Set[str]]:
    """paper id -> its set of cleaned concepts."""
    return {pid: set(clean_concepts(cell))
            for pid, cell in zip(papers[id_col], papers[concept_col])}


def trial_concept_sets(trials: pd.DataFrame, papers: pd.DataFrame,
                       pids_col: str = "pids") -> pd.Series:
    """For each trial, the UNION of the concepts of the UKB papers it cites.

    A trial has no concepts of its own — this is the trial -> cited-paper -> concept link
    that lets a *trial-level* row (e.g. a MeSH disease) be profiled by the *research* it
    draws on. Trials citing no matched paper get an empty set.
    """
    cmap = paper_concept_map(papers)
    return trials[pids_col].apply(
        lambda pids: sorted(set().union(*[cmap[p] for p in pids if p in cmap]) if
                            any(p in cmap for p in pids) else set()))


def count_items_frac(series: Iterable) -> pd.Series:
    """Fractional counts: each row spreads a weight of 1 across its distinct items."""
    w = defaultdict(float)
    for cell in series:
        items = set(cell if isinstance(cell, list) else _parse_listcol(cell))
        if not items:
            continue
        f = 1.0 / len(items)
        for it in items:
            w[it] += f
    return pd.Series(w, dtype="float64").sort_values(ascending=False)
