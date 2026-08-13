"""Data layer for the 03_academic_impact_* notebooks (both of them).

WHY THIS MODULE EXISTS
----------------------
The academic-impact analysis is run twice over the same count tables, once per
classification system:

    03_academic_impact_01_for_analysis.ipynb    ANZSRC Fields of Research
    03_academic_impact_02_rcdc_analysis.ipynb   NIH Research/Condition/Disease

Only the charts differ between them. Everything up to the charts — resolving which
partials belong to which arm, dropping unusable years, building the whole-database
denominator, the activity index, the citation windows and the impact table — is
identical, and was identical in two notebooks for exactly as long as it took someone
to fix a guard in one of them. It lives here instead, so a fix lands in both.

The notebooks keep what a reader wants to edit: the parameter block at the top and
the plotting code at the bottom. This module keeps what they must agree on.

USAGE (both notebooks do exactly this)

    from utils import data_analysis_03_academic_impact_analysis as AI
    CTX = AI.build(COUNTS_DIR, COL_TYPE, level=..., ...)
    globals().update(CTX)          # ukbb, whole, TOP_CODES, IMPACT, ...

`globals().update` is deliberate: it keeps the chart code reading `whole_ts` rather
than `CTX["whole_ts"]`, so a chart can be moved between the two notebooks unchanged.
Everything it injects is listed in `build`'s docstring and nothing else is added.

WHAT IT DOES NOT DO
-------------------
No plotting, and no matplotlib import. Charts belong to the notebook that draws them;
this module hands over frames and the guards that say what those frames can support.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# The measure columns the count job writes. Everything is derived from these two plus
# whichever citation weights the partials happen to carry.
BASE_MEASURES = ("n_papers", "n_frac")
CITATION_WEIGHTS = ("n_cit", "n_fcr", "n_top10")

WEIGHT_NAMES = {
    "n_cit": "raw citations",
    "n_fcr": "field-normalised citation ratio (FCR)",
    "n_top10": "top-decile rate",
}

# =============================================================================
# WHAT THE ACTIVITY INDEX IS MEASURED AGAINST
# =============================================================================
# The index is a ratio of shares — UK Biobank's share of its own output that sits in X,
# over the database's share of ITS output that sits in X — so the answer depends
# entirely on what "the database" is taken to mean. Three choices, and they are not
# variations on a theme:
#
#   "global"         every paper Dimensions indexes, art history included. The reference
#                    is "all published research". Every biomedical field therefore comes
#                    out high, because the comparison includes a world of research UK
#                    Biobank was never going to be part of.
#
#   "universe"       the same, restricted to divisions named as relevant. This is the
#                    obvious fix and it is worth knowing that it does LESS than it looks:
#                    dropping categories rescales every field's index by ONE constant
#                    (UK Biobank's share of its mass inside the universe, over the
#                    database's). Measured on FOR 2014-2024 with divisions 31/32/42/52
#                    that constant is 1.86 — identical to four decimals for every field.
#                    Nothing is re-ranked. What moves is where parity sits, which is not
#                    nothing: it makes "1.0" mean "as concentrated as HEALTH research".
#
#   "within_parent"  each L4 field against its own L2 division, both arms:
#
#                        (UKBB's L4 / UKBB's L2) / (world's L4 / world's L2)
#
#                    This is the one that removes "UK Biobank is a biomedical resource"
#                    from the comparison, because it divides that out per field instead
#                    of once globally — and it genuinely re-ranks (Spearman 0.45 against
#                    global over the top 40 fields). Public Health reads 3.3x globally
#                    and 0.9x within Health Sciences: against all research it looks like
#                    a speciality, against health research UK Biobank is slightly under-
#                    represented in it. Needs a hierarchy, so FOR only.
#
# Flat systems (RCDC) have no parent to normalise within. Their equivalent of the
# universe restriction is RCDC_VIEW, which selects a slice of the vocabulary — and by
# the same argument that too rescales the index rather than re-ranking it.
ACTIVITY_BASES = ("global", "universe", "within_parent")

# ANZSRC 2020 divisions that a UK Biobank paper could plausibly compete in. Used only
# by the "universe" base; edit in the notebook rather than here if you disagree.
HEALTH_DIVISIONS = ("31", "32", "42", "52")


# =============================================================================
# WHICH SYSTEM, WHICH FILES
# =============================================================================
# "{p}" is the file prefix, so counts / totals / coverage all resolve through one
# entry. The FOR arm predates the --category flag, so its legacy filenames carry no
# category token; each arm lists its patterns most-specific first and the first
# pattern that matches anything wins.
CATEGORY_SPECS = {
    "for": {
        "label": "Fields of Research (ANZSRC 2020)",
        "unit": "field",
        "levels": ("L2", "L4"),
        "patterns": {"ukbb": ("{p}.for.ukbb*.parquet", "{p}.ukbb*.parquet"),
                     "background": ("{p}.for.background*.parquet",
                                    "{p}.background*.parquet")},
    },
    "rcdc": {
        "label": "NIH Research, Condition and Disease Categories",
        "unit": "topic",
        "levels": ("L1",),
        "patterns": {"ukbb": ("{p}.rcdc.ukbb*.parquet",),
                     "background": ("{p}.rcdc.background*.parquet",)},
    },
}


def arm_files(counts_dir, col_type: str, arm: str, prefix: str = "counts") -> List:
    """Partials for one arm: the first filename pattern that matches anything."""
    patterns = [p.format(p=prefix) for p in CATEGORY_SPECS[col_type]["patterns"][arm]]
    for pattern in patterns:
        hits = sorted(counts_dir.glob(pattern))
        if hits:
            return hits
    raise FileNotFoundError(f"no {col_type} '{arm}' {prefix} partials in {counts_dir} "
                            f"(looked for {', '.join(patterns)})")


def short(label: str, n: int = 42) -> str:
    """Truncate a long category name for an axis. RCDC spells some tags out in full
    ("Alzheimer's Disease including Alzheimer's Disease Related Dementias")."""
    label = str(label)
    return label if len(label) <= n else label[:n - 1].rstrip() + "…"


def pct(value, nd: int = 1) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{nd}f}%"


# =============================================================================
# LOADING THE TWO ARMS
# =============================================================================
def load_arm(counts_dir, col_type, arm, level, year_min, year_max, verbose=True):
    """(counts, label-quality) for one arm. Blank-named categories are dropped here
    and reported back so the caller's guard can act on them.

    EVERY measure column the partials carry is summed, not a fixed pair, so the
    citation weights ride along without this function knowing their names — and a run
    whose partials predate --weights still works."""
    files = arm_files(counts_dir, col_type, arm)
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raw = raw[(raw.level == level) & raw.year.between(year_min, year_max)].copy()
    raw["code"] = raw.code.fillna("").str.strip()
    blank = raw.code == ""
    measures = [c for c in raw.columns if c.startswith("n_")]

    counts = (raw[~blank].groupby(["year", "code", "for_label"], as_index=False)
              [measures].sum())
    quality = pd.DataFrame({"tagged": raw.groupby("year").n_papers.sum(),
                            "blank": raw[blank].groupby("year").n_papers.sum()})
    quality = quality.fillna({"blank": 0.0})
    quality["blank_share"] = quality.blank / quality.tagged.where(quality.tagged > 0)
    if verbose:
        unit = CATEGORY_SPECS[col_type]["unit"]
        print(f"  {arm:<10s} {len(files)} partial(s), {len(counts):,} rows, "
              f"{counts.code.nunique():,} {unit}s, "
              f"blank labels {quality.blank.sum() / max(quality.tagged.sum(), 1):.1%}")
    return counts, quality


def load_side_table(counts_dir, col_type, arm, prefix, keys, year_min, year_max):
    """Sum one of the side tables (totals / coverage) over its shards."""
    raw = pd.concat([pd.read_parquet(f)
                     for f in arm_files(counts_dir, col_type, arm, prefix)],
                    ignore_index=True)
    raw = raw[raw.year.between(year_min, year_max)]
    return raw.groupby(keys, as_index=False).sum(numeric_only=True)


# =============================================================================
# CITATION WINDOWS — which years can carry a citation claim
# =============================================================================
# Paper counts are complete the moment a paper is indexed; citation numbers are not,
# and two different things go wrong at the recent end:
#
#   COVERAGE  the column is simply absent. Dimensions computes field_citation_ratio
#             only once a publication year has settled, so FCR is missing outright for
#             the newest years — not low, ABSENT.
#   ACCRUAL   the column is there but the papers have not been cited yet. A 2024 paper
#             averages ~1 citation against ~17 for 2014, so a citation-mass comparison
#             in the last year or two measures indexing lag, not impact. FCR is
#             normalised by year and is immune; times_cited and the top-decile
#             indicator built from it are not.
#
# Coverage decides which years EXIST for a weight; `lag` then trims the raw-citation
# weights. Both are read off the totals tables, so a fresher snapshot moves the
# windows by itself.
def citation_windows(totals: Dict[str, pd.DataFrame], weights: Sequence[str],
                     ukbb_year: int, year_max: int, cov_min: float, lag: int,
                     verbose: bool = True):
    """-> (cite_years, cite_win, cite_ok, coverage_frames)."""
    def coverage(weight):
        out = {}
        for arm, t in totals.items():
            t = t.set_index("year")
            out[arm] = t[f"{weight}_docs_total"] / t.n_papers.where(t.n_papers > 0)
        return pd.DataFrame(out)

    covs = {w: coverage(w) for w in weights}
    cite_years = {}
    for w in weights:
        cov = covs[w]
        ok = cov[(cov.ukbb >= cov_min) & (cov.background >= cov_min)].index
        ok = [int(y) for y in ok if ukbb_year <= y <= year_max]
        if w != "n_fcr" and ok and lag:
            ok = [y for y in ok if y <= max(ok) - lag]
        cite_years[w] = ok

    common = (set.intersection(*[set(v) for v in cite_years.values()])
              if cite_years else set())
    cite_ok = bool(common)
    cite_win = (min(common), max(common)) if cite_ok else (ukbb_year, year_max)

    if verbose and weights:
        print(f"citation weight coverage (share of papers carrying a value):\n")
        for w in weights:
            cov = covs[w].reindex(range(ukbb_year, year_max + 1))
            lost = [str(y) for y in cov.index if y not in cite_years[w]]
            span = (f"{min(cite_years[w])}-{max(cite_years[w])}" if cite_years[w]
                    else "no usable year")
            print(f"  {w:8s} ukbb {cov.ukbb.min():.0%}-{cov.ukbb.max():.0%}   "
                  f"background {cov.background.min():.0%}-{cov.background.max():.0%}   "
                  f"usable {span}"
                  + (f"   dropped {', '.join(lost)}" if lost else ""))
        print(f"\nimpact window {cite_win[0]}-{cite_win[1]} — the years every weight "
              f"supports.\nThe volume charts keep the full window; only the citation "
              f"measures are cut back to it.")
        # The arms are measured to different depths — FCR especially — and that gap is
        # a property of the corpus, not of UK Biobank. It biases a SUM comparison (one
        # arm's total is more complete) but not a MEAN, which is why every impact
        # measure below is a mean over the papers that carried a value.
        print("\nmean coverage inside the impact window (ukbb vs background):")
        for w in weights:
            s = covs[w].loc[cite_win[0]:cite_win[1]].mean()
            flag = ("  <- arms differ by >10pp; read means, not sums"
                    if abs(s.ukbb - s.background) > 0.10 else "")
            print(f"  {w:8s} {s.ukbb:.0%} vs {s.background:.0%}{flag}")
    return cite_years, cite_win, cite_ok, covs


# =============================================================================
# THE BUILD — one call, everything the charts need
# =============================================================================
def build(counts_dir, col_type, *, level, rcdc_view="all", year_min, year_max,
          ukbb_year, top_n, weight, palette, blank_max=0.05, mask_bad_denom=True,
          skew_max_share=5.0, cite_headline="n_fcr", cite_min_docs=50,
          cite_cov_min=0.5, cite_lag=1, activity_base="global",
          universe_divisions=HEALTH_DIVISIONS, verbose=True) -> dict:
    """Load, guard and derive everything downstream of the count job.

    Returns a dict meant for `globals().update(...)`, containing:

      frames      ukbb, background, whole            per (year, code) counts
                  whole_ts, ukbb_ts, share_ts        year x top-category pivots
                  IMPACT                             per-category volume + impact
      vocabulary  TOP_CODES, TOP_LABELS, FIELD_COLORS, CODE_LABEL, label_of
      guards      GOOD_YEARS, BAD_YEARS, DENOM_OK, WHOLE_DB_OK, DENOM_NOTE, SKEWED
                  CITE_YEARS, CITE_WIN, CITE_OK
      measures    MEASURES, WEIGHTS_PRESENT, VALUE, ACTIVITY, OVERALL
      labels      SYSTEM, UNIT, LEVEL, VIEW_NOTE, FILE_TAG
      helpers     share_at, pct, short, impact_table, rci_timeseries
    """
    spec = CATEGORY_SPECS[col_type]
    unit = spec["unit"]
    value = weight
    if col_type == "rcdc" and rcdc_view == "disease":
        unit = "disease"

    if verbose:
        print(f"loading {col_type} partials from {counts_dir}")
    ukbb, q_ukbb = load_arm(counts_dir, col_type, "ukbb", level, year_min, year_max,
                            verbose)
    background, q_background = load_arm(counts_dir, col_type, "background", level,
                                        year_min, year_max, verbose)

    measures = [c for c in ukbb.columns if c.startswith("n_")]
    weights_present = [w for w in CITATION_WEIGHTS if w in measures]
    if verbose:
        print(f"  measures: {', '.join(measures)}")
        print(f"  citation weights: {', '.join(weights_present) or 'none — re-run the '
              'count job with --weights cit,fcr,top10'}")

    # -- RCDC only: keep one slice of the mixed vocabulary --------------------------
    # Applied to BOTH arms, so numerator and denominator describe the same slice.
    view_note, file_tag = "", ""
    if col_type == "rcdc" and rcdc_view != "all":
        from utils.shared_rcdc import is_disease
        want = rcdc_view == "disease"
        before = ukbb.code.nunique()
        keep = lambda f: f[f.code.map(is_disease) == want]
        dropped = 1 - keep(ukbb)[value].sum() / ukbb[value].sum()
        ukbb, background = keep(ukbb), keep(background)
        view_note = f", {rcdc_view.replace('_', '-')} tags only"
        file_tag = f"_{rcdc_view}"
        if verbose:
            print(f"\nRCDC_VIEW={rcdc_view}: kept {ukbb.code.nunique():,} of {before:,}"
                  f" tags ({dropped:.0%} of UK Biobank's tag mass dropped — the "
                  f"cross-cutting tags are\n  the most-applied ones, which is exactly "
                  f"why they hide the disease signal). Stop list:\n"
                  f"  src/utils/shared_rcdc.py")

    # -- which years have a whole-database denominator worth dividing by ------------
    usable = q_background.blank_share.fillna(1.0) <= blank_max
    good_years = [y for y in range(year_min, year_max + 1) if bool(usable.get(y, False))]
    bad_years = [y for y in range(year_min, year_max + 1) if y not in good_years]
    denom_ok = not bad_years
    whole_db_ok = len(good_years) >= 3
    denom_note = "" if denom_ok else f"  —  denominator unusable in {len(bad_years)} year(s)"
    whole_db_msg = (f"no usable whole-database denominator for '{col_type}'\n"
                    "the background arm's labels are blank — see the check above")

    if not denom_ok and verbose:
        worst = q_background.blank_share.reindex(bad_years)
        print(f"\n!! BACKGROUND LABELS ARE BLANK IN {len(bad_years)} OF "
              f"{year_max - year_min + 1} YEARS ({bad_years[0]}-{bad_years[-1]})")
        print(f"   {worst.min():.0%}-{worst.max():.0%} of the background arm's tagged "
              f"papers come back with an empty\n   '{col_type}' name in those years, so "
              f"every share derived from them is meaningless.\n   The fix is upstream — "
              f"flat-category parsing in the count job, then a re-run of\n   the "
              f"background arm; the UK Biobank arm itself is unaffected.")
        print(f"   Those years are {'MASKED OUT' if mask_bad_denom else 'STILL PLOTTED'}"
              f" (MASK_BAD_DENOM={mask_bad_denom}).")
        print(f"   Usable years: {good_years if good_years else 'none'}\n")

    # -- drop tags the two snapshots disagree about ---------------------------------
    # A vocabulary revision between two snapshots leaves a tag with a real numerator
    # and a near-empty denominator. Inert when both arms come from one snapshot, which
    # is how the count job is meant to be run — kept as a tripwire.
    skewed = []
    if skew_max_share is not None:
        _u = ukbb.groupby("code")[value].sum()
        _w = pd.concat([ukbb, background]).groupby("code")[value].sum()
        share = (_u / _w * 100).dropna()
        skewed = sorted(share[share > skew_max_share].index)
        if skewed:
            if verbose:
                print(f"\n!! {len(skewed)} tag(s) where UK Biobank exceeds "
                      f"{skew_max_share}% of the whole database — the two snapshots use "
                      f"different vocabulary here,\n   so the denominator is missing, "
                      f"not small. Dropped (set SKEW_MAX_SHARE=None to keep them):")
                for t in skewed:
                    print(f"     {short(t, 66):66s} UKBB {int(_u[t]):>7,} / whole "
                          f"{int(_w[t]):>9,}  = {share[t]:5.1f}%")
            ukbb = ukbb[~ukbb.code.isin(skewed)]
            background = background[~background.code.isin(skewed)]

    # -- the whole database = both arms summed --------------------------------------
    whole = (pd.concat([ukbb, background])
             .groupby(["year", "code", "for_label"], as_index=False)[measures].sum())
    if mask_bad_denom:
        whole = whole[whole.year.isin(good_years)]

    # code -> label for the WHOLE vocabulary. FOR codes are numeric ("3215"), so a
    # chart that falls back to the code prints a number at the reader; RCDC hides the
    # same bug because there the code IS the label.
    code_label = (pd.concat([ukbb, background]).drop_duplicates("code")
                  .set_index("code")["for_label"].to_dict())

    def label_of(code):
        return code_label.get(code) or code

    # UK Biobank's own top categories, ranked over the after-window where it has mass.
    # Ranked on the UK Biobank arm alone, so the masking above never touches this list.
    rank = (ukbb[ukbb.year >= ukbb_year].groupby(["code", "for_label"])[value].sum()
            .sort_values(ascending=False).head(top_n))
    top_codes = [c for c, _ in rank.index]
    top_labels = [l for _, l in rank.index]
    field_colors = dict(zip(top_labels, palette(top_n)))

    def _pivot(frame, val):
        return (frame[frame.code.isin(top_codes)]
                .pivot_table(index="year", columns="for_label", values=val,
                             aggfunc="sum")
                .reindex(columns=top_labels))

    whole_ts = _pivot(whole, value)
    ukbb_ts = _pivot(ukbb, value)
    share_ts = (ukbb_ts.reindex(whole_ts.index) / whole_ts * 100).clip(lower=0)

    # -- ACTIVITY INDEX — the scale-free companion to the share ---------------------
    # A raw share can only ever be tiny, so "0.5% of Obesity" reads as no impact when
    # it is 3x what a random slice of the database that size would produce:
    #
    #     AI = (UKBB's papers in X / all UKBB papers)
    #          / (the database's papers in X / all its papers)
    #
    # Computed on n_frac whatever WEIGHT says, because a ratio of shares is only
    # meaningful when both arms' papers carry the same total weight. What "all papers"
    # means is set by activity_base — see ACTIVITY_BASES at the top of this file, and
    # read it before quoting a number, because the three answer different questions.
    after_win = (ukbb_year, year_max)
    if activity_base not in ACTIVITY_BASES:
        raise ValueError(f"activity_base={activity_base!r}; choose from {ACTIVITY_BASES}")

    hierarchical = len(spec["levels"]) > 1
    parent_level = spec["levels"][0] if hierarchical and level != spec["levels"][0] else None
    if activity_base == "within_parent" and parent_level is None:
        raise ValueError(
            f"activity_base='within_parent' needs a hierarchy: '{col_type}' at level "
            f"'{level}' has no parent level to normalise within. Use 'global' or "
            f"'universe' (for RCDC, RCDC_VIEW already restricts the vocabulary).")

    # The parent level is loaded whenever one exists, not only when it is the chosen
    # base, so the report below can put the two side by side — the difference between
    # them is the substantive choice, and it should not be invisible.
    ukbb_p = whole_p = None
    if parent_level:
        _up, _ = load_arm(counts_dir, col_type, "ukbb", parent_level, year_min,
                          year_max, verbose=False)
        _bp, _ = load_arm(counts_dir, col_type, "background", parent_level, year_min,
                          year_max, verbose=False)
        ukbb_p = _up
        whole_p = (pd.concat([_up, _bp])
                   .groupby(["year", "code", "for_label"], as_index=False)
                   [[c for c in _up.columns if c.startswith("n_")]].sum())
        if mask_bad_denom:
            whole_p = whole_p[whole_p.year.isin(good_years)]

    def parent_of(code):
        """The 2-digit ANZSRC division a 4-digit field belongs to."""
        return str(code)[:2]

    def activity_index(win=None, base=None, col="n_frac"):
        """The activity index over `win`, measured against `base`."""
        win = win or after_win
        base = base or activity_base
        u = ukbb[ukbb.year.between(*win)].groupby("code")[col].sum()
        w = whole[whole.year.between(*win)].groupby("code")[col].sum()
        u = u.reindex(w.index).fillna(0.0)

        if base == "within_parent":
            up = ukbb_p[ukbb_p.year.between(*win)].groupby("code")[col].sum()
            wp = whole_p[whole_p.year.between(*win)].groupby("code")[col].sum()
            parents = [parent_of(c) for c in w.index]
            u_den = pd.Series(up.reindex(parents).to_numpy(), index=w.index)
            w_den = pd.Series(wp.reindex(parents).to_numpy(), index=w.index)
            ai = (u / u_den.where(u_den > 0)) / (w / w_den.where(w_den > 0))
        else:
            if base == "universe":
                if not hierarchical:
                    raise ValueError("activity_base='universe' needs the division codes "
                                     "of a hierarchy; use 'global' for a flat system")
                keep = [c for c in w.index if parent_of(c) in set(universe_divisions)]
                u, w = u[keep], w[keep]
            ai = (u / u.sum()) / (w / w.sum())
        return ai.replace([np.inf, -np.inf], np.nan)

    activity = activity_index()
    activity_global = activity_index(base="global")
    activity_within = activity_index(base="within_parent") if parent_level else None
    activity_label = {
        "global": "against all published research",
        "universe": f"against divisions {'/'.join(universe_divisions)}",
        "within_parent": f"within its own {parent_level} division",
    }[activity_base]

    if verbose and parent_level:
        # Print both, because a reader who has seen "51x" needs to know the same field
        # is 14x once the biomedical skew is divided out per field rather than globally.
        _rank = (ukbb[ukbb.year.between(*after_win)].groupby("code").n_frac.sum()
                 .sort_values(ascending=False).head(top_n))
        print(f"\nactivity index, {after_win[0]}-{after_win[1]} — base={activity_base} "
              f"({activity_label}):")
        print(f"  {'field':38s} {'global':>8s} {'within ' + parent_level:>10s}")
        for c in _rank.index:
            print(f"  {short(code_label.get(c, c), 38):38s} "
                  f"{activity_global.get(c, np.nan):8.1f} "
                  f"{activity_within.get(c, np.nan):10.1f}")

    def share_at(label, year=year_max):
        """UK Biobank's share of one category in one year; NaN when that year is masked."""
        if year not in share_ts.index or label not in share_ts.columns:
            return float("nan")
        return float(share_ts.loc[year, label])

    # -- citation windows and the impact table --------------------------------------
    totals = {arm: load_side_table(counts_dir, col_type, arm, "totals", ["year"],
                                   year_min, year_max)
              for arm in ("ukbb", "background")} if weights_present else {}
    if weights_present:
        if verbose:
            print()
        cite_years, cite_win, cite_ok, _ = citation_windows(
            totals, weights_present, ukbb_year, year_max, cite_cov_min, cite_lag,
            verbose)
    else:
        cite_years, cite_win, cite_ok = {}, after_win, False

    def impact_table(win=None, min_docs=None):
        """Per-category volume and impact over `win`. Thin categories come back NaN.

        rci_<w>  RELATIVE CITATION IMPACT — the mean weight of UK Biobank's papers in
                 X over the mean weight of the whole database's papers in X. 1.0 =
                 cited exactly like everyone else's work in the same category. A mean
                 over the papers that carried a value, which is what makes it survive
                 the two arms being measured to different depths.
        act_<w>  CITATION ACTIVITY INDEX — the activity index with citation mass in
                 place of paper counts. Folds volume and impact together, so read it
                 beside activity_x, not instead of it.
        activity_x  the volume activity index RECOMPUTED on `win`, so the columns
                 describe the same years.
        """
        win = win or cite_win
        min_docs = cite_min_docs if min_docs is None else min_docs
        u = ukbb[ukbb.year.between(*win)].groupby("code")[measures].sum()
        w = whole[whole.year.between(*win)].groupby("code")[measures].sum()
        u = u.reindex(w.index).fillna(0.0)

        out = pd.DataFrame(index=w.index)
        out["ukbb_papers"] = u.n_papers
        # Same base as everywhere else, recomputed on the impact window so the volume
        # and impact columns describe the same years.
        out["activity_x"] = activity_index(win).reindex(w.index)
        for wt in weights_present:
            key = wt[2:]
            u_docs, w_docs = u[f"{wt}_docs_frac"], w[f"{wt}_docs_frac"]
            u_mean = u[f"{wt}_frac"] / u_docs.where(u_docs > 0)
            w_mean = w[f"{wt}_frac"] / w_docs.where(w_docs > 0)
            out[f"rci_{key}"] = (u_mean / w_mean).where(u_docs >= min_docs)
            out[f"act_{key}"] = ((u[f"{wt}_frac"] / u[f"{wt}_frac"].sum())
                                 / (w[f"{wt}_frac"] / w[f"{wt}_frac"].sum()))
            out[f"docs_{key}"] = u_docs
        return out.replace([np.inf, -np.inf], np.nan)

    def rci_timeseries(weight, min_docs=20):
        """Per-year relative citation impact, category by category."""
        def piv(frame, col):
            return frame.pivot_table(index="year", columns="code", values=col,
                                     aggfunc="sum")
        u_w, u_d = piv(ukbb, f"{weight}_frac"), piv(ukbb, f"{weight}_docs_frac")
        w_w, w_d = piv(whole, f"{weight}_frac"), piv(whole, f"{weight}_docs_frac")
        u_w, u_d = u_w.reindex_like(w_w), u_d.reindex_like(w_d)
        ts = (u_w / u_d.where(u_d > 0)) / (w_w / w_d.where(w_d > 0))
        ts = ts.where(u_d >= min_docs).replace([np.inf, -np.inf], np.nan)
        return ts.loc[ts.index.isin(cite_years.get(weight, ts.index))]

    def share_timeseries(column, codes=None, denominator=None):
        """UK Biobank's share (%) of the whole database, per year, per category, for any
        measure column: n_papers, n_cit, n_top10, ...

        `denominator` names the whole-database column to divide by, defaulting to the
        same one. Passing a different one asks a different question:

            share_timeseries("n_top10")                      UK Biobank's share of the
                                                             field's TOP-DECILE papers
            share_timeseries("n_top10", denominator="n_papers")
                                                             UK Biobank's top-decile
                                                             papers as a share of ALL
                                                             the field's papers

        The second is a piece of the first's numerator measured against the whole field,
        which is what lets a chart decompose UK Biobank's footprint into the part that
        landed in the top decile and the part that did not.
        """
        codes = top_codes if codes is None else codes
        denominator = denominator or column
        def piv(frame, col):
            return (frame[frame.code.isin(codes)]
                    .pivot_table(index="year", columns="code", values=col,
                                 aggfunc="sum").reindex(columns=codes))
        u, w = piv(ukbb, column), piv(whole, denominator)
        return (u.reindex(w.index) / w.where(w > 0) * 100).clip(lower=0)

    impact = impact_table() if cite_ok else None
    overall = {}
    if cite_ok:
        _u = ukbb[ukbb.year.between(*cite_win)]
        _w = whole[whole.year.between(*cite_win)]
        for wt in weights_present:
            overall[wt[2:]] = ((_u[f"{wt}_frac"].sum() / _u[f"{wt}_docs_frac"].sum())
                               / (_w[f"{wt}_frac"].sum() / _w[f"{wt}_docs_frac"].sum()))

    return dict(
        ukbb=ukbb, background=background, whole=whole,
        whole_ts=whole_ts, ukbb_ts=ukbb_ts, share_ts=share_ts,
        q_ukbb=q_ukbb, q_background=q_background,
        TOP_CODES=top_codes, TOP_LABELS=top_labels, FIELD_COLORS=field_colors,
        CODE_LABEL=code_label, label_of=label_of,
        GOOD_YEARS=good_years, BAD_YEARS=bad_years, DENOM_OK=denom_ok,
        WHOLE_DB_OK=whole_db_ok, DENOM_NOTE=denom_note, WHOLE_DB_MSG=whole_db_msg,
        SKEWED=skewed,
        MEASURES=measures, WEIGHTS_PRESENT=weights_present, WEIGHT_NAMES=WEIGHT_NAMES,
        VALUE=value, ACTIVITY=activity, AFTER_WIN=after_win,
        ACTIVITY_GLOBAL=activity_global, ACTIVITY_WITHIN=activity_within,
        ACTIVITY_BASE=activity_base, ACTIVITY_LABEL=activity_label,
        PARENT_LEVEL=parent_level, activity_index=activity_index,
        ukbb_parent=ukbb_p, whole_parent=whole_p,
        CITE_YEARS=cite_years, CITE_WIN=cite_win, CITE_OK=cite_ok,
        IMPACT=impact, OVERALL=overall,
        SYSTEM=spec["label"], UNIT=unit, LEVEL=level,
        VIEW_NOTE=view_note, FILE_TAG=file_tag,
        share_at=share_at, pct=pct, short=short,
        impact_table=impact_table, rci_timeseries=rci_timeseries,
        share_timeseries=share_timeseries,
    )


def denominator_flagger(style, whole_db_ok, message):
    """A `flag_denominator(target)` bound to one run's guards — stamps a figure or
    axes that has no whole-database series left to draw."""
    def flag_denominator(target):
        if whole_db_ok:
            return
        kwargs = dict(ha="center", va="center", fontsize=style["label_fs"],
                      color=style["colors"][0],
                      bbox=dict(boxstyle="round", fc="white", ec=style["colors"][0]))
        if hasattr(target, "transAxes"):
            target.text(0.5, 0.5, message, transform=target.transAxes, **kwargs)
        else:
            target.text(0.5, 0.5, message, **kwargs)
    return flag_denominator