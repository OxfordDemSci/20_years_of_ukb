"""Assembled panels for analysis 03 — the academic impact figure family.

Part IV of `03_academic_impact.ipynb` owns the selection (which chart is main-paper,
which is SI); this module owns the two halves that selection needs and that a notebook
cell is the wrong place for:

    build_panel_data()      every aggregate the panels draw, computed once from the two
                            arms and the author tables, returned as a dict
    draw_<name>(ax, D)      one chart into one caller-supplied axes, no figure of its own
    block_<name>(fig, spec) a small-multiple BLOCK into one gridspec cell (see below)
    figure_main(D)          the main-paper panel;      figure_si_<section>(D) the SI ones

**Why re-drawn rather than pasted.** Same rule as analysis 04 (D28). The two source
notebooks draw at their own figure sizes and, in the author arm's case, build a bespoke
`UKB_SEQ` blue ramp of their own; laid side by side those differences read as figures
stapled together. Every function here draws into an axes belonging to a figure the caller
has already sized, under the one `03_academic_impact_panels` style section.

**The palette.** Everything is drawn from `shared_style.PALETTE_COLORS` — the same seven
named colours analysis 04 is drawn in (D29), so the two figure families belong to one
paper. Where a chart needs more than seven categories or an ordered ramp, it is derived
from those anchors by `_shade()` rather than reaching for a new hue. Nothing
here reads `STYLE["colors"]` by integer index.

**What it reads.** Nothing here re-derives what a source notebook derives:

    both count arms   data/analysis/academic_impact/…  via AI.build (the API pathway)
    the cut-offs      field_thresholds.for.csv + api_whole.for.parquet
    the author arm    output/tables/03_academic_impact/author_{paper,summary}_*.csv

The author tables are read rather than rebuilt on purpose: rebuilding them means
re-exploding 267,571 author slots out of the 26,109-row corpus parquet, which is minutes
of work to reproduce a file Part II of `03_academic_impact.ipynb` has already written
and which is itself the artefact that notebook is responsible for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import data_analysis_03_academic_impact_analysis as AI
from utils import shared_paths as P
from .shared_analysis_window import ANALYSIS_START_YEAR, ANALYSIS_END_YEAR, filter_analysis_window

# =============================================================================
# 1. Analysis parameters — the FOR notebook's, verbatim
# =============================================================================
# These are copied from Part I, §2 of 03_academic_impact.ipynb rather than
# imported, because a notebook cannot be imported. THEY MUST NOT DRIFT: every one of them
# is argued in that notebook's §2, and `assert_parameters_match()` below is the tripwire.
# The window itself is D19 and is shared with the author arm.

COL_TYPE = "for"
FOR_LEVEL = "L4"
RCDC_VIEW = "all"

#: D19 — one window for every UK Biobank measure in analysis 03.
#:
#: 2015, NOT the project-wide `ANALYSIS_START_YEAR` of 2013. The shared constant is the
#: floor the corpus is cut at; D19's floor is the one this analysis can MEASURE at, and
#: three independent limits put it two years higher: UK Biobank published 27 papers in
#: 2014 against 63 in 2015, `percentiles --min-ukbb 10` measured no 2014 cell in any
#: field so `field_thresholds.csv` has no 2014 row at all, and `CITE_MIN_DOCS` would
#: drop 2014 regardless. D19 considered and rejected 2014 for a rounder window. Taking
#: the shared constant here silently widened the window to 2013 and put this module out
#: of step with the retained author tables, which are built on 2015–2025.
ANALYSIS_MIN, ANALYSIS_MAX = 2015, ANALYSIS_END_YEAR
UKBB_YEAR = ANALYSIS_MIN

#: How far back the BACKGROUND arm is carried. Not part of the analysis window: it exists
#: so a field's twenty-year trend has something to be a trend against.
#:
#: CURRENTLY INERT, and left at D19's value rather than rewritten to what it delivers.
#: `AI.load_arm` puts every partial through `filter_analysis_window` before applying
#: `year_min`, and that filter floors at the project-wide 2013 — so the 1,737 pre-2013
#: rows that `api_whole.for.parquet` really holds are dropped and a build at 2004 is
#: byte-identical to one at 2013. Nothing on the assembled pages reads them: the two
#: charts that wanted the twenty-year run were cut from the reporting set by D31. Relax
#: the shared filter for the background arm before relying on this number again.
YEAR_MIN, YEAR_MAX = 2004, ANALYSIS_MAX

TOP_N = 8                      # UK Biobank's own top fields, followed through both arms
WEIGHT = "n_papers"
CITE_HEADLINE = "n_mncs"       # D9/D10 — the measured headline, not the n_top10p proxy
CITE_SHARE_METRIC = "n_top10f"
CITE_MIN_DOCS = 50
CITE_COV_MIN = 0.5
CITE_LAG = 0                   # safe because every weight carried is year-normalised
ACTIVITY_BASE = "within_parent"
UNIVERSE_DIVISIONS = ("31", "32", "42", "52")
BLANK_MAX = 0.05
MASK_BAD_DENOM = True
SKEW_MAX_SHARE = 5.0

#: Growth windows for the SI growth figure (chart 11's).
GROW_NOW = (YEAR_MAX - 4, YEAR_MAX)
GROW_PREV = (YEAR_MAX - 8, YEAR_MAX - 4)
GROW_TOP_M = 12
GROW_MIN_BASE, GROW_MIN_LAST = 5, 20

#: Author entry cohorts, cut on the analysis window — the author notebook's `COHORT_BINS`.
#: The first bin opens at ANALYSIS_MIN, and its LABEL is built from the same number: the
#: retained author tables carry `entry_cohort` as a string, so a label written out by hand
#: here is a join key, and one that says 2013 while the tables say 2015 fails as a missing
#: column rather than as a wrong window.
COHORT_BINS = [
    (ANALYSIS_MIN, 2017, f"{ANALYSIS_MIN}–2017\nfoundational entrants"),
    (2018, 2020, "2018–2020\nexpansion entrants"),
    (2021, 2023, "2021–2023\nconsolidation entrants"),
    (2024, 2025, "2024–2025\nrecent entrants"),
]
COHORT_ORDER = [label for _, _, label in COHORT_BINS]

#: How many categories each panel ranks. The main figure's numbers are LOWER than the
#: source analyses': a lollipop of 20 fields and a scatter of 25 labelled points are
#: legible at full figure width and are not at a third or two thirds of it, which is what
#: a panel of the main page gets. The activity lollipop takes the narrowest slot on the
#: page — one column of three — so it is cut hardest.
ACT_TOP_M_MAIN, ACT_TOP_M_SI = 10, 20
MAP_M_MAIN, MAP_M_SI = 16, 25

#: How many fields the top-decile share panel draws on the MAIN page. It is read there
#: off a legend rather than off direct labels, so the limit is what a legend can carry in
#: one column of three without covering the lines it names: five leaves the bottom of the
#: panel clear for it. The SI draws the same chart for the impact map's own 25 fields,
#: direct-labelled at full width.
DEC_M_MAIN = 5

# =============================================================================
# 2. Colour — everything from the project palette
# =============================================================================
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb   # noqa: E402

from utils.shared_style import (                                        # noqa: E402
    PALETTE_COLORS, academic_impact_colormap, facet_ylabel, grid_on, panel_label, savefig,
    year_ticks,
)


def _shade(name: str, factor: float) -> str:
    """A palette colour darkened (`factor` < 1) or lightened (> 1) toward white.

    The palette has seven colours and this family needs eight distinguishable field
    lines, plus a four-step ordered cohort ramp. Rather than introduce a hue the rest of
    the paper does not use, the extra colours are stated shades of an anchor that is
    already in it — so a reader who has learned the palette still recognises them.
    """
    r, g, b = to_rgb(PALETTE_COLORS[name])
    if factor <= 1:
        return to_hex((r * factor, g * factor, b * factor))
    t = min(factor - 1.0, 1.0)
    return to_hex((r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t))


#: Eight categorical colours for the eight top fields, in assignment order.
#:
#: Two of the palette's seven cannot take this job. `cream` (#FEE7BA) is a fill colour —
#: as a 2pt line on white it is barely visible — and `blue` (#74ADD1) is within a few
#: percent of `light_blue` (#75BBD4), so a reader cannot tell two fields apart by them.
#: The five that can are used first, then three stated shades.
FIELD_CYCLE = [
    PALETTE_COLORS["red"],            # 1st field  — Epidemiology, the family's lead
    PALETTE_COLORS["navy"],
    PALETTE_COLORS["green"],
    PALETTE_COLORS["steel_blue"],
    PALETTE_COLORS["light_blue"],
    _shade("red", 0.62),              # deep brick — distinct from red at a glance
    _shade("green", 0.58),            # deep teal
    _shade("cream", 0.72),            # deep gold — cream itself is too pale for a line
]

#: The two-sided encoding every ratio panel uses: above the reference / below it. Red for
#: over-represented and steel blue for under is the same pairing analysis 04 uses for a
#: value against its benchmark.
C_ABOVE, C_BELOW = PALETTE_COLORS["red"], PALETTE_COLORS["steel_blue"]

#: Neutral ink for reference lines, parity rules and the "not one of the top N" dots.
C_INK = "#3D3D3D"
C_MUTED = "#BDBDBD"

#: Fields whose direct label on the impact map is drawn to the LEFT of its point.
#:
#: Every other label goes to the right, which is where a reader looks for it. Clinical
#: Sciences cannot: it sits just left of the parity rule with Public Health a few percent
#: to its right at almost the same impact, so its label — raised by `_nudge` to within a
#: hair of Public Health's row — runs straight over Public Health's dot and its leader.
#: There is nothing to its left, so that is where it goes. Keyed on the full label, so it
#: applies at both the main figure's 16 fields and the SI's 25.
MAP_LABEL_LEFT = {"Clinical Sciences"}


def field_palette(n: int) -> list:
    """`AI.build`'s `palette` argument: n categorical colours from FIELD_CYCLE."""
    if n <= len(FIELD_CYCLE):
        return FIELD_CYCLE[:n]
    ramp = LinearSegmentedColormap.from_list("field", FIELD_CYCLE, N=256)
    return FIELD_CYCLE + [ramp(i / (n - 1)) for i in range(len(FIELD_CYCLE), n)]




# =============================================================================
# 3. The data layer
# =============================================================================
def _cuts_table(counts_dir, level: str, say) -> pd.DataFrame | None:
    """The measured citation cut-offs per (year, field) — §6.1 of the FOR notebook.

    `field_thresholds` carries one row per (year, field, PERCENTILE): the top decile and
    the median are two measurements of the same cell, so each is picked out by its own
    percentile rather than merged blind. The MEASURED median is used wherever the
    percentile pass reached the cell; only otherwise the facet's interpolated
    `citations_median`, which is fractional and has no count of the field behind it.
    """
    try:
        thr = pd.read_csv(counts_dir / f"field_thresholds.{COL_TYPE}.csv",
                          dtype={"code": str})
        thr = filter_analysis_window(thr)
        if "percentile" not in thr.columns:
            thr["percentile"] = 10.0
        thr = thr[thr.level == level]
        med = pd.read_parquet(
            counts_dir / f"api_whole.{COL_TYPE}.parquet",
            columns=["year", "level", "code", "mean_cit", "api_citations_median"])
        med = filter_analysis_window(med)
        p10 = (thr.loc[thr.percentile == 10, ["year", "code", "threshold"]]
               .rename(columns={"threshold": "thr10"}))
        cols50 = ["year", "code", "threshold"] + (
            ["achieved_pct"] if "achieved_pct" in thr.columns else [])
        p50 = (thr.loc[thr.percentile == 50, cols50]
               .rename(columns={"threshold": "med_meas", "achieved_pct": "med_pct"}))
        if "med_pct" not in p50.columns:
            p50["med_pct"] = np.nan
        cuts = (med[med.level == level]
                .merge(p10, on=["year", "code"], how="left")
                .merge(p50, on=["year", "code"], how="left")
                .astype({"year": int}))
        cuts["median_measured"] = cuts.med_meas.notna()
        cuts["median"] = cuts.med_meas.fillna(cuts.api_citations_median)
        cuts = cuts.set_index(["year", "code"])[
            ["thr10", "median", "mean_cit", "med_pct", "median_measured"]]
        say(f"  cut-offs: top decile in {int(cuts.thr10.notna().sum())} cell(s), "
            f"median in {int(cuts.median_measured.sum())}")
        return cuts
    except (FileNotFoundError, KeyError) as exc:
        say(f"  !! no cut-off tables in {counts_dir} ({exc}) — panel D will be skipped")
        return None


def _author_arm(say) -> dict:
    """The author arm, read from the tables Part II of the consolidated notebook wrote.

    Two files, both under `output/tables/03_academic_impact/`. The paper-level one is
    62 MB and 266k rows; six columns of it answer everything the cohort panel asks, so it
    is read six columns wide rather than twenty-four.
    """
    table_dir = P.OUTPUT_TABLES / "03_academic_impact"
    paper_csv = table_dir / "author_paper_fractional_credit_table.csv"
    summary_csv = table_dir / "author_summary_with_impact_metrics.csv"
    missing = [p.name for p in (paper_csv, summary_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"the author arm needs {', '.join(missing)} under {P.raw_path(table_dir)}. "
            f"Run 03_academic_impact.ipynb (Part II) first — it writes them, and this "
            f"module reads rather than rebuilds them (267,571 author slots).")

    AI.require_author_impact_window(table_dir, ANALYSIS_MIN, ANALYSIS_MAX)
    ap = pd.read_csv(paper_csv, usecols=[
        "author_key", "year", "entry_cohort", "fractional_paper_credit",
        "fractional_citation_credit", "fractional_top_decile_credit",
        "leadership_author_credit"])
    summary = pd.read_csv(summary_csv)
    eligible_ap = filter_analysis_window(ap)
    eligible_summary = filter_analysis_window(summary, year_col="last_year")
    if len(eligible_ap) != len(ap) or len(eligible_summary) != len(summary):
        raise ValueError(
            "The retained author tables include papers outside the publication cutoff. "
            "Rerun 03_academic_impact.ipynb (Part II) to rebuild author metrics.")
    ap = eligible_ap[eligible_ap["year"].ge(ANALYSIS_MIN)].copy()
    say(f"  author arm: {len(ap):,} author-paper rows, {len(summary):,} distinct authors")

    ap = ap[ap.entry_cohort.isin(COHORT_ORDER)].copy()
    by_cohort = ap.groupby("entry_cohort", observed=True)
    contrib = pd.DataFrame({
        "Unique authors": by_cohort["author_key"].nunique(),
        "Fractional paper credit": by_cohort["fractional_paper_credit"].sum(),
        "Fractional citation credit": by_cohort["fractional_citation_credit"].sum(),
        "Top-decile paper credit": by_cohort["fractional_top_decile_credit"].sum(),
        "First/last-author credit": by_cohort["leadership_author_credit"].sum(),
    }).T[COHORT_ORDER].fillna(0)

    return {
        "contrib": contrib,
        "contrib_share": contrib.div(contrib.sum(axis=1), axis=0),
        "cohort_n": by_cohort["author_key"].nunique().reindex(COHORT_ORDER).fillna(0).astype(int),
        "summary": summary,
    }


def _growth_table(ukbb, whole, activity, code_label, say) -> tuple:
    """Chart 11's per-field growth rates, and UK Biobank's own pace over the same window."""
    uts = ukbb.pivot_table(index="year", columns="code", values="n_papers", aggfunc="sum")
    wts = whole.pivot_table(index="year", columns="code", values="n_papers",
                            aggfunc="sum").reindex(columns=uts.columns)

    def cagr(frame, win):
        a, b = win
        if a not in frame.index or b not in frame.index:
            return pd.Series(np.nan, index=frame.columns)
        lo, hi = frame.loc[a].replace(0, np.nan), frame.loc[b].replace(0, np.nan)
        return ((hi / lo) ** (1.0 / (b - a)) - 1.0) * 100

    # Speed is the only ranking criterion, but a rate needs a denominator that is not
    # noise: 2 papers -> 8 is 400% and means nothing.
    ok = ((uts.loc[GROW_NOW[0]].fillna(0) >= GROW_MIN_BASE)
          & (uts.loc[GROW_NOW[1]].fillna(0) >= GROW_MIN_LAST))
    codes = uts.columns
    growth = (pd.DataFrame({
        "ukbb_now": cagr(uts, GROW_NOW), "ukbb_prev": cagr(uts, GROW_PREV),
        "world_now": cagr(wts, GROW_NOW), "world_prev": cagr(wts, GROW_PREV),
        "ukbb_papers": uts.loc[GROW_NOW[1]], "world_papers": wts.loc[GROW_NOW[1]],
        "activity_x": pd.Series(activity).reindex(codes)}, index=codes)
        [ok.reindex(codes).fillna(False)]
        .assign(accel=lambda d: d.ukbb_now - d.ukbb_prev)
        .sort_values("ukbb_now", ascending=False))
    growth["label"] = [code_label.get(c, c) for c in growth.index]

    totals = ukbb.groupby("year").n_papers.sum()
    pace = float((totals[GROW_NOW[1]] / totals[GROW_NOW[0]])
                 ** (1 / (GROW_NOW[1] - GROW_NOW[0])) - 1) * 100
    say(f"  growth: {len(growth)} field(s) clear the floors; UK Biobank overall "
        f"{pace:.0f}%/yr over {GROW_NOW[0]}-{GROW_NOW[1]}")
    return growth, totals, uts, pace


def build_panel_data(verbose: bool = True) -> dict:
    """Every aggregate the panels draw, computed once.

    Roughly 20 seconds, nearly all of it `AI.build` loading both count arms and the
    62 MB author-paper table. Everything the panels draw comes out of this one dict, so a
    figure cannot quietly read a different vintage of the data than its neighbour.
    """
    def say(msg):
        if verbose:
            print(msg)

    counts_dir = AI.resolve_counts_dir(P.FOR_COUNTS_API)
    say(f"loading both arms from {P.raw_path(counts_dir)} (API pathway) …")
    ctx = AI.build(
        counts_dir, COL_TYPE,
        level=FOR_LEVEL, rcdc_view=RCDC_VIEW,
        year_min=YEAR_MIN, year_max=YEAR_MAX, ukbb_year=UKBB_YEAR,
        top_n=TOP_N, weight=WEIGHT, palette=field_palette,
        blank_max=BLANK_MAX, mask_bad_denom=MASK_BAD_DENOM,
        skew_max_share=SKEW_MAX_SHARE,
        cite_headline=CITE_HEADLINE, cite_min_docs=CITE_MIN_DOCS,
        cite_cov_min=CITE_COV_MIN, cite_lag=CITE_LAG,
        activity_base=ACTIVITY_BASE, universe_divisions=UNIVERSE_DIVISIONS,
        verbose=verbose,
    )

    D = dict(ctx)
    D["counts_dir"] = counts_dir
    D["CUTS"] = _cuts_table(counts_dir, ctx["LEVEL"], say)
    D["author"] = _author_arm(say)
    growth, totals, uts, pace = _growth_table(
        ctx["ukbb"], ctx["whole"], ctx["ACTIVITY"], ctx["CODE_LABEL"], say)
    D["GROWTH"], D["ukbb_totals"], D["ukbb_ts_all"], D["UKBB_PACE"] = growth, totals, uts, pace

    # The overall share the whole family is read against: UK Biobank is a fraction of a
    # percent of the literature, so a SHARE is small everywhere by construction and the
    # activity index is the measure that survives it.
    frac = ctx["FRAC_COL"]
    win = (ANALYSIS_MIN, ANALYSIS_MAX)
    D["ukbb_overall_share"] = 100 * (
        ctx["ukbb"][ctx["ukbb"].year.between(*win)][frac].sum()
        / ctx["whole"][ctx["whole"].year.between(*win)][frac].sum())
    say(f"\nUK Biobank is {D['ukbb_overall_share']:.3f}% of the whole database over "
        f"{win[0]}-{win[1]} ({frac}).")
    say("done.")
    return D


def assert_parameters_match(D) -> None:
    """Fail loudly if this module's parameters have drifted from the loaded run.

    §1's constants are copied from the FOR notebook rather than imported, so nothing
    stops the two diverging. This is the tripwire: it compares what the build actually
    used against what the ledger in doc/STATE.md records, and raises rather than drawing
    a figure whose caption would be wrong.
    """
    problems = []
    if D["LEVEL"] != FOR_LEVEL:
        problems.append(f"level {D['LEVEL']} != {FOR_LEVEL}")
    if len(D["TOP_CODES"]) != TOP_N:
        problems.append(f"{len(D['TOP_CODES'])} top codes != TOP_N={TOP_N}")
    if not D["CITE_OK"]:
        problems.append("no usable citation window — panels B, C and D have nothing to draw")
    for col, name in ((D.get("TOP10_COL"), "top-decile"), (D.get("TOP50_COL"), "median")):
        if col is None:
            problems.append(f"no {name} column in these partials — panel D needs both")
    if problems:
        raise ValueError("the loaded run does not match this module's parameters:\n  "
                         + "\n  ".join(problems))


# =============================================================================
# 4. Drawing primitives
# =============================================================================
# Every one of these takes an `ax` (or, for a block, a figure and a gridspec cell) and
# returns it. None creates a figure, sets a figure size, or calls savefig: the assembler
# owns the page, so a panel cannot quietly impose its own geometry on the one it shares.

from contextlib import contextmanager                                   # noqa: E402

import matplotlib.pyplot as plt                                         # noqa: E402
import matplotlib.ticker as mticker                                     # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402
from matplotlib.patches import Patch                                    # noqa: E402


def _style():
    """The active style, so a draw function can read type sizes without an argument."""
    from utils import shared_style
    return shared_style._resolve(None)


@contextmanager
def _font_scale(factor):
    """Draw the enclosed figure with every font size multiplied by `factor`.

    Same mechanism as the 04 family: the main figure is read at figure width in a paper
    and the SI panels full page on a screen, so they want different type. One number per
    figure in `fs_scale` rather than a second style section that would drift from this
    one. Restores the original style on the way out, including after an exception.
    """
    from utils import shared_style
    original = _style()
    if not factor or factor == 1:
        yield original
        return
    scaled = dict(original)
    for key in ("title_fs", "label_fs", "tick_fs", "annot_fs", "legend_fs",
                "body_fs", "panel_label_fs"):
        if key in scaled:
            scaled[key] = scaled[key] * factor
    shared_style.use_style(scaled)
    shared_style.apply_style(scaled)
    try:
        yield scaled
    finally:
        shared_style.use_style(original)
        shared_style.apply_style(original)


def _fs_scale(figure: str) -> float:
    """Type scale for one figure, from `fs_scale` in the style section (default 1.0)."""
    return float((_style().get("fs_scale") or {}).get(figure, 1.0))


def _times_fmt(v, _=None):
    """A ratio as a reader would say it: 12x, 1x, 1/4."""
    return f"{v:g}x" if v >= 1 else f"1/{1 / v:g}"


def _pct_fmt(v, _=None):
    return f"{v:g}%" if v >= 1 else f"{v:.2g}%"


def _pct(v):
    """A share as a direct label: one decimal at 1% and above, two figures below it.

    "0.2%" and "0.15%" are different fields, and one decimal prints both as 0.2%."""
    return f"{v:.1f}%" if v >= 1 else f"{v:.2g}%"


def _nudge(items, gap):
    """Space direct labels apart in log space, smallest first, preserving order.

    Several series in these charts finish within a few percent of each other; drawn at
    their true y they overprint. Each label is pushed to at least `gap` above the last,
    and the caller draws a leader line wherever the push was more than a hair.
    """
    out, prev = [], None
    for value, *rest in sorted(items):
        ly = np.log10(value) if prev is None else max(np.log10(value), prev + gap)
        prev = ly
        out.append((value, 10 ** ly, *rest))
    return out


def _field_color(D, label):
    return D["FIELD_COLORS"][label]


# ------------------------------------------------------------------ panel A
def draw_activity_index(ax, D, top_m: int = ACT_TOP_M_MAIN):
    """Chart 4, left panel — the activity index against parity, ranked.

    Chart 2's share is bounded by UK Biobank's size, so every field looks tiny there and
    the ranking mostly reproduces "which fields are big". This divides that size out:
    1.0 = UK Biobank works on the field exactly as much as the literature does. The axis
    is log, so 2x and 1/2 sit the same distance from parity, and dot area is UK Biobank
    papers — a large dot far right is both a big and a distinctive body of work.
    """
    ukbb, VALUE, ACTIVITY = D["ukbb"], D["VALUE"], D["ACTIVITY"]
    st = _style()

    rank = (ukbb[ukbb.year >= UKBB_YEAR].groupby(["code", "for_label"])[VALUE].sum()
            .sort_values(ascending=False).head(top_m))
    tbl = (pd.DataFrame({"label": [lab for _, lab in rank.index],
                         "ukbb": rank.to_numpy(),
                         "activity": [ACTIVITY.get(c, np.nan) for c, _ in rank.index]})
           .dropna(subset=["activity"]).sort_values("activity").reset_index(drop=True))

    ypos = np.arange(len(tbl))
    colors = [C_ABOVE if a >= 1 else C_BELOW for a in tbl.activity]
    sizes = 24 + 260 * np.sqrt(tbl.ukbb / tbl.ukbb.max())
    ax.hlines(ypos, 1.0, tbl.activity, color=colors, lw=1.6, alpha=0.55)
    ax.scatter(tbl.activity, ypos, s=sizes, color=colors, zorder=3,
               edgecolor="white", linewidth=0.8)
    ax.axvline(1.0, color=C_INK, lw=1.2)
    for y, a in zip(ypos, tbl.activity):
        left = a < 1.0
        ax.annotate(f"{a:.1f}x", (a, y), fontsize=st["annot_fs"],
                    color=C_BELOW if left else C_ABOVE, va="center",
                    ha="right" if left else "left",
                    xytext=(-9 if left else 9, 0), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_times_fmt))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(tbl.activity.min() / 3.0, tbl.activity.max() * 3.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([AI.short(lab, 34) for lab in tbl.label], fontsize=st["annot_fs"])
    ax.set_ylim(-0.9, len(tbl) - 0.1)
    ax.set_xlabel(f"activity index, log scale\n({D['ACTIVITY_LABEL']})")
    grid_on(ax, axis="x")
    return ax


# ------------------------------------------------------------------ panel B
def map_fields(D, top_m: int):
    """The fields the impact map draws at `top_m`: the largest with both axes measured.

    A helper rather than three lines inside the map, because SI 1's second panel follows
    the same fields through time and the two panels of one figure must not disagree about
    which fields they are.
    """
    rci = f"rci_{D['CITE_HEADLINE'][2:]}"
    return (D["IMPACT"].dropna(subset=[rci, "activity_x"])
            .nlargest(top_m, "ukbb_papers"))


def draw_impact_map(ax, D, top_m: int = MAP_M_MAIN, label_all: bool = False):
    """Chart 5 — how much UK Biobank publishes against how well that work lands.

    Two independent questions, so they get an axis each: horizontal is panel A's volume
    activity index, vertical the relative citation impact against the world's papers in
    the same field. Both are ratios and both are log, so "twice" and "half" sit the same
    distance from the parity lines.

    THE HORIZONTAL REFERENCE IS UK BIOBANK'S OWN AVERAGE, NOT 1.0. Against the world
    essentially every field is above parity — a resource whose papers average 2.5x world
    is not told apart by that line — so what is drawn solid is the field's own standard
    to beat, and world parity is the dotted one below it.
    """
    st = _style()
    head = D["CITE_HEADLINE"][2:]
    rci = f"rci_{head}"
    big = map_fields(D, top_m)
    labels = dict(zip(D["TOP_CODES"], D["TOP_LABELS"]))
    ref = D["OVERALL"][head]

    for code, r in big.iterrows():
        top = code in labels
        col = _field_color(D, labels[code]) if top else C_MUTED
        ax.scatter(r.activity_x, r[rci],
                   s=26 + 260 * np.sqrt(r.ukbb_papers / big.ukbb_papers.max()),
                   color=col, alpha=0.95 if top else 0.55, zorder=3 if top else 2,
                   edgecolor="white", linewidth=0.8)

    # Label the top-N (colour-matched to every other panel) plus whatever is extreme on
    # either axis — those are the points a reader will ask about.
    extreme = (set(big.nlargest(3, rci).index) | set(big.nsmallest(2, rci).index)
               | set(big.nlargest(3, "activity_x").index)
               | set(big.nsmallest(2, "activity_x").index))
    to_label = [(r[rci], r.activity_x, code) for code, r in big.iterrows()
                if label_all or code in labels or code in extreme]
    span = np.log10(big[rci].max() / big[rci].min())
    # _nudge raises each label to at least `gap` above the last, so the stack grows as
    # n x gap. A gap tuned for the main figure's 13 labels throws the SI's 25 clean off
    # the top of the axis, so it is capped to keep the stack inside the data's own span.
    gap = min(0.055 * (span + 0.4), 0.95 * span / max(len(to_label), 1))
    for value, ly, xpos, code in _nudge(to_label, gap):
        lab = D["label_of"](code)
        col = _field_color(D, labels[code]) if code in labels else C_INK
        if abs(np.log10(ly) - np.log10(value)) > 0.008:
            ax.plot([xpos, xpos], [value, ly], color=col, lw=0.7, alpha=0.5, zorder=1)
        to_the_left = lab in MAP_LABEL_LEFT
        ax.annotate(AI.short(lab, 26), (xpos, ly), color=col, fontsize=st["annot_fs"],
                    va="center", ha="right" if to_the_left else "left",
                    xytext=(-8 if to_the_left else 8, 0), textcoords="offset points",
                    bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.6))

    show_world = big[rci].min() < 1.3
    ax.axvline(1.0, color=C_INK, lw=1.2)
    ax.axhline(ref, color=C_ABOVE, lw=1.4)
    if show_world:
        ax.axhline(1.0, color=C_INK, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis, ticks in ((ax.xaxis, [0.25, 0.5, 1, 2, 4, 8]),
                        (ax.yaxis, [1, 1.5, 2, 3, 4, 5])):
        axis.set_major_formatter(mticker.FuncFormatter(_times_fmt))
        axis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    ax.set_yticks([1, 1.5, 2, 3, 4, 5])
    ax.set_xlim(big.activity_x.min() / 1.9, big.activity_x.max() * 2.4)
    ax.set_ylim((1.0 if show_world else big[rci].min()) / 1.12, big[rci].max() * 1.34)

    ax.annotate(f"UK Biobank average  {ref:.1f}x", (ax.get_xlim()[0], ref),
                color=C_ABOVE, fontsize=st["annot_fs"], va="bottom",
                xytext=(4, 3), textcoords="offset points")
    if show_world:
        ax.annotate("world average", (ax.get_xlim()[0], 1.0), color=C_INK,
                    fontsize=st["annot_fs"], va="bottom", xytext=(4, 3),
                    textcoords="offset points")
    # Name the halves rather than all four quadrants: with the reference at UK Biobank's
    # own average, left and right of the parity line is the whole message.
    ax.annotate("publishes MORE here than the literature does", (0.99, 0.015),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=st["annot_fs"], color=C_INK)
    ax.annotate("publishes less", (0.01, 0.015), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=st["annot_fs"], color=C_INK)
    ax.set_xlabel(f"volume activity index ({D['ACTIVITY_LABEL']}), log scale")
    ax.set_ylabel(f"relative citation impact ({head}),\nvs the world in the same field")
    grid_on(ax)
    return ax


# ------------------------------------------------------------------ panel C
def draw_top_decile_share(ax, D, label_chars: int = 22, pad_years: int = 6,
                          label_gap: float = 0.078, codes=None, top_m: int | None = None,
                          legend: bool = False, legend_loc: str = "lower right",
                          floor_ratio: float = 300):
    """Chart 7 — UK Biobank's share of each field's most-cited tenth, year by year.

    Panel A asks what share of a field's PAPERS are UK Biobank's. This asks what share of
    its BEST papers are, which is the number a funder actually wants: a resource can be
    a rounding error in a field's output and a visible fraction of its top decile, and
    that gap is the whole argument for the resource.

    A share is also the one place a raw citation count would be safe — numerator and
    denominator are the same papers in the same field-years, so the publication-year
    composition that rules out a pooled ratio (D10) cancels here.

    `codes` chooses the fields, defaulting to the family's top eight; `top_m` then keeps
    the strongest of them by their FINAL year's share, which is the axis the panel is
    ranked on. Any field outside the family's top eight is drawn in neutral grey, exactly
    as the impact map draws it, so the two panels of SI 1 read as one figure.

    **Direct labels or a legend, not both.** Direct labels are the better encoding and are
    what the SI panel uses, but they are paid for in axis: eight of them need `pad_years`
    of empty x to sit in, which at one column of three squeezes the decade the panel is
    about into two thirds of its width. `legend=True` gives that width back. The legend
    is stretched to the panel's full width, so `legend_loc` now chooses only the top or
    bottom edge; the "left"/"right" half of the name has nothing left to decide.

    `label_chars` / `pad_years` / `label_gap` / `floor_ratio` exist because this panel is
    drawn at two very different widths: full width in the SI, and one column of three on
    the main page. `floor_ratio` is how far below the largest share the axis reaches —
    the early years of a small field are a rounding error against today's largest, and
    letting them set the floor squashes the decade that matters into the top inch, but
    cut too close they leave lines dropping out of the bottom of the panel.
    """
    st = _style()
    m = CITE_SHARE_METRIC if CITE_SHARE_METRIC in D["MEASURES"] else "n_papers"
    codes = list(D["TOP_CODES"]) if codes is None else list(codes)
    share = D["share_timeseries"](m, codes=codes)
    if m in D["CITE_YEARS"]:
        share = share.loc[share.index.isin(D["CITE_YEARS"][m])]
    # A share of exactly zero has no place on a log axis — matplotlib draws it as a line
    # dropping off the bottom, which reads as a crash rather than as "no papers here".
    share = share.where(share > 0)

    # One row per field that has anything to draw, ranked on the final year's share: that
    # is what the panel is about, and it is the order the legend or the label stack reads
    # top-down. A field outside the family's eight has no colour of its own — grey, as on
    # the impact map — so the eight stay findable in the SI's 25.
    series = []
    for code in codes:
        s_ = share[code].dropna()
        if not len(s_):
            continue
        lab = D["label_of"](code)
        series.append((float(s_.iloc[-1]), lab, D["FIELD_COLORS"].get(lab, C_MUTED), s_))
    series.sort(key=lambda r: -r[0])
    if top_m is not None:
        series = series[:top_m]

    ends = []
    for last, lab, col, s_ in series:
        ax.plot(s_.index, s_.values, color=col, lw=2.2, marker="o",
                ms=st["marker_size"] * 0.62)
        ends.append((last, float(s_.index[-1]), lab))

    lo = min(v for v, _, _ in ends)
    hi = max(v for v, _, _ in ends)
    span = np.log10(hi / max(lo, 1e-6))
    if legend:
        # NAMES ONLY. The values are on the axis the lines are read against, and printing
        # them again turned a colour key into a column of numbers a reader has to match
        # back to the lines. The overall rule takes an entry of its own rather than a
        # second in-axes label, which is what makes the block three rows by two.
        entries = [(AI.short(lab, label_chars), col, len(lab))
                   for _, lab, col, _ in series]
        # THE LONGEST NAMES GO IN THE RIGHT-HAND COLUMN. matplotlib fills a legend column
        # by column, so which name lands where is ours to choose, and a column is only as
        # wide as its own longest entry: spreading the two long names across both columns
        # makes BOTH wide, where keeping them together leaves one narrow. Rank order is
        # kept inside each column, and the overall rule stays last, at the foot of the
        # right column.
        rows = -(-(len(entries) + 1) // 2)
        # Ranked on the FULL name, not the drawn one: two names cut to `label_chars` are
        # the same length as each other and the tie would be decided by nothing.
        long = sorted(sorted(range(len(entries)),
                             key=lambda i: -entries[i][2])[:rows - 1])
        order = [i for i in range(len(entries)) if i not in long] + long
        handles = [Line2D([], [], color=entries[i][1], lw=2.2, marker="o",
                          ms=st["marker_size"] * 0.62, label=entries[i][0])
                   for i in order]
        handles.append(Line2D([], [], color=C_INK, lw=1, ls="--",
                              label="UK Biobank overall"))
        # THE BLOCK IS STRETCHED TO THE PANEL'S FULL WIDTH rather than shrink-wrapped
        # around its text. Left to itself the legend sizes to its contents and then hangs
        # off whichever corner `legend_loc` names — at `label_chars` wide enough to be
        # worth reading that box was wider than the axes and overhung the y-spine, while
        # the panel's own left edge sat empty. `mode="expand"` against an axes-fraction
        # bbox pins the left column to the left spine and the right column to the right
        # one, so the width the panel already has is the width the names get to use, and
        # `label_chars` can be set from that rather than from what a floating box allows.
        ax.legend(handles=handles, loc=legend_loc, ncol=2,
                  bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), mode="expand",
                  fontsize=st["legend_fs"] * 0.95, handlelength=1.6,
                  handletextpad=0.5, labelspacing=0.3, columnspacing=1.2,
                  borderaxespad=0.0, frameon=True, facecolor="white",
                  edgecolor=C_INK, framealpha=0.95, fancybox=False,
                  borderpad=0.5).get_frame().set_linewidth(0.8)
    else:
        # _nudge stacks the labels upward, so n labels need n x gap of axis; a gap tuned
        # for eight of them throws the SI's 25 off the top. Cap it at the data's own span.
        gap = min(label_gap * (span + 0.4), 0.95 * span / max(len(ends), 1))
        for value, ly, xend, lab in _nudge(ends, gap):
            col = D["FIELD_COLORS"].get(lab, C_MUTED)
            if abs(np.log10(ly) - np.log10(value)) > 0.008:
                ax.plot([xend, xend], [value, ly], color=col, lw=0.7, alpha=0.5, zorder=1)
            ax.annotate(f"{AI.short(lab, label_chars)}  {_pct(value)}", (xend, ly),
                        color=col, fontsize=st["annot_fs"], va="center",
                        xytext=(7, 0), textcoords="offset points")

    # UK Biobank's overall share of this measure — the line a field has to beat to be one
    # of its stronger showings rather than simply a big field.
    win = D["CITE_WIN"]
    overall = 100 * (D["ukbb"][D["ukbb"].year.between(*win)][m].sum()
                     / D["whole"][D["whole"].year.between(*win)][m].sum())
    ax.axhline(overall, color=C_INK, lw=1, ls="--")
    # The axis is set from what was DRAWN, not from `share`: with `top_m` in force the
    # frame still holds the fields that were cut, and a cut field's floor or its extra
    # year would set limits nothing on the panel reaches.
    drawn_years = sorted({int(y) for _, _, _, s_ in series for y in s_.index})
    drawn_lo, drawn_hi = drawn_years[0], drawn_years[-1]
    # With a legend the rule is named there, so the axis carries the value alone rather
    # than the same phrase twice.
    ax.annotate(f"{overall:.2f}%" if legend else f"UK Biobank overall {overall:.2f}%",
                (drawn_lo, overall), color=C_INK, fontsize=st["annot_fs"], va="bottom",
                ha="left", xytext=(2, 4), textcoords="offset points",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    # The early years are a rounding error against today's shares, and letting them set
    # the floor squashes the decade that matters into the top inch.
    floor = max(hi / floor_ratio, min(float(s_.min()) for _, _, _, s_ in series))
    ax.set_yticks([t for t in (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25)
                   if floor / 1.2 <= t <= hi * 1.6])
    # An in-axes legend needs the room it covers to be room the data does not use. Three
    # rows of two take about a decade of a panel this size, so that is what the floor
    # drops by when one is drawn. The TICKS still stop at the data's own floor, so the
    # extra space stays visibly empty rather than labelled.
    ax.set_ylim(floor / (16.0 if legend else 1.3), hi * 2.4)
    ax.set_xlim(drawn_lo - 0.4, drawn_hi + pad_years)
    ax.set_xticks(year_ticks(drawn_lo, drawn_hi, 2))
    ax.set_xlabel("year")
    ax.set_ylabel("UK Biobank's share of the field's\ntop-decile papers (log scale)")
    grid_on(ax)
    return ax


# ------------------------------------------------------------------ panel D (a block)
def _contiguous_tail(index):
    """The longest run of CONSECUTIVE years ending at the last one.

    A cut-off exists only where it was measured, so a field can have 2015 measured, 2016
    and 2017 not, and 2018 onwards measured again. Plotting that index as-is draws a
    straight line across the hole, which reads as a smooth trend through years nobody
    looked at. The panel starts where the uninterrupted record starts.
    """
    years = [int(y) for y in index]
    if len(years) < 2:
        return index
    keep = [years[-1]]
    for y in reversed(years[:-1]):
        if keep[0] - y != 1:
            break
        keep.insert(0, y)
    return index[[years.index(y) for y in keep]]


def _cites(v):
    """A citation cut-off with its unit: "16 cites", "1 cite", "0.4 cites"."""
    if v != v:
        return "–"
    n = f"{v:.0f}" if v >= 1 else f"{v:.1f}"
    return f"{n} cite" + ("" if v == 1 else "s")


def _cut_note(D, code, year):
    """One line naming what the bands are worth in citations, for the final year.

    The cut-offs move every year — 2016's decile in Epidemiology is 59 citations and
    2025's is 8 — but a panel does not need the whole series to stay readable. Where the
    measured median kept far less than half the field, the note says so: in a recent year
    the cut lands at ">= 1 citation" and the band is "cited at all", not "above the
    median".
    """
    cuts = D["CUTS"]
    if cuts is None:
        return ""
    r = cuts.reindex([(int(year), code)]).iloc[0]
    if r["median"] != r["median"]:
        return ""
    ten = "" if r.thr10 != r.thr10 else f"top 10% >= {_cites(r.thr10)}   ·   "
    got = r["med_pct"]
    tail = "" if (got != got or abs(got - 50) <= 5) else f" (= top {got:.0f}%)"
    return f"{int(year)}:   {ten}median >= {_cites(r['median'])}{tail}"


def block_footprint_quality(fig, spec, D, nrow: int = 2, ncol: int = 4,
                            letters: str = ""):
    """Chart 9 as a BLOCK of small multiples inside one gridspec cell.

    Not a `draw_(ax, D)` like the others, because it is eight axes rather than one: it
    takes the figure and the cell of the parent grid it should fill, and builds its own
    sub-grid inside it. That is what lets the main figure carry all eight fields instead
    of trimming to the three that would fit a single slot.

    Each panel draws UK Biobank's slice of one field and cuts it at that field's own
    citation distribution:

        total height  UK Biobank's papers as a % of all the field's papers
        solid band    those in the field's most-cited tenth
        pale band     the rest of those above the field's median
        white         the rest — below the median of their own field

    The two reference points would be arithmetic if the cuts were exact (an ordinary
    paper is top-decile a tenth of the time and above the median half the time), so a
    solid band at 16% of the line reads as 1.6x ordinary. USE THE MEASURED SHARE, NOT THE
    ROUND NUMBER: the decile cut achieves 7.3-10.0% and 10% is close enough, but the
    median cut achieves 34-50%, so in the years where it falls short, dividing by 50
    understates the multiplier.

    EACH SUB-PANEL IS ITS OWN LETTERED PANEL. `letters` gives the run of letters to
    stamp on them, one each, in the order the fields are drawn — the block letters D
    through K on the main page.

    THE PANELS NO LONGER STATE THEIR OWN CUT-OFFS. Each carried a grey line giving what
    the two bars were worth in citations that year; it was removed on 2026-09-09 to leave
    the bands uncluttered. `cut_off_table()` is now the only place those numbers are
    written down, and the methods section has to carry it — the figure alone no longer
    says whether a median band means "above the median" or "cited at all".
    """
    st = _style()
    top10, top50 = D["TOP10_COL"], D["TOP50_COL"]
    gs = spec.subgridspec(nrow, ncol, hspace=0.52, wspace=0.30)

    all_ = D["share_timeseries"]("n_papers")
    best = D["share_timeseries"](top10, denominator="n_papers")
    half = D["share_timeseries"](top50, denominator="n_papers")

    # A cell whose cut-off was never measured comes back as a zero COUNT, not as a gap,
    # and zero here would draw as "none of its papers made the decile" — the opposite of
    # "we did not look". The _docs counters say which it is, so they are the mask.
    def measured(column):
        return (D["ukbb"][D["ukbb"].code.isin(D["TOP_CODES"])]
                .pivot_table(index="year", columns="code", values=f"{column}_docs",
                             aggfunc="sum").reindex(columns=D["TOP_CODES"]))

    best = best.where(measured(top10).reindex_like(best) > 0)
    half = half.where(measured(top50).reindex_like(half) > 0)
    yrs = sorted(set(all_.index) & set(best.index) & set(half.index))
    all_, best, half = (f.loc[f.index.isin(yrs)] for f in (all_, best, half))

    axes, drawn = [], []
    for i, (code, lab) in enumerate(zip(D["TOP_CODES"], D["TOP_LABELS"])):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        axes.append(ax)
        if i < len(letters):
            panel_label(ax, letters[i])
        col = _field_color(D, lab)
        a, b, h = all_[code].dropna(), best[code].dropna(), half[code].dropna()
        idx = _contiguous_tail(a.index.intersection(b.index).intersection(h.index))
        if not len(idx):
            continue
        a, b = a.reindex(idx), b.reindex(idx)
        # A paper above the decile is above the median too, so the bands nest; clip in
        # case a tie at the median cut-off makes the coarser count the smaller one.
        h = np.maximum(h.reindex(idx), b)
        drawn += [int(idx[0]), int(idx[-1])]

        ax.fill_between(idx, 0, b, color=col, alpha=0.85, lw=0)
        ax.fill_between(idx, b, h, color=col, alpha=0.18, lw=0)
        ax.plot(idx, a, color=col, lw=1.8)
        facet_ylabel(ax, lab, fontsize=st["annot_fs"] + 1)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
        ax.set_ylim(bottom=0)
        ax.margins(x=0.02)
        ax.tick_params(labelsize=st["tick_fs"])

        # Each share is written into the band it belongs to: white on the solid one, the
        # field's own colour on the pale one. No legend lookup, no second sentence.
        if len(a) and a.iloc[-1] > 0:
            if b.iloc[-1] > 0:
                ax.annotate(f"{100 * b.iloc[-1] / a.iloc[-1]:.0f}%",
                            xy=(idx[-1], b.iloc[-1] / 2), xytext=(-5, 0),
                            textcoords="offset points", ha="right", va="center",
                            color="white", fontsize=st["annot_fs"], fontweight="bold")
            if h.iloc[-1] > b.iloc[-1]:
                ax.annotate(f"{100 * h.iloc[-1] / a.iloc[-1]:.0f}%",
                            xy=(idx[-1], (h.iloc[-1] + b.iloc[-1]) / 2), xytext=(-5, 0),
                            textcoords="offset points", ha="right", va="center",
                            color=col, fontsize=st["annot_fs"], fontweight="bold")
        if i >= (nrow - 1) * ncol:
            ax.set_xlabel("year", fontsize=st["tick_fs"])
        if i % ncol == 0:
            ax.set_ylabel(f"% of the field's papers", fontsize=st["tick_fs"])
        grid_on(ax)

    if drawn:
        for ax in axes:
            ax.set_xticks(range(min(drawn) + 1, max(drawn) + 1, 3))

    return axes


def footprint_legend(fig, D, y=0.018):
    """The footprint block's key, as a FIGURE legend under it.

    Neutral grey patches: the encoding is what the legend describes, and every sub-panel
    already carries its field's own colour. It belongs to the figure rather than to
    `axes[0]` because anchored above that axes it lands in the row overhead, where panel
    the main page's third gridspec row already is.
    """
    st = _style()
    leg = fig.legend(handles=[
        Patch(color=C_INK, alpha=0.85, label="in the field's most-cited 10%"),
        Patch(color=C_INK, alpha=0.22, label="above the field's median (not top 10%)"),
        Line2D([], [], color=C_INK, lw=1.8, label="all UK Biobank papers in the field")],
        loc="lower center", bbox_to_anchor=(0.5, y), ncol=3, fontsize=st["legend_fs"],
        frameon=True, facecolor="white", edgecolor=C_INK, framealpha=1.0,
        fancybox=False, borderpad=0.6)
    leg.get_frame().set_linewidth(0.8)
    return leg


# --------------------------------------------------- the block's cut-offs, as a table
def cut_off_table(D) -> pd.DataFrame:
    """What the two bars in panels D-K were worth in citations, per (year, field).

    They used to print this one year at a time in the corner of each sub-panel. With
    that note removed the figure states no cut-offs at all, so this is what the methods
    section has to carry: for every measured cell, the top-decile threshold, the median
    threshold, and the share of the field the median cut actually kept.

    `achieved_pct` is the column to read before quoting a median band. Citations are small
    integers with enormous ties, so in a recent year the cut lands at ">= 1 citation" and
    keeps as little as 34% of the field: the count above the cut is still exact, but
    calling it the 50th percentile is not.
    """
    cuts = D["CUTS"]
    if cuts is None:
        return pd.DataFrame()
    want = set(D["TOP_CODES"])
    # The cells panel D DRAWS: the eight fields, inside the analysis window, and only
    # where the decile was actually measured. Without the last condition the table carries
    # earlier cache rows outside the study window, which may have no measured decile,
    # and no panel.
    out = (cuts.reset_index()
           .query("code in @want")
           .query("@ANALYSIS_MIN <= year <= @ANALYSIS_MAX")
           .dropna(subset=["thr10"])
           .assign(field=lambda d: [D["label_of"](c) for c in d.code])
           .rename(columns={"thr10": "top_decile_cites", "median": "median_cites",
                            "mean_cit": "mean_cites", "med_pct": "median_achieved_pct"})
           [["field", "year", "top_decile_cites", "median_cites", "mean_cites",
             "median_achieved_pct", "median_measured"]]
           .sort_values(["field", "year"])
           .reset_index(drop=True))
    return out.round(2)


# ------------------------------------------------------------------ SI: growth
def draw_growth_rates(ax, D):
    """Chart 11, left — UK Biobank's growth against the same fields in the world.

    Every other panel ranks fields by SIZE, which is stable by construction and so cannot
    show a field arriving. This ranks by SPEED and drops the size filter: a field that
    went from 6 papers to 126 belongs here beside one that went from 350 to 1,283.

    The hollow marker is the same field's growth in the whole database, so the gap
    between the markers is UK Biobank MOVING INTO a field rather than the field itself
    taking off. Read the acceleration column knowing UK Biobank's own output was doubling
    annually in the late 2010s from almost nothing, so nearly every field decelerates
    against that window; the honest reference is the dashed line, its own overall pace.
    """
    st = _style()
    top = D["GROWTH"].head(GROW_TOP_M)
    cols = dict(zip(top.label, field_palette(len(top))))
    ypos = np.arange(len(top))[::-1]

    for y, (_, r) in zip(ypos, top.iterrows()):
        col = cols[r.label]
        ax.plot([r.world_now, r.ukbb_now], [y, y], color=col, lw=2, alpha=0.45,
                zorder=2, solid_capstyle="round")
        ax.scatter(r.world_now, y, s=st["dot_marker_area"], facecolor="white",
                   edgecolor=C_INK, linewidth=1.4, zorder=3)
        ax.scatter(r.ukbb_now, y,
                   s=34 + 220 * np.sqrt(r.ukbb_papers / top.ukbb_papers.max()),
                   color=col, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(f"{r.ukbb_now:.0f}%", (r.ukbb_now, y), xytext=(10, 0),
                    textcoords="offset points", va="center",
                    fontsize=st["annot_fs"], color=col)

    ax.axvline(0, color=C_INK, lw=1)
    ax.axvline(D["UKBB_PACE"], color=C_ABOVE, lw=1.2, ls="--")
    ax.annotate(f"UK Biobank overall\n{D['UKBB_PACE']:.0f}%/yr",
                (D["UKBB_PACE"], len(top) - 0.35), xytext=(5, 0),
                textcoords="offset points", va="top", fontsize=st["annot_fs"],
                color=C_ABOVE)

    # The acceleration is a second measure, so it gets its own column rather than a
    # second glyph on a row that already carries three.
    ax.annotate(f"acceleration\nvs {GROW_PREV[0]}–{GROW_PREV[1]}", (0.995, 0.995),
                xycoords="axes fraction", ha="right", va="top",
                fontsize=st["annot_fs"] - 1, color=C_INK)
    for y, (_, r) in zip(ypos, top.iterrows()):
        txt, c = ("new", C_INK) if r.accel != r.accel else (
            f"{r.accel:+.0f} pp", C_ABOVE if r.accel > 0 else C_BELOW)
        ax.annotate(txt, (0.995, y), xycoords=("axes fraction", "data"), ha="right",
                    va="center", fontsize=st["annot_fs"] - 1, color=c)

    ax.set_yticks(ypos)
    ax.set_yticklabels([AI.short(l, 32) for l in top.label], fontsize=st["annot_fs"])
    ax.set_ylim(-0.9, len(top) - 0.1)
    # Zero is where a growth axis starts: the rule at 0 is the panel's reference, and
    # opening the axis below it spends width on nothing and reads as room for negative
    # rates that are not there. Two qualifications. A field whose WORLD rate is negative
    # would be cut off, so the floor follows the data down if one ever is; and a hollow
    # marker sitting AT zero is half outside an axis that ends there, so the limit takes
    # a marker's width of padding — a little over one percentage point, invisible on an
    # axis that runs to a hundred.
    right = top.ukbb_now.max() * 1.42
    left = min(0.0, top.world_now.min() - 1)
    ax.set_xlim(left - 0.015 * (right - left), right)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
    ax.set_xlabel(f"compound annual growth, {GROW_NOW[0]}–{GROW_NOW[1]} (%/yr)"
                  f"   ·   dot area = UK Biobank papers in {GROW_NOW[1]}")
    grid_on(ax, axis="x")
    return ax


def draw_growth_trajectories(ax, D):
    """Chart 11, right — the absolute trajectories those rates summarise.

    Counts on a log axis, so the SLOPE is the growth rate: parallel lines grow equally
    fast whatever their size, and a small fast field is legible beside a large one rather
    than flattened onto the axis. The dashed grey line is all UK Biobank papers.
    """
    st = _style()
    top = D["GROWTH"].head(GROW_TOP_M)
    cols = dict(zip(top.label, field_palette(len(top))))
    uts, totals = D["ukbb_ts_all"], D["ukbb_totals"]
    span = range(GROW_PREV[0], GROW_NOW[1] + 1)

    ends = []
    for code, r in top.iterrows():
        s = uts[code].reindex(span).replace(0, np.nan).dropna()
        ax.plot(s.index, s.values, color=cols[r.label], lw=2, marker="o",
                ms=st["marker_size"] * 0.60)
        ends.append((float(s.iloc[-1]), r.label))
    all_ukbb = totals.reindex(span)
    ax.plot(all_ukbb.index, all_ukbb.values, color=C_INK, lw=2.4, ls="--", alpha=0.8)
    ends.append((float(all_ukbb.iloc[-1]), "ALL UK Biobank papers"))

    lo, hi = min(v for v, _ in ends), max(v for v, _ in ends)
    for _, ly, lab in _nudge(ends, 0.032 * (np.log10(hi / lo) + 0.4)):
        ax.annotate(AI.short(lab, 28), (GROW_NOW[1], ly), color=cols.get(lab, C_INK),
                    fontsize=st["annot_fs"], va="center", xytext=(6, 0),
                    textcoords="offset points")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.axvline(GROW_NOW[0], color=C_INK, ls=":", lw=1)
    ax.annotate(f"{GROW_PREV[0]}–{GROW_PREV[1]}, comparison   |   "
                f"{GROW_NOW[0]}–{GROW_NOW[1]}, ranking",
                (GROW_NOW[0], 0.02), xycoords=("data", "axes fraction"), ha="center",
                va="bottom", fontsize=st["annot_fs"] - 1, color=C_INK)
    ax.set_xlim(GROW_PREV[0], GROW_NOW[1] + 5)
    ax.set_xticks(range(GROW_PREV[0], GROW_NOW[1] + 1, 2))
    ax.set_xlabel("year")
    ax.set_ylabel("UK Biobank papers per year (log scale)")
    grid_on(ax)
    return ax


# =============================================================================
# 5. Captions — written here so a figure and its caption cannot drift apart
# =============================================================================
MAIN_CAPTION = {
    "A": ("Where UK Biobank concentrates against where that work lands. Volume activity "
          "index — its share of a field against the field's share of the literature, "
          "measured within each field's own ANZSRC division — by relative citation "
          "impact, both log. The solid rule is UK Biobank's OWN average impact, not world "
          "parity: against the world essentially every field is above 1x, so its own "
          "standard is the informative reference. Dot area is UK Biobank papers."),
    "B": ("A's horizontal axis on its own, ranked: what UK Biobank is disproportionately "
          "about. 1x is parity with the literature; the axis is log, so 2x and 1/2 sit "
          "the same distance from it."),
    "C": ("A's vertical question over time, at the sharp end: UK Biobank's share of the "
          "papers clearing each field's measured top-decile citation cut-off, year by "
          "year, for the five of its top fields that reach furthest. The dashed rule is "
          "its share across all fields; SI 1B is the same chart at 25 fields."),
    "D-K": ("The footprint cut twice, one lettered panel per field. UK Biobank's share of "
            "the field (line), split into the part in the field's most-cited tenth "
            "(solid) and the rest above the field's median (pale). Both cut-offs are "
            "measured per (year, field) against the whole database and fall steeply with "
            "publication year because citations accrue; the values are tabulated in the "
            "methods, not on the panels."),
}

SI_CAPTIONS = {
    "impact_map": {
        "A": ("The impact map at UK Biobank's 25 largest fields, every point labelled — "
              "main-figure panel A carries 16 and labels only the top eight plus the "
              "extremes."),
    },
    "top_decile": {
        "A": ("Main-figure panel C undrawn-down: UK Biobank's share of each field's "
              "measured top-decile papers, year by year, for the same 25 fields as SI 1 "
              "and every year the citation measures support. The eight fields the family "
              "follows keep their colours; the other 17 are grey, as on the map."),
    },
    "growth": {
        "A": ("Compound annual growth 2021–2025 by field, UK Biobank (filled) against "
              "the same field in the whole database (hollow). Ranked on speed alone, "
              "with floors of 5 papers in 2021 and 20 in 2025."),
        "B": ("The absolute trajectories those rates summarise, log scale, so the slope "
              "is the growth rate. Dashed grey is all UK Biobank papers."),
    },
}


# =============================================================================
# 6. The pages
# =============================================================================
def _assemble_axes(fig, gs, slots):
    return [fig.add_subplot(gs[r, c]) for r, c in slots]


def figure_main(D, save=True):
    """The main-paper page: four lettered panels on a three-column grid.

        +---------------------------+-----------+
        |                           |     B     |   A     impact map      (2 rows x 2 cols)
        |             A             +-----------+   B     activity index  (top right)
        |                           |     C     |   C     top-decile share (below it)
        +---------------------------------------+   D-K   the footprint block,
        |                    D - K              |         full width, one letter each
        +---------------------------------------+

    The impact map leads and takes the largest slot: it is the one panel carrying both
    questions at once — how much UK Biobank publishes in a field, and how well that work
    is cited there — so the two panels beside it read as its margins. B is the volume axis
    on its own, ranked; C is the quality axis on its own, over time. The block then puts
    the quality claim back into each field's own citation distribution across eight
    fields, which needs the full width — and each of those eight is lettered D through K,
    because a reader citing "the genetics panel" needs an address for it.

    Drawn at `fs_scale.main` — larger type than the SI pages, because this one is read at
    figure width in a paper rather than full page on a screen.
    """
    assert_parameters_match(D)
    st = _style()
    with _font_scale(_fs_scale("main")):
        fig = plt.figure(figsize=st["figsize_main"])
        gs = fig.add_gridspec(4, 3, hspace=0.62, wspace=0.34,
                              height_ratios=[1.0, 1.0, 0.82, 0.82],
                              left=0.070, right=0.980, top=0.965, bottom=0.085)
        ax_a = fig.add_subplot(gs[0:2, 0:2])     # impact map, two rows and two columns
        ax_b = fig.add_subplot(gs[0, 2])         # activity index, top right
        ax_c = fig.add_subplot(gs[1, 2])         # top-decile share, under it

        draw_impact_map(ax_a, D, top_m=MAP_M_MAIN)
        draw_activity_index(ax_b, D, top_m=ACT_TOP_M_MAIN)
        # A third of the page width: eight direct labels and the empty years they need to
        # sit in would leave the decade this panel is about squeezed into two thirds of
        # its own axis, so here the fields go into a legend and the panel is cut to the
        # five that reach furthest. SI 1B carries all 25, direct-labelled, at full width.
        #
        # `label_chars` is set from the width the STRETCHED legend has (see the block in
        # `draw_top_decile_share`), not from what a shrink-wrapped box would allow. The
        # two columns still touch at 31, so the ceiling is legibility, not fit: 25 buys
        # back "Biological Psychology" whole and breaks "Cardiovascular Medicine and
        # Haematology" — the one name no width on this page can carry — after a word
        # instead of inside one, while leaving a gutter a reader can see.
        draw_top_decile_share(ax_c, D, label_chars=25, pad_years=1, top_m=DEC_M_MAIN,
                              legend=True, legend_loc="lower right")
        block_footprint_quality(fig, gs[2:4, 0:3], D, letters="DEFGHIJK")

        for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
            panel_label(ax, letter)
        footprint_legend(fig, D)

        if save:
            savefig(fig, P.MAIN_FIGURE_STEMS[5])
    return fig


def figure_si_impact_map(D, save=True):
    """SI 1 — the impact map undrawn-down: 25 fields, every one labelled.

    Main-figure panel A carries 16 points and labels the top eight plus whatever is
    extreme on either axis. This is the same map at full page with nothing left implicit.
    SI 2 follows these same 25 fields through time; `map_fields` serves both, so the two
    pages cannot disagree about which 25.
    """
    st = _style()
    with _font_scale(_fs_scale("si1_impact_map")):
        fig, ax = plt.subplots(figsize=st["figsize_si_wide"])
        draw_impact_map(ax, D, top_m=MAP_M_SI, label_all=True)
        if save:
            savefig(fig, "03_02_supplementary_figure_01_impact_map_full")
    return fig


def figure_si_top_decile(D, save=True):
    """SI 2 — the top-decile share undrawn-down: SI 1's 25 fields, direct-labelled.

    Main-figure panel C is cut to five fields and read off a legend, because it has one
    column of three to live in. At full page every field the map draws gets its line and
    its name, and the eight the family follows keep their colours against the other 17 in
    the map's grey — so a field's line here can be read straight off its point there.

    `floor_ratio` is wide enough that nothing is clipped: across 25 fields the smallest
    early share is about a thousandth of the largest, and the main page's tighter floor
    would leave four lines dropping out of the bottom of the panel.
    """
    st = _style()
    with _font_scale(_fs_scale("si2_top_decile")):
        fig, ax = plt.subplots(figsize=st["figsize_si_wide"])
        draw_top_decile_share(ax, D, codes=map_fields(D, MAP_M_SI).index,
                              label_chars=30, pad_years=5, label_gap=0.055,
                              floor_ratio=1200)
        if save:
            savefig(fig, "03_03_supplementary_figure_02_top_decile_share")
    return fig


def figure_si_growth(D, save=True):
    """SI 3 — growth, not size. A growth result inside an impact analysis, hence SI."""
    st = _style()
    with _font_scale(_fs_scale("si3_growth")):
        fig = plt.figure(figsize=st["figsize_si"])
        gs = fig.add_gridspec(1, 2, wspace=0.30, width_ratios=[1.15, 1.0])
        axes = _assemble_axes(fig, gs, [(0, 0), (0, 1)])
        draw_growth_rates(axes[0], D)
        draw_growth_trajectories(axes[1], D)
        for ax, letter in zip(axes, "AB"):
            panel_label(ax, letter)
        if save:
            savefig(fig, "03_04_supplementary_figure_03_growth")
    return fig
