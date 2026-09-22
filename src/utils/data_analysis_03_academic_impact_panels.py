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
paper. Extra categories reuse those colours with different markers and line styles;
continuous ramps interpolate only between shared named anchors. Nothing
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
from matplotlib.colors import to_rgb                                   # noqa: E402

from utils.shared_style import (                                        # noqa: E402
    PALETTE_COLORS, academic_impact_colormap, facet_ylabel, grid_on, panel_label, savefig,
    year_ticks, scatter_callouts, black_legend, align_legend_rows,
)


#: Exact project colours; additional fields reuse hues with different line/marker styles.
FIELD_CYCLE = [
    PALETTE_COLORS["red"],            # 1st field  — Epidemiology, the family's lead
    PALETTE_COLORS["navy"],
    PALETTE_COLORS["green"],
    PALETTE_COLORS["steel_blue"],
    PALETTE_COLORS["light_blue"],
    PALETTE_COLORS["blue"],
    PALETTE_COLORS["cream"],
]

#: The two-sided encoding every ratio panel uses: above the reference / below it. Red for
#: over-represented and steel blue for under is the same pairing analysis 04 uses for a
#: value against its benchmark.
C_ABOVE, C_BELOW = PALETTE_COLORS["red"], PALETTE_COLORS["steel_blue"]

#: Neutral ink for reference lines, parity rules and supporting annotations.
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
    return [FIELD_CYCLE[i % len(FIELD_CYCLE)] for i in range(n)]


def field_line_styles(D, labels=()):
    """Keep field identity consistent across figures without inventing extra colours."""
    names = list(dict.fromkeys([*D["FIELD_COLORS"], *labels]))
    markers = ("o", "s", "^", "D", "v", "P", "X", ">")
    lines = ("-", "--", "-.", ":")
    return {
        label: {
            "color": D["FIELD_COLORS"].get(label, FIELD_CYCLE[i % len(FIELD_CYCLE)]),
            "marker": markers[i % len(markers)],
            "linestyle": lines[(i // len(FIELD_CYCLE)) % len(lines)],
        }
        for i, label in enumerate(names)
    }




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
        .assign(accel=lambda d: d.ukbb_now - d.ukbb_prev,
                world_accel=lambda d: d.world_now - d.world_prev)
        .sort_values("ukbb_now", ascending=False))
    growth["label"] = [code_label.get(c, c) for c in growth.index]

    totals = ukbb.groupby("year").n_papers.sum()
    pace = float((totals[GROW_NOW[1]] / totals[GROW_NOW[0]])
                 ** (1 / (GROW_NOW[1] - GROW_NOW[0])) - 1) * 100
    say(f"  growth: {len(growth)} field(s) clear the floors; UK Biobank overall "
        f"{pace:.0f}%/yr over {GROW_NOW[0]}-{GROW_NOW[1]}")
    return growth, totals, uts, pace


def build_panel_data(verbose: bool = True, *, context=None) -> dict:
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
    ctx = dict(context) if context is not None else AI.build(
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
    D["CUTS"] = ctx["CUTS"] if "CUTS" in ctx else _cuts_table(counts_dir, ctx["LEVEL"], say)
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
from matplotlib.legend_handler import HandlerTuple                      # noqa: E402
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
               edgecolor="black", linewidth=0.6)
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
    from textwrap import fill
    ax.set_yticklabels([fill(lab, 36) for lab in tbl.label], fontsize=st["tick_fs"])
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
    styles = field_line_styles(D)
    ref = D["OVERALL"][head]

    for code, r in big.iterrows():
        top = code in labels
        col = _field_color(D, labels[code]) if top else PALETTE_COLORS["light_blue"]
        ax.scatter(r.activity_x, r[rci],
                   s=26 + 260 * np.sqrt(r.ukbb_papers / big.ukbb_papers.max()),
                   marker=styles[labels[code]]["marker"] if top else "o",
                   color=col, alpha=0.95 if top else 0.55, zorder=3 if top else 2,
                   edgecolor="black", linewidth=0.6)

    # Label the top-N (colour-matched to every other panel) plus whatever is extreme on
    # either axis — those are the points a reader will ask about.
    extreme = (set(big.nlargest(3, rci).index) | set(big.nsmallest(2, rci).index)
               | set(big.nlargest(3, "activity_x").index)
               | set(big.nsmallest(2, "activity_x").index))
    to_label = [(r[rci], r.activity_x, code) for code, r in big.iterrows()
                if label_all or code in labels or code in extreme]
    span = np.log10(big[rci].max() / big[rci].min())

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
    ax.set_xlabel("Within-division activity index (log scale)")
    ax.set_ylabel("Relative mean normalised citation impact\n(same-field reference; log scale)")
    grid_on(ax)
    scatter_callouts(ax, [(x, value) for value, x, _ in to_label],
                     [D["label_of"](code) for _, _, code in to_label],
                     obstacles=big[["activity_x", rci]].to_numpy(), width=28)
    return ax


# ------------------------------------------------------------------ panel C
def draw_top_decile_share(ax, D, label_chars: int = 22, pad_years: int = 6,
                          label_gap: float = 0.078, codes=None, top_m: int | None = None,
                          legend: bool = False, legend_loc: str = "lower right",
                          floor_ratio: float = 300, legend_fontsize=None,
                          legend_below: bool = False):
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
    ranked on. Additional fields use the same project palette with distinct markers.

    **Direct labels or a legend, not both.** Direct labels are the better encoding and are
    what the SI panel uses, but they are paid for in axis: eight of them need `pad_years`
    of empty x to sit in, which at one column of three squeezes the decade the panel is
    about into two thirds of its width. `legend=True` gives that width back. The legend
    is stretched to the panel's full width, so `legend_loc` now chooses only the top or
    bottom edge; the "left"/"right" half of the name has nothing left to decide.
    `legend_below` keeps the main-page key outside the data axes instead.

    `label_chars` / `pad_years` / `label_gap` / `floor_ratio` exist because this panel is
    drawn at two very different widths: full width in the SI, and one column of three on
    the main page. `floor_ratio` is how far below the largest share the axis reaches —
    the early years of a small field are a rounding error against today's largest, and
    letting them set the floor squashes the decade that matters into the top inch, but
    cut too close they leave lines dropping out of the bottom of the panel.
    """
    st = _style()
    m = CITE_SHARE_METRIC if CITE_SHARE_METRIC in D["MEASURES"] else "n_papers"
    win = D["CITE_WIN"]
    overall = 100 * (D["ukbb"][D["ukbb"].year.between(*win)][m].sum()
                     / D["whole"][D["whole"].year.between(*win)][m].sum())
    codes = list(D["TOP_CODES"]) if codes is None else list(codes)
    share = D["share_timeseries"](m, codes=codes)
    if m in D["CITE_YEARS"]:
        share = share.loc[share.index.isin(D["CITE_YEARS"][m])]
    # A share of exactly zero has no place on a log axis — matplotlib draws it as a line
    # dropping off the bottom, which reads as a crash rather than as "no papers here".
    share = share.where(share > 0)

    # One row per field that has anything to draw, ranked on the final year's share: that
    # is what the panel is about, and it is the order the legend or the label stack reads
    # top-down. The existing field-to-colour mapping is shared with the other panels.
    styles = field_line_styles(D, [D["label_of"](code) for code in codes])
    series = []
    for code in codes:
        s_ = share[code].dropna()
        if not len(s_):
            continue
        lab = D["label_of"](code)
        series.append((float(s_.iloc[-1]), lab, styles[lab]["color"], s_))
    series.sort(key=lambda r: -r[0])
    if top_m is not None:
        series = series[:top_m]

    ends = []
    for last, lab, col, s_ in series:
        ax.plot(s_.index, s_.values, color=col, lw=2.2,
                marker=styles[lab]["marker"], ls=styles[lab]["linestyle"],
                ms=st["marker_size"] * 0.8, markeredgecolor="black", markeredgewidth=.45)
        ends.append((last, float(s_.index[-1]), lab))

    lo = min(v for v, _, _ in ends)
    hi = max(v for v, _, _ in ends)
    span = np.log10(hi / max(lo, 1e-6))
    if legend:
        # Only the pooled reference needs a numeric label. Keep it in the key, not
        # over the early-year series where it obscures the points.
        from textwrap import fill
        entries = [(fill(lab, label_chars), col, len(lab))
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
        handles = [Line2D([], [], color=entries[i][1], lw=2.2,
                          marker=styles[series[i][1]]["marker"],
                          ls=styles[series[i][1]]["linestyle"],
                          markeredgecolor="black", markeredgewidth=.45,
                          ms=st["marker_size"] * 0.62, label=entries[i][0])
                   for i in order]
        handles.append(Line2D([], [], color=C_INK, lw=1, ls="--",
                              label=fill(f"UK Biobank overall ({overall:.2f}%)", label_chars)))
        # THE BLOCK IS STRETCHED TO THE PANEL'S FULL WIDTH rather than shrink-wrapped
        # around its text. Left to itself the legend sizes to its contents and then hangs
        # off whichever corner `legend_loc` names — at `label_chars` wide enough to be
        # worth reading that box was wider than the axes and overhung the y-spine, while
        # the panel's own left edge sat empty. `mode="expand"` against an axes-fraction
        # bbox pins the left column to the left spine and the right column to the right
        # one, so the width the panel already has is the width the names get to use, and
        # `label_chars` can be set from that rather than from what a floating box allows.
        leg = ax.legend(handles=handles, loc="upper left" if legend_below else legend_loc, ncol=2,
                  bbox_to_anchor=(0.0, -.23, 1.0, 0.0) if legend_below else (0.0, 0.0, 1.0, 1.0),
                  mode="expand", fontsize=legend_fontsize or st["legend_fs"], handlelength=1.6,
                  handletextpad=0.5, labelspacing=0.3, columnspacing=1.2,
                  borderaxespad=0.0, frameon=True, facecolor="white",
                  edgecolor="black", framealpha=1.0, fancybox=False,
                  borderpad=0.5)
        leg.get_frame().set_linewidth(0.8)
        align_legend_rows(leg)
    else:
        # _nudge stacks the labels upward, so n labels need n x gap of axis; a gap tuned
        # for eight of them throws the SI's 25 off the top. Cap it at the data's own span.
        gap = min(label_gap * (span + 0.4), 0.95 * span / max(len(ends), 1))
        for value, ly, xend, lab in _nudge(ends, gap):
            col = styles[lab]["color"]
            if abs(np.log10(ly) - np.log10(value)) > 0.008:
                ax.plot([xend, xend], [value, ly], color=col, lw=0.7, alpha=0.5, zorder=1)
            ax.annotate(f"{AI.short(lab, label_chars)}  {_pct(value)}", (xend, ly),
                        color=C_INK, fontsize=st["annot_fs"], va="center",
                        xytext=(7, 0), textcoords="offset points")

    # UK Biobank's overall share of this measure — the line a field has to beat to be one
    # of its stronger showings rather than simply a big field.
    ax.axhline(overall, color=C_INK, lw=1, ls="--")
    # The axis is set from what was DRAWN, not from `share`: with `top_m` in force the
    # frame still holds the fields that were cut, and a cut field's floor or its extra
    # year would set limits nothing on the panel reaches.
    drawn_years = sorted({int(y) for _, _, _, s_ in series for y in s_.index})
    drawn_lo, drawn_hi = drawn_years[0], drawn_years[-1]
    # The legend carries both the reference's name and value, away from the data.
    if not legend:
        ax.annotate(f"UK Biobank overall {overall:.2f}%",
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
    ax.set_ylim(floor / (16.0 if legend and not legend_below else 1.3), hi * 2.4)
    ax.set_xlim(drawn_lo - 0.4, drawn_hi + pad_years)
    ax.set_xticks(year_ticks(drawn_lo, drawn_hi, 2))
    ax.set_xlabel("Publication year")
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
        facet_ylabel(ax, lab + " (%)", fontsize=st["label_fs"])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
        ax.set_ylim(bottom=0)
        ax.margins(x=0.02)
        ax.tick_params(labelsize=st["tick_fs"])

        # Keep annotations readable on both the dark blue and pale cream fills.
        solid_ink = "black" if np.mean(to_rgb(col)) > .5 else "white"
        if len(a) and a.iloc[-1] > 0:
            if b.iloc[-1] > 0:
                ax.annotate(f"{100 * b.iloc[-1] / a.iloc[-1]:.0f}%",
                            xy=(idx[-1], b.iloc[-1] / 2), xytext=(-5, 0),
                            textcoords="offset points", ha="right", va="center",
                            color=solid_ink, fontsize=st["annot_fs"], fontweight="bold")
            if h.iloc[-1] > b.iloc[-1]:
                ax.annotate(f"{100 * h.iloc[-1] / a.iloc[-1]:.0f}%",
                            xy=(idx[-1], (h.iloc[-1] + b.iloc[-1]) / 2), xytext=(-5, 0),
                            textcoords="offset points", ha="right", va="center",
                            color=C_INK, fontsize=st["annot_fs"], fontweight="bold")
        if i >= (nrow - 1) * ncol:
            ax.set_xlabel("year", fontsize=st["tick_fs"])
        ax.grid(False, which="both")

    if drawn:
        for ax in axes:
            ax.set_xticks(year_ticks(min(drawn), max(drawn), 3))

    return axes


def footprint_legend(fig, D, y=0.018, *, fontsize=None):
    """The footprint block's key, as a FIGURE legend under it.

    Multi-colour swatches retain the field colours and match the two fill opacities.
    It belongs below the complete block, rather than inside any field's axes.
    """
    st = _style()
    colors = list(dict.fromkeys(_field_color(D, label) for label in D["TOP_LABELS"]))
    leg = fig.legend(handles=[
        tuple(Patch(color=color, alpha=.85) for color in colors),
        tuple(Patch(color=color, alpha=.18) for color in colors),
        tuple(Line2D([], [], color=color, lw=1.8) for color in colors)],
        labels=["in the field's most-cited 10%", "above the field's median (not top 10%)",
                "all UK Biobank papers in the field"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, handlelength=3.5,
        loc="lower center", bbox_to_anchor=(0.5, y), ncol=3, fontsize=fontsize or st["legend_fs"],
        frameon=True, facecolor="white", edgecolor="black", framealpha=1.0,
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
def _growth_encoding(D, top):
    """Keep shared field colours; distinguish reused hues with marker shape."""
    styles = field_line_styles(D, top.label)
    return ({lab: styles[lab]["color"] for lab in top.label},
            {lab: styles[lab]["marker"] for lab in top.label})


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
    cols, markers = _growth_encoding(D, top)
    ypos = np.arange(len(top))[::-1]

    for y, (_, r) in zip(ypos, top.iterrows()):
        col = cols[r.label]
        ax.plot([r.world_now, r.ukbb_now], [y, y], color=col, lw=2, alpha=0.45,
                zorder=2, solid_capstyle="round")
        ax.scatter(r.world_now, y, s=st["dot_marker_area"], facecolor="white",
                   edgecolor=C_INK, linewidth=1.4, zorder=3)
        ax.scatter(r.ukbb_now, y,
                   s=34 + 220 * np.sqrt(r.ukbb_papers / top.ukbb_papers.max()),
                   color=col, marker=markers[r.label], edgecolor="black", linewidth=0.6, zorder=4)
        ax.annotate(f"{r.ukbb_now:.0f}%", (r.ukbb_now, y), xytext=(10, 0),
                    textcoords="offset points", va="center",
                    fontsize=st["annot_fs"], color=C_INK)

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
    from textwrap import fill
    ax.set_yticklabels([fill(l, 32) for l in top.label], fontsize=st["tick_fs"])
    ax.set_ylim(-0.9, len(top) + .6)
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
    ax.set_xlabel(f"Compound annual growth, {GROW_NOW[0]}-{GROW_NOW[1]} (%/year)")
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
    cols, markers = _growth_encoding(D, top)
    uts, totals = D["ukbb_ts_all"], D["ukbb_totals"]
    styles = field_line_styles(D, top.label)
    span = range(GROW_PREV[0], GROW_NOW[1] + 1)

    ends = []
    for code, r in top.iterrows():
        s = uts[code].reindex(span).replace(0, np.nan).dropna()
        from textwrap import fill
        ax.plot(s.index, s.values, color=cols[r.label], lw=2, marker=markers[r.label],
                ls=styles[r.label]["linestyle"],
                ms=st["marker_size"] * 0.8, markeredgecolor="black", markeredgewidth=.45,
                label=fill(r.label, 30))
        ends.append((float(s.iloc[-1]), r.label))
    all_ukbb = totals.reindex(span)
    ax.plot(all_ukbb.index, all_ukbb.values, color=C_INK, lw=2.4, ls="--", alpha=0.8,
            label="All UK Biobank\nfield assignments")
    ends.append((float(all_ukbb.iloc[-1]), "ALL UK Biobank papers"))

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.axvline(GROW_NOW[0], color=C_INK, ls="--", lw=1)
    ax.set_xlim(GROW_PREV[0] - .2, GROW_NOW[1] + .2)
    ax.set_xticks(range(GROW_PREV[0], GROW_NOW[1] + 1, 2))
    ax.set_xlabel("Publication year")
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
          "its share across all fields; Supplementary Figure 2 shows 25 fields."),
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
        "A-Y": ("UK Biobank's share of each field's measured top-decile citation pool, "
                "for the same 25 fields as Supplementary Figure 1. Separate panels use "
                "common logarithmic axes, preserving all measured years. The eight "
                "principal fields retain their colours; other fields are steel blue."),
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
        # Reserve space below C for a readable key without extending its log scale.
        position = ax_c.get_position()
        ax_c.set_position([position.x0, position.y0 + .05,
                           position.width, position.height - .025])

        draw_impact_map(ax_a, D, top_m=MAP_M_MAIN)
        draw_activity_index(ax_b, D, top_m=ACT_TOP_M_MAIN)
        # A third of the page width: eight direct labels and the empty years they need to
        # sit in would leave the decade this panel is about squeezed into two thirds of
        # its own axis, so here the fields go into a legend and the panel is cut to the
        # five that reach furthest. SI 1B carries all 25, direct-labelled, at full width.
        #
        legend_fs = st.get("main_legend_fs", 14)
        draw_top_decile_share(ax_c, D, label_chars=22, pad_years=1, top_m=DEC_M_MAIN,
                              legend=True, legend_fontsize=legend_fs, legend_below=True)
        block_footprint_quality(fig, gs[2:4, 0:3], D, letters="DEFGHIJK")

        for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
            panel_label(ax, letter)
        footprint_legend(fig, D, fontsize=legend_fs)
        fig.canvas.draw()
        label_bottom = ax_c.xaxis.label.get_window_extent(fig.canvas.get_renderer()).y0
        legend_top = ax_c.transAxes.inverted().transform(
            (0, label_bottom - 8 * fig.dpi / 72))[1]
        ax_c.get_legend().set_bbox_to_anchor((0, legend_top, 1, 0))
        fig._ukb_caption = (
            "Academic position of UK Biobank research, 2015-2025. (A) Within-division "
            "activity index versus relative mean normalised citation impact for the 16 "
            "largest eligible fields; marker area reflects publication count. The red "
            "horizontal line marks overall UK Biobank impact and the vertical line marks "
            "activity parity. (B) Activity indices for the ten largest fields. "
            "(C) Annual UK Biobank shares of field-specific top-decile citation pools "
            "for the five leading fields by final-year share; the dashed line is the "
            "pooled UK Biobank share. (D-K) UK Biobank publications as a percentage of all "
            "publications in each of eight fields, partitioned into top-decile (solid) "
            "and other above-median (pale) citation credit. In-band percentages describe "
            "the corresponding fractions of UK Biobank's own field output. Citation "
            "thresholds are measured separately by field and publication year; integer "
            "ties can produce achieved fractions different from 10% and 50%."
        )

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
        panel_label(ax, "A")
        fig._ukb_caption = (
            "Research specialisation and citation impact of UK Biobank publications, "
            "2015-2025, across the 25 largest fields with both measures available. "
            "The horizontal axis compares the field's share of UK Biobank output with "
            "its reference-literature share within the same research division. The "
            "vertical axis compares mean normalised citation impact with the same-field "
            "reference. Both axes are logarithmic; marker area represents publication "
            "count. The red line marks overall UK Biobank impact, and parity lines mark "
            "a ratio of one. Curved arrows identify every field without covering its marker."
        )
        if save:
            savefig(fig, "03_02_supplementary_figure_01_impact_map_full")
    return fig


def figure_si_top_decile(D, save=True):
    """SI 2: the map's 25 fields, in separate panels with a common log scale."""
    st = _style()
    with _font_scale(_fs_scale("si2_top_decile")):
        codes = list(map_fields(D, MAP_M_SI).index)
        metric = CITE_SHARE_METRIC
        share = D["share_timeseries"](metric, codes=codes)
        share = share.loc[share.index.isin(D["CITE_YEARS"][metric])].where(lambda f: f > 0)
        values = share.to_numpy()
        lower, upper = np.nanmin(values) / 1.4, np.nanmax(values) * 1.5
        win = D["CITE_WIN"]
        overall = 100 * (D["ukbb"].loc[D["ukbb"].year.between(*win), metric].sum()
                         / D["whole"].loc[D["whole"].year.between(*win), metric].sum())
        fig, axes = plt.subplots(5, 5, figsize=st["figsize_si_grid"], sharex=True, sharey=True)
        fig.subplots_adjust(left=.09, right=.98, bottom=.10, top=.96, wspace=.60, hspace=.45)
        styles = field_line_styles(D, [D["label_of"](code) for code in codes])
        for i, (ax, code) in enumerate(zip(axes.flat, codes)):
            lab = D["label_of"](code)
            s = share[code]
            color = D["FIELD_COLORS"].get(lab, PALETTE_COLORS["steel_blue"])
            ax.plot(s.index, s, color=color, lw=2, marker=styles[lab]["marker"], ms=6,
                    markeredgecolor="black", markeredgewidth=.4)
            ax.axhline(overall, color="black", lw=1, ls="--")
            ax.set(yscale="log", ylim=(lower, upper))
            ax.set_xticks(year_ticks(share.index.min(), share.index.max(), 5))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            facet_ylabel(ax, lab, width=24)
            ax.tick_params(labelleft=True)
            grid_on(ax, which="major", alpha=.22)
            panel_label(ax, chr(65 + i))
            if i >= 20:
                ax.set_xlabel("Publication year")
        for ax in axes.flat[len(codes):]:
            ax.set_visible(False)
        fig.supylabel("UK Biobank share of the field's top-decile citation pool (log scale)", x=.005)
        black_legend(fig, handles=[Line2D([], [], color="black", ls="--",
                                         label=f"UK Biobank overall: {overall:.2f}%")],
                     loc="lower center", bbox_to_anchor=(.5, .015))
        fig._ukb_caption = (
            "UK Biobank's share of the measured top-decile citation pool in its 25 largest "
            "fields with available activity and citation-impact estimates (A-Y). All panels "
            "use the same logarithmic percentage scale. Each numerator is the field-year "
            "top-decile credit assigned to UK Biobank papers; its denominator is the "
            "corresponding whole-database pool. Dashed lines mark the pooled UK Biobank "
            "share across all fields in 2015-2025. Missing measurements and zeros are not "
            "drawn on the logarithmic axes. These are the same fields as Supplementary Figure 1."
        )
        summary = []
        for code in codes:
            measured = share[code].dropna()
            if not measured.empty:
                summary.append({"code": code, "field": D["label_of"](code),
                                "first_year": int(measured.index[0]),
                                "last_year": int(measured.index[-1]),
                                "first_share_pct": measured.iloc[0],
                                "last_share_pct": measured.iloc[-1],
                                "change_pp": measured.iloc[-1] - measured.iloc[0]})
        fig._ukb_tables = {
            "top_decile_share_endpoints.csv": pd.DataFrame(summary),
            "top_decile_share_by_field_year.csv": share.rename_axis("year").reset_index(),
        }
        if save:
            savefig(fig, "03_03_supplementary_figure_02_top_decile_share")
    return fig


def figure_si_growth(D, save=True):
    """SI 3 — growth, not size. A growth result inside an impact analysis, hence SI."""
    st = _style()
    with _font_scale(_fs_scale("si3_growth")):
        fig = plt.figure(figsize=st["figsize_si"])
        gs = fig.add_gridspec(1, 2, wspace=0.30, width_ratios=[1.15, 1.0],
                             left=.23, right=.98, top=.96, bottom=.21)
        axes = _assemble_axes(fig, gs, [(0, 0), (0, 1)])
        draw_growth_rates(axes[0], D)
        draw_growth_trajectories(axes[1], D)
        for ax, letter in zip(axes, "AB"):
            panel_label(ax, letter)
        handles, labels = axes[1].get_legend_handles_labels()
        black_legend(fig, handles=handles, labels=labels, loc="lower center", ncol=4,
                     bbox_to_anchor=(.55, .01), fontsize=st["legend_fs"])
        fig._ukb_caption = (
            "Growth of UK Biobank research by field. (A) Compound annual growth during "
            "2021-2025 for UK Biobank (filled markers) and the reference literature "
            "(hollow markers). Fields are ranked by UK Biobank growth, requiring at least "
            "five papers in 2021 and 20 in 2025. Filled-marker area reflects 2025 output. "
            "The right-hand column gives the change in annual growth relative to 2017-2021, "
            "in percentage points; 'new' denotes an unavailable earlier rate. "
            "(B) Publication trajectories for the same fields, on a logarithmic axis. "
            "The vertical dashed line marks 2021. The dashed curve sums UK Biobank "
            "publication-field assignments, so papers in multiple fields contribute more "
            "than once; the overall growth reference in A uses the same aggregate."
        )
        fig._ukb_tables = {"growth_rates_complete.csv": D["GROWTH"].reset_index()}
        if save:
            savefig(fig, "03_04_supplementary_figure_03_growth")
    return fig
