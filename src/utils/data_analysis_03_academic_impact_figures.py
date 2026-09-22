"""Shared-style diagnostic figures for the single academic-impact workflow.

The assembled manuscript figures remain in ``..._panels``. These diagnostics retain
distinct analyses without redrawing the impact map, top-decile shares or growth page.
All functions consume already-built tables; none refits or reloads an analysis.
"""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from . import shared_style as S
from . import data_analysis_03_academic_impact_panels as P


def _style():
    return S._resolve(None)


def _finish(fig, caption):
    fig._ukb_caption = caption
    return S.finalize_figure(fig)


def publish(fig, stem, registry, *, subdirectory=None):
    """Display image, relative export paths and caption; register this run only."""
    if fig is None:
        return
    style = dict(_style())
    if subdirectory:
        style["savedir"] = Path(style["savedir"]) / subdirectory
    paths = S.savefig(fig, stem, style=style)
    registry.record_figures(paths)
    for name, frame in getattr(fig, "_ukb_tables", {}).items():
        registry.save_table(frame, name)
    S.display_figure(fig, paths)
    plt.close(fig)


def _legend(fig, handles, ncol=4, *, fontsize=None):
    return S.black_legend(fig, handles=handles, loc="lower center", ncol=ncol,
                          bbox_to_anchor=(.5, .01),
                          fontsize=_style()["legend_fs"] if fontsize is None else fontsize)


def _field_handles(D):
    styles = P.field_line_styles(D)
    return [Line2D([], [], **styles[lab], lw=2, ms=6, markeredgecolor="black", markeredgewidth=.4,
                   label=fill(lab, 27)) for lab in D["TOP_LABELS"]]


def _field_grid(D, *, height=8.5, bottom=.15):
    n = len(D["TOP_LABELS"])
    fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(18, height), squeeze=False)
    fig.subplots_adjust(left=.08, right=.96, top=.94, bottom=bottom,
                        hspace=.48, wspace=.55)
    axes = axes.ravel()
    for i, (ax, label) in enumerate(zip(axes, D["TOP_LABELS"])):
        S.panel_label(ax, chr(65 + i))
        S.facet_ylabel(ax, label, width=24)
        S.style_axis(ax, grid_kws={"which": "major", "alpha": .25})
        ax.set_xlabel("Publication year")
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def whole_database_trends(D):
    fig, axes = _field_grid(D)
    for ax, lab in zip(axes, D["TOP_LABELS"]):
        whole = D["whole_ts"][lab]
        ukb = D["ukbb_ts"][lab].reindex(whole.index).fillna(0)
        ax.stackplot(whole.index, whole - ukb, ukb,
                     colors=[S.palette("light_blue"), S.palette("red")])
        ax.set_ylim(bottom=0)
        ax.set_xticks(S.year_ticks(whole.index.min(), whole.index.max(), 4))
        ax.yaxis.set_major_formatter(mticker.EngFormatter())
        ax.margins(x=.01)
    _legend(fig, [Patch(color=S.palette("light_blue"), label="Other publications"),
                  Patch(color=S.palette("red"), label="UK Biobank publications")], 2)
    fig.supylabel("Publications per year", x=.005)
    return _finish(fig, "Annual publication output in the eight leading UK Biobank "
                   "Fields of Research, shown in panels A-H in field order. Stacked areas "
                   "partition the whole database into UK Biobank and other publications. "
                   "A publication can contribute to multiple fields. Counts describe the "
                   "reference literature, not an estimated effect of UK Biobank on its growth.")


def publication_contribution(D):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(left=.07, right=.98, bottom=.23, top=.94, wspace=.25)
    styles = P.field_line_styles(D)
    for ax, frame, ylabel in zip(axes, [D["ukbb_ts"], D["share_ts"]],
                                 ["UK Biobank publications", "Share of field publications (%)"]):
        for lab in D["TOP_LABELS"]:
            s = frame[lab]
            ax.plot(s.index, s, **styles[lab], lw=2, ms=6,
                    markeredgecolor="black", markeredgewidth=.4)
        ax.set(xlabel="Publication year", ylabel=ylabel, ylim=(0, None))
        ax.set_xticks(S.year_ticks(frame.index.min(), frame.index.max(), 3))
        S.style_axis(ax, grid_kws={"which": "major", "alpha": .25})
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(100))
    S.label_panels(axes, "AB")
    S.align_legend_rows(_legend(fig, _field_handles(D), fontsize=14))
    return _finish(fig, "UK Biobank output in its eight leading Fields of Research. "
                   "(A) Annual publication counts. (B) UK Biobank publications as a percentage "
                   "of all publications in the same field and year. Whole counting is used "
                   "within each field; a paper may appear in more than one field.")


def annual_growth_speed(D):
    fig, axes = _field_grid(D, bottom=.12)
    styles = P.field_line_styles(D)
    for ax, lab in zip(axes, D["TOP_LABELS"]):
        s = D["whole_ts"][lab].diff().iloc[1:]
        ax.plot(s.index, s, **styles[lab], lw=2, ms=6,
                markeredgecolor="black", markeredgewidth=.4)
        ax.axhline(0, color="black", lw=.8)
        ax.set_xticks(S.year_ticks(s.index.min(), s.index.max(), 3))
        ax.yaxis.set_major_formatter(mticker.EngFormatter())
    fig.supylabel("Change in publications from previous year", x=.005)
    return _finish(fig, "Year-on-year changes in whole-database publication counts for "
                   "the eight leading UK Biobank fields (A-H). Values are first differences "
                   "in counts, not percentage growth rates or estimates of UK Biobank's contribution.")


def activity_over_time(D):
    fig, axes = plt.subplots(1, 2, figsize=(17, 9.6), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(left=.20, right=.98, bottom=.19, top=.95, wspace=.3)
    P.draw_activity_index(axes[0], D, top_m=20)
    annual = pd.DataFrame({y: D["activity_index"]((y, y))
                           for y in sorted(D["whole"].year.unique())}).T
    counts = D["ukbb"].pivot_table(index="year", columns="code", values=D["FRAC_COL"],
                                   aggfunc="sum")
    annual = annual.where(counts.reindex_like(annual) >= 10)
    annual = annual.reindex(columns=D["TOP_CODES"]).dropna(how="all")
    styles = P.field_line_styles(D)
    for code, lab in zip(D["TOP_CODES"], D["TOP_LABELS"]):
        axes[1].plot(annual.index, annual[code], **styles[lab],
                     lw=2, ms=6, markeredgecolor="black", markeredgewidth=.4)
    axes[1].set(yscale="log", xlabel="Publication year", ylabel="Within-division activity index")
    axes[1].axhline(1, color="black", ls="--", lw=1)
    axes[1].set_xticks(S.year_ticks(annual.index.min(), annual.index.max(), 3))
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    S.style_axis(axes[1], grid_kws={"which": "major", "alpha": .25})
    S.label_panels(axes, "AB")
    S.align_legend_rows(_legend(fig, _field_handles(D), fontsize=14))
    ranking = (D["ukbb"].loc[D["ukbb"].year >= P.ANALYSIS_MIN]
               .groupby(["code", "for_label"])[D["VALUE"]].sum()
               .nlargest(20).rename("publications").reset_index())
    ranking["activity_index"] = ranking.code.map(D["ACTIVITY"])
    fig._ukb_tables = {"activity_index_top20.csv": ranking}
    return _finish(fig, "Research specialisation by field. (A) Within-division activity "
                   "indices for the 20 largest UK Biobank fields; marker size encodes publication "
                   "count. (B) Annual activity indices for the leading eight fields, restricted "
                   "to field-years with at least ten UK Biobank publications. An index of one "
                   "indicates the same relative concentration as the reference literature.")


def citation_measure_comparison(D, min_docs=20):
    head = D["CITE_HEADLINE"][2:]
    order = D["IMPACT"].reindex(D["TOP_CODES"]).dropna(subset=[f"rci_{head}"])
    order = order.sort_values(f"rci_{head}")
    colors = [S.palette(name) for name in ("navy", "red", "steel_blue", "green")]
    names = {"n_mncs": "Mean normalised citations", "n_top10f": "Field top-decile credit",
             "n_top50f": "Above-median credit", "n_fcr": "Field citation ratio"}
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.8))
    fig.subplots_adjust(left=.18, right=.98, bottom=.125, top=.95, wspace=.3)
    y = np.arange(len(order))
    for i, metric in enumerate(D["RCI_WEIGHTS"]):
        color = colors[i % len(colors)]
        axes[0].scatter(order[f"rci_{metric[2:]}"] , y, s=80, color=color,
                        marker=["o", "D", "s", "^"][i % 4], edgecolor="black", lw=.5,
                        label=names.get(metric, metric), zorder=3)
        axes[0].axvline(D["OVERALL"][metric[2:]], color=color, ls="--", lw=1)
    axes[0].set_yticks(y, [fill(D["label_of"](c), 28) for c in order.index])
    axes[0].set_xlabel("Relative citation impact")
    S.black_legend(axes[0], loc="lower right")
    annual = D["rci_timeseries"](D["CITE_HEADLINE"], min_docs)
    styles = P.field_line_styles(D)
    for code, lab in zip(D["TOP_CODES"], D["TOP_LABELS"]):
        axes[1].plot(annual.index, annual[code], **styles[lab], lw=2,
                     ms=6, markeredgecolor="black", markeredgewidth=.4)
    axes[1].set(yscale="log", xlabel="Publication year", ylabel="Relative mean normalised citation impact")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    axes[1].set_xticks(S.year_ticks(annual.index.min(), annual.index.max(), 3))
    S.black_legend(axes[1], handles=_field_handles(D), loc="upper right", fontsize=10)
    for ax in axes:
        S.style_axis(ax, grid_kws={"which": "major", "alpha": .25})
    S.label_panels(axes, "AB")
    measures = [f"rci_{metric[2:]}" for metric in D["RCI_WEIGHTS"]]
    spread_measures = [f"rci_{metric[2:]}" for metric in D["RCI_WEIGHTS"]
                       if metric != "n_top50f"]
    comparison = order[measures].copy()
    comparison["field"] = [D["label_of"](code) for code in comparison.index]
    if spread_measures:
        denominator = comparison[spread_measures].min(axis=1).replace(0, np.nan)
        comparison["max_min_ratio_excluding_top50f"] = comparison[spread_measures].max(axis=1) / denominator
    fig._ukb_tables = {"citation_measure_agreement.csv": comparison.reset_index()}
    return _finish(fig, "Sensitivity of field-level impact to the citation measure. "
                   "(A) UK Biobank-to-reference ratios under each available normalised measure; "
                   "dashed lines mark the corresponding overall UK Biobank ratios. (B) Annual "
                   f"relative mean normalised citation impact, requiring at least {min_docs} "
                   "measured UK Biobank papers per field-year. Missing reference cells are not zeros.")


def citation_cutoffs(D, cuts):
    if cuts is None:
        return None
    fig, axes = _field_grid(D)
    for ax, code in zip(axes, D["TOP_CODES"]):
        frame = cuts.xs(code, level="code").sort_index()
        for column, color, marker in [("thr10", "navy", "o"), ("median", "red", "s")]:
            ax.plot(frame.index, frame[column], color=S.palette(color), lw=2, marker=marker,
                    ms=6, markeredgecolor="black", markeredgewidth=.4)
        ax.set_ylim(bottom=0)
        ax.set_xticks(S.year_ticks(frame.index.min(), frame.index.max(), 3))
    _legend(fig, [Line2D([], [], color=S.palette("navy"), marker="o", label="Top-decile cut-off"),
                  Line2D([], [], color=S.palette("red"), marker="s", label="Median cut-off")], 2)
    fig.supylabel("Citations needed to meet the field-year cut-off", x=.005)
    return _finish(fig, "Citation thresholds underlying the main figure's quality bands, "
                   "shown for the eight leading fields (A-H). Top-decile cut-offs are measured "
                   "within each publication year and field. Median cut-offs use the measured "
                   "threshold where available and the recorded reference median otherwise. "
                   "Integer citation ties mean that achieved fractions can differ from 10% and 50%. "
                   "These are snapshot thresholds, not annual citation accrual rates.")


def top_decile_pool(D):
    column = D["TOP10_COL"]
    if column is None:
        return None
    def pool(frame):
        values = frame.pivot_table(index="year", columns="code", values=column, aggfunc="sum")
        measured = frame.pivot_table(index="year", columns="code", values=f"{column}_docs", aggfunc="sum")
        return values.where(measured > 0)
    whole, ukb = pool(D["whole"]), pool(D["ukbb"])
    fig, axes = _field_grid(D, height=8.4, bottom=.13)
    legend_handles = []
    rows = []
    for ax, code, lab in zip(axes, D["TOP_CODES"], D["TOP_LABELS"]):
        w = whole[code].dropna()
        u = ukb[code].reindex(w.index)
        other_bars = ax.bar(w.index, w - u, bottom=u, color="white", edgecolor="black", lw=.5,
                            label="Other publications")
        ukbb_bars = ax.bar(w.index, u, color=S.palette("steel_blue"), edgecolor="black", lw=.5,
                           label="UK Biobank publications")
        ax.yaxis.set_major_formatter(mticker.EngFormatter())
        ax.set_xticks(S.year_ticks(w.index.min(), w.index.max(), 3))
        ax.grid(False)
        ratio = 100 * u / w.where(w > 0)
        twin = ax.twinx()
        share_line, = twin.plot(ratio.index, ratio, color=S.palette("red"), marker="o", ms=6,
                                markeredgecolor="black", markeredgewidth=.5, lw=2,
                                label="UK Biobank share (right axis)")
        if not legend_handles:
            legend_handles = [other_bars, ukbb_bars, share_line]
        twin.set_ylim(0, max(.6, ratio.max() * 1.2))
        twin.yaxis.set_major_formatter(mticker.PercentFormatter(100))
        twin.spines["right"].set_visible(True)
        twin.grid(False)
        measured_ratio = ratio.dropna()
        first, last = measured_ratio.index[0], measured_ratio.index[-1]
        rows.append({"field": lab, "year": last, "pool": w.loc[last], "ukbb": u.loc[last],
                     "ukbb_share_pct": ratio.loc[last], "measured_from": first,
                     "initial_ukbb_share_pct": ratio.loc[first],
                     "change_pp": ratio.loc[last] - ratio.loc[first],
                     "fold_change": ratio.loc[last] / ratio.loc[first] if ratio.loc[first] > 0 else np.nan})
    fig._ukb_tables = {"top_decile_pool_summary.csv": pd.DataFrame(rows)}
    fig.supylabel("Publications in the field's top-decile pool", x=.005)
    _legend(fig, legend_handles, 3, fontsize=14)
    return _finish(fig, "Size of each field's measured top-decile citation pool and UK "
                   "Biobank's contribution, across eight leading fields (A-H). Stacked white "
                   "and blue bars show other and UK Biobank publication counts, respectively; "
                   "red lines show UK Biobank's percentage of each pool "
                   "on the right axis. Unmeasured field-year cells remain missing.")

def author_cohort_profile(author_summary, author_paper, cohort_order):
    """Keep the legacy author-key cohort, with fractional credit unchanged."""
    authors = author_summary[author_summary.entry_cohort.isin(cohort_order)]
    papers = author_paper[author_paper.entry_cohort.isin(cohort_order)]
    by_author = authors.groupby("entry_cohort", observed=True)
    by_paper = papers.groupby("entry_cohort", observed=True)
    raw = pd.DataFrame({
        "Unique authors": by_author.author_key.nunique(),
        "Fractional paper credit": by_paper.fractional_paper_credit.sum(),
        "Fractional citation credit": by_paper.fractional_citation_credit.sum(),
        "Top-decile paper credit": by_paper.fractional_top_decile_credit.sum(),
        "First/last-author credit": by_paper.leadership_author_credit.sum(),
    }).T.reindex(columns=cohort_order).fillna(0)
    shares = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.subplots_adjust(left=.24, right=.98, bottom=.26, top=.92)
    colors = S.blue_colormap()(np.linspace(0, 1, len(cohort_order)))
    left = np.zeros(len(shares))
    for cohort, color in zip(cohort_order, colors):
        vals = shares[cohort].to_numpy()
        ax.barh(shares.index, vals, left=left, height=.64, color=color,
                edgecolor="black", lw=.5, label=cohort.replace("\n", " "))
        for y, x, v in zip(range(len(shares)), left, vals):
            if v >= .065:
                ink = "white" if np.mean(color[:3]) < .5 else "black"
                ax.text(x + v / 2, y, f"{100 * v:.0f}%", ha="center", va="center",
                        fontsize=_style()["annot_fs"], color=ink)
        left += vals
    ax.set(xlim=(0, 1), xlabel="Share of total contribution (%)")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1))
    S.style_axis(ax, grid=False)
    S.panel_label(ax, "A")
    S.black_legend(ax, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2)
    fig._ukb_tables = {
        "fig3a_author_entry_cohort_contribution_shares.csv": shares.rename_axis("measure").reset_index(),
        "fig3a_author_entry_cohort_contribution_raw.csv": raw.rename_axis("measure").reset_index(),
    }
    return _finish(fig, "Contributions to UK Biobank research by author-entry cohort, "
                   "2015-2025. Authors are grouped by their first observed publication within "
                   "this window. Rows show each cohort's share of authors, fractional papers, "
                   "fractional citations, field-year top-decile credit and first/last-author "
                   "credit. Author keys retain the original name-based fallback when a "
                   "researcher identifier is unavailable; this is not the resolved-only "
                   "author population analysed in the author-characteristics notebook.")


def author_fingerprint(author_summary, impact_short="MNCS", top_n=24):
    reference = author_summary[author_summary.showcase_paper_count >= 3]
    if reference.empty:
        reference = author_summary
    selected = author_summary.head(top_n)
    specs = [
        ("showcase_paper_count", "Papers", "{:,.0f}"),
        ("fractional_paper_credit", "Fractional\npapers", "{:.1f}"),
        ("fractional_citation_credit", "Fractional\ncitations", "{:,.0f}"),
        ("showcase_h_index", "UKB\nh-index", "{:,.0f}"),
        ("mean_impact_metric", f"Mean\n{impact_short}", "{:.2f}"),
        ("fractional_top_decile_credit", "Top-decile\ncredit", "{:.2f}"),
        ("leadership_share", "First/last\nshare", "{:.0%}"),
        ("active_span_years", "Active span\n(years)", "{:.0f}"),
    ]
    rank_columns = []
    for col, _, _ in specs:
        measured = reference[col].replace([np.inf, -np.inf], np.nan)
        ranks = measured.rank(pct=True, method="average")
        if measured.nunique(dropna=True) == 1:
            ranks = ranks.where(measured.isna(), .5)
        rank_map = pd.Series(ranks.to_numpy(), index=reference.author_key)
        rank_columns.append(selected.author_key.map(rank_map).to_numpy())
    ranks = np.asarray(rank_columns).T
    finite = ranks[np.isfinite(ranks)]
    if not len(finite):
        raise ValueError("No measured author percentiles are available.")
    cmap = S.blue_cream_red_colormap()
    norm = Normalize(finite.min(), finite.max())
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.subplots_adjust(left=.25, right=.90, bottom=.09, top=.95)
    im = ax.pcolormesh(np.ma.masked_invalid(ranks), cmap=cmap, norm=norm,
                      edgecolors="black", linewidth=.4)
    ax.set(ylim=(len(selected), 0), xlim=(0, len(specs)))
    ax.set_xticks(np.arange(len(specs)) + .5, [s[1] for s in specs])
    ax.set_yticks(np.arange(len(selected)) + .5, selected.author_name.fillna(""))
    ax.tick_params(length=4)
    ax.grid(False)
    for i, (_, row) in enumerate(selected.iterrows()):
        for j, (col, _, fmt) in enumerate(specs):
            if pd.notna(row[col]):
                rgba = cmap(norm(ranks[i, j])) if np.isfinite(ranks[i, j]) else (1, 1, 1, 1)
                luminance = np.dot(rgba[:3], [.2126, .7152, .0722])
                ax.text(j + .5, i + .5, fmt.format(row[col]), ha="center", va="center",
                        fontsize=_style()["annot_fs"], color="white" if luminance < .5 else "black")
    S.panel_label(ax, "A")
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.035)
    S.style_colorbar(cb, "Percentile rank among recurrent authors")
    cb.set_ticks(np.linspace(finite.min(), finite.max(), 5))
    cb.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))
    fig._ukb_tables = {
        "fig_author_influence_fingerprint_top_authors.csv":
            selected[["author_name", "ukb_influence_score"] + [s[0] for s in specs]]
    }
    return _finish(fig, "Multidimensional profiles of the 24 highest-ranked authors by "
                   "the composite UK Biobank influence score, 2015-2025. Numbers are observed "
                   "values; colours indicate metric-specific percentile ranks among authors "
                   "with at least three papers, using the observed minimum and maximum. "
                   "Missing ranks are uncoloured. Top-decile credit uses publication-year "
                   "and field-specific thresholds. Author identities retain the legacy "
                   "name-based fallback used in the cohort analysis.")


def _single(*, size=(9, 6), bottom=.15, left=.12, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        fig.subplots_adjust(left=left, right=.97, bottom=bottom, top=.92)
    else:
        fig = ax.figure
    S.panel_label(ax, "A")
    S.style_axis(ax, grid_kws={"which": "major", "alpha": .25})
    return fig, ax


def _table(fig, name, frame):
    if not hasattr(fig, "_ukb_tables"):
        fig._ukb_tables = {}
    fig._ukb_tables[name] = frame


def disciplinary_composition(analysis_papers, for_long, TOP_N=15, *, ax=None):
    discipline_summary = (
        for_long.groupby(["for_code", "for_name"], as_index=False)
        .agg(
            fractional_paper_count=("fractional_papers", "sum"),
            papers=("id", "nunique"),
        )
    )
    discipline_summary["percent_of_corpus"] = (
        100 * discipline_summary["fractional_paper_count"] / analysis_papers["id"].nunique()
    )

    discipline_plot = (
        discipline_summary.nlargest(TOP_N, "fractional_paper_count")
        .sort_values("fractional_paper_count")
        .copy()
    )


    fig, ax = _single(size=(13, 9), left=.34, bottom=.10, ax=ax)
    values = discipline_plot.fractional_paper_count
    colors = S.blue_colormap()(.15 + .85 * values / values.max())
    bars = ax.barh(np.arange(len(values)), values, color=colors, height=.7, edgecolor="black", lw=.5)
    ax.set_yticks(np.arange(len(values)), [fill(t, 36) for t in discipline_plot.for_name])
    ax.set(xlabel="Fractional publication count", xlim=(0, values.max() * 1.28))
    ax.grid(False)
    for bar, row in zip(bars, discipline_plot.itertuples()):
        ax.annotate(f"{row.fractional_paper_count:,.0f} ({row.percent_of_corpus:.1f}%)",
                    (bar.get_width(), bar.get_y() + bar.get_height()/2), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=_style()["annot_fs"])
    _table(fig, "fig01_disciplinary_composition.csv",
           discipline_summary.sort_values("fractional_paper_count", ascending=False))
    return _finish(fig, "Pooled disciplinary composition of Showcase+ publications, "
                   "2013-2025. Each paper contributes equally across its assigned four-digit "
                   "Fields of Research. Bars show the 15 largest fields; percentages use "
                   "all included publications, including unclassified papers, as denominator.")


def cumulative_citation_stock(analysis_papers, *, ax=None):
    yearly = (
        analysis_papers.groupby("year", as_index=False)
        .agg(
            papers=("id", "nunique"),
            citations=("times_cited", "sum"),
        )
        .sort_values("year")
    )
    yearly["cumulative_papers"] = yearly["papers"].cumsum()
    yearly["cumulative_citations"] = yearly["citations"].cumsum()
    yearly["cumulative_paper_share"] = (
        100 * yearly["cumulative_papers"] / yearly["cumulative_papers"].iloc[-1]
    )
    yearly["cumulative_citation_share"] = (
        100 * yearly["cumulative_citations"] / yearly["cumulative_citations"].iloc[-1]
    )


    fig, ax = _single(ax=ax)
    for column, name, color in [
            ("cumulative_paper_share", "Publications", "navy"),
            ("cumulative_citation_share", "Citation stock", "red")]:
        ax.plot(yearly.year, yearly[column], color=S.palette(color), lw=2, marker="o",
                ms=7, markeredgecolor="black", markeredgewidth=.5, label=name)
    ax.set(xlim=(yearly.year.min(), yearly.year.max()), ylim=(0, 103),
           xlabel="Publication year", ylabel="Cumulative share of 2013-2025 total (%)")
    ax.set_xticks(S.year_ticks(yearly.year.min(), yearly.year.max(), 3))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
    S.black_legend(ax, loc="upper left")
    _table(fig, "fig02_cumulative_output_and_citation_stock.csv", yearly)
    return _finish(fig, "Cumulative shares of Showcase+ publications and their citation "
                   "stock by publication year, 2013-2025. Citations are those recorded at "
                   "the data snapshot, assigned to the year of the cited paper; this is "
                   "not a historical time series of citations received in each year.")


def field_citation_stock(for_long, TOP_N=15, *, ax=None, cax=None):
    field_impact = (
        for_long.groupby(["for_code", "for_name"], as_index=False)
        .agg(
            fractional_papers=("fractional_papers", "sum"),
            papers=("id", "nunique"),
            fractional_citations=("fractional_citations", "sum"),
        )
    )
    field_impact["citations_per_fractional_paper"] = (
        field_impact["fractional_citations"] / field_impact["fractional_papers"]
    )

    field_plot = (
        field_impact.nlargest(TOP_N, "fractional_citations")
        .sort_values("fractional_citations")
        .copy()
    )

    fig, ax = _single(size=(14, 9), left=.32, bottom=.10, ax=ax)
    norm = Normalize(field_plot.fractional_papers.min(), field_plot.fractional_papers.max())
    cmap = S.blue_cream_red_colormap()
    bars = ax.barh(np.arange(len(field_plot)), field_plot.fractional_citations,
                   color=cmap(norm(field_plot.fractional_papers)), edgecolor="black", lw=.5)
    ax.set_yticks(np.arange(len(field_plot)), [fill(t, 34) for t in field_plot.for_name])
    maximum = field_plot.fractional_citations.max()
    ax.set(xlim=(0, maximum * 1.22), xlabel="Fractionally allocated citations")
    ax.xaxis.set_major_formatter(mticker.EngFormatter())
    ax.grid(False)
    for bar in bars:
        ax.annotate(f"{bar.get_width():,.0f}", (bar.get_width(), bar.get_y()+bar.get_height()/2),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=_style()["annot_fs"])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, cax=cax,
                      fraction=.035, pad=.04)
    S.style_colorbar(cb, "Fractional publication count")
    cb.set_ticks(np.linspace(norm.vmin, norm.vmax, 5))
    cb.ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    _table(fig, "fig04_field_citation_impact.csv",
           field_impact.sort_values("fractional_citations", ascending=False))
    return _finish(fig, "The 15 Fields of Research with the largest allocated citation "
                   "stock among Showcase+ papers published in 2013-2025. Both papers and "
                   "their snapshot citations are divided equally across assigned four-digit "
                   "fields. Bar length shows citations and colour shows fractional publication "
                   "count, scaled to the observed range among the displayed fields.")


def citation_concentration(analysis_papers, *, ax=None):
    ordered_citations = analysis_papers["times_cited"].clip(lower=0).sort_values(
        ascending=False
    ).reset_index(drop=True)
    total_citations = ordered_citations.sum()
    if total_citations <= 0:
        raise ValueError("Citation concentration cannot be calculated because all counts are zero.")

    concentration = pd.DataFrame({
        "top_paper_share": 100 * np.arange(1, len(ordered_citations) + 1) / len(ordered_citations),
        "citation_share": 100 * ordered_citations.cumsum() / total_citations,
    })

    checkpoints = []
    for top_share in (1, 5, 10, 20, 50):
        index = max(0, int(np.ceil(len(ordered_citations) * top_share / 100)) - 1)
        checkpoints.append({
            "top_paper_percent": top_share,
            "citation_percent": concentration.iloc[index]["citation_share"],
        })
    checkpoint_table = pd.DataFrame(checkpoints)


    fig, ax = _single(ax=ax)
    ax.plot(concentration.top_paper_share, concentration.citation_share,
            color=S.palette("navy"), lw=2.3)
    ax.fill_between(concentration.top_paper_share, concentration.citation_share,
                    concentration.top_paper_share, color=S.palette("light_blue"), alpha=.18)
    ax.plot([0, 100], [0, 100], color="black", ls="--", lw=1, label="Equal citation counts")
    shown = checkpoint_table[checkpoint_table.top_paper_percent.isin([1, 5, 10])]
    ax.scatter(shown.top_paper_percent, shown.citation_percent, s=85,
               color=S.palette("cream"), edgecolor="black", zorder=4)
    # A separate key avoids labels crossing the steep part of the concentration curve.
    S.summary_box(ax, "\n".join(f"Top {r.top_paper_percent:g}%: {r.citation_percent:.1f}% of citations"
                  for r in shown.itertuples()), x=.97, y=.08, ha="right", va="bottom")
    ax.set(xlim=(0, 100), ylim=(0, 103), xlabel="Most-cited publications included (%)",
           ylabel="Cumulative share of citations (%)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
    S.black_legend(ax, loc="lower right", bbox_to_anchor=(1, .33))
    _table(fig, "fig05_citation_concentration_curve.csv", concentration)
    _table(fig, "fig05_citation_concentration_checkpoints.csv", checkpoint_table)
    return _finish(fig, "Concentration of snapshot citations across Showcase+ papers "
                   "published in 2013-2025, ranked from most to least cited. Highlighted points "
                   "mark the leading 1%, 5% and 10% of publications. The dashed diagonal "
                   "represents equal citation counts across papers.")


def citation_cohorts(analysis_papers, CITATION_SNAPSHOT_YEAR=2026, MIN_YEAR=2013, *, ax=None):
    IMPACT_MAX_YEAR = CITATION_SNAPSHOT_YEAR - 2
    cohort_papers = analysis_papers[analysis_papers["year"].le(IMPACT_MAX_YEAR)].copy()
    cohort_papers["log_citations_per_year"] = np.log10(
        cohort_papers["citations_per_year"] + 1
    )

    cohort_edges = list(range(MIN_YEAR, IMPACT_MAX_YEAR + 2, 3))
    if cohort_edges[-1] <= IMPACT_MAX_YEAR:
        cohort_edges.append(IMPACT_MAX_YEAR + 1)
    cohort_labels = [
        f"{start}-{end - 1}" for start, end in zip(cohort_edges[:-1], cohort_edges[1:])
    ]
    cohort_papers["cohort"] = pd.cut(
        cohort_papers["year"],
        bins=cohort_edges,
        labels=cohort_labels,
        right=False,
        include_lowest=True,
    )

    top_decile_threshold = cohort_papers["citations_per_year"].quantile(0.90)
    cohort_summary = (
        cohort_papers.dropna(subset=["cohort"])
        .groupby("cohort", observed=True)
        .agg(
            papers=("id", "nunique"),
            median_citations_per_year=("citations_per_year", "median"),
            p90_citations_per_year=("citations_per_year", lambda values: values.quantile(0.90)),
            top_decile_share=("citations_per_year", lambda values: 100 * values.ge(top_decile_threshold).mean()),
        )
        .reset_index()
    )

    plot_labels = cohort_summary["cohort"].astype(str).tolist()
    plot_values = [
        cohort_papers.loc[
            cohort_papers["cohort"].astype(str).eq(label),
            "log_citations_per_year",
        ].dropna().to_numpy()
        for label in plot_labels
    ]
    positions = np.arange(len(plot_values))


    fig, ax = _single(size=(10, 6.5), bottom=.19, ax=ax)
    violins = ax.violinplot(plot_values, positions=positions, widths=.75,
                           showmeans=False, showmedians=False, showextrema=False)
    colors = S.palette("red", "cream", "steel_blue", "green")
    for index, body in enumerate(violins["bodies"]):
        body.set(facecolor=colors[index % len(colors)], edgecolor="black", linewidth=.7, alpha=1)
    ax.boxplot(plot_values, positions=positions, widths=.18, patch_artist=True, showfliers=False,
               medianprops={"color": "black", "linewidth": 1.5},
               boxprops={"facecolor": "white", "edgecolor": "black"})
    ax.set_xticks(positions, [f"{r.cohort}\nn = {r.papers:,}" for r in cohort_summary.itertuples()])
    raw_ticks = np.array([0, .5, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000])
    upper = max(np.nanmax(v) for v in plot_values) * 1.08
    keep = np.log10(raw_ticks + 1) <= upper
    ax.set_yticks(np.log10(raw_ticks[keep] + 1), [f"{v:g}" for v in raw_ticks[keep]])
    ax.set(ylim=(0, upper), xlabel="Publication cohort",
           ylabel="Citations per year since publication\n(log-transformed scale)")
    _table(fig, "fig06_age_adjusted_citation_cohorts.csv", cohort_summary)
    return _finish(fig, f"Citation-rate distributions by publication cohort through "
                   f"{IMPACT_MAX_YEAR}. Snapshot citations are divided by "
                   f"{CITATION_SNAPSHOT_YEAR} minus publication year plus one. Violins show "
                   "log10(1 + annualised citations); boxes show the median and interquartile "
                   "range, with whiskers extending to 1.5 interquartile ranges. Tick labels "
                   "are on the original citation-rate scale. Annualisation is not field "
                   "normalisation and does not remove all citation-age differences.")


def citation_concentration_and_cohorts(analysis_papers, CITATION_SNAPSHOT_YEAR=2026, MIN_YEAR=2013):
    """Combine concentration and cohort distributions without changing either analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.subplots_adjust(left=.07, right=.98, bottom=.18, top=.92, wspace=.30)
    citation_concentration(analysis_papers, ax=axes[0])
    concentration_caption = fig._ukb_caption
    citation_cohorts(analysis_papers, CITATION_SNAPSHOT_YEAR, MIN_YEAR, ax=axes[1])
    cohort_caption = fig._ukb_caption
    S.label_panels(axes, "AB")
    return _finish(fig, f"(A) {concentration_caption} (B) {cohort_caption}")


def citation_survival(analysis_papers, *, ax=None):
    metric_candidates = [
        ("field_citation_ratio", "Field Citation Ratio", "FCR"),
        ("relative_citation_ratio", "Relative Citation Ratio", "RCR"),
        ("citations_per_year", "Citations per year", "annualised citations"),
    ]
    for metric_column, metric_label, metric_short in metric_candidates:
        valid_metric = analysis_papers[metric_column].replace([np.inf, -np.inf], np.nan).dropna()
        valid_metric = valid_metric[valid_metric.ge(0)]
        if len(valid_metric) >= 100:
            break
    else:
        raise ValueError("No citation impact metric has enough valid values.")

    positive_metric = valid_metric[valid_metric.gt(0)]
    if positive_metric.empty:
        raise ValueError(f"{metric_label} has no positive values.")

    x_min = max(float(positive_metric.quantile(0.005)), 0.02)
    x_max = float(positive_metric.quantile(0.995))
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 320)
    y_share = np.array([100 * valid_metric.ge(value).mean() for value in x_grid])

    thresholds = [1, 2, 5, 10]
    threshold_summary = pd.DataFrame({
        "threshold": thresholds,
        "papers_at_or_above": [int(valid_metric.ge(value).sum()) for value in thresholds],
        "percent_at_or_above": [100 * valid_metric.ge(value).mean() for value in thresholds],
    })


    fig, ax = _single(size=(10, 6.5), ax=ax)
    ax.plot(x_grid, y_share, color=S.palette("navy"), lw=2.4)
    ax.fill_between(x_grid, y_share, color=S.palette("light_blue"), alpha=.15)
    visible = threshold_summary[threshold_summary.threshold.between(x_min, x_max)]
    ax.scatter(visible.threshold, visible.percent_at_or_above, s=85,
               color=S.palette("cream"), edgecolor="black", zorder=4)
    ax.set(xscale="log", xlim=(x_min, x_max), ylim=(0, 103),
           xlabel=metric_label + " (log scale)", ylabel="Publications at or above threshold (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100))
    S.summary_box(ax, f"Median: {valid_metric.median():.2f}\n90th percentile: "
                  f"{valid_metric.quantile(.9):.2f}\nn = {len(valid_metric):,}",
                  x=.97, y=.96, ha="right", va="top")
    S.summary_box(ax, "\n".join(f"At least {r.threshold:g}: {r.percent_at_or_above:.1f}%"
                  for r in visible.itertuples()), x=.97, y=.08, ha="right", va="bottom")
    _table(fig, "fig07_normalised_citation_thresholds.csv", threshold_summary)
    benchmark = ("Values above one exceed the metric's reference benchmark."
                 if metric_short in {"FCR", "RCR"} else
                 "A field-normalised ratio was unavailable, so annualised citations are shown.")
    return _finish(fig, f"Exceedance distribution of {metric_label.lower()} for Showcase+ "
                   f"papers, 2013-2025 ({len(valid_metric):,} non-negative measured values). "
                   "The curve spans the 0.5th to 99.5th percentiles of positive values; "
                   "percentages include measured zeros in their denominator. " + benchmark)


def citation_overview(analysis_papers, for_long, TOP_N=15):
    """Assemble the four publication-level diagnostics without changing their tables."""
    fig, axes = plt.subplots(2, 2, figsize=(19, 13.3))
    fig.subplots_adjust(left=.20, right=.98, bottom=.07, top=.96, wspace=.34, hspace=.22)
    position = axes[1, 0].get_position()
    cax = fig.add_axes([position.x1 + .01, position.y0 + .05 * position.height,
                       .012, .9 * position.height])
    captions = []
    for letter, plotter, args, kwargs in [
        ("A", disciplinary_composition, (analysis_papers, for_long, TOP_N), {"ax": axes[0, 0]}),
        ("B", cumulative_citation_stock, (analysis_papers,), {"ax": axes[0, 1]}),
        ("C", field_citation_stock, (for_long, TOP_N), {"ax": axes[1, 0], "cax": cax}),
        ("D", citation_survival, (analysis_papers,), {"ax": axes[1, 1]}),
    ]:
        plotter(*args, **kwargs)
        captions.append(f"({letter}) {fig._ukb_caption}")
    S.label_panels(axes.ravel(), "ABCD")
    return _finish(fig, " ".join(captions))
