"""Consistent rendering and notebook publication for the non-academic analyses.

Keep data aggregation in the source modules. This layer owns presentation only;
in particular, directional overlap matrices are never triangularised.
"""

from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.container import BarContainer
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

from . import shared_style as S
from .shared_figure_captions import caption_for_name, suggest_caption


SECTOR_COLORS = dict(zip(
    ["University/HEI", "Hospital/Clinical", "Government/Public",
     "Research institute/Centre", "Nonprofit/Charity", "Company (non-UK)",
     "UK company", "Other/Unknown"],
    [*S.palette("blue", "light_blue", "steel_blue"), S.NON_ACADEMIC_PALETTE[6],
     *S.palette("cream", "navy", "red"),
     "white"],
))

PATENT_STATUS_COLORS = dict(zip(
    ["Active", "Application Pending", "Application Granted", "Granted Patent Expired",
     "Application Ceased", "Application Withdrawn", "Application Abandoned"],
    [S.NON_ACADEMIC_PALETTE[6],
     *S.palette("steel_blue", "blue", "light_blue", "navy", "cream", "red")],
))

TRIAL_SECTOR_COLORS = dict(zip(
    ["Academia", "Healthcare", "Industry", "Government", "Nonprofit", "Other"],
    [*S.palette("navy", "light_blue", "red", "steel_blue", "cream"), "white"],
))

# Kept in this module, rather than notebook globals, across the four isolated parts.
EXPORTS = {}

CAPTIONS = {
    "ct_mesh_rcdc_categories": "Broad classifications of UK Biobank-linked clinical trials using Dimensions MeSH tags (A, B) and all RCDC tags (C, D). Each panel shows the 15 highest-ranked categories for its measure. Left panels count trials carrying each category; right panels divide each trial's unit weight equally across its distinct tags before aggregation. Categories can therefore differ between the raw and fractional rankings. Dimensions MeSH includes tree ancestors and intervention-related terms; RCDC includes research areas and methods as well as diseases. These are classification portfolios, not disease-only rankings.",
    "ct_disease_classifications": "Disease-focused classifications of UK Biobank-linked clinical trials using RCDC (A, B) and registry condition-leaf MeSH terms (C, D). The explicit stop lists documented in the notebook remove cross-cutting RCDC tags and non-disease MeSH leaf terms before counting. Each panel shows the 15 highest-ranked categories for its measure: numbers of trials on the left and fractional trial weights on the right. Fractional weights divide each trial's unit weight equally across all its distinct retained tags before selecting the top 15. Different disease vocabularies and coverage mean that the RCDC and MeSH counts are not interchangeable.",
    "ct_diseases_mesh_ancestors": "MeSH tree-ancestor categories recorded for UK Biobank-linked clinical trials, after the documented exclusions of non-disease terms. The 15 highest-ranked ancestors are shown by trial count (A) and fractional trial weight (B). Each trial's unit weight is divided equally across its distinct retained ancestor tags before aggregation. Ancestors describe the classification hierarchy and should not be interpreted as directly matched disease terms.",
    "ct_combined_figure": "(A) Lifecycle stages of UK Biobank-linked clinical trials, split by study type. (B) Number of linked trials by start year. Counts refer to trial records, not publications.",
    "ct_icd_bodymap": "Clinical-trial disease coverage mapped to ICD-10 chapters using the keyword mapping documented above. On the schematic body outline, marker area is directly proportional to the number of trials touching a chapter, using the displayed size key. Positions indicate broad body systems, not precise organ locations; neoplasms and infectious disease are shown separately as systemic conditions. A trial may map to more than one chapter.",
    "ct_rcdc_icd": "Clinical-trial coverage by ICD-10 chapter using the documented keyword rules. (A) Counts mapped from disease-focused RCDC tags, with percentages of all included trials. (B) Counts mapped from MeSH terms (blue) and disease-focused RCDC tags (red), using the same mapping rules. Both panels share chapter order and count scale; chapters found by either vocabulary are retained, including those with zero RCDC counts. Each trial is counted once per chapter it touches, so chapters are not mutually exclusive.",
    "ct_where_sector": "Location and organisation sectors of UK Biobank-linked clinical trials. Each trial's unit weight is divided equally across its distinct country-sector pairs before aggregation.",
    "ct_country_maps_trials_vs_papers": "Geographical distribution of UK Biobank-linked clinical trials and the publications they cite, based on recorded research-organisation countries. (A, B) Trial-level country counts, with each country counted once per trial; these two panels repeat the same distribution. (C) Countries of the cited UK Biobank publications, counting each country once per publication. All panels use the same Blues colourmap. Trials and publications have separate count scales. White countries have no matched observations in the corresponding data.",
    "ct_ukbb_papers_impact_and_timeliness": "Characteristics of UK Biobank publications cited by clinical trials. (A) Raw citation counts of the cited publications on a logarithmic axis; counts below one are displayed at one, and the median is calculated from the original counts. (B) Number of distinct UK Biobank publications cited per trial, with the most-linked trial identified. (C) Publication years of the cited papers. (D) Positive paper-to-trial lags, calculated as trial start year minus publication year for each trial-paper link; same-year and negative lags are excluded and reported below. The median positive lag is marked. (E) The ten leading journals by number of cited UK Biobank papers. Paper-level distributions count each publication once, whereas panel D counts trial-paper links. These are descriptive citation links, not evidence that a publication caused a trial to start.",
    "ct_ukbb_paper_titles": "Terms in the titles of UK Biobank publications cited by clinical trials. Word size reflects frequency after the documented stop-word filtering.",
}


def trial_geography_figure(rows, *, style, world=None):
    """Stack count maps with the author main-map ramp and no country callouts."""
    if world is None:
        import geopandas as gpd
        from . import shared_paths as P
        world = gpd.read_file(P.WORLD_SHP)
    world = world.copy()
    world.columns = [column.lower() for column in world.columns]
    world = world.loc[world["admin"] != "Antarctica"].copy()
    world["iso_key"] = world["iso_a3"].where(world["iso_a3"] != "-99", world["adm0_a3"])
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 4.2 * len(rows)),
                             squeeze=False, layout="constrained")
    fig.get_layout_engine().set(h_pad=.08, hspace=.04)
    cmap = S.author_geography_colormap()
    for index, (ax, (counts, label)) in enumerate(zip(axes.flat, rows)):
        merged = world.merge(counts.rename_axis("iso3").reset_index(name="value"),
                             how="left", left_on="iso_key", right_on="iso3")
        measured = merged.loc[merged["value"].gt(0)]
        merged.plot(color="white", ax=ax, edgecolor=style["edgecolor"], linewidth=.3)
        if not measured.empty:
            measured.plot(column="value", cmap=cmap, ax=ax, edgecolor=style["edgecolor"],
                          linewidth=.3, legend=True,
                          legend_kwds={"label": label, "shrink": .65, "pad": .02})
        ax.set_xlim(-180, 180)
        ax.set_ylim(-58, 90)
        S.set_title(ax, chr(65 + index), fontsize=20, pad=8)
        ax.set_axis_off()
    finalize_figure(fig)
    return fig


def trial_classification_figure(rows, *, style):
    """Pair raw/fractional rankings by vocabulary, preserving every input value."""
    fig, axes = plt.subplots(len(rows), 2, figsize=(17, 5.5 * len(rows)),
                             squeeze=False, layout="constrained")
    fig._ukb_annotation_fs = 12
    fig.get_layout_engine().set(w_pad=.06, h_pad=.06, wspace=.025, hspace=.025)
    panels = [(vocabulary, series, bool(column))
              for vocabulary, raw, fractional in rows
              for column, series in enumerate((raw, fractional))]
    for index, (ax, (vocabulary, series, fractional)) in enumerate(zip(axes.flat, panels)):
        ranked = series.sort_values(ascending=False, kind="stable")
        positions = np.arange(len(ranked))
        color = style["c_gold"] if fractional else style["c_primary"]
        bars = ax.barh(positions, ranked.to_numpy(), height=.78, color=color,
                       edgecolor="black", linewidth=.6)
        ax.set_yticks(positions, [textwrap.fill(str(label), 34, break_long_words=False,
                                             break_on_hyphens=False)
                                 for label in ranked.index])
        ax.invert_yaxis()
        ax.set_xlabel("Fractional trial weight" if fractional else "Number of trials")
        if not fractional:
            ax.set_ylabel(vocabulary, labelpad=10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=not fractional))
        ax.set_xlim(0, max(float(ranked.max()) * 1.12, 1.) if len(ranked) else 1.)
        ax.margins(y=.025)
        ax.grid(False)
        for bar, value in zip(bars, ranked):
            ax.annotate(f"{value:.1f}" if fractional else f"{value:,.0f}",
                        (value, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=12)
        S.set_title(ax, chr(65 + index), fontsize=20, pad=8)
    finalize_figure(fig)
    return fig


def trial_icd_comparison_figure(rcdc_counts, comparison, *, total_trials, style):
    """Compare vocabularies on aligned rows, with one legible set of chapter labels."""
    if not rcdc_counts.index.isin(comparison.index).all():
        raise ValueError("The comparison must include every RCDC chapter.")
    paired = comparison.sort_values("RCDC", ascending=False, kind="stable")
    ranked = rcdc_counts.reindex(paired.index, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(17, max(10, .57 * len(paired) + 1)),
                             sharex=True, sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=.12, h_pad=.12, wspace=.035)
    for ax in axes:
        ax._ukb_title_fs = 24
        ax._ukb_label_fs = 17
        ax._ukb_tick_fs = 14
        ax._ukb_annotation_fs = 13
        ax._ukb_legend_fs = 14
    colors = dict(zip(("MeSH", "RCDC"), S.palette("steel_blue", "red")))
    ax = axes[0]
    positions = np.arange(len(ranked))
    bars = ax.barh(positions, ranked.to_numpy(), height=.70, color=colors["RCDC"])
    labels = [f"{value:,.0f} ({value / total_trials:.0%})" if total_trials else f"{value:,.0f}"
              for value in ranked]
    ax.bar_label(bars, labels=labels, padding=5, fontsize=13)
    chapter_labels = [textwrap.fill(" ".join(str(chapter).split()), 34,
                                   break_long_words=False, break_on_hyphens=False)
                      for chapter in paired.index]
    ax.set_yticks(positions, chapter_labels)
    ax.set_xlabel("Trials mapped from RCDC")
    ax.set_ylabel("ICD-10 chapter", labelpad=12)

    ax = axes[1]
    positions, height = np.arange(len(paired)), .34
    for offset, source in [(-.19, "MeSH"), (.19, "RCDC")]:
        values = paired[source].to_numpy()
        bars = ax.barh(positions + offset, values, height=height, label=source, color=colors[source])
        ax.bar_label(bars, labels=[f"{value:,.0f}" if value else "" for value in values],
                     padding=5, fontsize=13)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel("Trials mapped from MeSH or RCDC")
    maximum = max(float(paired[["MeSH", "RCDC"]].to_numpy().max()), float(ranked.max())) if len(paired) else 0
    ax.set_xlim(0, max(maximum * 1.30, 1.))
    axes[0].set_ylim(max(len(paired) - .45, .55), -.55)
    ax.legend(loc="lower right", ncol=2, columnspacing=1.2, handlelength=1.4)
    for index, ax in enumerate(axes):
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        S.set_title(ax, chr(65 + index), fontsize=24, pad=8)
    fig._ukb_caption_notes = [f"Panel A percentages use all {total_trials:,} included trials as the denominator."]
    finalize_figure(fig)
    return fig


TRIAL_PAPERS_EXPORT = "ct_ukbb_papers_impact_and_timeliness"


def trial_papers_overview_figure(citations, papers_per_trial, most_linked_title,
                                year_counts, positive_lags, journal_counts, *,
                                style, start_year, end_year):
    """Four distributions beside one tall journal panel, with consistent callouts."""
    fig = plt.figure(figsize=(20, 11), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1.05))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[:, 2])]
    fig.get_layout_engine().set(w_pad=.10, h_pad=.10, wspace=.06, hspace=.10)
    blue, cream, red, navy = S.palette("steel_blue", "cream", "red", "navy")
    box_style = dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1, alpha=1)

    def callout(ax, label, xy, position, *, curve=.22):
        return ax.annotate(
            label, xy=xy, xytext=position, textcoords="axes fraction",
            ha="left", va="top", fontsize=13, color="black", bbox=dict(box_style),
            arrowprops=dict(arrowstyle="-|>", color=navy, lw=1.3,
                            connectionstyle=f"arc3,rad={curve}", shrinkA=6, shrinkB=8),
        )

    ax = axes[0]
    raw = np.asarray(citations, dtype=float)
    raw = raw[np.isfinite(raw)]
    if raw.size:
        plotted = np.clip(raw, 1, None)
        ax.hist(plotted, bins=np.logspace(0, np.log10(plotted.max()) + .05, 20),
                color=blue, edgecolor="black", linewidth=.6)
        median = float(np.median(raw))
        marker = max(median, 1.)
        ax.axvline(marker, ls="--", color=red, lw=1.5)
        callout(ax, f"Median: {median:,.10g}", (marker, ax.get_ylim()[1] * .78), (.66, .88))
    ax.set_xscale("log")
    ax.set_xlabel("Times cited (log scale)")
    ax.set_ylabel("Number of papers")

    ax = axes[1]
    counts = papers_per_trial.sort_index()
    bars = ax.bar(counts.index, counts.to_numpy(), color=cream, edgecolor="black", linewidth=.6)
    ax.bar_label(bars, labels=[f"{value:,.0f}" for value in counts], padding=4, fontsize=13)
    if len(counts):
        largest = int(counts.index.max())
        ax.set_xlim(.2, largest + .65)
        ax.set_ylim(0, max(float(counts.max()) * 1.16, 1.))
        if most_linked_title:
            callout(ax, "Most-linked trial:\n" + textwrap.fill(str(most_linked_title), 25,
                    break_long_words=False, break_on_hyphens=False),
                    (largest + .35, counts.loc[largest]), (.37, .94), curve=-.22)
    if len(counts) <= 12:
        ax.set_xticks(counts.index)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.set_xlabel("UK Biobank papers cited\nper trial")
    ax.set_ylabel("Number of trials")

    ax = axes[2]
    ax.bar(year_counts.index.astype(int), year_counts.to_numpy(), color=blue,
           edgecolor="black", linewidth=.6)
    ax.set_xlim(start_year - .9, end_year + 1)
    ax.set_xticks(S.year_ticks(start_year, end_year, step=3))
    ax.set_xlabel("Publication year of cited papers")
    ax.set_ylabel("Number of papers")

    ax = axes[3]
    lags = np.asarray(positive_lags, dtype=float)
    lags = lags[np.isfinite(lags) & (lags > 0)]
    if lags.size:
        ax.hist(lags, bins=range(1, int(lags.max()) + 2), align="left",
                color=cream, edgecolor="black", linewidth=.6)
        median = float(np.median(lags))
        years = "three years" if median == 3 else f"{median:g} year{'s' if median != 1 else ''}"
        ax.axvline(median, ls="--", color=red, lw=1.5)
        callout(ax, f"Median: {years}", (median, ax.get_ylim()[1] * .78), (.40, .93), curve=-.22)
    else:
        ax.text(.5, .5, "No positive lags", transform=ax.transAxes, ha="center", va="center",
                fontsize=13, bbox=dict(box_style))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlabel("Paper-to-trial lag (years)\nPositive lags only")
    ax.set_ylabel("Trial-paper links")

    ax = axes[4]
    positions = np.arange(len(journal_counts))
    bars = ax.barh(positions, journal_counts.to_numpy(), color=red,
                   edgecolor="black", linewidth=.6)
    labels = [textwrap.fill(journal if "JAMA" not in journal else "JAMA", 23,
                            break_long_words=False, break_on_hyphens=False)
              for journal in journal_counts.index.astype(str)]
    ax.set_yticks(positions, labels)
    ax.bar_label(bars, labels=[f"{value:,.0f}" for value in journal_counts], padding=5, fontsize=13)
    ax.margins(x=.18, y=.035)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_xlabel("Number of papers")
    ax.set_ylabel("Journals of cited papers")

    for index, ax in enumerate(axes):
        ax._ukb_label_fs = max(17, style["label_fs"])
        ax._ukb_tick_fs = max(14, style["tick_fs"])
        ax._ukb_annotation_fs = max(13, style["annot_fs"])
        ax.grid(False, which="both")
        if index != 4:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        S.set_title(ax, chr(65 + index), fontsize=20, pad=8)
    finalize_figure(fig)
    return fig


def start_run():
    """Clear the in-memory inventory, never files from a previous execution."""
    EXPORTS.clear()


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _heatmaps(fig):
    images = [im for ax in fig.axes for im in ax.images
              if np.ndim(im.get_array()) == 2]
    # Shared colourbars require a common normalization across their panels.
    groups = {}
    for im in images:
        groups.setdefault(id(im.norm), []).append(im)
    for group in groups.values():
        values = np.concatenate([np.ma.masked_invalid(im.get_array()).compressed()
                                 for im in group])
        if not len(values):
            continue
        lo, hi = float(values.min()), float(values.max())
        norm = Normalize(lo, hi if hi > lo else lo + 1)
        for im in group:
            im.set_norm(norm)
            cmap = S.blue_cream_red_colormap()
            cmap.set_bad("white")
            im.set_cmap(cmap)
            ax = im.axes
            rows, cols = im.get_array().shape
            left, right, bottom, top = im.get_extent()
            if not getattr(ax, "_ukb_heatmap_edges", True):
                for artist in getattr(ax, "_ukb_cell_edge_artists", []):
                    artist.remove()
                ax._ukb_cell_edge_artists = []
                ax._ukb_cell_edges = False
            elif not getattr(ax, "_ukb_cell_edges", False):
                ax._ukb_cell_edge_artists = [
                    ax.vlines(np.linspace(left, right, cols + 1), min(bottom, top),
                              max(bottom, top), color="black", lw=.45, alpha=.65),
                    ax.hlines(np.linspace(bottom, top, rows + 1), left, right,
                              color="black", lw=.45, alpha=.65),
                ]
                ax._ukb_cell_edges = True
            ax.grid(False, which="both")
            ax.tick_params(which="major", bottom=True, left=True, length=4)
            # These plots use integer cell centres, including origin='lower'.
            for text in ax.texts:
                x, y = text.get_position()
                if text.get_transform() != ax.transData:
                    continue
                j, i = int(round(x)), int(round(y))
                if (abs(x-j) < .01 and abs(y-i) < .01
                        and 0 <= i < rows and 0 <= j < cols):
                    value = im.get_array()[i, j]
                    if not np.ma.is_masked(value) and np.isfinite(value):
                        r, g, b, _ = cmap(norm(value))
                        text.set_color("black" if .299*r + .587*g + .114*b > .56 else "white")
            cb = im.colorbar
            if cb is not None:
                ticks = MaxNLocator(nbins=4).tick_values(lo, hi)
                ticks = ticks[(ticks > lo) & (ticks < hi)]
                cb.set_ticks(np.unique(np.r_[lo, ticks, hi]))
                cb.formatter = FormatStrFormatter("%.1f")
                cb.update_ticks()
                cb.ax.tick_params(labelsize=getattr(cb.ax, "_ukb_tick_fs", 11), length=4)
                cb.ax.xaxis.label.set_size(getattr(cb.ax, "_ukb_label_fs", 13))
                cb.ax.yaxis.label.set_size(getattr(cb.ax, "_ukb_label_fs", 13))


def finalize_figure(fig):
    """Apply the manuscript typography before layout, export and display."""
    S.finalize_figure(fig)
    primary = [ax for ax in fig.axes if not hasattr(ax, "_colorbar")
               and not ax.get_label().startswith("inset")]
    # A twin axis shares its position with the primary axis; it gets no second letter.
    occupied = []
    for ax in primary:
        bounds = ax.get_position().bounds
        twin = any(np.allclose(bounds, other, atol=1e-5) for other in occupied)
        if not twin:
            letter = ax.get_title(loc="left") or chr(65 + len(occupied))
            S.set_title(ax, letter.upper(), fontsize=getattr(ax, "_ukb_title_fs", 20), pad=8)
            occupied.append(bounds)
        ax.xaxis.label.set_size(getattr(ax, "_ukb_label_fs", 13))
        ax.yaxis.label.set_size(getattr(ax, "_ukb_label_fs", 13))
        ax.tick_params(axis="both", labelsize=getattr(ax, "_ukb_tick_fs", 11))
        for text in ax.texts:
            x, y = text.get_position()
            if (text.get_transform() == ax.transAxes and y >= 1
                    and re.fullmatch(r"[A-Za-z][.)]?", text.get_text().strip())):
                text.remove()
                continue
            text.set_fontsize(getattr(ax, "_ukb_annotation_fs",
                                     getattr(fig, "_ukb_annotation_fs", 10)))
        for line in ax.lines:
            from matplotlib.colors import to_rgba
            if to_rgba(line.get_color())[:3] == (1., 1., 1.):
                line.set_color("black")
                line.set_linestyle(":")
                line.set_markerfacecolor("white")
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(7)
                line.set_markeredgecolor("black")
                line.set_markeredgewidth(.45)
        ax.set_facecolor("white")
        for container in ax.containers:
            if isinstance(container, BarContainer):
                for patch in container.patches:
                    patch.set_edgecolor("black")
                    patch.set_linewidth(.6)
    _heatmaps(fig)
    for ax in fig.axes:
        if hasattr(ax, "_colorbar"):
            ax.xaxis.label.set_size(getattr(ax, "_ukb_label_fs", 13))
            ax.yaxis.label.set_size(getattr(ax, "_ukb_label_fs", 13))
            ax.tick_params(labelsize=getattr(ax, "_ukb_tick_fs", 11))
    for legend in [*fig.legends, *[ax.get_legend() for ax in fig.axes]]:
        if legend is None:
            continue
        font_size = getattr(legend.axes, "_ukb_legend_fs",
                            getattr(fig, "_ukb_legend_fs", 11))
        for text in [*legend.get_texts(), legend.get_title()]:
            text.set_fontsize(font_size)
        legend.set_frame_on(True)
        legend.get_frame().set(facecolor="white", edgecolor="black", alpha=1, linewidth=.8)
        # This helper is idempotent; multiline labels get equal-height rows.
        S.align_legend_rows(legend)
        if legend.axes is not None:
            handles, labels = legend.axes.get_legend_handles_labels()
            originals = dict(zip(labels, handles))
            for handle, text in zip(legend.legend_handles, legend.get_texts()):
                original = originals.get(text.get_text())
                if isinstance(handle, plt.Line2D) and isinstance(original, plt.Line2D):
                    handle.set_color(original.get_color())
                    handle.set_linestyle(original.get_linestyle())
                    handle.set_marker(original.get_marker())
                    handle.set_markersize(original.get_markersize())
                    handle.set_markerfacecolor(original.get_markerfacecolor())
                    handle.set_markeredgecolor(original.get_markeredgecolor())
                    handle.set_markeredgewidth(original.get_markeredgewidth())
    return fig


def _caption(fig, name, caption=None):
    result = (caption or getattr(fig, "_ukb_caption", None) or CAPTIONS.get(name)
              or caption_for_name(name) or suggest_caption(fig))
    for note in getattr(fig, "_ukb_caption_notes", []):
        if note not in result:
            result += " " + note
    return result


def bubble_callouts(ax, points, labels, sizes):
    """Keep callouts off the full marker areas, not just their centre coordinates."""
    ax.figure.canvas.draw()
    points = np.asarray(points)
    display = ax.transData.transform(points)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    obstacles = [display]
    for (x, y), size in zip(display, sizes):
        radius = np.sqrt(size) / 2 * ax.figure.dpi / 72 + 5
        obstacles.append(np.column_stack([x + radius * np.cos(angles),
                                          y + radius * np.sin(angles)]))
    obstacles = ax.transData.inverted().transform(np.vstack(obstacles))
    font_size = getattr(ax, "_ukb_annotation_fs", getattr(ax.figure, "_ukb_annotation_fs", 10))
    return S.scatter_callouts(ax, points, labels, obstacles=obstacles, fontsize=font_size, width=27)


def savefig(fig, name, style=None, caption=None, **kwargs):
    finalize_figure(fig)
    caption = _caption(fig, name, caption)
    paths = S.savefig(fig, name, style=style, caption=caption, **kwargs)
    for path in paths:
        EXPORTS[str(path)] = path
    return paths


def save_figure_file(fig, path, **kwargs):
    finalize_figure(fig)
    fig._ukb_caption = _caption(fig, Path(path).stem)
    result = S.save_figure_file(fig, path, **kwargs)
    for exported in fig._ukb_export_paths:
        EXPORTS[str(exported)] = exported
    return result


def show_figures(name=None, caption=None):
    """Save every otherwise-unsaved figure, then show paths and its caption below it."""
    for number in plt.get_fignums():
        fig = plt.figure(number)
        if caption:
            fig._ukb_caption = caption
        if not getattr(fig, "_ukb_export_paths", None):
            descriptions = [getattr(ax, "_ukb_description", "") for ax in fig.axes]
            stem = name or slug("_".join(filter(None, descriptions)))
            if not stem:
                raise ValueError("Supply a descriptive export name for this figure.")
            savefig(fig, stem, caption=caption)
        S.display_figure(fig)
        plt.close(fig)


def export_manifest():
    """Only files actually exported in this kernel run, with repository-relative paths."""
    import pandas as pd
    return pd.DataFrame([
        {"file": str(path.relative_to(S.ROOT)), "kb": path.stat().st_size // 1024}
        for path in sorted(EXPORTS.values()) if path.exists()
    ])
