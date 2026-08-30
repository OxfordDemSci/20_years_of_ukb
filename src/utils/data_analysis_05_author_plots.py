"""Publication figures for analysis 05 author characteristics."""

from __future__ import annotations

import math
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import sparse as sp
from scipy.sparse.csgraph import breadth_first_tree

from . import data_analysis_05_author_characteristics as A
from . import shared_name_gender as NG
from . import shared_paths as P
from .shared_style import (
    black_legend,
    compact_count,
    extended_palette,
    gridspec_figure,
    label_panels,
    marker_area,
    marker_size,
    mask_grid_region,
    panel_grid,
    panel_label,
    percent_axis,
    save_figure,
    semantic_colors,
    sequential_colormap,
    style_axis,
    style_colorbar,
    summary_box,
    year_ticks,
)

NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/"
    "ne_110m_admin_0_countries.zip"
)
WORLD_CACHE = P.AUTHOR_ANALYSIS / "natural_earth_110m.geojson"
WORLD_DOWNLOADED_SHP = (
    P.AUTHOR_ANALYSIS
    / "natural_earth"
    / "ne_110m_admin_0_countries.shp"
)
_NETWORK_LAYOUT_CACHE = {}


def _map_colormap(style):
    """Use the colourblind-safe navy-to-yellow geography scale."""
    return plt.get_cmap("cividis")


def _component_colors():
    """Return the component-class colours shared by all network panels."""
    return {
        "Isolate": "#B80C09",
        "Small (2-5)": "#D4AF37",
        "Intermediate (6-55)": "#6E8B3D",
        "Giant component": "#345995",
    }


def _short_label(value, width=28):
    text = "Unknown" if value is None or pd.isna(value) else str(value)
    replacements = {
        "University of Oxford": "Oxford",
        "University of Cambridge": "Cambridge",
        "University College London": "UCL",
        "Imperial College London": "Imperial",
        "King's College London": "King's College London",
        "Queen Mary University of London": "Queen Mary London",
        "Massachusetts General Hospital": "Massachusetts General",
        "Broad Institute": "Broad Institute",
    }
    text = replacements.get(text, text)
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def _field_label(value):
    """Use concise, unambiguous FOR labels without clipped ellipses."""
    text = "Unclassified" if value is None or pd.isna(value) else str(value)
    replacements = {
        "Agricultural, Veterinary and Food Sciences": "Agric., Veterinary & Food",
        "Biomedical and Clinical Sciences": "Biomedical & Clinical",
        "Information and Computing Sciences": "Information & Computing",
    }
    return replacements.get(text, text)


def _draw_top_country_bars(ax, core: A.CoreTables, style, n=8):
    """Draw the leading affiliation countries using geolocated fractional credit."""
    countries = core.country_metrics.copy()
    countries["label"] = countries["country"].fillna(countries["iso3"])
    countries["share"] = 100 * countries["fractional_paper_credit"] / countries[
        "fractional_paper_credit"
    ].sum()
    top = countries.nlargest(n, "share").sort_values("share")
    bars = ax.barh(
        top["label"],
        top["share"],
        color=semantic_colors("domain_colors", style)["geography"],
        edgecolor="black",
        linewidth=0.65,
    )
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in top["share"]], padding=4)
    ax.set_xlabel("Share of geolocated fractional credit")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax.margins(x=0.20)
    style_axis(ax, style, grid_axis="x")
    return top


def plot_headline_figure(core: A.CoreTables, network: A.NetworkTables, style, world=None):
    """Six headline views anchored by the community network and geographic map."""
    colors = semantic_colors("domain_colors", style)
    fig, outer = gridspec_figure(
        1,
        2,
        style,
        figsize_key="figsize_main",
        left=0.045,
        right=0.985,
        bottom=0.085,
        top=0.965,
        wspace=0.11,
        width_ratios=[1.45, 1],
    )
    left = outer[0].subgridspec(
        2,
        1,
        height_ratios=[1.72, 1],
        hspace=0.14,
    )
    right = outer[1].subgridspec(3, 1, hspace=0.44)
    network_row = left[0].subgridspec(
        1,
        2,
        width_ratios=[0.72, 1],
        wspace=0.03,
    )
    sidebar = network_row[0].subgridspec(
        2,
        1,
        height_ratios=[0.72, 1],
        hspace=0.18,
    )
    ax_a_meta = fig.add_subplot(sidebar[0])
    ax_b = fig.add_subplot(sidebar[1])
    ax_a = fig.add_subplot(network_row[1])
    ax_e = fig.add_subplot(left[1])
    ax_c = fig.add_subplot(right[0])
    ax_d = fig.add_subplot(right[1])
    ax_f = fig.add_subplot(right[2])

    # A: component-aware author network with an actual-edge topology backbone.
    _draw_component_network(
        ax_a,
        network,
        style,
        compact=True,
        meta_ax=ax_a_meta,
    )

    # B: direct concentration profile of resolved-author publication credit.
    concentration = A.author_credit_concentration(core.author_metrics)
    y = np.arange(len(concentration))
    bars = ax_b.barh(
        y,
        concentration["credit_share_percent"],
        height=0.62,
        color=colors["author_metrics"],
        edgecolor="black",
        linewidth=0.65,
    )
    labels = [
        f"Top {percentage:g}%"
        for percentage in concentration["top_author_percent"]
    ]
    ax_b.set_yticks(y, labels)
    ax_b.invert_yaxis()
    ax_b.bar_label(
        bars,
        labels=[f"{share:.1f}%" for share in concentration["credit_share_percent"]],
        padding=4,
        fontsize=style["annot_fs"] + 1,
        fontweight="bold",
    )
    ax_b.set_xlabel("Share of fractional\npublication credit")
    ax_b.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax_b.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax_b.set_xlim(0, 110)
    style_axis(ax_b, style, grid_axis="x")

    # C: cumulative geographic reach through each publication year.
    _draw_cumulative_country_reach(ax_c, core, style)

    # D: annual female-name share, with binomial interval among classified names.
    gender = core.gender_by_year.sort_values("year")
    ax_d.fill_between(
        gender["year"],
        gender["female_name_ci_low"],
        gender["female_name_ci_high"],
        color=colors["name_gender"],
        alpha=0.16,
        linewidth=0,
        label="95% CI",
    )
    ax_d.plot(
        gender["year"],
        gender["female_name_share"],
        color=colors["name_gender"],
        marker="o",
        markeredgecolor="black",
        markeredgewidth=0.5,
        markersize=marker_size(style, scale=0.85),
        linewidth=2.3,
        label="Female-name share",
    )
    ax_d.set(
        xlabel="Publication year",
        ylabel="Female-name share",
        xticks=year_ticks(gender["year"].min(), gender["year"].max(), 3),
    )
    percent_axis(ax_d)
    style_axis(ax_d, style)
    black_legend(ax_d, style, loc="lower right")

    # E: global distribution of geolocated fractional publication credit.
    world = load_world_geometries() if world is None else world
    _draw_country_map(
        ax_e,
        world,
        core.country_metrics,
        style,
        colorbar_label="Fractional publication credit",
        scale="linear",
        colorbar_orientation="vertical",
    )

    # F: institutional concentration over time.
    institutions = core.institution_by_year.sort_values("year")
    ax_f.plot(
        institutions["year"],
        institutions["top_10_share"],
        color=colors["institutions"],
        marker="o",
        markeredgecolor="black",
        markeredgewidth=0.5,
        linewidth=2.4,
        label="Top 10 institutions",
    )
    ax_f.plot(
        institutions["year"],
        institutions["top_1_share"],
        color=colors["author_metrics"],
        marker="s",
        markeredgecolor="black",
        markeredgewidth=0.5,
        linewidth=2.0,
        label="Leading institution",
    )
    ax_f.set(
        xlabel="Publication year",
        ylabel="Share of annual institutional credit",
        xticks=year_ticks(institutions["year"].min(), institutions["year"].max(), 2),
    )
    percent_axis(ax_f)
    style_axis(ax_f, style)
    black_legend(ax_f, style, loc="upper right")

    panel_label(
        ax_a_meta,
        "A",
        style,
        x=-0.04,
        y=1.075,
    )
    for ax, label in zip([ax_b, ax_c, ax_d, ax_f], "BCDF"):
        panel_label(
            ax,
            label,
            style,
            x=0.01,
            y=1.025,
        )
    panel_label(
        ax_e,
        "E",
        style,
        x=-0.08,
        y=0.93,
    )
    return save_figure(fig, "05_01_figure_01_author_characteristics", style)


def plot_author_metrics_supplement(core: A.CoreTables, style):
    metrics = core.author_metrics.copy()
    colors = semantic_colors("author_metric_colors", style)
    fig, axes = panel_grid(
        2,
        2,
        style,
        adjust={
            "left": 0.08,
            "right": 0.98,
            "bottom": 0.10,
            "top": 0.95,
            "wspace": 0.32,
            "hspace": 0.38,
        },
    )

    ax = axes[0, 0]
    thresholds = np.arange(1, int(metrics["n_ukb_papers"].max()) + 1)
    survival = np.array([(metrics["n_ukb_papers"] >= value).mean() * 100 for value in thresholds])
    ax.plot(thresholds, survival, color=colors["productivity"], linewidth=2.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        xlabel="UK Biobank publications per author",
        ylabel="Resolved authors at or above threshold",
    )
    ax.yaxis.set_major_locator(
        mticker.FixedLocator([100, 10, 1, 0.1, 0.01, 0.001])
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{value:g}%")
    )
    style_axis(ax, style)
    paper_quartiles = metrics["n_ukb_papers"].quantile([0.25, 0.5, 0.75])
    summary_box(
        ax,
        [
            (
                f"Median {paper_quartiles.loc[0.5]:.0f} paper "
                f"(IQR {paper_quartiles.loc[0.25]:.0f}-{paper_quartiles.loc[0.75]:.0f})"
            ),
            f"10+ papers: {100 * metrics['n_ukb_papers'].ge(10).mean():.1f}%",
            f"Maximum: {metrics['n_ukb_papers'].max():,.0f} papers",
        ],
        style,
        x=0.04,
        y=0.06,
        ha="left",
        va="bottom",
    )

    ax = axes[0, 1]
    max_h = int(metrics["ukb_h_index"].quantile(0.995))
    bins = np.arange(-0.5, max_h + 1.5, 1)
    ax.hist(
        metrics.loc[metrics["ukb_h_index"].le(max_h), "ukb_h_index"],
        bins=bins,
        color=colors["h_index"],
        edgecolor="black",
        linewidth=0.55,
    )
    ax.set(xlabel="UK Biobank h-index", ylabel="Resolved authors")
    style_axis(ax, style)
    h_quartiles = metrics["ukb_h_index"].quantile([0.25, 0.5, 0.75])
    shown_share = 100 * metrics["ukb_h_index"].le(max_h).mean()
    summary_box(
        ax,
        [
            (
                f"Median {h_quartiles.loc[0.5]:.0f} "
                f"(IQR {h_quartiles.loc[0.25]:.0f}-{h_quartiles.loc[0.75]:.0f})"
            ),
            f"95th percentile: {metrics['ukb_h_index'].quantile(0.95):.0f}",
            f"{shown_share:.1f}% shown; maximum {metrics['ukb_h_index'].max():,.0f}",
        ],
        style,
        x=0.96,
        y=0.94,
        ha="right",
        va="top",
    )

    ax = axes[1, 0]
    ax.scatter(
        metrics["n_ukb_papers"],
        metrics["ukb_h_index"],
        s=16,
        facecolor=colors["association"],
        edgecolor="black",
        linewidth=0.25,
        alpha=0.24,
        rasterized=True,
    )
    association = metrics[["n_ukb_papers", "ukb_h_index"]].corr(
        method="spearman"
    ).iloc[0, 1]
    summary_box(
        ax,
        [
            f"Spearman rho = {association:.2f}",
            f"n = {len(metrics):,} authors",
        ],
        style,
        x=0.04,
        y=0.94,
        ha="left",
        va="top",
    )
    ax.set_xscale("log")
    ax.set(xlabel="UK Biobank publications", ylabel="UK Biobank h-index")
    style_axis(ax, style)

    ax = axes[1, 1]
    leaders = metrics.nlargest(
        10,
        ["ukb_h_index", "n_ukb_papers", "total_ukb_citations"],
    ).sort_values(
        ["ukb_h_index", "n_ukb_papers", "total_ukb_citations"]
    )
    labels = [_short_label(value, 26) for value in leaders["full_name"]]
    h_values = leaders["ukb_h_index"].to_numpy(dtype=float)
    h_span = np.ptp(h_values)
    blue_positions = (
        0.38 + 0.62 * (h_values - h_values.min()) / h_span
        if h_span > 0
        else np.ones_like(h_values)
    )
    blue_scale = sequential_colormap(colors["leaders"])
    bars = ax.barh(
        labels,
        leaders["ukb_h_index"],
        color=blue_scale(blue_positions),
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xlabel("UK Biobank h-index")
    x_max = max(120, 1.55 * leaders["ukb_h_index"].max())
    annotation_start = leaders["ukb_h_index"].max() + 2
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    style_axis(ax, style, grid_axis="x")
    mask_grid_region(ax, annotation_start, x_max)
    ax.bar_label(
        bars,
        labels=[
            f"h={h:d} | {n:d} papers | {compact_count(c)} citations"
            for h, n, c in zip(
                leaders["ukb_h_index"],
                leaders["n_ukb_papers"],
                leaders["total_ukb_citations"],
            )
        ],
        padding=4,
        fontsize=style["annot_fs"] - 1,
        zorder=3,
    )
    label_panels(axes, "ABCD", style)
    return save_figure(
        fig,
        "05_02_supplementary_figure_01_author_metrics",
        style,
    )


def plot_gender_supplement(core: A.CoreTables, style):
    colors = semantic_colors("gender_colors", style)
    fig, axes = panel_grid(
        2,
        2,
        style,
        adjust={
            "left": 0.09,
            "right": 0.98,
            "bottom": 0.10,
            "top": 0.95,
            "wspace": 0.34,
            "hspace": 0.40,
        },
    )
    year = core.gender_by_year.sort_values("year")

    ax = axes[0, 0]
    ax.stackplot(
        year["year"],
        year["Female"],
        year["Male"],
        year["Unknown"],
        labels=["Female", "Male", "Unknown/androgynous"],
        colors=[colors["Female"], colors["Male"], colors["Unknown"]],
        alpha=0.88,
        linewidth=0.4,
        edgecolor="black",
    )
    ax.set(xlabel="Publication year", ylabel="Author-paper pairs", xticks=year_ticks(year["year"].min(), year["year"].max(), 3))
    style_axis(ax, style)
    black_legend(ax, style, loc="upper left")

    ax = axes[0, 1]
    coverage = NG.inference_coverage(core.authorships, "year")
    coverage_styles = {
        "Strict dictionary": {"color": "#6B6B6B", "marker": "^", "linestyle": "--"},
        "Expanded dictionary": {"color": colors["Male"], "marker": "s", "linestyle": "-"},
        "Offline ensemble": {"color": "#D4AF37", "marker": "D", "linestyle": "-"},
        "Primary + identity linkage": {"color": colors["Female"], "marker": "o", "linestyle": "-"},
    }
    coverage_labels = {
        "Strict dictionary": "Strict dictionary",
        "Expanded dictionary": "Expanded dictionary",
        "Offline ensemble": "Offline ensemble",
        "Primary + identity linkage": "Primary enhanced",
    }
    for stage, plot_style in coverage_styles.items():
        values = coverage[coverage["stage"].eq(stage)]
        ax.plot(
            values["year"],
            values["coverage_percent"],
            linewidth=2.0,
            markersize=marker_size(style, scale=0.80),
            markeredgecolor="black",
            markeredgewidth=0.65,
            label=coverage_labels[stage],
            **plot_style,
        )
    ax.set(
        xlabel="Publication year",
        ylabel="Names classified",
        xticks=year_ticks(coverage["year"].min(), coverage["year"].max(), 3),
    )
    percent_axis(ax)
    style_axis(ax, style)
    black_legend(ax, style, loc="lower left")

    ax = axes[1, 0]
    order = ["First author", "Middle author", "Last author", "Single author", "Corresponding author"]
    role = core.gender_by_role.set_index("authorship_role").reindex(order).dropna(subset=["classified"])
    y = np.arange(len(role))
    x = role["female_name_share"].to_numpy()
    xerr = np.vstack([x - role["female_name_ci_low"], role["female_name_ci_high"] - x])
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        color=colors["Female"],
        markeredgecolor="black",
        markeredgewidth=0.7,
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        markersize=marker_size(style, scale=1.30),
    )
    role_labels = [label.replace(" ", "\n", 1) for label in role.index]
    ax.set_yticks(y, role_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Female share of classified names")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    style_axis(ax, style)

    ax = axes[1, 1]
    field = core.gender_by_field.nlargest(12, "classified").sort_values("female_name_share")
    labels = [_field_label(value) for value in field["for_l2"]]
    blue_scale = sequential_colormap(colors["Male"])
    bars = ax.barh(
        labels,
        field["female_name_share"],
        color=blue_scale(np.linspace(0.38, 1.0, len(field))),
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in field["female_name_share"]], padding=4)
    ax.set_xlabel("Female share of classified names")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax.margins(x=0.14)
    style_axis(ax, style, grid=False)

    label_panels(axes, "ABCD", style)
    return save_figure(
        fig,
        "05_03_supplementary_figure_02_name_inferred_gender",
        style,
    )


def load_world_geometries():
    """Load Natural Earth from the project cache, local legacy path, or source URL."""
    import geopandas as gpd

    if WORLD_CACHE.exists():
        world = gpd.read_file(WORLD_CACHE)
    elif P.WORLD_SHP.exists():
        world = gpd.read_file(P.WORLD_SHP)
    elif WORLD_DOWNLOADED_SHP.exists():
        world = gpd.read_file(WORLD_DOWNLOADED_SHP)
    else:
        world = gpd.read_file(NATURAL_EARTH_URL)
        WORLD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        world.to_file(WORLD_CACHE, driver="GeoJSON")
    world.columns = [column.lower() for column in world.columns]
    admin = "admin" if "admin" in world else "name"
    world = world[world[admin].ne("Antarctica")].copy()
    iso_candidates = ["iso_a3_eh", "iso_a3", "adm0_a3"]
    iso = next(column for column in iso_candidates if column in world.columns)
    world["iso3"] = world[iso]
    if "adm0_a3" in world:
        world["iso3"] = world["iso3"].where(world["iso3"].ne("-99"), world["adm0_a3"])
    return world


def _draw_country_map(
    ax,
    world,
    country_values,
    style,
    value_col="fractional_paper_credit",
    colorbar_label="Fractional publication credit (log scale)",
    scale="log",
    colorbar_orientation="horizontal",
    norm=None,
):
    cmap = _map_colormap(style)
    merged = world.merge(country_values[["iso3", value_col]], on="iso3", how="left")
    positive = merged[value_col].dropna()
    positive = positive[positive > 0]
    if positive.empty:
        raise ValueError(f"Map variable {value_col!r} has no positive values")
    if norm is None:
        if scale == "log":
            norm = LogNorm(vmin=max(float(positive.min()), 0.05), vmax=float(positive.max()))
        elif scale == "linear":
            norm = Normalize(vmin=0, vmax=float(positive.max()))
        else:
            raise ValueError("Map scale must be 'linear' or 'log'")
    merged.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        norm=norm,
        missing_kwds={"color": "#F2F2F2", "edgecolor": "black"},
        edgecolor="black",
        linewidth=0.38,
    )
    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    if colorbar_orientation == "vertical":
        colorbar_ax = ax.inset_axes([-0.055, 0.08, 0.025, 0.84])
        cbar = ax.figure.colorbar(scalar, cax=colorbar_ax, orientation="vertical")
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    elif colorbar_orientation == "horizontal":
        colorbar_ax = ax.inset_axes([0.02, -0.13, 0.96, 0.05])
        cbar = ax.figure.colorbar(scalar, cax=colorbar_ax, orientation="horizontal")
    else:
        raise ValueError("Colorbar orientation must be 'vertical' or 'horizontal'")
    style_colorbar(cbar, colorbar_label)
    ax.set_ylim(-58, 88)
    ax.set_axis_off()
    ax.set_anchor("C" if colorbar_orientation == "vertical" else "N")
    return merged


def plot_geography_maps_supplement(core: A.CoreTables, style, world=None):
    """Three vertically stacked maps with directly comparable paper-count scales."""
    world = load_world_geometries() if world is None else world
    fig, gs = gridspec_figure(
        3,
        1,
        style,
        figsize=(11.5, 12.5),
        left=0.04,
        right=0.98,
        bottom=0.045,
        top=0.975,
        hspace=0.34,
    )
    axes = [fig.add_subplot(gs[row, 0]) for row in range(3)]
    paper_values = pd.concat(
        [
            core.country_metrics["author_basis_unique_papers"],
            core.country_metrics["org_basis_unique_papers"],
        ]
    ).dropna()
    paper_values = paper_values[paper_values.gt(0)]
    paper_norm = LogNorm(
        vmin=max(float(paper_values.min()), 0.05),
        vmax=float(paper_values.max()),
    )
    _draw_country_map(
        axes[0],
        world,
        core.country_metrics,
        style,
        value_col="author_basis_unique_papers",
        colorbar_label="Papers: author-affiliation basis (shared log scale)",
        norm=paper_norm,
    )
    _draw_country_map(
        axes[1],
        world,
        core.country_metrics,
        style,
        value_col="org_basis_unique_papers",
        colorbar_label="Papers: research-organisation basis (shared log scale)",
        norm=paper_norm,
    )
    intensity = core.country_metrics.copy()
    intensity.loc[
        intensity["author_basis_unique_papers"].lt(20), "authors_per_paper"
    ] = np.nan
    _draw_country_map(
        axes[2],
        world,
        intensity,
        style,
        value_col="authors_per_paper",
        colorbar_label="Author-paper rows per paper\n(>=20 papers; log scale)",
    )
    label_panels(axes, "ABC", style, x=-0.04, y=1.02)
    return save_figure(
        fig,
        "05_04_supplementary_figure_03_geography_maps",
        style,
    )


def _draw_cumulative_country_reach(ax, core: A.CoreTables, style):
    """Plot cumulative affiliation-country reach for the headline figure."""
    annual = core.country_by_year.sort_values("year")
    years = annual["year"].astype(int)
    values = annual["cumulative_entities"].astype(int).to_numpy()
    color = semantic_colors("domain_colors", style)["geography"]
    ax.plot(
        years,
        values,
        color=color,
        linewidth=2.4,
        marker="o",
        markersize=marker_size(style, scale=0.9),
        markeredgecolor="black",
        markeredgewidth=0.6,
    )
    ax.fill_between(years, 0, values, color=color, alpha=0.10, linewidth=0)
    ax.annotate(
        f"{values[-1]:,} countries",
        xy=(years.iloc[-1], values[-1]),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=color,
        fontsize=style["annot_fs"] + 1,
        fontweight="bold",
    )
    ax.set(
        xlabel="Publication year",
        ylabel="Cumulative affiliation countries",
        xticks=year_ticks(years.min(), years.max(), 3),
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.set_ylim(0, values[-1] * 1.12)
    style_axis(ax, style)


def _draw_country_basis_agreement(ax, core: A.CoreTables, style):
    """Compare country paper counts under two independent metadata definitions."""
    x_col = "author_basis_unique_papers"
    y_col = "org_basis_unique_papers"
    countries = core.country_metrics.dropna(subset=[x_col, y_col]).copy()
    countries = countries[countries[x_col].gt(0) & countries[y_col].gt(0)]
    project_blue = semantic_colors("domain_colors", style)["geography"]
    project_yellow = style["colors"][1]

    lower = 0.5 * countries[[x_col, y_col]].min().min()
    upper = 1.8 * countries[[x_col, y_col]].max().max()
    ax.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )
    ax.scatter(
        countries[x_col],
        countries[y_col],
        s=marker_area(style, scale=0.72),
        color=project_blue,
        edgecolor="black",
        linewidth=0.55,
        alpha=0.72,
        zorder=2,
    )

    eligible = countries[countries[[x_col, y_col]].min(axis=1).ge(10)].copy()
    eligible["log_discrepancy"] = np.abs(
        np.log10(eligible[x_col] / eligible[y_col])
    )
    outliers = eligible.nlargest(4, "log_discrepancy")
    ax.scatter(
        outliers[x_col],
        outliers[y_col],
        s=marker_area(style, scale=1.05),
        color=project_yellow,
        edgecolor="black",
        linewidth=0.75,
        zorder=3,
    )
    label_layout = {
        "Serbia": ((0.34, 0.11), 0.32),
        "Nigeria": ((0.51, 0.23), -0.24),
        "Lithuania": ((0.24, 0.52), -0.24),
        "Turkey": ((0.45, 0.75), -0.32),
    }
    for row in outliers.itertuples(index=False):
        x_value = getattr(row, x_col)
        y_value = getattr(row, y_col)
        country_label = _short_label(row.country or row.iso3, 18)
        text_position, curvature = label_layout[country_label]
        ax.annotate(
            f"{country_label}  {int(x_value):,} / {int(y_value):,}",
            (x_value, y_value),
            xytext=text_position,
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=style["annot_fs"],
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.96, "pad": 1},
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#3F3F3F",
                "connectionstyle": f"arc3,rad={curvature}",
                "linewidth": 1.0,
                "mutation_scale": 11,
                "shrinkA": 2,
                "shrinkB": 2,
            },
        )

    countries["paper_count_scale"] = np.sqrt(countries[x_col] * countries[y_col])
    minimum_scale = countries["paper_count_scale"].min()
    minima = countries[countries["paper_count_scale"].eq(minimum_scale)]
    minimum = minima.sort_values([x_col, y_col, "iso3"]).iloc[0]
    maximum = countries.sort_values(
        ["paper_count_scale", x_col, y_col], ascending=False
    ).iloc[0]
    extrema = pd.DataFrame([minimum, maximum])
    ax.scatter(
        extrema[x_col],
        extrema[y_col],
        s=marker_area(style, scale=0.95),
        marker="D",
        facecolor=project_blue,
        edgecolor="black",
        linewidth=0.9,
        zorder=4,
    )
    endpoint_labels = [
        (
            minimum,
            (
                f"Minimum: {len(minima)}-country tie  "
                f"({int(minimum[x_col]):,} / {int(minimum[y_col]):,})"
            ),
            (8, -4),
            "offset points",
            "left",
            "top",
            0.14,
        ),
        (
            maximum,
            (
                f"Maximum: "
                f"{_short_label(maximum['country'] or maximum['iso3'], 18)}\n"
                f"{int(maximum[x_col]):,} / {int(maximum[y_col]):,}"
            ),
            (0.98, 1.01),
            "axes fraction",
            "right",
            "bottom",
            -0.12,
        ),
    ]
    for row, label, offset, coordinates, horizontal, vertical, curvature in endpoint_labels:
        ax.annotate(
            label,
            (row[x_col], row[y_col]),
            xytext=offset,
            textcoords=coordinates,
            ha=horizontal,
            va=vertical,
            fontsize=style["annot_fs"],
            fontweight="bold",
            zorder=6,
            annotation_clip=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1},
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#555555",
                "connectionstyle": f"arc3,rad={curvature}",
                "linewidth": 0.9,
                "mutation_scale": 10,
                "shrinkA": 2,
                "shrinkB": 2,
            },
        )

    association = countries[[x_col, y_col]].corr(method="spearman").iloc[0, 1]
    summary_box(
        ax,
        [
            f"Spearman rho = {association:.3f}  (n = {len(countries):,})",
            "Yellow: largest proportional discrepancies",
            "Dashed: equal counts",
        ],
        style,
        x=0.03,
        y=0.97,
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(lower, upper),
        ylim=(lower, upper),
        xlabel="Papers: author-affiliation basis",
        ylabel="Papers: research-organisation basis",
    )
    style_axis(ax, style)


def _draw_country_period_composition(ax, core: A.CoreTables, style, n=8):
    """Show how the leading countries' fractional-credit shares changed by period."""
    periods = ["2013-15", "2016-18", "2019-21", "2022-25"]
    credits = core.country_credits[["year", "iso3", "country", "credit"]].copy()
    credits["period"] = pd.cut(
        credits["year"],
        bins=[2012, 2015, 2018, 2021, 2025],
        labels=periods,
    )
    overall = (
        credits.groupby(["iso3", "country"], observed=True)["credit"]
        .sum()
        .nlargest(n)
    )
    top_iso3 = overall.index.get_level_values("iso3").tolist()
    country_names = dict(overall.index.tolist())

    period_country = (
        credits.groupby(["period", "iso3"], observed=True)["credit"]
        .sum()
        .rename("credit")
        .reset_index()
    )
    period_country["share"] = 100 * period_country["credit"] / period_country.groupby(
        "period", observed=True
    )["credit"].transform("sum")
    matrix = (
        period_country[period_country["iso3"].isin(top_iso3)]
        .pivot(index="iso3", columns="period", values="share")
        .reindex(index=top_iso3, columns=periods)
        .fillna(0)
    )

    positive = matrix.to_numpy()[matrix.to_numpy() > 0]
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    cmap = sequential_colormap(
        semantic_colors("domain_colors", style)["geography"]
    )
    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    for row, iso3 in enumerate(matrix.index):
        for column, period in enumerate(matrix.columns):
            value = float(matrix.loc[iso3, period])
            red, green, blue, _ = cmap(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "black" if luminance > 0.52 else "white"
            ax.text(
                column,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=style["annot_fs"],
            )
    ax.set(
        xticks=np.arange(len(periods)),
        xticklabels=periods,
        yticks=np.arange(len(top_iso3)),
        yticklabels=[country_names[iso3] for iso3 in top_iso3],
        xlabel="Publication-year period",
    )
    ax.tick_params(axis="both", which="both", length=0)
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            len(periods),
            len(top_iso3),
            fill=False,
            edgecolor="black",
            linewidth=0.8,
        )
    )
    style_axis(ax, style, grid=False)

    colorbar_ax = ax.inset_axes([1.025, 0.08, 0.025, 0.84])
    colorbar = ax.figure.colorbar(image, cax=colorbar_ax, orientation="vertical")
    ticks = [value for value in [0.5, 1, 5, 10, 25, 50] if norm.vmin <= value <= norm.vmax]
    colorbar.set_ticks(ticks)
    colorbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _position: f"{value:g}%")
    )
    style_colorbar(colorbar, "Fractional-credit share")


def _draw_country_diversity(ax, core: A.CoreTables, style):
    """Plot period-averaged effective geographic diversity."""
    color = semantic_colors("domain_colors", style)["geography"]
    annual = core.country_by_year.sort_values("year")

    period_labels = ["2013-15", "2016-18", "2019-21", "2022-25"]
    period = pd.cut(
        annual["year"],
        bins=[2012, 2015, 2018, 2021, 2025],
        labels=period_labels,
    )
    diversity = (
        annual.assign(period=period)
        .groupby("period", observed=True)["effective_entities"]
        .mean()
        .reindex(period_labels)
    )
    x = np.arange(len(diversity))
    ax.vlines(
        x,
        0,
        diversity,
        color="#9A9A9A",
        linewidth=1.0,
        zorder=1,
    )
    ax.scatter(
        x,
        diversity,
        s=marker_area(style),
        color=color,
        edgecolor="black",
        linewidth=0.55,
        zorder=2,
    )
    ax.set(
        xlabel="Publication-year period",
        ylabel="Mean effective number of countries",
        xticks=x,
        xticklabels=period_labels,
    )
    style_axis(ax, style, grid_axis="y")


def plot_geography_metrics_supplement(core: A.CoreTables, style):
    """Four quantitative views of geographic reach, composition, and diversity."""
    fig, axes = panel_grid(
        2,
        2,
        style,
        figsize=(14.5, 9.5),
        adjust={
            "left": 0.105,
            "right": 0.98,
            "bottom": 0.10,
            "top": 0.95,
            "wspace": 0.34,
            "hspace": 0.40,
        },
    )
    _draw_top_country_bars(axes[0, 0], core, style, n=8)
    _draw_country_basis_agreement(axes[0, 1], core, style)
    _draw_country_period_composition(axes[1, 0], core, style)
    _draw_country_diversity(axes[1, 1], core, style)

    label_panels(axes, "ABCD", style)
    return save_figure(
        fig,
        "05_05_supplementary_figure_04_geography_metrics",
        style,
    )


def field_color_map(core: A.CoreTables, style, n=7):
    fields = (
        core.authorships[["paper_id", "for_l2"]]
        .drop_duplicates("paper_id")
        .explode("for_l2")["for_l2"]
        .value_counts()
        .head(n)
        .index
        .tolist()
    )
    palette = extended_palette(len(fields), style)
    return dict(zip(fields, palette))


def _institution_field_matrix(core: A.CoreTables, top_ids, field_colors):
    data = core.institution_credits[core.institution_credits["institution_id"].isin(top_ids)].copy()
    data["field"] = data["for_l2"].apply(lambda values: values if values else ["Unclassified"])
    data["n_fields"] = data["field"].apply(len)
    data = data.explode("field")
    data["field_credit"] = data["credit"] / data["n_fields"]
    data["field_display"] = data["field"].where(data["field"].isin(field_colors), "Other")
    return data.pivot_table(index="institution_id", columns="field_display", values="field_credit", aggfunc="sum", fill_value=0)


def plot_institution_supplement(core: A.CoreTables, style):
    color = semantic_colors("domain_colors", style)["institutions"]
    fig, gs = gridspec_figure(
        2,
        2,
        style,
        width_ratios=[1.7, 1],
        left=0.14,
        right=0.98,
        bottom=0.09,
        top=0.95,
        hspace=0.22,
        wspace=0.18,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    top = core.institution_metrics.head(18).copy()
    field_colors = field_color_map(core, style, n=6)
    matrix = _institution_field_matrix(core, top["institution_id"], field_colors)
    order = top.sort_values("fractional_paper_credit")["institution_id"].tolist()
    matrix = matrix.reindex(order).fillna(0)
    display = top.set_index("institution_id")["institution"].map(
        lambda value: _short_label(value, 40)
    )
    left = np.zeros(len(matrix))
    columns = [field for field in field_colors if field in matrix] + (["Other"] if "Other" in matrix else [])
    for field in columns:
        values = matrix[field].to_numpy()
        ax_a.barh(
            [display.get(key, key) for key in matrix.index],
            values,
            left=left,
            color=field_colors.get(field, "#B7B7B7"),
            edgecolor="black",
            linewidth=0.45,
            label=_field_label(field),
        )
        left += values
    ax_a.set_xlabel("Fractional publication credit")
    style_axis(ax_a, style, grid_axis="x")
    black_legend(
        ax_a,
        style,
        title="FOR division",
        loc="lower right",
        fontsize=style["legend_fs"] - 1,
    )

    annual = core.institution_by_year.sort_values("year")
    ax_b.plot(annual["year"], annual["cumulative_entities"], color=color, marker="o", markeredgecolor="black", linewidth=2.3)
    ax_b.set(xlabel="Publication year", ylabel="Cumulative institutions", xticks=year_ticks(annual["year"].min(), annual["year"].max(), 3))
    style_axis(ax_b, style)

    metrics = core.institution_metrics
    ax_c.scatter(
        metrics["fractional_paper_credit"],
        metrics["ukb_h_index"],
        s=np.clip(
            np.sqrt(metrics["unique_resolved_authors"]) * 3.5,
            marker_area(style, scale=0.24),
            marker_area(style, scale=1.35),
        ),
        color=color,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.45,
        rasterized=True,
    )
    association = metrics[["fractional_paper_credit", "ukb_h_index"]].corr(
        method="spearman"
    ).iloc[0, 1]
    summary_box(
        ax_c,
        [
            f"Spearman rho = {association:.2f}",
            f"n = {len(metrics):,} institutions",
        ],
        style,
        x=0.04,
        y=0.94,
    )
    ax_c.set_xscale("log")
    ax_c.set(xlabel="Fractional publication credit", ylabel="Institutional UKB h-index")
    style_axis(ax_c, style)

    label_panels([ax_a, ax_b, ax_c], "ABC", style)
    return save_figure(
        fig,
        "05_06_supplementary_figure_05_institutions",
        style,
    )


def _component_network_layout(network: A.NetworkTables):
    """Lay out every author by connected component using an actual-edge backbone."""
    key = (id(network.adjacency), network.adjacency.shape, network.adjacency.nnz)
    if key in _NETWORK_LAYOUT_CACHE:
        return _NETWORK_LAYOUT_CACHE[key]

    import igraph as ig

    adjacency = network.adjacency.tocsr()
    author = network.author_metrics.reset_index(drop=True)
    if len(author) != adjacency.shape[0]:
        raise ValueError("Network author order does not match the adjacency matrix")
    components = author["component"].astype(int).to_numpy()
    component_sizes = author["component_size"].astype(int).to_numpy()
    degrees = author["coauthor_count"].fillna(0).to_numpy(float)
    giant_label = int(author.loc[author["component_size"].idxmax(), "component"])
    giant_indices = np.flatnonzero(components == giant_label)
    giant_adjacency = adjacency[giant_indices][:, giant_indices].tocsr()

    # A breadth-first tree keeps every giant-component author connected. Repeated
    # coauthorship ties then restore the strongest local structure without the full
    # 3.9-million-edge overplotting burden.
    root = int(np.argmax(np.asarray(giant_adjacency.sum(axis=1)).ravel()))
    tree = breadth_first_tree(giant_adjacency, root, directed=False).tocoo()
    tree_edges = np.sort(np.column_stack([tree.row, tree.col]), axis=1)
    upper_giant = sp.triu(giant_adjacency, k=1).tocoo()
    repeated = np.flatnonzero(upper_giant.data > 1)
    repeat_limit = A.NETWORK_BACKBONE_REPEAT_LIMIT
    if len(repeated) > repeat_limit:
        keep = np.argpartition(upper_giant.data[repeated], -repeat_limit)[
            -repeat_limit:
        ]
        repeated = repeated[keep]
    repeated_edges = np.column_stack(
        [upper_giant.row[repeated], upper_giant.col[repeated]]
    )
    giant_edges_local = np.unique(
        np.vstack([tree_edges, repeated_edges]), axis=0
    ).astype(np.int32)

    ig.set_random_number_generator(random.Random(A.LEIDEN_SEED))
    graph = ig.Graph(
        n=len(giant_indices),
        edges=giant_edges_local.tolist(),
        directed=False,
    )
    giant_xy = np.asarray(
        graph.layout_lgl(maxiter=A.NETWORK_LAYOUT_ITERATIONS, root=root).coords,
        dtype=float,
    )
    if not np.isfinite(giant_xy).all():
        raise ValueError("Large Graph Layout returned non-finite coordinates")

    # Robustly orient and compress long branches so the topology remains visible in
    # both the compact headline panel and the full-size supplement.
    giant_xy -= np.median(giant_xy, axis=0)
    covariance = np.cov(giant_xy, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    giant_xy = giant_xy @ eigenvectors[:, ::-1]
    axis_scale = np.quantile(np.abs(giant_xy), 0.995, axis=0)
    giant_xy /= np.where(axis_scale > 0, axis_scale, 1)
    radius = np.linalg.norm(giant_xy, axis=1)
    compressed = np.tanh(1.15 * radius) / np.tanh(1.15)
    giant_xy *= np.divide(
        compressed,
        radius,
        out=np.ones_like(radius),
        where=radius > 0,
    )[:, None]
    giant_xy *= np.array([1.30, 1.30])

    positions = np.zeros((len(author), 2), dtype=np.float32)
    positions[giant_indices] = giant_xy.astype(np.float32)
    categories = np.full(len(author), "intermediate", dtype=object)
    categories[component_sizes == 1] = "isolate"
    categories[(component_sizes >= 2) & (component_sizes <= 5)] = "small"
    categories[components == giant_label] = "giant"

    component_table = (
        author.loc[components != giant_label, ["component", "component_size"]]
        .drop_duplicates("component")
        .sort_values(["component_size", "component"], ascending=[False, True])
    )
    component_table["category"] = np.select(
        [
            component_table["component_size"].eq(1),
            component_table["component_size"].between(2, 5),
        ],
        ["isolate", "small"],
        default="intermediate",
    )
    golden = math.pi * (3 - math.sqrt(5))
    rng = np.random.default_rng(A.LEIDEN_SEED)
    radial_bounds = {
        "intermediate": (1.20, 2.12),
        "small": (1.24, 2.24),
        "isolate": (1.30, 2.32),
    }
    for row in component_table.itertuples(index=False):
        component = int(row.component)
        category = row.category
        node_indices = np.flatnonzero(components == component)
        count = len(node_indices)
        inner, outer = radial_bounds[category]
        centre_radius = math.sqrt(rng.uniform(inner**2, outer**2))
        angle = rng.uniform(0, 2 * math.pi)
        centre = centre_radius * np.array(
            [math.cos(angle), 0.98 * math.sin(angle)]
        )
        if count == 1:
            positions[node_indices[0]] = centre
            continue
        order = node_indices[np.argsort(-degrees[node_indices], kind="stable")]
        local_rank = np.arange(count)
        local_radius = (
            0.012 + 0.060 * math.sqrt(count / 55)
        ) * np.sqrt((local_rank + 0.5) / count)
        local_angle = local_rank * golden + component * 0.19
        offsets = np.column_stack(
            [local_radius * np.cos(local_angle), local_radius * np.sin(local_angle)]
        )
        offsets -= offsets.mean(axis=0)
        positions[order] = centre + offsets

    upper = sp.triu(adjacency, k=1).tocoo()
    non_giant = components[upper.row] != giant_label
    other_edges = np.column_stack(
        [upper.row[non_giant], upper.col[non_giant]]
    ).astype(np.int32)
    giant_edges = giant_indices[giant_edges_local]
    result = {
        "positions": positions,
        "categories": categories,
        "degrees": degrees,
        "giant_edges": giant_edges,
        "other_edges": other_edges,
        "giant_size": len(giant_indices),
        "total_ties": int(adjacency.nnz // 2),
    }
    _NETWORK_LAYOUT_CACHE[key] = result
    return result


def _draw_component_network(ax, network, style, *, compact=False, meta_ax=None):
    """Draw a static, component-aware coauthorship network for all resolved authors."""
    layout = _component_network_layout(network)
    positions = layout["positions"]
    categories = layout["categories"]
    degrees = layout["degrees"]
    shared_colors = _component_colors()
    colors = {
        "isolate": shared_colors["Isolate"],
        "small": shared_colors["Small (2-5)"],
        "intermediate": shared_colors["Intermediate (6-55)"],
        "giant": shared_colors["Giant component"],
    }

    ax.add_collection(
        LineCollection(
            positions[layout["giant_edges"]],
            colors=[(0.35, 0.40, 0.45, 0.045 if compact else 0.060)],
            linewidths=0.09 if compact else 0.13,
            rasterized=True,
            zorder=1,
        )
    )
    if len(layout["other_edges"]):
        ax.add_collection(
            LineCollection(
                positions[layout["other_edges"]],
                colors=[(0.30, 0.34, 0.37, 0.18 if compact else 0.22)],
                linewidths=0.16 if compact else 0.22,
                rasterized=True,
                zorder=1,
            )
        )

    maximum_degree = max(float(np.sqrt(degrees.max())), 1)
    for category in ["giant", "intermediate", "small", "isolate"]:
        mask = categories == category
        if category == "giant":
            sizes = (0.45 if compact else 0.75) + (
                2.8 if compact else 5.0
            ) * np.sqrt(degrees[mask]) / maximum_degree
        else:
            base = {
                "intermediate": 2.4,
                "small": 3.2,
                "isolate": 4.2,
            }[category]
            sizes = np.full(mask.sum(), base * (1.0 if compact else 1.55))
        ax.scatter(
            positions[mask, 0],
            positions[mask, 1],
            s=sizes,
            color=colors[category],
            edgecolors="black",
            linewidths=0.06 if category == "giant" else 0.14,
            alpha=0.76 if category == "giant" else 0.86,
            rasterized=True,
            zorder=2,
        )

    labels = {
        "isolate": "Isolates",
        "small": "Small components (2-5)",
        "intermediate": "Intermediate components (6-55)",
        "giant": "Giant component",
    }
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor=colors[key],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=marker_size(style),
            label=labels[key],
        )
        for key in ["isolate", "small", "intermediate", "giant"]
    ]
    legend_target = ax if meta_ax is None else meta_ax
    if meta_ax is not None:
        meta_ax.set_axis_off()
    legend_kwargs = (
        {
            "loc": "upper right",
            "bbox_to_anchor": (1.0, 0.95),
            "borderaxespad": 0,
            "ncol": 1,
            "fontsize": style["legend_fs"] - 1,
        }
        if meta_ax is not None
        else {
            "loc": "lower left",
            "ncol": 2 if compact else 4,
            "fontsize": style["legend_fs"] - (2 if compact else 1),
        }
    )
    black_legend(
        legend_target,
        style,
        handles=handles,
        **legend_kwargs,
    )
    giant_share = 100 * layout["giant_size"] / len(positions)
    if meta_ax is not None:
        meta_ax.text(
            0.0,
            0.88,
            f"{len(positions):,}",
            transform=meta_ax.transAxes,
            ha="left",
            va="top",
            fontsize=style["annot_fs"] + 2,
            fontweight="bold",
        )
        meta_ax.text(
            0.0,
            0.71,
            "resolved authors",
            transform=meta_ax.transAxes,
            ha="left",
            va="top",
            fontsize=style["annot_fs"] - 1,
        )
        meta_ax.text(
            0.0,
            0.54,
            f"{layout['total_ties'] / 1_000_000:.2f}m",
            transform=meta_ax.transAxes,
            ha="left",
            va="top",
            fontsize=style["annot_fs"] + 2,
            fontweight="bold",
        )
        meta_ax.text(
            0.0,
            0.40,
            "coauthor ties",
            transform=meta_ax.transAxes,
            ha="left",
            va="top",
            fontsize=style["annot_fs"] - 1,
        )
        meta_ax.text(
            0.0,
            0.20,
            "Authors by component class",
            transform=meta_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=style["annot_fs"] - 1,
        )
        left = 0.0
        for key in ["isolate", "small", "intermediate", "giant"]:
            share = float(np.mean(categories == key))
            meta_ax.add_patch(
                Rectangle(
                    (left, 0.03),
                    share,
                    0.11,
                    transform=meta_ax.transAxes,
                    facecolor=colors[key],
                    edgecolor="black",
                    linewidth=0.6,
                    clip_on=False,
                )
            )
            if share >= 0.08:
                meta_ax.text(
                    left + share / 2,
                    0.085,
                    f"{100 * share:.1f}%",
                    transform=meta_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=style["annot_fs"] - 2,
                    fontweight="bold",
                    color="white" if key in {"intermediate", "giant"} else "black",
                )
            left += share
    else:
        summary_text = (
            f"Giant component\n{layout['giant_size']:,} of {len(positions):,} authors "
            f"({giant_share:.1f}%)"
            if compact
            else (
                f"{len(positions):,} authors | Giant component: "
                f"{layout['giant_size']:,} ({giant_share:.1f}%)"
            )
        )
        summary_box(
            ax,
            summary_text,
            style,
            x=0.99,
            y=0.985 if not compact else 0.01,
            ha="right",
            va="top" if not compact else "bottom",
            fontsize=style["annot_fs"] - (1 if compact else 0),
            bbox_kws={"alpha": 0.92},
        )
    ax.set_aspect("equal")
    ax.set_anchor("E" if compact and meta_ax is not None else ("W" if compact else "C"))
    ax.set_axis_off()
    ax.set_xlim(-2.48, 2.48)
    ax.set_ylim(-2.48, 2.48)


def _draw_community_composition(ax, core, network, style, n=12):
    """Show field composition within the largest Leiden communities."""
    top_communities = (
        network.community_summary.head(n)["community"].astype(int).tolist()
    )
    members = network.community_membership[
        network.community_membership["community"].isin(top_communities)
    ].copy()
    fields = field_color_map(core, style, n=7)
    if {"Biological Sciences", "Psychology"}.issubset(fields):
        fields["Biological Sciences"], fields["Psychology"] = (
            fields["Psychology"],
            fields["Biological Sciences"],
        )
    members["field_display"] = members["modal_for_l2"].where(
        members["modal_for_l2"].isin(fields),
        "Other",
    )
    matrix = pd.crosstab(members["community"], members["field_display"])
    matrix = matrix.reindex(top_communities).fillna(0)
    order = list(reversed(top_communities))
    matrix = matrix.reindex(order)
    left = np.zeros(len(matrix), dtype=float)
    columns = [field for field in fields if field in matrix] + (
        ["Other"] if "Other" in matrix else []
    )
    for field in columns:
        values = matrix[field].to_numpy(float)
        ax.barh(
            [f"C{community}" for community in matrix.index],
            values,
            left=left,
            color=fields.get(field, "#C8C8C8"),
            edgecolor="black",
            linewidth=0.35,
            label=_field_label(field),
        )
        left += values
    ax.set_xlabel("Resolved authors")
    style_axis(ax, style, grid_axis="x")
    black_legend(
        ax,
        style,
        title="Modal FOR division",
        loc="lower right",
        fontsize=style["legend_fs"] - 2,
        ncol=1,
    )


def plot_network_supplement(core: A.CoreTables, network: A.NetworkTables, style):
    colors = semantic_colors("domain_colors", style)
    project_red = colors["name_gender"]
    project_blue = colors["author_metrics"]
    project_yellow = colors["institutions"]
    component_colors = _component_colors()
    summaries = A.network_figure_tables(network)
    fig, gs = gridspec_figure(
        3,
        3,
        style,
        width_ratios=[1.35, 1, 1],
        left=0.075,
        right=0.98,
        bottom=0.09,
        top=0.95,
        hspace=0.64,
        wspace=0.56,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    _draw_community_composition(ax_a, core, network, style, n=12)

    # B: annual additions avoid another cumulative trajectory while retaining growth.
    ax_b = fig.add_subplot(gs[0, 1])
    growth = summaries["network_new_authors_by_year.csv"]
    bars = ax_b.bar(
        growth["year"],
        growth["new_resolved_authors"],
        width=0.72,
        color=component_colors["Giant component"],
        edgecolor="black",
        linewidth=0.55,
    )
    ax_b.bar_label(
        bars,
        labels=[""] * (len(bars) - 1) + [f"{growth['new_resolved_authors'].iloc[-1]:,}"],
        padding=3,
        fontsize=style["annot_fs"],
    )
    ax_b.set(
        xlabel="Publication year",
        ylabel="New resolved authors",
        xticks=year_ticks(growth["year"].min(), growth["year"].max(), 3),
    )
    ax_b.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    style_axis(ax_b, style, grid_axis="y")

    # C: density of author productivity against distinct collaboration reach.
    ax_c = fig.add_subplot(gs[0, 2])
    author_network = core.author_metrics[["researcher_id", "n_ukb_papers"]].merge(
        network.author_metrics[["researcher_id", "coauthor_count"]],
        on="researcher_id",
        how="inner",
        validate="one_to_one",
    )
    connected = author_network[author_network["coauthor_count"].gt(0)]
    density = ax_c.hexbin(
        connected["n_ukb_papers"],
        connected["coauthor_count"],
        xscale="log",
        yscale="log",
        gridsize=34,
        mincnt=1,
        bins="log",
        cmap="Blues",
        edgecolors="none",
        rasterized=True,
    )
    association = connected[["n_ukb_papers", "coauthor_count"]].corr(
        method="spearman"
    ).iloc[0, 1]
    summary_box(
        ax_c,
        [
            f"Spearman rho = {association:.2f}",
            f"n = {len(connected):,} authors",
        ],
        style,
        x=0.96,
        y=0.06,
        ha="right",
        va="bottom",
    )
    ax_c.set(xlabel="UK Biobank publications", ylabel="Distinct coauthors")
    style_axis(ax_c, style)
    colorbar = fig.colorbar(density, ax=ax_c, pad=0.025, fraction=0.055)
    style_colorbar(colorbar, "Authors per hexagon\n(log scale)")

    # D: most unique collaborations occur on only one shared paper.
    ax_d = fig.add_subplot(gs[1, 1])
    tie_distribution = summaries["network_tie_strength_distribution.csv"].iloc[::-1]
    tie_palette = ["#345995", "#5F7FB2", "#93AAD0", "#C8D5E8"]
    bars = ax_d.barh(
        tie_distribution["strength_band"].astype(str),
        tie_distribution["tie_share_percent"],
        color=tie_palette,
        edgecolor="black",
        linewidth=0.55,
    )
    ax_d.bar_label(
        bars,
        labels=[f"{value:.1f}%" for value in tie_distribution["tie_share_percent"]],
        padding=3,
        fontsize=style["annot_fs"],
    )
    ax_d.set_xlabel("Share of unique coauthor ties")
    ax_d.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax_d.margins(x=0.16)
    style_axis(ax_d, style, grid_axis="x")

    # E: rank-size structure makes the giant/non-giant discontinuity explicit.
    ax_e = fig.add_subplot(gs[1, 2])
    components = summaries["network_component_rank_size.csv"]
    giant = components.iloc[0]
    non_giant = components.iloc[1:].copy()
    non_giant["non_giant_rank"] = np.arange(1, len(non_giant) + 1)
    component_markers = {
        "Isolate": "o",
        "Small (2-5)": "s",
        "Intermediate (6-55)": "^",
    }
    rank_size_colors = {
        "Isolate": project_red,
        "Small (2-5)": project_yellow,
        "Intermediate (6-55)": project_blue,
    }
    for component_class, values in non_giant.groupby("component_class", sort=False):
        ax_e.scatter(
            values["non_giant_rank"],
            values["component_size"],
            s=marker_area(style, scale=0.22),
            marker=component_markers.get(component_class, "o"),
            color=rank_size_colors.get(component_class, project_blue),
            edgecolor="black",
            linewidth=0.25,
            alpha=0.78,
            rasterized=True,
            label=component_class,
        )
    giant_share = 100 * giant["component_size"] / network.adjacency.shape[0]
    summary_box(
        ax_e,
        (
            f"Giant component\n"
            f"{int(giant['component_size']):,} authors ({giant_share:.1f}%)"
        ),
        style,
        x=0.96,
        y=1.04,
        ha="right",
        va="bottom",
        bbox_kws={"alpha": 0.9},
    )
    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set(xlabel="Non-giant component rank", ylabel="Authors per component")
    style_axis(ax_e, style)
    black_legend(
        ax_e,
        style,
        loc="lower left",
        fontsize=style["legend_fs"] - 3,
        ncol=1,
    )

    # F: endpoint sensitivity is clearer as a normalized dumbbell than two curves.
    ax_f = fig.add_subplot(gs[2, 1:3])
    sensitivity = summaries["network_hyperauthorship_sensitivity.csv"].iloc[::-1]
    y = np.arange(len(sensitivity))
    retained = sensitivity["retained_percent"].to_numpy()
    ax_f.hlines(y, retained, 100, color="#8B8B8B", linewidth=1.5, zorder=1)
    ax_f.scatter(
        np.full(len(y), 100),
        y,
        s=marker_area(style),
        facecolor=project_blue,
        edgecolor="black",
        linewidth=0.9,
        zorder=3,
        label="All papers",
    )
    ax_f.scatter(
        retained,
        y,
        s=marker_area(style),
        marker="s",
        color=project_yellow,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label=f"Teams <= {A.HYPERAUTHOR_THRESHOLD}",
    )
    for yi, value in zip(y, retained):
        ax_f.annotate(
            f"{value:.1f}%",
            (value, yi),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=style["annot_fs"],
        )
    wrapped_labels = sensitivity["label"].replace({
        "Resolved authors": "Resolved\nauthors",
        "Unique coauthor ties": "Unique coauthor\nties",
        "Median coauthor count": "Median coauthor\ncount",
        "Giant-component share": "Giant-component\nshare",
        "Mean normalized strength": "Mean normalized\nstrength",
    })
    ax_f.set_yticks(y, wrapped_labels)
    ax_f.set_xlim(20, 104)
    ax_f.set_xlabel("2025 value retained relative to all papers")
    ax_f.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    style_axis(ax_f, style, grid_axis="x")
    black_legend(ax_f, style, loc="lower left")

    panel_label(ax_a, "A", style, x=-0.03)
    for ax, label in zip([ax_b, ax_c, ax_d, ax_e, ax_f], "BCDEF"):
        panel_label(ax, label, style, x=-0.13, y=1.05)
    return save_figure(
        fig,
        "05_07_supplementary_figure_06_networks",
        style,
    )
