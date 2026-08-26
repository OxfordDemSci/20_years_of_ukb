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
from scipy import sparse as sp
from scipy.sparse.csgraph import breadth_first_tree

from . import data_analysis_05_author_characteristics as A
from . import shared_paths as P
from .shared_style import extended_palette, savefig, year_ticks

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


def _domain_colors(style):
    return dict(style["domain_colors"])


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


def _axis(ax, grid_axis="both", zero=False):
    """Apply the analysis-wide black-axis and dashed-grid treatment."""
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(colors="black")
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color="#D2D2D2", linestyle="--", linewidth=0.65, alpha=0.75)
    if zero:
        ax.axhline(0, color="black", linewidth=0.8)
    return ax


def _panel_label(ax, label, x=-0.12, y=1.07, fontsize=18):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def _black_legend(ax, **kwargs):
    legend = ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1, **kwargs)
    legend.get_frame().set_linewidth(0.9)
    return legend


def _save(fig, name, style):
    paths = savefig(fig, name, style=style)
    return fig, paths


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


def _percent_axis(ax):
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))


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
        color=_domain_colors(style)["geography"],
        edgecolor="black",
        linewidth=0.65,
    )
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in top["share"]], padding=4)
    ax.set_xlabel("Share of geolocated fractional credit")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax.margins(x=0.20)
    _axis(ax, grid_axis="x")
    return top


def plot_headline_figure(core: A.CoreTables, network: A.NetworkTables, style, world=None):
    """Six headline views anchored by the community network and geographic map."""
    colors = _domain_colors(style)
    fig = plt.figure(figsize=style["figsize_main"])
    outer = fig.add_gridspec(
        1,
        2,
        left=0.045,
        right=0.985,
        bottom=0.085,
        top=0.965,
        wspace=0.14,
        width_ratios=[1.62, 1],
    )
    left = outer[0].subgridspec(
        2,
        1,
        height_ratios=[2.05, 0.92],
        hspace=0.14,
    )
    right = outer[1].subgridspec(3, 1, hspace=0.44)
    network_row = left[0].subgridspec(
        1,
        2,
        width_ratios=[0.34, 1],
        wspace=0.04,
    )
    lower_row = left[1].subgridspec(
        1,
        2,
        width_ratios=[2.55, 0.9],
        wspace=0.30,
    )
    ax_a_meta = fig.add_subplot(network_row[0])
    ax_a = fig.add_subplot(network_row[1])
    ax_d = fig.add_subplot(lower_row[0])
    ax_f = fig.add_subplot(lower_row[1])
    ax_b = fig.add_subplot(right[0])
    ax_c = fig.add_subplot(right[1])
    ax_e = fig.add_subplot(right[2])

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
        f"Top {percentage:g}% of authors"
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
    ax_b.set_xlabel("Share of fractional publication credit")
    ax_b.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax_b.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax_b.set_xlim(0, 100)
    _axis(ax_b, grid_axis="x")

    # C: annual female-name share, with binomial interval among classified names.
    gender = core.gender_by_year.sort_values("year")
    ax_c.fill_between(
        gender["year"],
        gender["female_name_ci_low"],
        gender["female_name_ci_high"],
        color=colors["name_gender"],
        alpha=0.16,
        linewidth=0,
        label="95% CI",
    )
    ax_c.plot(
        gender["year"],
        gender["female_name_share"],
        color=colors["name_gender"],
        marker="o",
        markeredgecolor="black",
        markeredgewidth=0.5,
        markersize=4.5,
        linewidth=2.3,
        label="Female-name share",
    )
    ax_c.set(
        xlabel="Publication year",
        ylabel="Female-name share",
        xticks=year_ticks(gender["year"].min(), gender["year"].max(), 3),
    )
    _percent_axis(ax_c)
    _axis(ax_c)
    _black_legend(ax_c, loc="lower right")

    # D: global distribution of geolocated fractional publication credit.
    world = load_world_geometries() if world is None else world
    _draw_country_map(
        ax_d,
        world,
        core.country_metrics,
        style,
        colorbar_label="Fractional publication credit",
        scale="linear",
        colorbar_orientation="vertical",
    )

    # E: compact headline distribution; the full survival curve is supplementary.
    productivity = A.author_productivity_bands(core.author_metrics)
    bars = ax_f.barh(
        productivity["publication_band"],
        productivity["author_share_percent"],
        height=0.62,
        color=colors["author_metrics"],
        edgecolor="black",
        linewidth=0.65,
    )
    ax_f.invert_yaxis()
    ax_f.bar_label(
        bars,
        labels=[f"{share:.1f}%" for share in productivity["author_share_percent"]],
        padding=3,
        fontsize=style["annot_fs"],
        fontweight="bold",
    )
    ax_f.set_xlim(0, 65)
    ax_f.set_xlabel("Share of resolved authors")
    ax_f.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax_f.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    _axis(ax_f, grid_axis="x")

    # F: institutional concentration over time.
    institutions = core.institution_by_year.sort_values("year")
    ax_e.plot(
        institutions["year"],
        institutions["top_10_share"],
        color=colors["institutions"],
        marker="o",
        markeredgecolor="black",
        markeredgewidth=0.5,
        linewidth=2.4,
        label="Top 10 institutions",
    )
    ax_e.plot(
        institutions["year"],
        institutions["top_1_share"],
        color=colors["author_metrics"],
        marker="s",
        markeredgecolor="black",
        markeredgewidth=0.5,
        linewidth=2.0,
        label="Leading institution",
    )
    ax_e.set(
        xlabel="Publication year",
        ylabel="Share of annual institutional credit",
        xticks=year_ticks(institutions["year"].min(), institutions["year"].max(), 2),
    )
    _percent_axis(ax_e)
    _axis(ax_e)
    _black_legend(ax_e, loc="upper right")

    _panel_label(
        ax_a_meta,
        "A",
        x=-0.04,
        y=1.025,
        fontsize=style["title_fs"],
    )
    for ax, label in zip([ax_b, ax_c, ax_d, ax_f, ax_e], "BCDEF"):
        _panel_label(
            ax,
            label,
            x=-0.02 if ax is ax_d else 0.01,
            y=1.025,
            fontsize=style["title_fs"],
        )
    return _save(fig, "05_01_figure_01_author_characteristics", style)


def plot_author_metrics_supplement(core: A.CoreTables, style):
    metrics = core.author_metrics.copy()
    color = _domain_colors(style)["author_metrics"]
    fig, axes = plt.subplots(2, 2, figsize=style["figsize_panel"])
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.95, wspace=0.32, hspace=0.38)

    ax = axes[0, 0]
    thresholds = np.arange(1, int(metrics["n_ukb_papers"].max()) + 1)
    survival = np.array([(metrics["n_ukb_papers"] >= value).mean() * 100 for value in thresholds])
    ax.plot(thresholds, survival, color=color, linewidth=2.5)
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
    _axis(ax)

    ax = axes[0, 1]
    max_h = int(metrics["ukb_h_index"].quantile(0.995))
    bins = np.arange(-0.5, max_h + 1.5, 1)
    ax.hist(
        metrics.loc[metrics["ukb_h_index"].le(max_h), "ukb_h_index"],
        bins=bins,
        color=color,
        edgecolor="black",
        linewidth=0.55,
    )
    ax.set(xlabel="UK Biobank h-index", ylabel="Resolved authors")
    _axis(ax)

    ax = axes[1, 0]
    ax.scatter(
        metrics["n_ukb_papers"],
        metrics["ukb_h_index"],
        s=16,
        facecolor=color,
        edgecolor="black",
        linewidth=0.25,
        alpha=0.24,
        rasterized=True,
    )
    association = metrics[["n_ukb_papers", "ukb_h_index"]].corr(
        method="spearman"
    ).iloc[0, 1]
    ax.text(
        0.04,
        0.94,
        f"Spearman rho = {association:.2f}\nn = {len(metrics):,} authors",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style["annot_fs"],
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9, "pad": 3},
    )
    ax.set_xscale("log")
    ax.set(xlabel="UK Biobank publications", ylabel="UK Biobank h-index")
    _axis(ax)

    ax = axes[1, 1]
    leaders = metrics.nlargest(20, ["ukb_h_index", "n_ukb_papers"]).sort_values(
        ["ukb_h_index", "n_ukb_papers"]
    )
    labels = [_short_label(value, 24) for value in leaders["full_name"]]
    bars = ax.barh(
        labels,
        leaders["ukb_h_index"],
        color=color,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar_label(
        bars,
        labels=[f"{h:d}  |  {n:d} papers" for h, n in zip(leaders["ukb_h_index"], leaders["n_ukb_papers"])],
        padding=4,
        fontsize=style["annot_fs"] - 1,
    )
    ax.set_xlabel("UK Biobank h-index")
    ax.margins(x=0.22)
    _axis(ax, grid_axis="x")

    for ax, label in zip(axes.flat, "ABCD"):
        _panel_label(ax, label, fontsize=style["title_fs"])
    return _save(fig, "05_02_supplementary_figure_01_author_metrics", style)


def plot_gender_supplement(core: A.CoreTables, style):
    colors = style["gender_colors"]
    fig, axes = plt.subplots(2, 2, figsize=style["figsize_panel"])
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.95, wspace=0.34, hspace=0.40)
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
    _axis(ax)
    _black_legend(ax, loc="upper left")

    ax = axes[0, 1]
    strict = (
        core.authorships.assign(
            classified=core.authorships["name_gender"].isin(["Female", "Male"]),
            strict_classified=core.authorships["name_gender_strict"].isin(["Female", "Male"]),
        )
        .groupby("year")[["classified", "strict_classified"]]
        .mean()
        .mul(100)
        .reset_index()
    )
    ax.plot(strict["year"], strict["classified"], color=colors["Female"], marker="o", linewidth=2.2, label="Expanded rule")
    ax.plot(strict["year"], strict["strict_classified"], color=colors["Male"], marker="s", linewidth=2.0, label="Strict rule")
    ax.set(xlabel="Publication year", ylabel="Names classified", xticks=year_ticks(strict["year"].min(), strict["year"].max(), 3))
    _percent_axis(ax)
    _axis(ax)
    _black_legend(ax, loc="lower left")

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
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        markersize=7,
    )
    ax.set_yticks(y, role.index)
    ax.invert_yaxis()
    ax.set_xlabel("Female share of classified names")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    _axis(ax, grid_axis="x")

    ax = axes[1, 1]
    field = core.gender_by_field.nlargest(12, "classified").sort_values("female_name_share")
    labels = [_field_label(value) for value in field["for_l2"]]
    bars = ax.barh(
        labels,
        field["female_name_share"],
        color=colors["Female"],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in field["female_name_share"]], padding=4)
    ax.set_xlabel("Female share of classified names")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax.margins(x=0.14)
    _axis(ax, grid_axis="x")

    for ax, label in zip(axes.flat, "ABCD"):
        _panel_label(ax, label, fontsize=style["title_fs"])
    return _save(fig, "05_03_supplementary_figure_02_name_inferred_gender", style)


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
):
    cmap = _map_colormap(style)
    merged = world.merge(country_values[["iso3", value_col]], on="iso3", how="left")
    positive = merged[value_col].dropna()
    positive = positive[positive > 0]
    if positive.empty:
        raise ValueError(f"Map variable {value_col!r} has no positive values")
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
    cbar.set_label(colorbar_label)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.8)
    ax.set_ylim(-58, 88)
    ax.set_axis_off()
    ax.set_anchor("C" if colorbar_orientation == "vertical" else "N")
    return merged


def plot_geography_supplement(core: A.CoreTables, style, world=None):
    world = load_world_geometries() if world is None else world
    color = _domain_colors(style)["geography"]
    width, height = style["figsize_panel"]
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.72, 1],
        left=0.045,
        right=0.985,
        bottom=0.08,
        top=0.96,
        hspace=0.28,
        wspace=0.24,
    )
    axes = [fig.add_subplot(gs[row, column]) for row in range(2) for column in range(3)]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes
    _draw_top_country_bars(ax_a, core, style, n=8)
    _draw_country_map(
        ax_b,
        world,
        core.country_metrics,
        style,
        value_col="author_basis_unique_papers",
        colorbar_label="Papers: author-affiliation basis (log scale)",
    )
    _draw_country_map(
        ax_c,
        world,
        core.country_metrics,
        style,
        value_col="org_basis_unique_papers",
        colorbar_label="Papers: research-org basis (log scale)",
    )
    intensity = core.country_metrics.copy()
    intensity.loc[
        intensity["author_basis_unique_papers"].lt(20), "authors_per_paper"
    ] = np.nan
    _draw_country_map(
        ax_d,
        world,
        intensity,
        style,
        value_col="authors_per_paper",
        colorbar_label="Author-paper rows per paper\n(>=20 papers; log scale)",
    )

    annual = core.country_by_year.sort_values("year")
    bars = ax_e.bar(
        annual["year"],
        annual["cumulative_entities"],
        width=0.72,
        color=color,
        edgecolor="black",
        linewidth=0.55,
    )
    ax_e.bar_label(
        bars,
        labels=[""] * (len(bars) - 1) + [f"{annual['cumulative_entities'].iloc[-1]:,.0f}"],
        padding=3,
        fontsize=style["annot_fs"],
    )
    ax_e.set(xlabel="Publication year", ylabel="Cumulative affiliation countries", xticks=year_ticks(annual["year"].min(), annual["year"].max(), 3))
    _axis(ax_e, grid_axis="y")

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
    ax_f.vlines(
        x,
        0,
        diversity,
        color="#9A9A9A",
        linewidth=1.0,
        zorder=1,
    )
    ax_f.scatter(
        x,
        diversity,
        s=36,
        color=color,
        edgecolor="black",
        linewidth=0.55,
        zorder=2,
    )
    ax_f.set(
        xlabel="Publication-year period",
        ylabel="Mean effective number of countries",
        xticks=x,
        xticklabels=period_labels,
    )
    _axis(ax_f, grid_axis="y")

    for ax, label in zip(axes, "ABCDEF"):
        _panel_label(
            ax,
            label,
            x=-0.12 if ax is ax_a else (-0.04 if ax in {ax_b, ax_c, ax_d} else -0.12),
            fontsize=style["title_fs"],
        )
    return _save(fig, "05_04_supplementary_figure_03_geography", style)


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
    color = _domain_colors(style)["institutions"]
    fig = plt.figure(figsize=style["figsize_panel"])
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.7, 1],
        left=0.14,
        right=0.98,
        bottom=0.09,
        top=0.95,
        hspace=0.40,
        wspace=0.42,
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
    _axis(ax_a, grid_axis="x")
    _black_legend(ax_a, title="FOR division", loc="lower right", fontsize=style["legend_fs"] - 1)

    annual = core.institution_by_year.sort_values("year")
    ax_b.plot(annual["year"], annual["cumulative_entities"], color=color, marker="o", markeredgecolor="black", linewidth=2.3)
    ax_b.set(xlabel="Publication year", ylabel="Cumulative institutions", xticks=year_ticks(annual["year"].min(), annual["year"].max(), 3))
    _axis(ax_b)

    metrics = core.institution_metrics
    ax_c.scatter(
        metrics["fractional_paper_credit"],
        metrics["ukb_h_index"],
        s=np.clip(np.sqrt(metrics["unique_resolved_authors"]) * 3, 8, 80),
        color=color,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.45,
        rasterized=True,
    )
    association = metrics[["fractional_paper_credit", "ukb_h_index"]].corr(
        method="spearman"
    ).iloc[0, 1]
    ax_c.text(
        0.04,
        0.94,
        f"Spearman rho = {association:.2f}\nn = {len(metrics):,} institutions",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=style["annot_fs"],
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9, "pad": 3},
    )
    ax_c.set_xscale("log")
    ax_c.set(xlabel="Fractional publication credit", ylabel="Institutional UKB h-index")
    _axis(ax_c)

    for ax, label in zip([ax_a, ax_b, ax_c], "ABC"):
        _panel_label(ax, label, fontsize=style["title_fs"])
    return _save(fig, "05_05_supplementary_figure_04_institutions", style)


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
    giant_xy *= np.array([1.42, 1.15])

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
            [math.cos(angle), 0.82 * math.sin(angle)]
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
            markersize=6.5,
            label=labels[key],
        )
        for key in ["isolate", "small", "intermediate", "giant"]
    ]
    legend_target = ax if meta_ax is None else meta_ax
    if meta_ax is not None:
        meta_ax.set_axis_off()
    legend_kwargs = (
        {
            "loc": "upper left",
            "bbox_to_anchor": (0.0, 0.88),
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
    legend = legend_target.legend(
        handles=handles,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
        **legend_kwargs,
    )
    legend.get_frame().set_linewidth(0.9)
    giant_share = 100 * layout["giant_size"] / len(positions)
    summary_target = ax if meta_ax is None else meta_ax
    if meta_ax is not None:
        summary_text = (
            f"Giant component\n{layout['giant_size']:,} authors ({giant_share:.1f}%)"
        )
        summary_x, summary_y = 0.0, 0.68
        summary_ha, summary_va = "left", "top"
        summary_bbox = None
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
        summary_x = 0.99
        summary_y = 0.985 if not compact else 0.01
        summary_ha = "right"
        summary_va = "top" if not compact else "bottom"
        summary_bbox = {
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.92,
            "pad": 3,
        }
    summary_target.text(
        summary_x,
        summary_y,
        summary_text,
        transform=summary_target.transAxes,
        ha=summary_ha,
        va=summary_va,
        fontsize=style["annot_fs"] - (1 if compact else 0),
        bbox=summary_bbox,
    )
    ax.set_aspect("equal")
    ax.set_anchor("E" if compact and meta_ax is not None else ("W" if compact else "C"))
    ax.set_axis_off()
    ax.set_xlim(-2.48, 2.48)
    ax.set_ylim(-2.08, 2.18)


def _draw_community_composition(ax, core, network, style, n=12):
    """Show field composition within the largest Leiden communities."""
    top_communities = (
        network.community_summary.head(n)["community"].astype(int).tolist()
    )
    members = network.community_membership[
        network.community_membership["community"].isin(top_communities)
    ].copy()
    fields = field_color_map(core, style, n=7)
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
    _axis(ax, grid_axis="x")
    _black_legend(
        ax,
        title="Modal FOR division",
        loc="lower right",
        fontsize=style["legend_fs"] - 2,
        ncol=1,
    )


def plot_network_supplement(core: A.CoreTables, network: A.NetworkTables, style):
    colors = _domain_colors(style)
    network_color = colors["networks"]
    component_colors = _component_colors()
    summaries = A.network_figure_tables(network)
    fig = plt.figure(figsize=style["figsize_panel"])
    gs = fig.add_gridspec(
        3,
        3,
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
    _axis(ax_b, grid_axis="y")

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
    ax_c.text(
        0.04,
        0.94,
        f"Spearman rho = {association:.2f}\nn = {len(connected):,} authors",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=style["annot_fs"],
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9, "pad": 3},
    )
    ax_c.set(xlabel="UK Biobank publications", ylabel="Distinct coauthors")
    _axis(ax_c)
    colorbar = fig.colorbar(density, ax=ax_c, pad=0.025, fraction=0.055)
    colorbar.set_label("Authors per hexagon\n(log scale)")
    colorbar.outline.set_edgecolor("black")
    colorbar.outline.set_linewidth(0.8)

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
    _axis(ax_d, grid_axis="x")

    # E: rank-size structure makes the giant/non-giant discontinuity explicit.
    ax_e = fig.add_subplot(gs[1, 2])
    components = summaries["network_component_rank_size.csv"]
    giant = components.iloc[0]
    non_giant = components.iloc[1:].copy()
    non_giant["non_giant_rank"] = np.arange(1, len(non_giant) + 1)
    for component_class, values in non_giant.groupby("component_class", sort=False):
        ax_e.scatter(
            values["non_giant_rank"],
            values["component_size"],
            s=9,
            color=component_colors[component_class],
            edgecolor="black",
            linewidth=0.25,
            alpha=0.78,
            rasterized=True,
            label=component_class,
        )
    giant_share = 100 * giant["component_size"] / network.adjacency.shape[0]
    ax_e.text(
        0.96,
        1.04,
        f"Giant component\n{int(giant['component_size']):,} authors ({giant_share:.1f}%)",
        transform=ax_e.transAxes,
        ha="right",
        va="bottom",
        fontsize=style["annot_fs"],
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9, "pad": 3},
    )
    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set(xlabel="Non-giant component rank", ylabel="Authors per component")
    _axis(ax_e)
    _black_legend(ax_e, loc="lower left", fontsize=style["legend_fs"] - 3, ncol=1)

    # F: endpoint sensitivity is clearer as a normalized dumbbell than two curves.
    ax_f = fig.add_subplot(gs[2, 1:3])
    sensitivity = summaries["network_hyperauthorship_sensitivity.csv"].iloc[::-1]
    y = np.arange(len(sensitivity))
    retained = sensitivity["retained_percent"].to_numpy()
    ax_f.hlines(y, retained, 100, color="#8B8B8B", linewidth=1.5, zorder=1)
    ax_f.scatter(
        np.full(len(y), 100),
        y,
        s=42,
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
        zorder=3,
        label="All papers",
    )
    ax_f.scatter(
        retained,
        y,
        s=42,
        marker="s",
        color=network_color,
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
    _axis(ax_f, grid_axis="x")
    _black_legend(ax_f, loc="lower left")

    _panel_label(ax_a, "A", x=-0.03, fontsize=style["title_fs"])
    for ax, label in zip([ax_b, ax_c, ax_d, ax_e, ax_f], "BCDEF"):
        _panel_label(ax, label, x=-0.13, y=1.05, fontsize=style["title_fs"])
    return _save(fig, "05_06_supplementary_figure_05_networks", style)
