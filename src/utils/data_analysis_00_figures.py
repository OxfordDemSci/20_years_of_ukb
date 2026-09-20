"""Publication panels for saved candidate-label diagnostics; no model inference."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np

from .shared_style import (
    PNG_DPI, apply_typography, blue_cream_red_colormap, finalize_figure,
    grid_on, palette, set_title,
)

TRUE_GROUP = "Three-model TRUE agreement"
REST_GROUP = "Rest: not three-model TRUE agreement"
GROUPS = (TRUE_GROUP, REST_GROUP)
GROUP_LABELS = ("Unanimous TRUE", "Other candidates")
GROUP_COLORS = palette("red", "steel_blue")
MODEL_COLORS = palette("red", "navy", "light_blue")


def _axis(ax, title, *, grid="both"):
    set_title(ax, title, fontsize=11, y=1.035)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.grid(False, which="both")
    ax.minorticks_off()
    if grid:
        grid_on(ax, axis=grid, which="major", alpha=.35, linewidth=.6, log=True)


def _years(ax):
    ax.set_xlim(2012.7, 2025.3)
    ax.set_xticks([2013, 2016, 2019, 2022, 2025])
    ax.set_xlabel("Publication year", fontsize=10)


def _legend(ax, **kwargs):
    return ax.legend(frameon=True, edgecolor="black", facecolor="white",
                     framealpha=.95, fontsize=8, **kwargs)


def _percent(ax, *, axis="y"):
    target = getattr(ax, f"{axis}axis")
    target.set_major_locator(MaxNLocator(nbins=4))
    target.set_major_formatter(PercentFormatter(100, decimals=0))


def _export(fig, directory, stem, caption, *, show):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig)
    files = []
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
        files.append(path)
    (directory / f"{stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")
    if show:
        from IPython.display import display
        display(fig)
    plt.close(fig)
    return files


def plot_candidate_agreement(model_summary, vote_distribution, agreement_matrix, yearly):
    """Six complementary panels; agreement uses only jointly parsed predictions."""
    apply_typography()
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=.10, h_pad=.12, wspace=.07, hspace=.13)
    ax = axes[0, 0]
    x = np.arange(len(model_summary))
    ax.bar(x - .19, model_summary.true_percent_among_parsed, width=.36,
           color=palette("red"), edgecolor=palette("navy"), linewidth=.6, label="TRUE among parsed")
    ax.bar(x + .19, model_summary.parse_rate_percent, width=.36,
           color=palette("light_blue"), edgecolor=palette("navy"), linewidth=.6, label="Parsed among all")
    ax.set_xticks(x, [name.replace("-", "\n", 1) for name in model_summary.display_name])
    ax.set(ylabel="Share of candidates", ylim=(0, 122))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    _legend(ax, loc="upper left", ncol=1)
    _axis(ax, "A  Model predictions", grid="y")

    ax = axes[0, 1]
    votes = vote_distribution.set_index("n_true_votes").n_candidates.reindex(range(4), fill_value=0)
    bars = ax.bar(range(4), votes, color=[palette("steel_blue")] * 3 + [palette("red")],
                  edgecolor=palette("navy"), linewidth=.6)
    ax.bar_label(bars, labels=[f"{value:,}" for value in votes], padding=4, fontsize=9)
    ax.set(xlabel="Number of TRUE predictions", ylabel="Candidates", xticks=range(4))
    ax.set_ylim(0, max(float(votes.max()), 1) * 1.22)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    _axis(ax, "B  Three-model consensus", grid="y")

    ax = axes[0, 2]
    values = np.ma.masked_invalid(100 * agreement_matrix.to_numpy(dtype=float))
    image = ax.pcolormesh(values, cmap=blue_cream_red_colormap(), vmin=0, vmax=100,
                         edgecolors=palette("navy"), linewidth=.6)
    labels = [name.replace("-", "\n", 1) for name in model_summary.display_name]
    ax.set_xticks(np.arange(len(labels)) + .5, labels)
    ax.set_yticks(np.arange(len(labels)) + .5, model_summary.display_name)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = values[row, column]
            ax.text(column + .5, row + .5, "–" if np.ma.is_masked(value) else f"{value:.1f}",
                    ha="center", va="center", fontsize=10, color="black")
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=.07, pad=.09,
                        ticks=[0, 50, 100], format=PercentFormatter(100, decimals=0))
    cbar.set_label("Agreement among jointly parsed labels", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    _axis(ax, "C  Pairwise agreement", grid=None)

    # Missing years remain gaps, not artificial zeros or interpolated observations.
    series = yearly.set_index("year_int").reindex(range(2013, 2026))
    ax = axes[1, 0]
    for key, color, label in zip(
            ("three_model_TRUE_agreement", "rest_NOT_three_model_TRUE_agreement"),
            GROUP_COLORS, GROUP_LABELS):
        counts = series[key].astype(float)
        ax.plot(series.index, counts.where(counts.gt(0)), marker="o", markersize=4,
                linewidth=1.7, color=color, label=label)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}" if value >= 1 else f"{value:g}"))
    ax.set_ylabel("Candidates (log scale)")
    _years(ax)
    _legend(ax, loc="best")
    _axis(ax, "D  Annual candidate counts")

    ax = axes[1, 1]
    ax.plot(series.index, series.three_model_TRUE_agreement_rate_percent,
            color=palette("red"), marker="o", markersize=4, linewidth=1.7)
    ax.set(ylabel="Unanimous TRUE / all candidates", ylim=(0, 105))
    _years(ax)
    _percent(ax)
    _axis(ax, "E  Annual consensus rate")

    ax = axes[1, 2]
    for model, color, label in zip(model_summary.model, MODEL_COLORS, model_summary.display_name):
        ax.plot(series.index, series[f"{model}_TRUE"], marker="o", markersize=4,
                linewidth=1.7, color=color, label=label)
    ax.set_ylabel("TRUE predictions")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    _years(ax)
    _legend(ax, loc="best")
    _axis(ax, "F  Annual model predictions")
    return fig


def plot_candidate_text(category_summary, tfidf_terms):
    """Combine mention/keyword profiles and signed, descriptive TF-IDF contrasts."""
    apply_typography()
    fig = plt.figure(figsize=(13.2, 8.5), layout="constrained")
    fig.get_layout_engine().set(w_pad=.10, h_pad=.12, wspace=.12, hspace=.12)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 1])
    ax = fig.add_subplot(grid[:, 0])
    pivot = category_summary.pivot(index="category", columns="binary_split", values="percent_with_category")
    pivot = pivot.sort_values(TRUE_GROUP, ascending=False)
    y = np.arange(len(pivot))
    sizes = category_summary.groupby("binary_split").n_group.first()
    for offset, group, label, color in zip((-.19, .19), GROUPS, GROUP_LABELS, GROUP_COLORS):
        ax.barh(y + offset, pivot[group], height=.36, color=color,
                edgecolor=palette("navy"), linewidth=.5, label=f"{label} (n={sizes[group]:,})")
    ax.set_yticks(y, [name.replace("explicit UKB", "Explicit UK Biobank mention") for name in pivot.index])
    ax.set(ylim=(len(pivot) + .8, -.8), xlim=(0, 105), xlabel="Candidates with cue in title or abstract")
    _percent(ax, axis="x")
    _legend(ax, loc="lower left")
    _axis(ax, "A  Mentions and keyword profiles", grid="x")

    difference = "difference_TRUE_minus_rest"
    subsets = (
        tfidf_terms.loc[tfidf_terms[difference].gt(0)].nlargest(12, difference),
        tfidf_terms.loc[tfidf_terms[difference].lt(0)].nsmallest(12, difference),
    )
    max_difference = max(float(tfidf_terms[difference].abs().max()), .001)
    for row, (data, title, color, sign) in enumerate(zip(subsets,
            ("B  Terms enriched in unanimous TRUE", "C  Terms enriched in other candidates"),
            GROUP_COLORS, (1, -1))):
        ax = fig.add_subplot(grid[row, 1])
        positions = np.arange(len(data))
        ax.barh(positions, sign * data[difference], color=color,
                edgecolor=palette("navy"), linewidth=.5)
        ax.set_yticks(positions, data.term)
        ax.invert_yaxis()
        ax.set_xlim(0, max_difference * 1.05)
        ax.set_xlabel("Mean TF-IDF difference: " + ("TRUE minus other" if sign == 1 else "other minus TRUE"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if data.empty:
            ax.text(.5, .5, "No terms enriched in this group", transform=ax.transAxes, ha="center")
        _axis(ax, title, grid="x")
    return fig


def plot_semantic_diagnostics(data):
    """Show the same cached projection by consensus group and publication year."""
    apply_typography()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharex=True, sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=.12, h_pad=.1, wspace=.10)
    for group, label, color in zip(GROUPS, GROUP_LABELS, GROUP_COLORS):
        subset = data.loc[data.binary_split.eq(group)]
        axes[0].scatter(subset.semantic_x, subset.semantic_y, s=9, alpha=.65, color=color,
                        linewidths=0, rasterized=True, label=f"{label} (n={len(subset):,})")
    _legend(axes[0], loc="best")
    points = axes[1].scatter(data.semantic_x, data.semantic_y, c=data.year_int.astype(int),
                             norm=Normalize(2013, 2025), cmap=blue_cream_red_colormap(),
                             s=9, alpha=.8, linewidths=0, rasterized=True)
    cbar = fig.colorbar(points, ax=axes[1], pad=.02, fraction=.04, ticks=[2013, 2016, 2019, 2022, 2025])
    cbar.set_label("Publication year", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    for ax, title in zip(axes, ("A  Consensus groups", "B  Publication year")):
        ax.set(xlabel="Embedding principal component 1", ylabel="Embedding principal component 2")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        _axis(ax, title)
    return fig


def render_agreement_figures(*, model_summary, vote_distribution, agreement_matrix, yearly,
                             category_summary, tfidf_terms, tfidf_data,
                             semantic_data=None, semantic_metrics=None,
                             figure_dir, show_figures=True):
    """Export only the current grouped figures and their manuscript captions."""
    count = int(vote_distribution.n_candidates.sum())
    caption = (
        f"Candidate identification and three-model agreement, 2013–2025 (n={count:,} candidates). "
        "A, TRUE prediction rate among parsed responses and parsing rate among all candidates. "
        "B, Number of TRUE predictions across the three models; an unparsed response is not a TRUE vote. "
        "C, Pairwise agreement among candidates with parsed labels from both models. "
        "D, Annual numbers receiving three TRUE predictions and all other candidates (log scale; zero counts are omitted). "
        "E, Annual proportion of candidates receiving three TRUE predictions. F, Annual TRUE predictions by model. "
        "Missing years are left blank. Agreement measures consistency, not validation accuracy. "
        "Other candidates include disagreements and unparsed responses, not just unanimous FALSE labels. "
        "Pairwise denominators and detailed consensus categories are supplied in the accompanying CSV tables."
    )
    files = _export(plot_candidate_agreement(model_summary, vote_distribution, agreement_matrix, yearly),
                    figure_dir, "00_01_figure_candidate_agreement", caption, show=show_figures)
    sizes = tfidf_data.binary_split.value_counts()
    caption = (
        "Text characteristics of candidate publications, 2013–2025. "
        "A, Prevalence of explicit UK Biobank mentions and prespecified keyword cues in titles or abstracts, "
        "comparing unanimous three-model TRUE predictions with all other candidates. Cues are not mutually exclusive. "
        "B,C, Up to 12 terms or bigrams with the largest positive mean TF-IDF differences for each group; "
        f"sample sizes are {sizes.get(TRUE_GROUP, 0):,} unanimous TRUE and {sizes.get(REST_GROUP, 0):,} other candidates. "
        "Only terms enriched in the stated direction are plotted. Full term scores and exact cue counts are available as CSV. "
        "These are descriptive contrasts in model-selected groups, not independent validation of classification accuracy."
    )
    files.extend(_export(plot_candidate_text(category_summary, tfidf_terms), figure_dir,
                          "00_02_figure_candidate_text", caption, show=show_figures))
    if semantic_data is not None and semantic_metrics is not None:
        metrics = semantic_metrics.set_index("metric").value
        method = metrics.get("embedding_method", "saved embeddings")
        silhouette = float(metrics["SI_silhouette_index_cosine"])
        caption = (
            f"Semantic diagnostics of candidate publications, 2013–2025 (n={len(semantic_data):,}). "
            f"A,B, The same two-dimensional principal-component projection of {method} representations, "
            "coloured by three-model consensus group (A) and publication year (B). "
            f"The cosine silhouette index in the full representation is {silhouette:.3f}; "
            "this measures separation of model-defined groups, not classification accuracy. "
            "Coordinates and sample-specific metrics are provided as CSV."
        )
        files.extend(_export(plot_semantic_diagnostics(semantic_data), figure_dir,
                              "00_05_figure_semantic_diagnostics", caption, show=show_figures))
    return files
