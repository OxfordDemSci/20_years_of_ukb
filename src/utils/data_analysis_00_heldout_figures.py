"""Shared manuscript styling for the locked held-out evaluation figures.

These functions only plot supplied results: prompt selection, inference, thresholds
and the held-out confusion counts remain in the validation workflow.
"""
from string import ascii_uppercase

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

from .data_analysis_00_figures import _export, format_heatmap_colorbar, heatmap_limits
from .shared_style import (
    black_legend, blue_cream_red_colormap, label_panels, load_style, palette,
    panel_grid, panel_label, style_axis,
)


def _letter(index):
    label = ""
    while index >= 0:
        index, remainder = divmod(index, 26)
        label = ascii_uppercase[remainder] + label
        index -= 1
    return label


def _heatmap(ax, values, style, *, square=False, lower_triangle=False, vmin=0, vmax=1):
    mask = np.triu(np.ones(values.shape, dtype=bool), k=1) if lower_triangle else None
    sns.heatmap(
        values, vmin=vmin, vmax=vmax, cmap=blue_cream_red_colormap(),
        mask=mask,
        annot=True, fmt=".2f", annot_kws={"fontsize": style["annot_fs"]},
        square=square, cbar=False, ax=ax,
    )
    ax.set(xlabel="", ylabel="")
    ax.tick_params(axis="both", labelsize=style["tick_fs"], length=0)
    if lower_triangle:
        ax.tick_params(axis="both", bottom=True, left=True, top=False, right=False,
                       direction="out", length=style.get("tick_length", 3.5),
                       width=style.get("tick_width", .8), colors="black")
        # Border only displayed cells, leaving the upper triangle completely blank.
        for row, column in np.argwhere(~mask & values.notna().to_numpy()):
            ax.add_patch(Rectangle((column, row), 1, 1, fill=False,
                                   edgecolor="black", linewidth=.6, clip_on=False))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.grid(False, which="both")
    return ax.collections[0]


def _colorbar(fig, image, axes, label):
    bar = fig.colorbar(image, ax=axes, fraction=.025, pad=.025, shrink=.9)
    format_heatmap_colorbar(bar, label)
    return bar


def _save(fig, directory, stem, caption, *, show):
    return _export(fig, directory, stem, caption, show=show)


def plot_heldout_agreement(agreements, model_labels):
    """One prompt or six prompts with a shared observed-range colour scale."""
    style = load_style("00_dataset")
    single = len(agreements) == 1
    if not single and len(agreements) != 6:
        raise ValueError("Expected one or six held-out prompt agreement tables.")
    visible = np.concatenate([
        frame.to_numpy(dtype=float)[np.tril_indices(len(frame))]
        for frame in agreements.values()
    ])
    vmin, vmax = heatmap_limits(visible)
    fig, axes = panel_grid(
        1 if single else 2, 1 if single else 3, style,
        figsize=(8.2, 7.1) if single else (19, 12),
        squeeze=False, layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=.12, h_pad=.12, wspace=.06, hspace=.08)
    for ax, agreement in zip(axes.flat, agreements.values()):
        labelled = agreement.rename(index=model_labels, columns=model_labels)
        image = _heatmap(ax, labelled, style, square=True, lower_triangle=True,
                         vmin=vmin, vmax=vmax)
    label_panels(axes, ascii_uppercase, style)
    _colorbar(fig, image, axes, "Pairwise agreement")
    return fig


def render_heldout_agreement(agreements, prompt_labels, model_labels, directory,
                             *, show_figures=True):
    files = []
    caveat = ("Only the lower triangle, including the diagonal, is shown. "
              "The colour scale spans the observed minimum and maximum within each figure. "
              "Agreement uses jointly parsed predictions, not accuracy against ground-truth labels.")
    for prompt, agreement in agreements.items():
        caption = f"A, Held-out pairwise agreement: {prompt_labels[prompt]}. {caveat}"
        files += _save(
            plot_heldout_agreement({prompt: agreement}, model_labels), directory,
            f"pairwise_agreement_{prompt}", caption, show=show_figures,
        )
    key = "; ".join(f"{_letter(i)}, {prompt_labels[p]}" for i, p in enumerate(agreements))
    files += _save(
        plot_heldout_agreement(agreements, model_labels), directory,
        "pairwise_agreement_all_six_prompts", f"Held-out pairwise agreement. {key}. {caveat}",
        show=show_figures,
    )
    return files


def plot_heldout_performance(metrics, prompts, models, prompt_labels, model_labels):
    style = load_style("00_dataset")
    selected = metrics.loc[metrics.prompt.isin(prompts) & metrics.model.isin(models)]
    low, high = heatmap_limits(selected[["precision", "recall", "f1"]].to_numpy())
    fig, axes = panel_grid(1, 3, style, figsize=(20, 6.5), layout="constrained")
    fig.get_layout_engine().set(w_pad=.12, h_pad=.12, wspace=.06)
    for ax, metric in zip(axes, ("precision", "recall", "f1")):
        table = metrics.pivot(index="prompt", columns="model", values=metric)
        table = table.reindex(index=list(prompts), columns=list(models))
        table = table.rename(index=prompt_labels, columns=model_labels)
        image = _heatmap(ax, table, style, vmin=low, vmax=high)
    label_panels(axes, "ABC", style)
    _colorbar(fig, image, axes, "Held-out score")
    return fig


def plot_selected_prompt_performance(metrics):
    style = load_style("00_dataset")
    frame = metrics.melt(
        id_vars=["model_display"], value_vars=["precision", "recall", "f1"],
        var_name="metric", value_name="score",
    )
    fig, ax = panel_grid(1, 1, style, figsize=(11, 5.8), layout="constrained")
    sns.barplot(
        data=frame, x="model_display", y="score", hue="metric", errorbar=None,
        palette=palette("red", "cream", "light_blue"), edgecolor="black", linewidth=.6, ax=ax,
    )
    ax.set(ylim=(0, 1.04), xlabel="", ylabel="Held-out score")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    style_axis(ax, style, grid_axis="y", grid_kws={"which": "major"})
    black_legend(ax, style, loc="lower right", bbox_to_anchor=(1, 1.01), ncol=3)
    panel_label(ax, "A", style)
    return fig


def render_heldout_performance(metrics, primary_metrics, prompts, models,
                              prompt_labels, model_labels, selected_prompt, directory,
                              *, show_figures=True):
    files = _save(
        plot_heldout_performance(metrics, prompts, models, prompt_labels, model_labels),
        directory, "heldout_precision_recall_f1_heatmaps",
        "Held-out performance across prompts and classifiers. A, Precision; B, Recall; C, F1. "
        "Rows identify prompt strategies and columns identify classifiers; all panels share the observed score range.",
        show=show_figures,
    )
    files += _save(
        plot_selected_prompt_performance(primary_metrics), directory,
        "heldout_selected_prompt_performance",
        f"A, Primary held-out performance for {prompt_labels[selected_prompt]}. "
        "This prompt was selected on the development set, before evaluation on the held-out set.",
        show=show_figures,
    )
    return files


def _draw_confusion(ax, matrix, style, *, show_y=True, limits):
    matrix = np.asarray(matrix)
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    image = ax.imshow(normalised, cmap=blue_cream_red_colormap(), vmin=limits[0], vmax=limits[1])
    for row, column in np.ndindex(2, 2):
        ax.text(
            column, row, f"{matrix[row, column]:,}\n({normalised[row, column]:.1%})",
            ha="center", va="center", fontsize=style["annot_fs"],
        )
    ax.set_xticks([0, 1], ["Pred. negative", "Pred. positive"])
    ax.set_yticks([0, 1], ["True negative", "True positive"] if show_y else ["", ""])
    ax.tick_params(labelsize=style["tick_fs"], length=0)
    ax.grid(False, which="both")
    return image


def plot_heldout_confusion(lookup, prompts, models, prompt_labels, model_labels):
    style = load_style("00_dataset")
    prompts, models = list(prompts), list(models)
    single = len(prompts) == 1
    if len(models) != 6 or len(prompts) not in (1, 6):
        raise ValueError("Expected six classifiers and one or six held-out prompts.")
    fig, axes = panel_grid(
        2 if single else 6, 3 if single else 6, style,
        figsize=(15, 10) if single else (24, 23), layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=.10, h_pad=.12, wspace=.03, hspace=.04)
    pairs = [(prompt, model) for prompt in prompts for model in models]
    matrices = [np.asarray(lookup[pair]) for pair in pairs]
    limits = heatmap_limits(*(m / np.maximum(m.sum(axis=1, keepdims=True), 1) for m in matrices))
    for index, (ax, (prompt, model)) in enumerate(zip(axes.flat, pairs)):
        image = _draw_confusion(ax, lookup[(prompt, model)], style,
                                show_y=single or index % len(models) == 0, limits=limits)
        if single or index // len(models) == len(prompts) - 1:
            ax.set_xlabel(model_labels[model])
        if not single and index % len(models) == 0:
            ax.set_ylabel(prompt_labels[prompt])
        panel_label(ax, _letter(index), style)
    _colorbar(fig, image, axes, "Share within true-label row")
    return fig


def render_heldout_confusion(lookup, prompts, models, prompt_labels, model_labels,
                            directory, *, show_figures=True):
    prompts, models = list(prompts), list(models)
    details = (
        "Each matrix shows counts and row-normalised percentages. "
        "Unparsed LLM responses count as negative, matching the deployment rule; "
        "parsing coverage is reported separately."
    )
    key = "; ".join(
        f"{_letter(i * len(models))}-{_letter((i + 1) * len(models) - 1)}, {prompt_labels[p]}"
        for i, p in enumerate(prompts)
    )
    classifiers = ", ".join(model_labels[m] for m in models)
    files = _save(
        plot_heldout_confusion(lookup, prompts, models, prompt_labels, model_labels),
        directory, "heldout_confusion_matrices_6_prompts_6_models",
        f"Held-out confusion matrices. Rows: {key}. Columns, left to right: {classifiers}. {details}",
        show=show_figures,
    )
    for prompt in prompts:
        key = "; ".join(f"{_letter(i)}, {model_labels[m]}" for i, m in enumerate(models))
        files += _save(
            plot_heldout_confusion(lookup, [prompt], models, prompt_labels, model_labels),
            directory, f"heldout_confusion_matrices_{prompt}",
            f"Held-out confusion matrices for {prompt_labels[prompt]}. {key}. {details}",
            show=show_figures,
        )
    return files
