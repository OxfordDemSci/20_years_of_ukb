"""Shared plotting style + helpers for the analysis notebooks.

The style itself lives in ``universal_settings.yml`` at the repo root, not in the
notebooks. Notebooks used to each carry their own ``STYLE = {...}`` literal, which is
how the palettes and font sizes drifted apart; now every notebook asks this module for
its section of that one file:

    from utils.shared_style import load_style
    STYLE = load_style("03_academic_impact")     # base + that notebook's overrides

``load_style`` merges ``style.base`` with ``style.notebooks.<name>``, registers the
result as the active style and pushes it into rcParams, so the three original helpers
(``apply_style``, ``extended_palette``, ``savefig``) keep working when called without an
explicit ``style=`` argument — which is how the notebooks call them.

``semantic_colors`` exposes resolved ``*_colors`` mappings from the active style so
metric identities can remain consistent across every figure in an analysis. The
remaining public helpers centralize figure construction, panel lettering, axes,
legends, statistical annotations, colorbars, saving, and notebook registration so
analysis modules only need to define data-specific marks and layouts.

Notebooks that still build a ``STYLE`` dict by hand keep working: register it with
``use_style(STYLE)`` exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# utils/ -> src/ -> repo root. Kept local rather than imported from shared_paths so this
# module stays usable on its own.
ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = ROOT / "universal_settings.yml"
DEFAULT_MARKER_SIZE = 8.5
DEFAULT_DOT_MARKER_AREA = 76.0

# The style most recently registered by a notebook via use_style(); used whenever a
# helper is called without an explicit `style=` argument (the common case).
_ACTIVE: dict | None = None


def use_style(style: dict) -> dict:
    """Register `style` as the active style for later style-less helper calls. Returns it."""
    global _ACTIVE
    _ACTIVE = style
    return style


def _resolve(style):
    if style is not None:
        return style
    if _ACTIVE is None:
        raise RuntimeError(
            "shared_style: no style given and none registered — call use_style(STYLE) first."
        )
    return _ACTIVE


def marker_size(style=None, *, scale=1.0):
    """Return the shared line/errorbar marker diameter in points.

    Legacy plotting modules can call this before registering a style; active notebooks
    receive their configured value. ``scale`` is reserved for genuinely compact or
    emphasized marks rather than introducing another hard-coded diameter.
    """
    resolved = _ACTIVE if style is None else style
    base = (resolved or {}).get("marker_size", DEFAULT_MARKER_SIZE)
    return float(base) * float(scale)


def marker_area(style=None, *, scale=1.0):
    """Return the shared analytical-dot area in points squared."""
    resolved = _ACTIVE if style is None else style
    base = (resolved or {}).get("dot_marker_area", DEFAULT_DOT_MARKER_AREA)
    return float(base) * float(scale)


def apply_style(style=None):
    """Push the shared style into matplotlib rcParams (call once / after edits).

    Body is verbatim from the notebooks (identical across 01/04/05)."""
    style = _resolve(style)
    font_family = style.get("font_family", "DejaVu Sans")
    plt.rcParams.update({
        "figure.dpi": 110,            # on-screen
        "savefig.dpi": style["dpi"],  # exported
        "font.size": style.get("body_fs", style["annot_fs"]),
        "axes.titlesize": style["title_fs"],
        "axes.labelsize": style["label_fs"],
        "axes.linewidth": style.get("axes_linewidth", 0.8),
        "xtick.labelsize": style["tick_fs"],
        "ytick.labelsize": style["tick_fs"],
        "xtick.major.size": style.get("tick_length", 3.5),
        "ytick.major.size": style.get("tick_length", 3.5),
        "xtick.major.width": style.get("tick_width", 0.8),
        "ytick.major.width": style.get("tick_width", 0.8),
        "xtick.major.pad": style.get("tick_pad", 3.5),
        "ytick.major.pad": style.get("tick_pad", 3.5),
        "xtick.minor.size": style.get("minor_tick_length", 2.0),
        "ytick.minor.size": style.get("minor_tick_length", 2.0),
        "xtick.minor.width": style.get("minor_tick_width", 0.6),
        "ytick.minor.width": style.get("minor_tick_width", 0.6),
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": style["legend_fs"],
        "legend.title_fontsize": style["legend_fs"],
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "legend.framealpha": 1.0,
        "lines.markersize": marker_size(style),
        "grid.linestyle": style.get("grid_linestyle", "-"),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": font_family,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    sns.set_palette(style["colors"])


def extended_palette(n, style=None):
    """N colours interpolated across the brand colours (for >len(colors) categories)."""
    style = _resolve(style)
    from matplotlib.colors import LinearSegmentedColormap
    if n <= len(style["colors"]):
        return style["colors"][:n]
    cmap = LinearSegmentedColormap.from_list("brand", style["colors"], N=256)
    return [cmap(i / (n - 1)) for i in range(n)]


def semantic_colors(name="metric_colors", style=None):
    """Return a copy of a configured semantic colour map.

    Maps whose setting name ends in ``_colors`` are resolved by ``load_style`` before
    this helper is called, so callers receive concrete Matplotlib colour values rather
    than palette indexes.
    """
    style = _resolve(style)
    mapping = style.get(name)
    if not isinstance(mapping, dict) or not mapping:
        raise KeyError(f"shared_style: no non-empty {name!r} mapping in the active style")
    return dict(mapping)


def new_figure(style=None, *, figsize=None, figsize_key="figsize_panel", **kwargs):
    """Create a figure using a configured size unless one is supplied explicitly."""
    style = _resolve(style)
    resolved_size = style[figsize_key] if figsize is None else figsize
    return plt.figure(figsize=resolved_size, **kwargs)


def gridspec_figure(
    nrows,
    ncols,
    style=None,
    *,
    figsize=None,
    figsize_key="figsize_panel",
    figure_kwargs=None,
    **gridspec_kwargs,
):
    """Create a styled figure and its top-level GridSpec together."""
    fig = new_figure(
        style,
        figsize=figsize,
        figsize_key=figsize_key,
        **dict(figure_kwargs or {}),
    )
    return fig, fig.add_gridspec(nrows, ncols, **gridspec_kwargs)


def panel_grid(
    nrows,
    ncols,
    style=None,
    *,
    figsize=None,
    figsize_key="figsize_panel",
    adjust=None,
    **subplot_kwargs,
):
    """Create a regular styled subplot grid and apply optional fixed margins."""
    style = _resolve(style)
    resolved_size = style[figsize_key] if figsize is None else figsize
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=resolved_size,
        **subplot_kwargs,
    )
    if adjust:
        fig.subplots_adjust(**dict(adjust))
    return fig, axes


def style_axis(
    ax,
    style=None,
    *,
    grid_axis="both",
    grid=True,
    zero=False,
    grid_kws=None,
):
    """Apply the shared black-axis and dashed-grid treatment to an axis."""
    style = _resolve(style)
    linewidth = style.get("axes_linewidth", 1.0)
    for name in ("left", "bottom"):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color("black")
        ax.spines[name].set_linewidth(linewidth)
    ax.tick_params(colors="black")
    ax.set_axisbelow(True)
    if grid:
        grid_style = {
            "color": "#D2D2D2",
            "linestyle": style.get("grid_linestyle", "--"),
            "linewidth": 0.65,
            "alpha": 0.75,
        }
        grid_style.update(dict(grid_kws or {}))
        ax.grid(True, axis=grid_axis, **grid_style)
    else:
        ax.grid(False, axis=grid_axis)
    if zero:
        ax.axhline(0, color="black", linewidth=0.8)
    return ax


def panel_label(
    ax,
    label,
    style=None,
    *,
    x=-0.12,
    y=1.07,
    ha="left",
    va="bottom",
    fontsize=None,
    in_layout=None,
    clip_on=False,
):
    """Add a consistently styled panel letter in axes coordinates."""
    style = _resolve(style)
    artist = ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=style["title_fs"] if fontsize is None else fontsize,
        fontweight="bold",
        color="black",
        clip_on=clip_on,
    )
    if in_layout is not None:
        artist.set_in_layout(in_layout)
    return artist


def label_panels(axes, labels, style=None, **kwargs):
    """Apply sequential panel letters to any flat or array-like axes collection."""
    flattened = axes.flat if hasattr(axes, "flat") else axes
    return [
        panel_label(ax, label, style=style, **kwargs)
        for ax, label in zip(flattened, labels)
    ]


def black_legend(ax, style=None, **kwargs):
    """Create an opaque white legend with the project-standard black border."""
    style = _resolve(style)
    legend = ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
        **kwargs,
    )
    legend.get_frame().set_linewidth(style.get("legend_linewidth", 0.9))
    return legend


def summary_box(
    ax,
    lines,
    style=None,
    *,
    x,
    y,
    ha="left",
    va="top",
    fontsize=None,
    zorder=5,
    bbox_kws=None,
):
    """Place a consistently styled statistical summary inside an axis."""
    style = _resolve(style)
    text = lines if isinstance(lines, str) else "\n".join(lines)
    box = {
        "boxstyle": "square,pad=0.35",
        "facecolor": "white",
        "edgecolor": "k",
        "linewidth": 0.9,
        "alpha": 0.94,
    }
    box.update(dict(bbox_kws or {}))
    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=style["annot_fs"] if fontsize is None else fontsize,
        bbox=box,
        zorder=zorder,
    )


def compact_count(value, digits=1):
    """Format a large count compactly for dense figure annotations."""
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{digits}f}m"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.{digits}f}k"
    return f"{value:,.0f}"


def percent_axis(ax, *, axis="y", xmax=100, decimals=0):
    """Apply Matplotlib's percentage formatter to one axis."""
    import matplotlib.ticker as mticker

    formatter = mticker.PercentFormatter(xmax=xmax, decimals=decimals)
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(formatter)
    return ax


def style_colorbar(colorbar, label=None, *, edgecolor="black", linewidth=0.8):
    """Apply the standard outline and optional label to a colorbar."""
    if label is not None:
        colorbar.set_label(label)
    colorbar.outline.set_edgecolor(edgecolor)
    colorbar.outline.set_linewidth(linewidth)
    return colorbar


def mask_grid_region(ax, start, end=None, *, color="white", zorder=1.5):
    """Mask gridlines in a reserved annotation region without hiding text."""
    end = ax.get_xlim()[1] if end is None else end
    return ax.axvspan(
        start,
        end,
        facecolor=color,
        edgecolor="none",
        zorder=zorder,
    )


def sequential_colormap(color):
    """Return a light sequential map whose maximum is exactly ``color``."""
    return sns.light_palette(color, as_cmap=True)


def savefig(fig, name, style=None, formats=None, dpi=None, **kwargs):
    """Optionally persist a figure in one or more configured formats.

    `savedir` is anchored on the repo root when relative, so a figure lands in the same
    place whether the notebook was launched from the root or from src/data_analysis/.
    Explicit ``formats=`` and ``dpi=`` values override the active style. Saved paths are
    returned, while log messages always use repository-relative paths.
    """
    style = _resolve(style)
    if not style.get("save"):
        return []
    outdir = Path(style["savedir"])
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    requested = formats or style.get("formats") or ["pdf"]
    requested = list(dict.fromkeys(str(ext).lower().lstrip(".") for ext in requested))
    stem = str(Path(name).with_suffix("")) if Path(name).suffix else str(name)
    saved = []
    for ext in requested:
        dest = outdir / f"{stem}.{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white", **kwargs}
        if ext in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs.setdefault("dpi", dpi or style["dpi"])
        fig.savefig(dest, format=ext, **save_kwargs)
        saved.append(dest)
        try:
            shown = dest.relative_to(ROOT).as_posix()
        except ValueError:
            shown = dest.name
        print("saved", shown)
    return saved


def save_figure(fig, name, style=None, **kwargs):
    """Save a figure and return the `(figure, paths)` contract used by notebooks."""
    return fig, savefig(fig, name, style=style, **kwargs)


def render_figure(plotter, *args, registry=None, show=True, **kwargs):
    """Build, register, and optionally display a saved figure in one call."""
    fig, paths = plotter(*args, **kwargs)
    if registry is not None:
        registry.record_figures(paths)
    if show:
        plt.show()
    return fig, paths


# =============================================================================
# universal_settings.yml
# =============================================================================
def load_settings(path=None) -> dict:
    """Parse universal_settings.yml (the whole file) into a dict."""
    import yaml
    with open(Path(path) if path else SETTINGS_FILE) as fh:
        return yaml.safe_load(fh) or {}


def _resolve_refs(mapping, colors):
    """Turn integer colour references into palette entries; leave strings alone."""
    return {k: (colors[v] if isinstance(v, int) and not isinstance(v, bool) else v)
            for k, v in (mapping or {}).items()}


def load_style(notebook, path=None, activate=True) -> dict:
    """Build one notebook's STYLE from universal_settings.yml.

    `style.base` merged with `style.notebooks.<notebook>`; integer values under `roles`
    and any `*_colors` mapping are resolved against the palette; `figsize*` lists become
    tuples. With `activate` (the default) the result is registered via use_style and
    pushed into rcParams, so `savefig(fig, name)` and `extended_palette(n)` need no
    further argument.
    """
    cfg = (load_settings(path).get("style") or {})
    known = cfg.get("notebooks") or {}
    if notebook not in known:
        raise KeyError(f"universal_settings.yml has no style.notebooks.{notebook} "
                       f"(known: {sorted(known)})")
    style = {**(cfg.get("base") or {}), **(known[notebook] or {})}

    colors = list(style.get("colors") or [])
    style["colors"] = colors
    for key in [k for k in style if k.endswith("_colors")]:
        style[key] = _resolve_refs(style[key], colors)
    style.update(_resolve_refs(style.pop("roles", None), colors))
    for key, val in style.items():
        if key.startswith("figsize") and isinstance(val, list):
            style[key] = tuple(val)

    if activate:
        use_style(style)
        apply_style(style)
    return style


def year_ticks(lo, hi, step: int = 2) -> list:
    """Tick years for a time axis, anchored on the LAST year rather than the first.

    `range(lo, hi + 1, step)` and `MultipleLocator(step)` both anchor at the low end, so
    whether the final year gets a label is an accident of parity: a 2014-2023 axis at
    step 2 labels 2014, 2016, 2018, 2020, 2022 and leaves the last point sitting past an
    unlabelled tick. A reader then takes the axis at its word and reads the series as
    ending in 2022 — which is exactly the misreading this analysis has to avoid, because
    the last year is the one every citation measure is arguing about.

    Anchoring at `hi` instead guarantees the endpoint is labelled and costs nothing: the
    ticks are just as evenly spaced, they simply run backwards from the end.

        year_ticks(2014, 2023)      -> [2015, 2017, 2019, 2021, 2023]
        year_ticks(2013, 2024, 3)   -> [2015, 2018, 2021, 2024]
    """
    lo, hi = int(lo), int(hi)
    if hi < lo:
        lo, hi = hi, lo
    return list(range(hi, lo - 1, -int(step)))[::-1]
