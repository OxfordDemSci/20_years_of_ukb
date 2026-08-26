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
metric identities can remain consistent across every figure in an analysis.

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
