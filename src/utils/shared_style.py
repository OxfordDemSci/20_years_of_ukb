"""Small shared plotting helpers used by the non-patent analysis notebooks.

Only the three helpers whose bodies were byte-identical across notebooks live here:
``apply_style``, ``extended_palette`` and ``savefig``. The per-notebook ``STYLE``
dictionaries are deliberately NOT centralised — they differ on purpose (01_authors
uses a 4-colour palette saving to ``fig/authors``; 05_clinical_trials uses a 6-colour
LCDS palette at dpi 400 with black edges) — so each notebook keeps its own ``STYLE``
literal and registers it here once with ``use_style(STYLE)``.

Because the notebooks call ``apply_style()`` / ``savefig(fig, name)`` WITHOUT passing a
style (relying on the old ``style=STYLE`` default), these functions resolve an omitted
style to the last one registered via ``use_style`` — preserving the original behaviour
exactly while removing the duplicated function bodies.

Usage (right after the notebook's ``STYLE = {...}`` block):

    from utils.shared_style import apply_style, extended_palette, savefig, use_style
    use_style(STYLE)
    apply_style()            # same as before
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.rcParams.update({
        "figure.dpi": 110,            # on-screen
        "savefig.dpi": style["dpi"],  # exported
        "axes.titlesize": style["title_fs"],
        "axes.labelsize": style["label_fs"],
        "xtick.labelsize": style["tick_fs"],
        "ytick.labelsize": style["tick_fs"],
        "legend.fontsize": style["legend_fs"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
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


def savefig(fig, name, style=None):
    """Optionally persist a figure to the configured directory."""
    style = _resolve(style)
    if style.get("save"):
        outdir = Path(style["savedir"]); outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{name}.pdf", dpi=style["dpi"], bbox_inches="tight")
        print("saved", outdir / f"{name}.pdf")
