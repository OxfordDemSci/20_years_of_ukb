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

Every size, stroke and colour a draw function needs comes from a helper rather than a
literal (D55): ``figsize(<rung>)`` for the 18 in width and height ladder,
``font_size(<role>)``, ``stroke(<role>)``, ``neutral(<role>)`` and ``figure_colormap()``.
``tests/test_style_uniformity.py`` fails on a literal that bypasses them.

``semantic_colors`` exposes resolved ``*_colors`` mappings from the active style so
metric identities can remain consistent across every figure in an analysis. The
remaining public helpers centralize figure construction, panel lettering, axes,
legends, statistical annotations, colorbars, saving, and notebook registration so
analysis modules only need to define data-specific marks and layouts.

Panel headings are letters only. Descriptions belong in axis labels, legends or
notebook captions; ``finalize_figure`` also removes native and third-party titles
before export or shared inline display.

Notebooks that still build a ``STYLE`` dict by hand keep working: register it with
``use_style(STYLE)`` exactly as before.
"""

from __future__ import annotations

from pathlib import Path
import re
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.text import Text
import seaborn as sns

# utils/ -> src/ -> repo root. Kept local rather than imported from shared_paths so this
# module stays usable on its own.
ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = ROOT / "universal_settings.yml"
DEFAULT_MARKER_SIZE = 8.5
DEFAULT_DOT_MARKER_AREA = 76.0
PNG_DPI = 500
FONT_FAMILY = "Helvetica"

# The style most recently registered by a notebook via use_style(); used whenever a
# helper is called without an explicit `style=` argument (the common case).
_ACTIVE: dict | None = None


# =============================================================================
# The project palette, by name
# =============================================================================
# `universal_settings.yml` carries palettes as bare ordered lists, which is what
# matplotlib wants but not what a figure author wants: "colors[4]" says nothing about
# whether panel C shares an identity with panel A. These are the same colours keyed on
# their names, so a draw function can ask for `palette("navy")` and a reader of the code
# can see the intent. Order matters as well as name — `PALETTE` below is the list form,
# and it is what `style.notebooks.04_non_academic*` uses, so an integer role index in the
# yaml and a name here address the same colour.
PALETTE_COLORS = {
    "red": "#E66859",
    "cream": "#FEE7BA",
    "light_blue": "#75BBD4",
    "blue": "#74ADD1",
    "steel_blue": "#416FA0",
    "navy": "#274668",
    # A 50% white tint of the red. It replaced the old green (#74D1AF) everywhere, so the
    # palette is blues, cream and reds only.
    "light_red": "#F2B4AC",
}

#: The same seven colours as an ordered list (the shape `style["colors"]` takes).
PALETTE = list(PALETTE_COLORS.values())

#: Analysis 04's palette. Once 04 alone swapped green for light red; now every analysis
#: does, so this is the same list under its older name.
NON_ACADEMIC_PALETTE = list(PALETTE)

#: The only greys a figure may use (D55). Mirrors `style.base.neutral_colors`.
NEUTRALS = {
    "missing": "#B7B7B7",   # an absence (Unknown, no data), never a category
    "muted": "#555555",     # secondary text, de-emphasised marks
    "light": "#E5E5E5",     # backgrounds, inactive bands
}

#: Separators between filled areas (stacked segments, map borders); every other edge is black.
EDGE_COLOR = "black"
SEPARATOR_COLOR = "white"

# D55: every exported figure is FIG_WIDTH inches wide and one of these height/width ratios,
# so one set of font sizes prints at one size wherever the figure is scaled to page width.
FIG_WIDTH = 18.0
FIG_ASPECTS = {"wide": 0.5, "landscape": 0.625, "standard": 0.8, "square": 1.0, "tall": 1.25}


def palette(*names):
    """Look colours up by name: ``palette("navy")`` -> str, ``palette("navy", "red")`` -> list.

    Raises on an unknown name rather than falling back, for the same reason `load_style`
    raises on an unknown notebook section (D4): a silent substitution is a figure that is
    wrong in a way nobody sees.
    """
    unknown = [n for n in names if n not in PALETTE_COLORS]
    if unknown:
        raise KeyError(f"shared_style: unknown palette colour(s) {unknown}; "
                       f"known: {sorted(PALETTE_COLORS)}")
    picked = [PALETTE_COLORS[n] for n in names]
    return picked[0] if len(picked) == 1 else picked


def neutral(name):
    """Look a grey up by role (``missing``, ``muted``, ``light``); raises on an unknown role."""
    if name not in NEUTRALS:
        raise KeyError(f"shared_style: unknown neutral {name!r}; known: {sorted(NEUTRALS)}")
    return NEUTRALS[name]


def use_style(style: dict) -> dict:
    """Register `style` as the active style for later style-less helper calls. Returns it."""
    global _ACTIVE
    _ACTIVE = style
    return style


_BASE: dict | None = None


def _style_or_base(style=None):
    """The given style, else the active one, else `style.base` — so legacy modules that
    draw before any notebook registered a style still get the project sizes."""
    global _BASE
    if style is not None:
        return style
    if _ACTIVE is not None:
        return _ACTIVE
    if _BASE is None:
        _BASE = (load_settings().get("style") or {}).get("base") or {}
    return _BASE


#: Text roles and the style key each reads. Four sizes in all: 22 · 16 · 14 · 13.
FONT_ROLES = {
    "panel_label": "panel_label_fs",
    "title": "title_fs",
    "label": "label_fs",
    "tick": "tick_fs",
    "legend": "legend_fs",
    "annot": "annot_fs",
    "small": "small_fs",
}


def font_size(role, style=None):
    """The project font size (points) for a text role — never a literal at the call site."""
    if role not in FONT_ROLES:
        raise KeyError(f"shared_style: unknown font role {role!r}; known: {sorted(FONT_ROLES)}")
    return _style_or_base(style)[FONT_ROLES[role]]


#: Stroke roles: `edge` outlines filled marks, `line` draws a data series, `ref` a reference line.
STROKE_ROLES = {"edge": "edge_linewidth", "line": "line_width", "ref": "ref_linewidth"}


def stroke(role, style=None):
    """The project line width (points) for a stroke role."""
    if role not in STROKE_ROLES:
        raise KeyError(f"shared_style: unknown stroke role {role!r}; known: {sorted(STROKE_ROLES)}")
    return _style_or_base(style)[STROKE_ROLES[role]]


def nearest_aspect(ratio):
    """The ladder rung (name) closest to a height/width ratio."""
    return min(FIG_ASPECTS, key=lambda name: abs(FIG_ASPECTS[name] - float(ratio)))


def figsize(aspect="standard", style=None):
    """``(width, height)`` in inches: the project width and one rung of the height ladder.

    ``aspect`` is a rung name or a height/width ratio, which is snapped to the nearest
    rung — so a data-driven layout (one bar row per field) still lands on the ladder.
    """
    style = _style_or_base(style)
    width = float(style.get("fig_width", FIG_WIDTH))
    aspects = style.get("fig_aspects") or FIG_ASPECTS
    name = aspect if isinstance(aspect, str) else nearest_aspect(aspect)
    if name not in aspects:
        raise KeyError(f"shared_style: unknown figure aspect {name!r}; known: {sorted(aspects)}")
    return (width, width * float(aspects[name]))


def _resolve(style):
    """The given style, else the registered one, else `style.base` (D55 keeps every size there)."""
    return _style_or_base(style)


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


def apply_typography():
    """Apply the project font and title defaults without changing sizes or colours."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY],
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        # The installed Helvetica faces use the standard hyphen for negative ticks.
        "axes.unicode_minus": False,
        "figure.titleweight": "bold",
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
        "mathtext.bfit": f"{FONT_FAMILY}:bold:italic",
        "mathtext.sf": FONT_FAMILY,
        "mathtext.tt": FONT_FAMILY,
        "mathtext.cal": FONT_FAMILY,
    })


def uppercase_title(label):
    """Uppercase title prose, preserving case-sensitive mathematical notation."""
    parts = re.split(r"(\$.*?\$)", str(label), flags=re.DOTALL)
    return "".join(part if i % 2 else part.upper() for i, part in enumerate(parts))


def _title_kwargs(kwargs):
    # Remove aliases so an older font/alignment argument cannot defeat the policy.
    for key in ("family", "fontname", "weight", "horizontalalignment", "position"):
        kwargs.pop(key, None)
    kwargs.update(fontfamily=FONT_FAMILY, fontweight="bold", ha="left")
    return kwargs


def set_title(ax, label, fontdict=None, **kwargs):
    """Set only a panel letter; descriptive prose belongs in labels or captions."""
    apply_typography()
    letter = panel_title_letter(label)
    if str(label).strip() and not re.fullmatch(r"\(?[A-Za-z]\)?[.:]?", str(label).strip()):
        description = re.sub(r"^\(?[A-Za-z]\)?[).:]?\s+", "", str(label).strip()) if letter else str(label).strip()
        if len(description) > 1:
            ax._ukb_description = description
    existing = panel_title_letter(ax.get_title(loc="left"))
    if not letter and str(label).strip() and existing:
        # A legacy drawing helper must not overwrite or shrink an existing letter.
        ax.title.set_text("")
        ax._right_title.set_text("")
        ax._left_title.set_text(existing)
        return ax._left_title
    kwargs = _title_kwargs(kwargs)
    kwargs.update(loc="left", x=0.0)
    ax.title.set_text("")
    ax._right_title.set_text("")
    return ax.set_title(letter, fontdict=fontdict, **kwargs)


def panel_title_letter(label):
    """Extract explicit legacy panel prefixes without turning prose into initials."""
    text = str(label).strip()
    match = re.fullmatch(r"\(?([A-Za-z])\)?[.:]?", text)
    if match is None:
        match = re.match(r"^(?:\(([A-Za-z])\)|([A-Za-z])[.):]|([A-Z])\s)\s*", text)
    if match is None:
        return ""
    return next(group.upper() for group in match.groups() if group)


def facet_ylabel(ax, label, *, width=26, **kwargs):
    """Keep full facet names on the axis without overrunning small-multiple rows."""
    return ax.set_ylabel(fill(str(label), width=width, break_long_words=False,
                             break_on_hyphens=False), **kwargs)


def set_figure_title(fig, label, **kwargs):
    """Compatibility wrapper: manuscript figures have no figure-wide title."""
    apply_typography()
    fig._ukb_description = str(label).strip()
    kwargs = _title_kwargs(kwargs)
    kwargs.pop("loc", None)  # Figure.suptitle uses x/ha rather than loc.
    kwargs["x"] = 0.01
    return fig.suptitle("", **kwargs)


def finalize_figure(fig):
    """Enforce typography and letter-only headings before display or export.

    Third-party plotters can create text with their own font properties. Native axes
    titles (including inset axes) and figure titles also receive the title policy;
    axis labels, legends and data annotations keep their wording and weights.
    """
    apply_typography()
    for ax in fig.findobj(match=Axes):
        titles = [title for title in (ax._left_title, ax.title, ax._right_title)
                  if title.get_text()]
        if not titles:
            continue
        base = titles[0]
        letter = next((panel_title_letter(title.get_text()) for title in titles
                       if panel_title_letter(title.get_text())), "")
        properties = {"fontsize": base.get_fontsize(), "color": base.get_color()}
        if not ax._autotitlepos:
            properties["y"] = base.get_position()[1]
        ax.title.set_text("")
        ax._right_title.set_text("")
        ax._left_title.set_text("")
        set_title(ax, letter, **properties)
    for container in [fig, *fig.findobj(match=lambda artist: hasattr(artist, "_suptitle"))]:
        title = getattr(container, "_suptitle", None)
        if title is not None and title.get_text():
            title.set_text("")
    for artist in fig.findobj(match=Text):
        properties = artist.get_fontproperties().copy()
        properties.set_file(None)
        properties.set_family(FONT_FAMILY)
        artist.set_fontproperties(properties)
    return fig


def apply_style(style=None):
    """Apply shared figure settings and the project-wide typography policy."""
    style = _resolve(style)
    plt.rcParams.update({
        "figure.dpi": 110,            # on-screen
        "savefig.dpi": style["dpi"],  # exported
        "font.size": style.get("body_fs", style["annot_fs"]),
        "lines.markersize": style.get("marker_size", DEFAULT_MARKER_SIZE),
        "lines.linewidth": style.get("line_width", 2.0),
        # Edges are drawn only where a call asks for them (no force_edgecolor), but when
        # they are drawn they are black at one width.
        "patch.edgecolor": style.get("edgecolor", EDGE_COLOR),
        "patch.linewidth": style.get("edge_linewidth", 0.6),
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
        # Dashed everywhere, and the same dash for minor gridlines as for major:
        # matplotlib has ONE grid.linestyle, so the only way a minor gridline ends up
        # solid is a per-call `ax.grid(which="minor", linestyle="-")`. Use grid_on() below
        # rather than raw ax.grid() and that cannot happen.
        "grid.linestyle": style.get("grid_linestyle", "--"),
        "grid.linewidth": style.get("grid_linewidth", 0.6),
        "grid.color": style.get("grid_color", "#B0B0B0"),
        "grid.alpha": style.get("grid_alpha", 0.6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": FONT_FAMILY,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    sns.set_palette(style["colors"])
    apply_typography()


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


LOG_SCALES = ("log", "symlog", "logit")


def is_log_axis(ax, axis):
    """True if the named axis ("x" or "y") is on a logarithmic-family scale."""
    return getattr(ax, f"get_{axis}scale")() in LOG_SCALES


def grid_on(ax, axis="both", which="both", style=None, *, log=False, **kwargs):
    """Turn on dashed gridlines behind the data, on both tick levels by default.

    Every panel in this project is meant to carry the same grid, and "the same" has to
    include the minor ticks: an axis with solid minor gridlines under dashed major ones
    reads as two different figures pasted together. This wraps `ax.grid` so a caller
    cannot forget the `which="minor"` call or give it a different dash, and it puts the
    grid below the artists (`set_axisbelow`) which `ax.grid` alone does not do.

    Logarithmic axes are the one exception and get NO gridlines: a decade carries eight
    minor ticks, so the same treatment that reads as a scale on a linear axis reads as
    hatching behind the data on a log one. The gridlines are turned off rather than left
    alone, so a log scale set after an earlier `grid_on` still ends up bare. Only the log
    axis loses its grid — a log-x/linear-y panel keeps its horizontal lines — and
    `log=True` forces the grid on anyway. Set the scale BEFORE calling this.
    """
    style = _resolve(style)
    ls = kwargs.pop("linestyle", style.get("grid_linestyle", "--"))
    lw = kwargs.pop("linewidth", style.get("grid_linewidth", 0.6))
    color = kwargs.pop("color", style.get("grid_color", "#B0B0B0"))
    alpha = kwargs.pop("alpha", style.get("grid_alpha", 0.6))
    ax.set_axisbelow(True)
    levels = ("major", "minor") if which == "both" else (which,)
    for target in (("x", "y") if axis == "both" else (axis,)):
        if not log and is_log_axis(ax, target):
            ax.grid(False, axis=target, which="both")
            continue
        for lvl in levels:
            ax.grid(True, axis=target, which=lvl, linestyle=ls, color=color,
                    linewidth=lw if lvl == "major" else lw * 0.7,
                    alpha=alpha if lvl == "major" else alpha * 0.6, **kwargs)
    return ax


def panel_label(ax, letter, style=None, *, x=None, y=None, ha="left", va="bottom",
                fontsize=None, in_layout=None, clip_on=False, **kwargs):
    """Stamp the bold panel letter (A, B, C ...) on a sub-panel.

    Two placements, and which one you get depends on whether you name a position:

    * **No `x`/`y` (the default).** The letter IS the title — no descriptive text beside
      it — so the caption carries the description and the panel carries only its address
      in that caption. Set as a left-aligned axes title rather than as free text at fixed
      axes coordinates: a horizontal bar panel with long category labels and a scatter
      panel with a narrow y axis then put their letters in visually the same place, which
      free text at `x=-0.08` does not.
    * **`x` and/or `y` given.** The letter is free text in axes coordinates. That is the
      escape hatch for a hand-built gridspec whose panels are not on a common grid — 05's
      headline figure letters a map, a network sidebar and three stacked charts inside one
      page, and no single title offset lands all five where a reader expects them.
    """
    style = _resolve(style)
    if fontsize is None:
        fontsize = style.get("panel_label_fs", style.get("title_fs", 14))
    if x is None and y is None:
        kwargs.setdefault("fontweight", "bold")
        kwargs.setdefault("loc", "left")
        kwargs.setdefault("pad", 8)
        return set_title(ax, letter, fontsize=fontsize, **kwargs)
    artist = ax.text(
        -0.12 if x is None else x,
        1.07 if y is None else y,
        str(letter).upper(),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontfamily=FONT_FAMILY,
        fontweight=kwargs.pop("fontweight", "bold"),
        color=kwargs.pop("color", "black"),
        clip_on=clip_on,
        **kwargs,
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


# =============================================================================
# Figure and axis construction
# =============================================================================
def new_figure(style=None, *, aspect="standard", **kwargs):
    """Create a figure at the project width and one rung of the height ladder (D55)."""
    return plt.figure(figsize=figsize(aspect, style), **kwargs)


def gridspec_figure(
    nrows,
    ncols,
    style=None,
    *,
    aspect="standard",
    figure_kwargs=None,
    **gridspec_kwargs,
):
    """Create a styled figure and its top-level GridSpec together."""
    fig = new_figure(style, aspect=aspect, **dict(figure_kwargs or {}))
    return fig, fig.add_gridspec(nrows, ncols, **gridspec_kwargs)


def panel_grid(
    nrows,
    ncols,
    style=None,
    *,
    aspect="standard",
    adjust=None,
    **subplot_kwargs,
):
    """Create a regular styled subplot grid and apply optional fixed margins."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize(aspect, style),
                             **subplot_kwargs)
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
    """Apply the shared black-axis and dashed-grid treatment to an axis.

    The grid goes on through `grid_on`, so an axis styled here carries the one project
    grid (D29) — dashed, behind the data, on both tick levels — rather than a second
    dashed-major-only variant that would read as a different figure family.
    """
    style = _resolve(style)
    linewidth = style.get("axes_linewidth", 1.0)
    for name in ("left", "bottom"):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color("black")
        ax.spines[name].set_linewidth(linewidth)
    ax.tick_params(colors="black")
    ax.set_axisbelow(True)
    if grid:
        grid_on(ax, axis=grid_axis, style=style, **dict(grid_kws or {}))
    else:
        ax.grid(False, axis=grid_axis, which="both")
    if zero:
        ax.axhline(0, color="black", linewidth=stroke("ref", style))
    return ax


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
    legend.get_frame().set_linewidth(stroke("edge", style))
    return legend


def align_legend_rows(legend):
    """Give multi-column legend rows equal heights, with centred text and handles.

    Invisible renderer-sized spacers preserve alignment in both raster and vector
    exports without adding blank lines to labels or shrinking the legend font.
    """
    from matplotlib.offsetbox import DrawingArea, HPacker
    from matplotlib.transforms import Bbox

    class RowSpacer(DrawingArea):
        def __init__(self, peers):
            super().__init__(0, 0)
            self.peers = peers
            self._ukb_legend_padding = True

        def get_bbox(self, renderer):
            # Font metrics vary with the export renderer, not just its DPI.
            height = max(child.get_bbox(renderer).height for child in self.peers)
            return Bbox.from_bounds(0, 0, 0, height)

    figure = legend.get_figure()
    columns = legend._legend_handle_box.get_children()
    for column in columns:
        column._children[:] = [row for row in column.get_children()
                               if not getattr(row, "_ukb_legend_padding", False)]
        for row in column.get_children():
            row._children[:] = [child for child in row.get_children()
                                if not getattr(child, "_ukb_legend_padding", False)]
            row.align = "center"
    rows_per_column = [column.get_children() for column in columns]
    for index in range(max(map(len, rows_per_column), default=0)):
        rows = [rows[index] for rows in rows_per_column if index < len(rows)]
        peers = [child for row in rows for child in row.get_children()]
        for column in columns:
            spacer = RowSpacer(peers)
            spacer.set_figure(figure)
            if index < len(column.get_children()):
                column.get_children()[index]._children.append(spacer)
            else:
                row = HPacker(children=[spacer], pad=0, sep=0, align="center")
                row._ukb_legend_padding = True
                row.set_figure(figure)
                column._children.append(row)
    legend.stale = True
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
        "edgecolor": "black",
        "linewidth": stroke("edge", style),
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


def scatter_callouts(ax, points, labels, *, obstacles=None, fontsize=None, width=28):
    """Place wrapped callouts in free axes space, with curved arrows to data points.

    Candidate boxes are tested in display coordinates, so this also works on log
    axes. Set the final axes limits and layout before calling this helper.
    ``obstacles`` can include additional unlabelled points to keep labels off data.
    """
    from textwrap import fill
    from matplotlib.transforms import Bbox
    import numpy as np

    points = np.asarray(points, dtype=float)
    if len(points) != len(labels):
        raise ValueError("Each callout needs one point and one label.")
    if not len(points):
        return []
    fontsize = _resolve(None)["annot_fs"] if fontsize is None else fontsize
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer).padded(-5)
    obstacle_points = points if obstacles is None else np.asarray(obstacles, dtype=float)
    dots = ax.transData.transform(obstacle_points)
    occupied = [Bbox.from_bounds(x - 11, y - 11, 22, 22) for x, y in dots]
    occupied.extend(t.get_window_extent(renderer).padded(3)
                    for t in ax.texts if t.get_visible() and t.get_text())
    artists = []
    pixels_per_point = fig.dpi / 72
    for point, label in zip(points, labels):
        anchor = ax.transData.transform(point)
        text = fill(str(label), width=width)
        probe = ax.text(*point, text, fontsize=fontsize)
        box = probe.get_window_extent(renderer)
        probe.remove()
        candidates = []
        for radius in (24, 40, 60, 85, 115, 150, 190):
            for angle in (0, 180, 45, 135, -45, -135, 90, -90):
                dx, dy = radius * np.array([np.cos(np.deg2rad(angle)),
                                           np.sin(np.deg2rad(angle))])
                ha = "left" if dx > 1 else "right" if dx < -1 else "center"
                x, y = anchor + pixels_per_point * np.array([dx, dy])
                left = x if ha == "left" else x - box.width if ha == "right" else x - box.width / 2
                candidate = Bbox.from_bounds(left, y - box.height / 2, box.width, box.height).padded(3)
                if not (bounds.contains(*candidate.p0) and bounds.contains(*candidate.p1)):
                    continue
                overlap = 0.0
                for other in occupied:
                    intersection = Bbox.intersection(candidate, other)
                    if intersection is not None:
                        overlap += intersection.width * intersection.height
                candidates.append((overlap * 1000 + radius, dx, dy, ha, candidate))
        if not candidates:
            raise ValueError("Callouts do not fit; enlarge the axes or wrap labels more narrowly.")
        _, dx, dy, ha, chosen = min(candidates, key=lambda item: item[0])
        artist = ax.annotate(text, point, xytext=(dx, dy), textcoords="offset points",
                             ha=ha, va="center", fontsize=fontsize, color="black",
                             bbox={"facecolor": "white", "edgecolor": "none", "pad": .8},
                             arrowprops={"arrowstyle": "->", "color": "black", "lw": .8,
                                         "connectionstyle": "arc3,rad=0.15", "shrinkB": 7},
                             zorder=5)
        occupied.append(chosen)
        artists.append(artist)
    return artists


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


def style_colorbar(colorbar, label=None, *, edgecolor="black", linewidth=None):
    """Apply the standard outline and optional label to a colorbar."""
    if label is not None:
        colorbar.set_label(label)
    colorbar.outline.set_edgecolor(edgecolor)
    colorbar.outline.set_linewidth(_style_or_base()["axes_linewidth"] if linewidth is None else linewidth)
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


#: The one continuous ramp (D55): light blue -> cream -> red, walked through the palette.
FIGURE_COLORMAP_ANCHORS = ("light_blue", "cream", "red")


def palette_colormap(*names, name="project_palette", reverse=False, n=256):
    """Build a continuous map from named anchors in the shared project palette."""
    from matplotlib.colors import LinearSegmentedColormap

    if len(names) < 2:
        raise ValueError("palette_colormap requires at least two named colours")
    colors = palette(*names)
    if reverse:
        colors = colors[::-1]
    return LinearSegmentedColormap.from_list(name, colors, N=n)


#: CIE L* added at the middle of the blue -> cream half of the ramp (none at the anchors).
FIGURE_COLORMAP_LIFT = 8.0

_D65 = (0.95047, 1.0, 1.08883)
_SRGB_TO_XYZ = ((0.4124564, 0.3575761, 0.1804375),
                (0.2126729, 0.7151522, 0.0721750),
                (0.0193339, 0.1191920, 0.9503041))


def _srgb_to_lab(rgb):
    import numpy as np

    rgb = np.asarray(rgb, dtype=float)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = np.asarray(_SRGB_TO_XYZ) @ linear / np.asarray(_D65)
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _lab_to_srgb(lab):
    import numpy as np

    fy = (lab[0] + 16) / 116
    f = np.array([fy + lab[1] / 500, fy, fy - lab[2] / 200])
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * np.asarray(_D65)
    linear = np.clip(np.linalg.solve(np.asarray(_SRGB_TO_XYZ), xyz), 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def figure_colormap():
    """The project's only continuous colormap (D55), for every heatmap, map and ramp.

    Light blue -> cream -> red, blended in CIE Lab rather than sRGB. A straight blend
    from the blue to the cream passes through a muddy sage that is in no palette, so
    that half is also lifted in lightness (`FIGURE_COLORMAP_LIFT`) and stays clean.
    The anchors themselves are unchanged. Masked and NaN cells are drawn white, so
    "no data" never reads as a low value.
    """
    import numpy as np
    from matplotlib.colors import ListedColormap, to_rgb

    anchors = [_srgb_to_lab(to_rgb(color)) for color in palette(*FIGURE_COLORMAP_ANCHORS)]
    colors = []
    for t in np.linspace(0, 1, 256):
        half = 0 if t <= 0.5 else 1
        u = 2 * t - half
        lab = (1 - u) * anchors[half] + u * anchors[half + 1]
        if half == 0:
            lab[0] += FIGURE_COLORMAP_LIFT * np.sin(np.pi * u)
        colors.append(_lab_to_srgb(lab))
    cmap = ListedColormap(colors, name="blue_cream_red")
    cmap.set_bad("white")
    return cmap


#: The name every caption, test and older call site knows the one map by.
blue_cream_red_colormap = figure_colormap


def ramp_colors(n, *, low=0.0, high=1.0):
    """``n`` discrete colours sampled evenly from the one colormap (ordered categories)."""
    import numpy as np
    return [figure_colormap()(x) for x in np.linspace(low, high, n)]


def figure_export_formats(formats):
    """Keep PNG/PDF exports, falling back to PDF for legacy format requests."""
    if isinstance(formats, str):
        formats = [formats]
    normalized = (str(ext).lower().lstrip(".") for ext in formats)
    return list(dict.fromkeys(ext for ext in normalized if ext in {"png", "pdf"})) or ["pdf"]


def fixed_width_bbox(fig, width=None):
    """The tight bounding box of `fig`, widened to exactly `width` inches (D55).

    `bbox_inches="tight"` crops each export to its own content, so two figures drawn at the
    same width export at different widths and their text prints at different sizes. This
    keeps the tight crop vertically but pads the horizontal crop out to the project width.
    Content wider than that is a layout defect: it is exported uncropped and a warning names
    the overflow, so the figure audit can find it.
    """
    import warnings
    from matplotlib.transforms import Bbox

    width = float(_style_or_base().get("fig_width", FIG_WIDTH)) if width is None else width
    fig.canvas.draw()
    pad = plt.rcParams["savefig.pad_inches"]
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    if tight.width > width + 1e-6:
        warnings.warn(f"shared_style: figure content is {tight.width:.2f}in wide, wider than "
                      f"the {width:g}in project width", stacklevel=3)
        return tight.padded(pad)
    # The horizontal margin is whatever the width leaves; only the vertical one is `pad`.
    extra = (width - tight.width) / 2
    return Bbox([[tight.x0 - extra, tight.y0 - pad], [tight.x1 + extra, tight.y1 + pad]])


def savefig(fig, name, style=None, formats=None, dpi=None, caption=None, **kwargs):
    """Optionally persist a figure as PNG and/or PDF.

    `savedir` is anchored on the repo root when relative, so a figure lands in the same
    place whether the notebook was launched from the root or from src/data_analysis/.
    PNG exports always use the project-wide ``PNG_DPI``, and every export is exactly the
    project width wide (`fixed_width_bbox`) unless the caller passes ``bbox_inches``. Legacy format requests
    cannot create SVG or JSON figure files. Saved paths are returned, while log
    messages always use repository-relative paths.
    """
    style = _resolve(style)
    from .shared_figure_captions import caption_for_name
    if caption is not None:
        fig._ukb_caption = caption
    elif not getattr(fig, "_ukb_caption", None):
        fig._ukb_caption = caption_for_name(name)
    finalize_figure(fig)
    if not style.get("save"):
        return []
    outdir = Path(style["savedir"])
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    requested = formats or style.get("formats") or ["pdf"]
    requested = figure_export_formats(requested)
    stem = str(Path(name).with_suffix("")) if Path(name).suffix else str(name)
    saved = []
    if "bbox_inches" not in kwargs:
        kwargs["bbox_inches"] = fixed_width_bbox(fig, style.get("fig_width", FIG_WIDTH))
    for ext in requested:
        dest = outdir / f"{stem}.{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"facecolor": "white", **kwargs}
        if ext == "png":
            save_kwargs["dpi"] = PNG_DPI
        fig.savefig(dest, format=ext, **save_kwargs)
        saved.append(dest)
    fig._ukb_export_paths = list(dict.fromkeys([*getattr(fig, "_ukb_export_paths", []), *saved]))
    return saved


def save_figure(fig, name, style=None, **kwargs):
    """Save a figure and return the `(figure, paths)` contract used by notebooks."""
    return fig, savefig(fig, name, style=style, **kwargs)


def save_figure_file(fig, path, **kwargs):
    """Legacy explicit-path export with metadata for the unified notebook display."""
    from .shared_figure_captions import caption_for_name
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig)
    if kwargs.get("bbox_inches") == "tight" or "bbox_inches" not in kwargs:
        kwargs["bbox_inches"] = fixed_width_bbox(fig)
    if path.suffix.lower() == ".png":
        kwargs["dpi"] = PNG_DPI
    fig.savefig(path, **kwargs)
    fig._ukb_export_paths = list(dict.fromkeys([*getattr(fig, "_ukb_export_paths", []), path]))
    if not getattr(fig, "_ukb_caption", None):
        fig._ukb_caption = caption_for_name(path)
    return path


def display_figure(fig, paths=(), caption=None, *, width=1100):
    """Show a figure, repository-relative export paths, and an inline caption."""
    from IPython.display import Image, Markdown, display
    from .shared_paths import raw_path
    from .shared_figure_captions import caption_for_name, suggest_caption

    if isinstance(fig, (str, Path)):
        caption = caption or caption_for_name(fig) or "The plotted quantities and groups are identified by the figure labels and legend."
        display(Image(filename=str(fig), width=width))
    else:
        paths = paths or getattr(fig, "_ukb_export_paths", [])
        caption = caption or suggest_caption(fig)
        finalize_figure(fig)
        display(fig)
    for path in paths:
        print("Saved:", raw_path(path))
    display(Markdown("**Suggested caption:** " + caption))


def show_figures():
    """Display open figures with paths and suggested captions, then close them."""
    for number in plt.get_fignums():
        fig = plt.figure(number)
        display_figure(fig)
        plt.close(fig)


def render_figure(plotter, *args, registry=None, show=True, caption=None, **kwargs):
    """Build, register, and optionally display a saved figure in one call."""
    fig, paths = plotter(*args, **kwargs)
    if registry is not None:
        registry.record_figures(paths)
    if show:
        display_figure(fig, paths, caption)
        plt.close(fig)
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


#: The only keys a `style.notebooks.<stem>` section may set, besides `*_colors` and `c_*`.
NOTEBOOK_KEYS = {"colors", "roles", "save", "formats", "savedir"}


def load_style(notebook, path=None, activate=True) -> dict:
    """Build one notebook's STYLE from universal_settings.yml.

    `style.base` merged with `style.notebooks.<notebook>`; integer values under `roles`
    and any `*_colors` mapping are resolved against the palette. With `activate` (the
    default) the result is registered via use_style and pushed into rcParams, so
    `savefig(fig, name)` and `extended_palette(n)` need no further argument.

    A notebook section may set only its output location, its palette and its semantic
    colours (`NOTEBOOK_KEYS`, `*_colors`, `c_*`). Typography, strokes and figure sizes
    live in `base` alone (D55); a section that overrides one is a crash, not a drift.
    """
    cfg = (load_settings(path).get("style") or {})
    known = cfg.get("notebooks") or {}
    if notebook not in known:
        raise KeyError(f"universal_settings.yml has no style.notebooks.{notebook} "
                       f"(known: {sorted(known)})")
    section = known[notebook] or {}
    illegal = sorted(k for k in section if k not in NOTEBOOK_KEYS
                     and not k.endswith("_colors") and not k.startswith("c_"))
    if illegal:
        raise KeyError(f"style.notebooks.{notebook} overrides base-only key(s) {illegal}; "
                       "typography, strokes and figure sizes are set once in style.base (D55)")
    style = {**(cfg.get("base") or {}), **section}

    colors = list(style.get("colors") or [])
    style["colors"] = colors
    for key in [k for k in style if k.endswith("_colors")]:
        style[key] = _resolve_refs(style[key], colors)
    style.update(_resolve_refs(style.pop("roles", None), colors))

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
