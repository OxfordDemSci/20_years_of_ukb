"""Publication-ready content figures for the combined content analysis notebook.

The main figure places FOR Level 4 composition (A) and RCDC composition (B) above
BERTopic thematic waves (C). Supplements show annual category shares, rank flows,
vocabulary coverage and breadth, and topic-model robustness. Every panel uses the same
dated corpus, 2013–2025 inclusive. Topic results are read from verified, completed
caches only; this module never imports or fits a topic model.

FOR composition counts paper–field assignments; RCDC divides each classified paper
fractionally across its tags. Topics count assigned papers. Missing classifications
are missing composition, not zero-percent observations. Their coverage is shown
explicitly in the second supplement. Figure exports use shared Helvetica typography,
the project palette, and PNG (500 dpi) / PDF output only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import shared_for as F
from utils import shared_paths as P
from utils import shared_rcdc as RCDC
from utils.shared_showcase import item_names, load_showcase
from utils.shared_analysis_window import ANALYSIS_START_YEAR, ANALYSIS_END_YEAR, filter_analysis_window
from utils.data_analysis_02_content_window import require_topic_window_provenance

# =============================================================================
# 1. The window and the weights
# =============================================================================
#: The shared inclusive composition window. Early years contain very few papers, so
#: the panels print their populations rather than leaving that to be remembered.
FLOW_MIN, FLOW_MAX = ANALYSIS_START_YEAR, ANALYSIS_END_YEAR
FLOW_YEARS = list(range(FLOW_MIN, FLOW_MAX + 1))

#: FOR shares use distinct paper–field assignments. RCDC gives each classified paper
#: total credit one across its tags. Topic shares count assigned papers once each.
WEIGHTS = {"topics": "n_papers", "for": "n_papers", "rcdc": "n_frac"}

#: RCDC mixes conditions with research areas, methods and populations. "all" is the whole
#: vocabulary; `shared_rcdc.is_disease` splits it (D11, D24).
RCDC_VIEW = "all"

OTHER_LABEL = "All other categories"

# =============================================================================
# 2. Reproducible, bounded band selection
# =============================================================================
# Eight leading fields/tags fit the half-width panels. Topic waves show up to twelve
# leading topics; a small mean-share floor avoids invisible bands. All categories and
# the exact coverage of each selection remain available in the exported tables.
BAND_RULES = {
    "topics": {"target_other": 45.0, "min_band": 0.8, "max_bands": 12, "min_bands": 8},
    "for":    {"target_other": 10.0, "min_band": 0.7, "max_bands": 8, "min_bands": 8},
    "rcdc":   {"target_other": 10.0, "min_band": 1.2, "max_bands": 8, "min_bands": 8},
}

# Presentation only: descriptors checked against this model's keywords and paper
# titles. Exact full-label keys prevent a new model's topic IDs inheriting names.
# Aggregation always retains the original model label.
TOPIC_DISPLAY_LABELS = {
    "T0: cognitive / ad / dementia / alzheimers": "Cognition, dementia and brain imaging",
    "T1: mdd / anxiety / depressive / psychiatric": "Depression and mental health",
    "T2: sleep / sleep duration / insomnia / daytime": "Sleep and circadian rhythms",
    "T9: complex traits / heritability / finemapping / simulations": "Complex-trait genetics and methods",
    "T3: retinal / glaucoma / amd / myopia": "Vision and eye disease",
    "T6: dietary / meat / food / fish": "Diet and nutrition",
    "T4: air / pollution / air pollution / pm": "Air pollution and environmental exposure",
    "T5: asthma / copd / lung function / ipf": "Lung function and respiratory disease",
    "T10: prostate / prostate cancer / pca / breast": "Cancer risk and genetic susceptibility",
    "T8: covid19 / infection / sarscov2 / covid19 infection": "COVID-19 and infection",
    "T7: masld / nafld / liver disease / fatty liver": "Metabolic liver disease",
    "T11: mvpa / pa / sedentary / intensity": "Physical activity and sedentary behaviour",
}


def category_weights(long, cat_col, weight, year_col="year", id_col="id"):
    """year x category weight matrix. Whole for `n_papers`, spread for `n_frac`.

    Preserves the counting conventions of the archived category-flow notebook.
    """
    pairs = long.drop_duplicates([id_col, cat_col]).copy()
    if weight == "n_papers":
        pairs["w"] = 1.0
    elif weight == "n_frac":
        pairs["w"] = 1.0 / pairs.groupby(id_col)[cat_col].transform("size")
    elif weight in pairs.columns:
        pairs["w"] = pd.to_numeric(pairs[weight], errors="coerce").fillna(0.0)
    else:
        raise KeyError(f"unknown weight {weight!r}; column not in the pair table")
    return pairs.pivot_table(index=year_col, columns=cat_col, values="w",
                             aggfunc="sum", fill_value=0.0)


def flow_shares(long, cat_col, weight, years=None, **kw):
    """(weight matrix, share matrix, per-year paper count); shares sum to 100 per year."""
    years = FLOW_YEARS if years is None else years
    w = category_weights(long, cat_col, weight, **kw).reindex(years).fillna(0.0)
    w.index.name = "year"
    totals = w.sum(axis=1)
    # Missing classifications are not a zero-percent composition. Retain those years
    # as NaN so the plots show a gap and the coverage supplement reports zero coverage.
    share = w.div(totals.where(totals > 0), axis=0) * 100
    assert np.allclose(share.loc[totals > 0].sum(axis=1), 100)
    n_papers = (long[long["year"].isin(years)].drop_duplicates("id")
                .groupby("year").size().reindex(years).fillna(0).astype(int))
    return w, share, n_papers


def choose_bands(share, w, rule) -> list:
    """The categories that get their own band, by the rule in `BAND_RULES`.

    Ordered by total in-window weight, which is the order the stack is drawn in. Returns
    the kept labels; everything else pools into the closing remainder.
    """
    order = w.sum(axis=0).sort_values(ascending=False)
    means = share.mean(axis=0)
    lo = min(int(rule["min_bands"]), len(order))
    hi = min(int(rule["max_bands"]), len(order))

    n = hi
    for candidate in range(lo, hi + 1):
        covered = means[order.index[:candidate]].sum()
        if 100 - covered <= rule["target_other"]:
            n = candidate
            break

    keep = list(order.index[:n])
    # Trim from the END only: a hairline in the middle of the order would leave a gap in
    # the stack's size ordering, which is a worse read than one band under the floor.
    while len(keep) > lo and means[keep[-1]] < rule["min_band"]:
        keep.pop()
    return keep


def top_bands(share, w, rule, other_label=OTHER_LABEL):
    """(band matrix, kept labels, the full weight order). The bands close at 100%."""
    keep = choose_bands(share, w, rule)
    band = share[keep].copy()
    rest = share.drop(columns=keep)
    if len(rest.columns):
        band[other_label] = rest.sum(axis=1, min_count=1)
    present = share.notna().any(axis=1)
    assert np.allclose(band.loc[present].sum(axis=1), 100)
    return band, keep, w.sum(axis=0).sort_values(ascending=False)


def _flow_block(long, cat_col, key, n_total=None, *, view_note="") -> dict:
    """Everything one panel row needs: the matrices, the bands and the self-description."""
    weight = WEIGHTS[key]
    w, share, n_papers = flow_shares(long, cat_col, weight)
    # Growth tables use distinct paper counts even when composition is fractional.
    paper_counts = (w.copy() if weight == "n_papers" else
                    category_weights(long, cat_col, "n_papers").reindex(
                        index=w.index, columns=w.columns, fill_value=0).fillna(0))
    band, keep, order = top_bands(share, w, BAND_RULES[key])
    covered = band[keep].sum(axis=1, min_count=1)
    return {
        "weights": w,
        "paper_counts": paper_counts,
        "share": share,
        "band": band,
        "keep": keep,
        "order": order,
        "n_papers": n_papers,
        "n_categories": int(n_total if n_total is not None else share.shape[1]),
        "n_bands": len(keep),
        "n_other": int((n_total if n_total is not None else share.shape[1]) - len(keep)),
        "coverage_mean": float(covered.mean()),
        "coverage_min": float(covered.min()),
        "coverage_max": float(covered.max()),
        "weight": weight,
        "view_note": view_note,
        "available": True,
    }


def start_end_table(share, keep):
    """Start-year vs end-year share and rank for the drawn bands (§6/§7's own table)."""
    y0, y1 = FLOW_YEARS[0], FLOW_YEARS[-1]
    ranks = share.rank(axis=1, ascending=False, method="first")
    out = pd.DataFrame({
        f"share_{y0}_%": share.loc[y0, keep].round(2),
        f"share_{y1}_%": share.loc[y1, keep].round(2),
        f"rank_{y0}": ranks.loc[y0, keep].astype("Int64"),
        f"rank_{y1}": ranks.loc[y1, keep].astype("Int64"),
    })
    out["delta_pp"] = (out[f"share_{y1}_%"] - out[f"share_{y0}_%"]).round(2)
    out["rank_change"] = out[f"rank_{y0}"] - out[f"rank_{y1}"]      # +ve = moved up
    out.index.name = "category"
    return out.sort_values("delta_pp", ascending=False)


# =============================================================================
# 3. The sources
# =============================================================================
CORPUS_COLUMNS = ["id", "year", "date", "category_for_2020", "category_rcdc"]

#: Per-paper assignments carry all topics and retain the correct denominator when
#: selecting a subset for the waves. A selected-only matrix cannot recover that
#: denominator and is therefore not accepted as a completed topic result here.
#: Every source also requires its matching .analysis_window.json sidecar.
TOPIC_SOURCES = (
    P.OUTPUT / "bertopic" / "bertopic_document_topic_assignments.csv",
    P.OUTPUT / "bertopic" / "showcase_plus_id_topics.csv",
    P.OUTPUT / "bertopic" / "tables" / "bertopic_document_topic_assignments.csv",
    P.OUTPUT / "bertopic" / "tables" / "showcase_plus_id_topics.csv",
    P.TOPIC_ASSIGNMENTS,                                        # id, topics
    P.CONTENT / "bertopic_document_topic_assignments.csv",      # id, year, topic, topics
    P.ACADEMIC_IMPACT / "bertopic" / "tables" / "bertopic_document_topic_assignments.csv",
)

#: Missing outputs must not silently trigger a model fit inside a figure builder.
TOPIC_MISSING_NOTE = (
    "BERTopic assignments not found.\n\n"
    "Run 02_content.ipynb to train on publications\n"
    f"dated {FLOW_MIN}–{FLOW_MAX}, then use its output/bertopic/ tables with their\n"
    ".analysis_window.json sidecars. Legacy copied outputs require the\n"
    "same provenance; filtering their rows cannot correct a different training window."
)


def load_corpus() -> pd.DataFrame:
    """The dated analysis corpus, with numeric years and parsed category records."""
    df = load_showcase(columns=CORPUS_COLUMNS,
                       parse=["category_for_2020", "category_rcdc"])
    df = filter_analysis_window(df)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df


def build_for_pairs(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per (paper, L2 division, L4 field), through `shared_for` (D22)."""
    frame = filter_analysis_window(corpus)
    F.add_for_columns(frame, fold_parents=True)
    return F.for_long(frame, carry=("year",))


def build_rcdc_pairs(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per (paper, RCDC tag). RCDC is flat — no levels, so a plain explode."""
    corpus = filter_analysis_window(corpus)
    tags = corpus["category_rcdc"].apply(lambda cell: sorted(set(item_names(cell))))
    return (corpus[["id", "year"]]
            .assign(rcdc=tags)
            .explode("rcdc")
            .dropna(subset=["rcdc"])
            .reset_index(drop=True))


def _topic_label(raw: str) -> str:
    """Retain complete model labels (including IDs) until the presentation stage."""
    text = str(raw).strip()
    if text.replace(".0", "").isdigit():
        return f"Topic {int(float(text))}"
    # Keep all terms for aggregation; shorten only the displayed legend. Otherwise
    # different topics sharing their first three keywords would be silently merged.
    return text


def load_topic_year_matrix(corpus: pd.DataFrame, topic_results=None):
    """The BERTopic (paper, topic, year) table, or `None` when the run's output is absent.

    Returns a long frame with `id`, `year`, `topic_label` so the caller can use the same
    `flow_shares` / `top_bands` path as the other vocabularies. The third return value is
    reserved for compatibility with the older selected-matrix reader and is always None.
    """
    corpus = filter_analysis_window(corpus)
    sources = (Path(topic_results),) if topic_results is not None else TOPIC_SOURCES
    for path in sources:
        if not Path(path).exists():
            continue
        # Filtering exported rows cannot correct a different model training window.
        require_topic_window_provenance(path)
        frame = pd.read_csv(path)
        columns = set(frame.columns)

        id_column = next((c for c in ("id", "showcase_plus_id") if c in columns), None)
        label_column = next((c for c in ("topics", "topic_label", "topic") if c in columns), None)
        if id_column is not None and label_column is not None:
            long = frame[[id_column, label_column]].rename(
                columns={id_column: "id", label_column: "topics"}).copy()
            if long["id"].isna().any() or long["id"].duplicated().any():
                raise ValueError(f"Topic results contain missing or duplicate publication IDs: {path}")
            if long["topics"].isna().any():
                raise ValueError(f"Topic results contain missing assignments: {path}")
            # The canonical publication dates control plotting, even where a copied
            # output carries stale year metadata. Training provenance is checked above.
            long["id"] = long["id"].astype(str)
            dated = corpus[["id", "year"]].copy()
            dated["id"] = dated["id"].astype(str)
            long = long.merge(dated, on="id", how="inner",
                              validate="one_to_one")
            long = long[~long["topics"].astype(str).str.strip().str.lower().isin(
                ("outlier", "-1", "-1.0"))]
            if long.empty:
                raise ValueError(f"Topic results have no assigned papers in the analysis corpus: {path}")
            long["topic_label"] = long["topics"].map(_topic_label)
            long["year"] = pd.to_numeric(long["year"], errors="coerce")
            long = long.dropna(subset=["year"])
            long["year"] = long["year"].astype(int)
            return long[["id", "year", "topic_label"]], Path(path), None

        raise ValueError(f"Unrecognised per-paper topic-results columns in {path}: {sorted(columns)}")

    return None, None, None


def _topics_block(corpus: pd.DataFrame, topic_results=None) -> dict:
    """The topic row's block, or a `available=False` stub carrying why it is empty."""
    loaded, source, _ = load_topic_year_matrix(corpus, topic_results=topic_results)
    if loaded is None:
        return {"available": False, "note": TOPIC_MISSING_NOTE, "source": None,
                "searched": [P.raw_path(p) for p in TOPIC_SOURCES]}

    block = _flow_block(loaded, "topic_label", "topics")
    block["from_assignments"] = True
    block["source"] = P.raw_path(source)
    return block


# =============================================================================
# 4. The one build
# =============================================================================
def build_panel_data(verbose: bool = True, *, topic_results=None) -> dict:
    """Every aggregate the panels draw, computed once from the corpus (~25 s).

    Everything the figures read comes out of this dict, so two panels on one page cannot
    be drawn on different vintages of the data.
    """
    corpus = load_corpus()
    for_pairs = build_for_pairs(corpus)
    rcdc_pairs = build_rcdc_pairs(corpus)

    if RCDC_VIEW == "all":
        rcdc_view, view_note = rcdc_pairs, f"all {rcdc_pairs.rcdc.nunique()} tags"
    else:
        want = RCDC_VIEW == "disease"
        rcdc_view = rcdc_pairs[rcdc_pairs["rcdc"].map(RCDC.is_disease) == want]
        view_note = (f"{rcdc_view.rcdc.nunique()} {RCDC_VIEW} tags of "
                     f"{rcdc_pairs.rcdc.nunique()}")

    D = {
        "years": FLOW_YEARS,
        "corpus": corpus,
        "topics": _topics_block(corpus, topic_results=topic_results),
        "for": _flow_block(for_pairs, "l4_label", "for"),
        "rcdc": _flow_block(rcdc_view, "rcdc", "rcdc", view_note=view_note),
        "counts": {
            "papers": len(corpus),
            "for_pairs": len(for_pairs),
            "for_papers": int(for_pairs.id.nunique()),
            "for_fields": int(for_pairs.l4_label.nunique()),
            "rcdc_pairs": len(rcdc_pairs),
            "rcdc_papers": int(rcdc_pairs.id.nunique()),
            "rcdc_tags": int(rcdc_pairs.rcdc.nunique()),
        },
    }

    if verbose:
        counts = D["counts"]
        print(f"corpus  : {counts['papers']:,} papers, "
              f"{int(corpus.year.min())}-{int(corpus.year.max())}")
        print(f"FOR L4  : {counts['for_pairs']:,} pairs | {counts['for_papers']:,} papers "
              f"| {counts['for_fields']} fields "
              f"({counts['for_pairs'] / counts['for_papers']:.2f} codes/paper)")
        print(f"RCDC    : {counts['rcdc_pairs']:,} pairs | {counts['rcdc_papers']:,} papers "
              f"| {counts['rcdc_tags']} tags "
              f"({counts['rcdc_pairs'] / counts['rcdc_papers']:.1f} tags/paper)")
        for key, title in (("topics", "topics"), ("for", "FOR L4"), ("rcdc", "RCDC  ")):
            block = D[key]
            if not block["available"]:
                print(f"{title:<8}: NOT AVAILABLE — all {len(block['searched'])} topic sources absent")
                continue
            print(f"{title:<8}: {block['n_bands']} bands of {block['n_categories']} "
                  f"hold {block['coverage_min']:.1f}-{block['coverage_max']:.1f}% "
                  f"of each year (mean {block['coverage_mean']:.1f}%, "
                  f"remainder {100 - block['coverage_mean']:.1f}% over "
                  f"{block['n_other']} categories)")
    return D


def band_summary(D) -> pd.DataFrame:
    """One row per vocabulary: the rule it was banded by and the coverage that produced.

    Written beside the figures so the threshold is a recorded number rather than something
    to be read back off the picture.
    """
    rows = []
    for key, name in (("topics", "BERTopic topics"), ("for", "FOR 2020 Level 4"),
                      ("rcdc", "RCDC tags")):
        block = D[key]
        rule = BAND_RULES[key]
        if not block["available"]:
            rows.append({"vocabulary": name, "available": False, **rule})
            continue
        rows.append({
            "vocabulary": name, "available": True,
            "weight": block["weight"],
            "categories": block["n_categories"],
            "bands_drawn": block["n_bands"],
            "categories_in_remainder": block["n_other"],
            "coverage_mean_pct": round(block["coverage_mean"], 1),
            "coverage_min_pct": round(block["coverage_min"], 1),
            "coverage_max_pct": round(block["coverage_max"], 1),
            "remainder_mean_pct": round(100 - block["coverage_mean"], 1),
            **rule,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 5. Drawing and publication pages
# =============================================================================
from collections import Counter
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

from utils.shared_style import (
    PALETTE, blue_cream_red_colormap, extended_palette, grid_on,
    palette, savefig, set_title,
)

OTHER_COLOR = "#DCDCDC"
VOCABULARY_COLORS = {"for": palette("navy"), "rcdc": palette("red"),
                     "topics": palette("steel_blue")}
VOCABULARY_LABELS = {"for": "FOR Level 4", "rcdc": "RCDC", "topics": "BERTopic"}


def _style():
    from utils import shared_style
    return shared_style._resolve(None)


def _colors(count):
    """Categorical colours anchored in the shared project palette."""
    return extended_palette(count, style={"colors": PALETTE})


def _axes(ax, *, grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_visible(True)
    ax.minorticks_off()
    ax.grid(False, which="both")
    if grid:
        grid_on(ax, axis="both", which="major", linestyle="--", alpha=0.35)
    return ax


def _years(ax, *, every=3):
    ticks = sorted(set([FLOW_MIN, *range(FLOW_MIN + every, FLOW_MAX + 1, every), FLOW_MAX]))
    ax.set_xticks(ticks)
    ax.set_xlim(FLOW_MIN, FLOW_MAX)
    ax.set_xlabel("Publication year")


def _heading(ax, letter, title):
    set_title(ax, f"{letter}  {title}", fontsize=_style()["title_fs"], pad=20)


def _short_label(text, limit=46):
    text = str(text)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _legend_label(text, width=29):
    # Keep the full category names in the exported tables; long keyword lists are
    # abbreviated here rather than allowing legends to change the page dimensions.
    text = str(text)
    if ":" in text and text.split(":", 1)[0].strip().upper().startswith("T"):
        text = text.split(":", 1)[1].strip()
    return fill(_short_label(text, width * 2 - 2), width=width)


def _topic_legend_labels(labels, width=36):
    """Use reviewed descriptors or keywords while preserving distinct model topics."""
    shown = {label: fill(TOPIC_DISPLAY_LABELS[label], width=width)
             if label in TOPIC_DISPLAY_LABELS else _legend_label(label, width)
             for label in labels}
    duplicates = Counter(shown.values())
    for label, display in shown.items():
        if duplicates[display] > 1:
            prefix = str(label).split(":", 1)[0].strip()
            if ":" in str(label) and prefix.upper().startswith("T"):
                shown[label] = f"{display} [{prefix}]"
            else:
                # Imported labels without model IDs retain their unabridged text.
                shown[label] = fill(str(label), width=width)
    return shown


def _missing_panel(ax, title, note):
    """A clearly incomplete preview; the notebook reports the absent source."""
    _axes(ax, grid=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#FAFAFA")
    set_title(ax, title, fontsize=_style()["title_fs"], pad=20)
    ax.text(.5, .56, "Topic results unavailable", transform=ax.transAxes,
            ha="center", va="center", fontsize=_style()["label_fs"], fontweight="bold")
    ax.text(.5, .40, "No verified completed topic assignments were found.\n"
            "This panel will populate when the saved results are available.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=_style()["annot_fs"], linespacing=1.5, color="#555555")
    return ax


def draw_share_stream(ax, block, title, *, label_min=None, callout_gap=None,
                      show_year_n=False, xlabel="Publication year"):
    """A compact 100% stream with a bounded legend, suitable for a half-page panel."""
    band = block["band"]
    named = [label for label in band if label != OTHER_LABEL]
    colors = _colors(len(named))
    if OTHER_LABEL in band:
        colors.append(OTHER_COLOR)
    _axes(ax, grid=False)
    ax.stackplot(band.index, band.T.to_numpy(dtype=float), colors=colors,
                 edgecolor="white", linewidth=.35)
    _years(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_ylabel("Share of field assignments" if block["weight"] == "n_papers"
                  else "Share of fractional paper credit")
    handles = []
    for color, label in zip(colors, band.columns):
        name = f"Other ({block['n_other']} categories)" if label == OTHER_LABEL else label
        handles.append(Patch(facecolor=color, edgecolor=palette("navy"), linewidth=.4,
                             label=_legend_label(name)))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-.015, -.20),
              ncol=2, frameon=False, borderaxespad=0, handlelength=1.25,
              columnspacing=1.4, labelspacing=.40, fontsize=_style()["legend_fs"])
    set_title(ax, title, fontsize=_style()["title_fs"], pad=20)
    return ax


def draw_for_stream(ax, D):
    return draw_share_stream(ax, D["for"], "Fields of Research, Level 4")


def draw_rcdc_stream(ax, D):
    return draw_share_stream(ax, D["rcdc"], "RCDC categories")


def draw_topic_stream(ax, D):
    """Selected-topic widths are annual shares of all assigned papers, not renormalised."""
    block = D["topics"]
    if not block["available"]:
        return _missing_panel(ax, "Thematic waves", block["note"])
    keep = block["keep"]
    selected = block["share"][keep]
    colors = _colors(len(keep))
    _axes(ax, grid=False)
    # A centred streamgraph retains actual share widths. The vertical displacement
    # conveys no additional quantity; omitting the numerical y scale makes that clear.
    values = selected.T.to_numpy(dtype=float)
    ax.stackplot(selected.index, values, baseline="sym", colors=colors,
                 edgecolor="white", linewidth=.35)
    _years(ax)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Relative topic prominence")
    grid_on(ax, axis="x", which="major", linestyle="--", alpha=.3)
    legend_labels = _topic_legend_labels(block["share"].columns)
    handles = [Patch(facecolor=color, edgecolor=palette("navy"), linewidth=.4,
                     label=legend_labels[label])
               for color, label in zip(colors, keep)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.025, .5),
              frameon=False, borderaxespad=0, handlelength=1.3,
              labelspacing=.5, fontsize=_style()["legend_fs"])
    set_title(ax, "Thematic waves", fontsize=_style()["title_fs"], pad=20)
    coverage = selected.sum(axis=1, min_count=1)
    noun = "assigned papers" if block.get("from_assignments") else "cached topic selection"
    ax.text(0, 1.01, f"{len(keep)} topics; {coverage.mean():.1f}% of {noun} per year on average",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=_style()["annot_fs"], color="#555555")
    return ax


def thin_years_note(D):
    counts = D["corpus"].groupby("year").size().reindex(FLOW_YEARS, fill_value=0)
    return (f"Publication window: {FLOW_MIN}–{FLOW_MAX}; n = {len(D['corpus']):,}. "
            f"Early-year estimates use {int(counts.iloc[0]):,} papers in {FLOW_MIN} "
            f"and {int(counts.iloc[1]):,} in {FLOW_MIN + 1}. "
            "Missing classifications are excluded from composition denominators.")


def figure_main(D, save=True):
    """FOR and RCDC side by side, with thematic waves across the lower row."""
    figsize = _style().get("figsize_main", (16, 12))
    fig = plt.figure(figsize=figsize)
    ax_for = fig.add_axes([.075, .64, .39, .29])
    ax_rcdc = fig.add_axes([.575, .64, .39, .29])
    ax_topic = fig.add_axes([.075, .095, .65, .31])
    draw_for_stream(ax_for, D)
    draw_rcdc_stream(ax_rcdc, D)
    draw_topic_stream(ax_topic, D)
    for ax, letter, title in ((ax_for, "A", "Fields of Research, Level 4"),
                              (ax_rcdc, "B", "RCDC categories")):
        _heading(ax, letter, title)
    # An explicit title position survives shared export finalisation and reserves
    # a separate line for the coverage annotation above the streamgraph.
    set_title(ax_topic, "C  Thematic waves", fontsize=_style()["title_fs"], y=1.075)
    fig.text(.075, .018, thin_years_note(D), ha="left", va="bottom",
             fontsize=_style()["annot_fs"], color="#555555")
    if save:
        savefig(fig, P.MAIN_FIGURE_STEMS[4], formats=("pdf", "png"))
    return fig


def annual_vocabulary_metrics(D):
    """Annual denominators, assignment density and exp(Shannon) diversity for export."""
    paper_n = D["corpus"].groupby("year").size().reindex(FLOW_YEARS, fill_value=0)
    rows = []
    for key in ("for", "rcdc", "topics"):
        block = D[key]
        if not block["available"]:
            continue
        probabilities = block["share"] / 100.0
        entropy = -(probabilities * np.log(probabilities.where(probabilities > 0))).sum(
            axis=1, min_count=1)
        present = block["weights"].sum(axis=1) > 0
        effective = np.exp(entropy).where(present)
        cumulative = (block["weights"].cumsum(axis=0) > 0).sum(axis=1)
        for year in FLOW_YEARS:
            n = block["n_papers"].get(year, 0)
            paper_denominator_known = key != "topics" or block.get("from_assignments", False)
            rows.append({
                "year": year, "vocabulary": key, "corpus_papers": int(paper_n.loc[year]),
                "classified_papers": int(n) if paper_denominator_known else np.nan,
                "coverage_pct": 100 * n / paper_n.loc[year]
                    if paper_denominator_known and paper_n.loc[year] else np.nan,
                "effective_categories": effective.loc[year],
                "active_categories": int((block["weights"].loc[year] > 0).sum()),
                "cumulative_categories": int(cumulative.loc[year]),
            })
    return pd.DataFrame(rows)


def category_share_table(D):
    """Full annual distributions underlying the main page and category heatmaps."""
    frames = []
    for key in ("for", "rcdc", "topics"):
        block = D[key]
        if not block["available"]:
            continue
        share = block["share"].rename_axis(columns="category").stack(
            future_stack=True).rename("share_pct")
        weight = block["weights"].rename_axis(columns="category").stack(
            future_stack=True).rename("weight")
        frames.append(pd.concat([share, weight], axis=1).reset_index().assign(
            vocabulary=key, counting=block["weight"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def category_change_table(D):
    """Start/end shares for every category, preserving missing starting-year data."""
    frames = []
    for key in ("for", "rcdc", "topics"):
        block = D[key]
        if block["available"]:
            frames.append(start_end_table(block["share"], list(block["share"].columns))
                          .reset_index().assign(vocabulary=key))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def topic_label_table(D):
    """Model labels, displayed abbreviations, counts and main-panel selection."""
    block = D["topics"]
    columns = ["topic_label", "display_label", "label_source", "papers", "selected_main_figure"]
    if not block["available"]:
        return pd.DataFrame(columns=columns)
    legend_labels = _topic_legend_labels(block["order"].index)
    return pd.DataFrame([
        {"topic_label": label, "display_label": legend_labels[label].replace("\n", " "),
         "label_source": "Reviewed descriptor" if label in TOPIC_DISPLAY_LABELS else "Model keywords",
         "papers": count, "selected_main_figure": label in block["keep"]}
        for label, count in block["order"].items()
    ], columns=columns)


def draw_category_heatmap(ax, block, title, *, n_categories=12):
    """Leading categories across all years, on an explicit shared percentage scale."""
    keep = list(block["order"].index[:n_categories])
    values = block["share"][keep].T
    cmap = blue_cream_red_colormap().with_extremes(bad="#EEEEEE")
    vmax = max(1.0, float(np.nanmax(values.to_numpy())))
    norm = Normalize(vmin=0, vmax=vmax)
    mesh = ax.pcolormesh(np.ma.masked_invalid(values.to_numpy(dtype=float)),
                        cmap=cmap, norm=norm, edgecolors=palette("navy"), linewidth=.45)
    _axes(ax, grid=False)
    ax.set_xlim(0, len(values.columns))
    ax.set_ylim(len(keep), 0)
    ax.set_xticks(np.arange(len(values.columns)) + .5, labels=values.columns,
                  rotation=45, ha="right")
    ax.set_yticks(np.arange(len(keep)) + .5,
                  labels=[_legend_label(label, 34) for label in keep])
    ax.set_xlabel("Publication year")
    ax.tick_params(axis="y", length=0, labelsize=_style()["tick_fs"])
    for row in range(len(keep)):
        for col in range(len(values.columns)):
            value = values.iloc[row, col]
            if not np.isfinite(value):
                continue
            red, green, blue, _ = cmap(norm(value))
            text_color = "white" if .2126 * red + .7152 * green + .0722 * blue < .52 else "#222222"
            ax.text(col + .5, row + .5, f"{value:.1f}", ha="center", va="center",
                    fontsize=max(7.0, _style()["annot_fs"] - .5), color=text_color)
    cax = ax.inset_axes([1.025, .07, .028, .86])
    cb = ax.figure.colorbar(mesh, cax=cax)
    cb.set_label("Annual share (%)")
    cb.set_ticks(np.linspace(0, vmax, 5))
    cb.ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, vmax, 5)])
    cb.outline.set_edgecolor(palette("navy"))
    set_title(ax, title, fontsize=_style()["title_fs"], pad=18)
    return ax


def figure_si_category_detail(D, save=True):
    """Supplement 1: the leading twelve fields and tags without stacked-band occlusion."""
    figsize = _style().get("figsize_si", (15, 10))
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.subplots_adjust(left=.27, right=.88, top=.95, bottom=.115, hspace=.48)
    draw_category_heatmap(axes[0], D["for"], "A  Fields of Research, Level 4")
    draw_category_heatmap(axes[1], D["rcdc"], "B  RCDC categories")
    fig.text(.27, .018, "Numbers give percentages. Grey cells indicate no classified publications. "
             "The complete category distributions are supplied as CSV.",
             fontsize=_style()["annot_fs"], color="#555555")
    if save:
        savefig(fig, "02_02_supplementary_figure_01_category_composition", formats=("pdf", "png"))
    return fig


def _metric_lines(ax, metrics, column, ylabel, *, keys=("for", "rcdc", "topics")):
    _axes(ax)
    for key in keys:
        frame = metrics.loc[metrics["vocabulary"].eq(key)]
        if frame.empty or not frame[column].notna().any():
            continue
        ax.plot(frame["year"], frame[column], label=VOCABULARY_LABELS[key],
                color=VOCABULARY_COLORS[key], marker="o", linewidth=1.8,
                markersize=4.5, markeredgecolor=palette("navy"), markeredgewidth=.5)
    _years(ax)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    return ax


def figure_si_coverage(D, save=True):
    """Supplement 2: coverage, diversity, active categories and cumulative breadth."""
    metrics = annual_vocabulary_metrics(D)
    fig, axes = plt.subplots(2, 2, figsize=_style().get("figsize_si", (15, 10)))
    fig.subplots_adjust(left=.085, right=.975, top=.94, bottom=.11, hspace=.40, wspace=.32)
    panels = (
        ("coverage_pct", "Publications classified (%)", "A  Classification coverage"),
        ("effective_categories", "Effective number of categories", "B  Category diversity"),
        ("active_categories", "Categories represented", "C  Annual category breadth"),
        ("cumulative_categories", "Cumulative categories represented", "D  Cumulative category breadth"),
    )
    for ax, (metric, label, title) in zip(axes.flat, panels):
        _metric_lines(ax, metrics, metric, label)
        set_title(ax, title, fontsize=_style()["title_fs"], pad=16)
        if metric == "coverage_pct":
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        else:
            ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .01),
               ncol=len(handles), frameon=False, fontsize=_style()["legend_fs"])
    if save:
        savefig(fig, "02_03_supplementary_figure_02_coverage_and_breadth", formats=("pdf", "png"))
    return fig


def draw_rank_flow(ax, block, title):
    """Draw annual ranks for the same leading categories used in the main figure."""
    share, keep = block["share"], list(block["keep"])
    ranks = share[keep].rank(axis=1, ascending=False, method="first")
    colors = dict(zip(keep, _colors(len(keep))))
    mean_share = share[keep].mean()
    scale = max(float(mean_share.max()), 1e-9)
    outside = ax.get_yaxis_transform()

    _axes(ax, grid=False)
    for category in keep:
        values = ranks[category].dropna()
        if values.empty:
            continue
        color = colors[category]
        width = 1.2 + 2.2 * float(mean_share[category]) / scale
        marker_size = max(4.0, _style()["marker_size"] * 0.55)
        ax.plot(
            values.index,
            values,
            color=color,
            linewidth=width,
            marker="o",
            markersize=marker_size,
            markeredgecolor="black",
            markeredgewidth=0.45,
            solid_capstyle="round",
            zorder=3,
        )
        label = _short_label(category, 30)
        first_rank, last_rank = int(values.iloc[0]), int(values.iloc[-1])
        ax.text(
            -0.015,
            first_rank,
            f"#{first_rank}  {label}",
            transform=outside,
            ha="right",
            va="center",
            fontsize=_style()["annot_fs"] - 0.5,
            color=color,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            1.015,
            last_rank,
            f"{label}  #{last_rank}",
            transform=outside,
            ha="left",
            va="center",
            fontsize=_style()["annot_fs"] - 0.5,
            color=color,
            fontweight="bold",
            clip_on=False,
        )

    ax.set_ylim(len(keep) + 0.55, 0.45)
    ax.set_yticks(range(1, len(keep) + 1), labels=[])
    ax.tick_params(axis="y", length=0)
    _years(ax)
    grid_on(ax, axis="y", which="major", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    set_title(ax, title, fontsize=_style()["title_fs"], pad=16)
    return ax


def figure_si_rank_flow(D, save=True):
    """Supplement: rank trajectories for leading FOR and RCDC categories."""
    fig, axes = plt.subplots(2, 1, figsize=_style().get("figsize_si", (15, 10)))
    fig.subplots_adjust(left=.24, right=.76, top=.95, bottom=.08, hspace=.40)
    draw_rank_flow(
        axes[0], D["for"],
        f"A  FOR Level 4 fields, rank by annual share ({FLOW_MIN}–{FLOW_MAX})",
    )
    draw_rank_flow(
        axes[1], D["rcdc"],
        f"B  RCDC categories, rank by annual share ({FLOW_MIN}–{FLOW_MAX})",
    )
    fig.text(
        .24,
        .018,
        "Lines cover the leading categories selected for the main composition figure; "
        "line width is proportional to mean annual share.",
        fontsize=_style()["annot_fs"],
        color="#555555",
    )
    if save:
        savefig(
            fig,
            "02_02_supplementary_figure_01_category_rank_flow",
            formats=("pdf", "png"),
        )
    return fig


def load_topic_diagnostics(D):
    """Optional diagnostics from the same directory as the completed topic result.

    Historic, unscoped diagnostics are not evidence for the current model. Partial
    diagnostic sets are omitted; invalid sidecars raise rather than being ignored.
    """
    if not D["topics"]["available"] or not D["topics"].get("source"):
        return None
    source = Path(D["topics"]["source"])
    if not source.is_absolute():
        source = P.ROOT / source
    files = {
        "runs": "seed_grid_runs.csv",
        "summary": "bertopic_seed_robustness_summary.csv",
        "pairs": "bertopic_seed_pair_stability.csv",
        "persistence": "bertopic_final_cluster_persistence.csv",
    }
    paths = {key: source.parent / name for key, name in files.items()}
    if not all(path.is_file() and path.with_suffix(".analysis_window.json").is_file()
               for path in paths.values()):
        return None
    for path in paths.values():
        require_topic_window_provenance(path)
    return {key: pd.read_csv(path) for key, path in paths.items()}


def figure_si_topic_robustness(D, save=True):
    """Supplement 3, when verified diagnostics exist: seeds, stability and persistence."""
    diagnostics = D.get("topic_diagnostics") or load_topic_diagnostics(D)
    if diagnostics is None:
        return None
    runs = diagnostics["runs"].query('status == "ok"').copy()
    summary, pairs = diagnostics["summary"], diagnostics["pairs"]
    persistence = diagnostics["persistence"]
    parameters = sorted(runs["parameter_index"].unique())
    positions = {parameter: i for i, parameter in enumerate(parameters)}
    # The summary is ordered by the exact configured selection score by the model
    # pipeline. Its first row identifies the selected parameter setting.
    selected = int(summary.iloc[0]["parameter_index"])
    labels = []
    for parameter in parameters:
        row = runs.loc[runs["parameter_index"].eq(parameter)].iloc[0]
        labels.append(f"{int(row.n_neighbors)}/{int(row.min_cluster_size)}/{int(row.min_samples)}")
    fig, axes = plt.subplots(2, 2, figsize=_style().get("figsize_si", (15, 10)))
    fig.subplots_adjust(left=.085, right=.975, top=.94, bottom=.115, hspace=.47, wspace=.33)

    def configuration_axis(ax):
        _axes(ax)
        ax.axvspan(positions[selected] - .38, positions[selected] + .38,
                   color=palette("cream"), alpha=.6, zorder=0)
        ax.set_xticks(range(len(parameters)), labels=labels)
        ax.set_xlim(-.55, len(parameters) - .45)
        ax.set_xlabel("Neighbours / minimum cluster size / minimum samples")
        ax.tick_params(axis="x", labelsize=_style()["tick_fs"] - .5)

    for ax, metric, title, ylabel in (
        (axes[0, 0], "coherence_cv", "A  Topic coherence across seeds", r"Topic coherence ($c_v$)"),
        (axes[0, 1], "n_topics", "B  Topic count across seeds", "Number of topics"),
    ):
        configuration_axis(ax)
        for parameter, group in runs.groupby("parameter_index"):
            group = group.sort_values("seed")
            x = positions[parameter]
            offsets = np.linspace(-.18, .18, len(group)) if len(group) > 1 else [0]
            ax.scatter(x + np.asarray(offsets), group[metric], s=30,
                       color=palette("steel_blue"), edgecolor=palette("navy"),
                       linewidth=.5, alpha=.8, zorder=3)
            ax.plot([x - .26, x + .26], [group[metric].mean()] * 2,
                    color=palette("red"), linewidth=2.4, zorder=4)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        set_title(ax, title, fontsize=_style()["title_fs"], pad=16)

    ax = axes[1, 0]
    configuration_axis(ax)
    for metric, offset, color, label in (("ari", -.12, palette("red"), "Adjusted Rand index"),
                                         ("nmi", .12, palette("steel_blue"), "Normalised mutual information")):
        for parameter, group in pairs.groupby("parameter_index"):
            x = positions[parameter] + offset
            jitter = np.linspace(-.055, .055, len(group))
            ax.scatter(x + jitter, group[metric], s=15, color=color, alpha=.40,
                       edgecolor="none", zorder=2)
        means = pairs.groupby("parameter_index")[metric].mean().reindex(parameters)
        ax.plot(np.arange(len(parameters)) + offset, means, color=color, marker="o",
                markersize=4.5, linewidth=1.5, label=label, zorder=4)
    ax.set_ylabel("Agreement between seed pairs")
    ax.set_ylim(min(0, float(pairs[["ari", "nmi"]].min().min()) - .03), 1.03)
    ax.legend(loc="lower left", frameon=True, facecolor="white",
              edgecolor=palette("navy"), fontsize=_style()["legend_fs"] - 1)
    set_title(ax, "C  Assignment stability", fontsize=_style()["title_fs"], pad=16)

    ax = axes[1, 1]
    _axes(ax)
    ax.scatter(persistence["raw_cluster_size"], persistence["cluster_persistence"],
               s=42, color=palette("steel_blue"), alpha=.75,
               edgecolor=palette("navy"), linewidth=.6, zorder=3)
    median = float(persistence["cluster_persistence"].median())
    ax.axhline(median, color=palette("red"), linestyle="--", linewidth=1.4,
               label=f"Median persistence: {median:.2f}")
    ax.set_xscale("log")
    ax.minorticks_off()
    grid_on(ax, axis="both", which="major", log=True, alpha=.35)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("HDBSCAN cluster size before outlier reassignment")
    ax.set_ylabel("Cluster persistence")
    ax.legend(loc="upper right", frameon=False, fontsize=_style()["legend_fs"] - 1)
    set_title(ax, "D  Final-model cluster persistence", fontsize=_style()["title_fs"], pad=16)
    fig.text(.085, .018, "Points in A–B: individual seeds; red bars: means. "
             "Cream shading: selected parameter setting. C includes all seed pairs.",
             fontsize=_style()["annot_fs"], color="#555555")
    if save:
        savefig(fig, "02_04_supplementary_figure_03_topic_robustness", formats=("pdf", "png"))
    return fig


# Descriptive aliases used by the combined notebook/export orchestrator.
figure_si_category_changes = figure_si_category_detail
figure_si_breadth_coverage = figure_si_coverage
breadth_coverage_time_series = annual_vocabulary_metrics


MAIN_CAPTION = {
    "A": f"Annual composition of UK Biobank research by Fields of Research 2020 Level 4, "
         f"{FLOW_MIN}–{FLOW_MAX}. Papers count once in each assigned field; percentages "
         "use all paper–field assignments in each year as the denominator. The eight "
         "leading fields by total assignment count are shown separately; grey pools the remainder.",
    "B": "Annual Research, Condition and Disease Categorisation (RCDC) composition. "
         "Each classified paper contributes a total weight of one, "
         "divided equally across its distinct tags. Shares are fractions of classified papers, "
         "with the eight leading tags displayed separately and remaining tags pooled in grey.",
    "C": "Thematic waves from cached BERTopic assignments. Band thickness shows the annual shares of "
         "all topic-assigned publications attributable to the selected leading topics; the "
         "stream is centred for display, so vertical position carries no meaning. Topics "
         "are selected using total publication counts across the analysis window, with "
         "up to twelve shown. Descriptive labels summarise model keywords and example publications.",
}

SI_CAPTIONS = {
    "category_detail": {
        "A": "Annual shares of the twelve leading FOR Level 4 fields, ranked by total "
             "paper–field assignments across the analysis window. Each paper counts once "
             "per assigned field; yearly shares sum to 100% across the full vocabulary.",
        "B": "Annual shares of the twelve leading RCDC tags, using fractional assignment "
             "across each paper's tags. Cells are annotated as percentages; grey denotes "
             "years with no classified publications. Colour scales span each panel's observed range.",
    },
    "rank_flow": {
        "A": "Annual rank trajectories for the leading FOR Level 4 fields shown in "
             "the main composition figure. Line width is proportional to mean annual share.",
        "B": "Annual rank trajectories for the leading fractionally weighted RCDC "
             "categories shown in the main composition figure.",
    },
    "coverage": {
        "A": "Percentage of dated publications carrying at least one classification or "
             "a non-outlier topic assignment. Coverage uses the full yearly corpus as denominator.",
        "B": "Effective number of categories, calculated as the exponential of Shannon "
             "entropy from each vocabulary's annual composition shares.",
        "C": "Number of distinct categories represented among each year's publications.",
        "D": "Cumulative number of distinct categories observed. Vocabularies differ in "
             "granularity and are presented separately; their category counts are not equivalent units.",
    },
    "topic_robustness": {
        "A": "Topic coherence (c_v) for every successful random-seed fit within each "
             "parameter setting. Points show seeds and red bars their means.",
        "B": "Topic counts from the same fits. Cream shading in A–C identifies the "
             "selected parameter setting, chosen using the recorded quality and stability score.",
        "C": "Adjusted Rand index and normalised mutual information for every pair of "
             "seed-specific assignment vectors; solid lines show the corresponding means.",
        "D": "Cluster persistence against cluster size for the final selected HDBSCAN model. "
             "Sizes and persistence are matched using original HDBSCAN cluster IDs before "
             "outlier reassignment; the dashed line denotes median persistence.",
    },
}
