"""Assembled panels for analysis 02 — the content / composition figure family.

`02_content_99_all.ipynb` owns the selection (which chart is main-paper, which is SI);
this module owns the two halves that selection needs and that a notebook cell is the
wrong place for, following the shape analyses 03 and 04 already use (D28, D31):

    build_panel_data()      every aggregate the panels draw, computed once, returned
                            as a dict of small frames
    draw_<name>(ax, D)      one chart into one caller-supplied axes, no figure of its own
    figure_main(D)          the main-paper panel;  figure_si_<name>(D) the SI ones

**What the main panel is.** One page, three stacked rows, one grammar: a 100%-normalised
composition stream over 2014–2025, drawn once per vocabulary.

    A   BERTopic topics      what the papers are about, learned from the text
    B   FOR 2020 Level 4     what field the publisher's classifier assigns them to
    C   RCDC tags            what condition / research area the funder vocabulary names

Reading them stacked is the point: three independent vocabularies describing one corpus
over one window, so a shift that shows up in all three is a shift in the literature and
one that shows up in only one is a property of that vocabulary.

**Why re-drawn rather than pasted.** `02_content_1_bert_topic.ipynb` is a Colab notebook
that carries its own `STREAM_COLOURS` literal, its own rcParams block and a `wiggle`
baseline; `02_content_2_other_category_flow.ipynb` draws through `shared_style` with a
zero baseline and a closing remainder. Laid side by side those read as two figures
stapled together. Every function here draws into an axes belonging to a figure the caller
has already sized, under the one `02_content_99_all` style section.

**The band threshold is per vocabulary, and it is a rule, not a literal.** D23 fixed
twelve bands plus a remainder for both flows. That is the wrong constant for both: twelve
L4 fields already hold ~87.5% of a year, while twelve RCDC tags hold ~52.3%, because a
paper carries 1.5 L4 codes and 7.4 RCDC tags spread fractionally over 271 of them. So
`choose_bands` takes a coverage target and a legibility floor per vocabulary
(`BAND_RULES`) and the drawn count follows from the data. Where the target cannot be met
at any readable band count — RCDC, whose tail is irreducible — the panel says so on its
own face rather than leaving an unexplained grey slab: the remainder band is labelled with
how many categories are inside it, and each panel prints the coverage it achieved.

**Sources.** The FOR and RCDC arms re-derive from the corpus, exactly as
`02_content_2_other_category_flow.ipynb` does, through `shared_for.for_long` and a plain
explode of `category_rcdc` (D2, D22, D24). The topic arm reads what the BERTopic run
wrote; `load_topic_year_matrix` looks in `output/bertopic/` and the legacy
`data/analysis/content/` directory, and requires verified training-window provenance.
It returns `None`
rather than inventing topics when it is absent — the panel then draws the reason it is
empty, the way 04's RCDC macro-cluster panel does for its missing partition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import shared_for as F
from utils import shared_paths as P
from utils import shared_rcdc as RCDC
from utils.shared_showcase import item_names, load_showcase
from utils.shared_analysis_window import ANALYSIS_END_YEAR, filter_analysis_window
from utils.data_analysis_02_content_window import require_topic_window_provenance

# =============================================================================
# 1. The window and the weights
# =============================================================================
#: The composition window. 2014 is where the corpus starts; 2026 is a partial year and is
#: excluded from every panel here (D23). The first two years rest on 26 and 61 papers, so
#: the panels print their populations rather than leaving that to be remembered.
FLOW_MIN, FLOW_MAX = 2014, ANALYSIS_END_YEAR
FLOW_YEARS = list(range(FLOW_MIN, FLOW_MAX + 1))

#: Whole counting for FOR L4 (1.48 codes/paper, so it barely inflates) and fractional for
#: RCDC (7.4 tags/paper, where whole counting would let a paper tagged with twenty
#: conditions outvote one tagged with two). D24. BERTopic assigns exactly one topic per
#: paper after forced reassignment (D6), so the two coincide there.
WEIGHTS = {"topics": "n_papers", "for": "n_papers", "rcdc": "n_frac"}

#: RCDC mixes conditions with research areas, methods and populations. "all" is the whole
#: vocabulary; `shared_rcdc.is_disease` splits it (D11, D24).
RCDC_VIEW = "all"

OTHER_LABEL = "All other categories"

# =============================================================================
# 2. The band rule — one per vocabulary, because the tails differ by 4x
# =============================================================================
# `target_other` is the remainder share the panel would like to be left with; `min_band`
# is the mean share below which a band is a hairline nobody can read or label; `max_bands`
# is where a stream stops being a composition and becomes a colour chart. The rule takes
# the SMALLEST band count that meets the coverage target, then trims trailing bands that
# fall under the legibility floor. When the target is unreachable — RCDC — it stops at
# `max_bands` and the panel reports the coverage it actually got.
#
# Measured on the current corpus, over 2014-2025:
#
#   FOR L4   119 fields   12 bands -> 87.5% mean cover   15 bands -> 90.6%   (rule picks 15)
#   RCDC     271 tags     12 bands -> 52.3%              18 bands -> 62.3%   (rule picks 18)
#
# The RCDC remainder is irreducible, not a tuning failure: 24 bands still leave 31% and 40
# bands leave 19%. That is what a flat 271-tag vocabulary at 7.4 fractional tags per paper
# looks like, and the panel states it.
BAND_RULES = {
    "topics": {"target_other": 45.0, "min_band": 0.8, "max_bands": 16, "min_bands": 8},
    "for":    {"target_other": 10.0, "min_band": 0.7, "max_bands": 16, "min_bands": 8},
    "rcdc":   {"target_other": 10.0, "min_band": 1.2, "max_bands": 18, "min_bands": 10},
}


def category_weights(long, cat_col, weight, year_col="year", id_col="id"):
    """year x category weight matrix. Whole for `n_papers`, spread for `n_frac`.

    Lifted unchanged from §6 of `02_content_2_other_category_flow.ipynb` so the panels and
    that notebook cannot disagree about what a share is.
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
    if (totals <= 0).any():
        raise ValueError(f"no weight at all in year(s) {list(totals[totals <= 0].index)}")
    share = w.div(totals, axis=0) * 100
    assert np.allclose(share.sum(axis=1), 100), "shares do not sum to 100%"
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
        band[other_label] = rest.sum(axis=1)
    assert np.allclose(band.sum(axis=1), 100), "the drawn bands do not close at 100%"
    return band, keep, w.sum(axis=0).sort_values(ascending=False)


def _flow_block(long, cat_col, key, n_total=None, *, view_note="") -> dict:
    """Everything one panel row needs: the matrices, the bands and the self-description."""
    weight = WEIGHTS[key]
    w, share, n_papers = flow_shares(long, cat_col, weight)
    band, keep, order = top_bands(share, w, BAND_RULES[key])
    covered = band[keep].sum(axis=1)
    return {
        "weights": w,
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
        f"rank_{y0}": ranks.loc[y0, keep].astype(int),
        f"rank_{y1}": ranks.loc[y1, keep].astype(int),
    })
    out["delta_pp"] = (out[f"share_{y1}_%"] - out[f"share_{y0}_%"]).round(2)
    out["rank_change"] = out[f"rank_{y0}"] - out[f"rank_{y1}"]      # +ve = moved up
    out.index.name = "category"
    return out.sort_values("delta_pp", ascending=False)


# =============================================================================
# 3. The sources
# =============================================================================
CORPUS_COLUMNS = ["id", "year", "date", "category_for_2020", "category_rcdc"]

#: Where the BERTopic run's output is looked for, best first. Per-paper assignments
#: are preferred because they carry EVERY topic rather than the fourteen the notebook
#: pre-selected for its wave figure — so the same band rule can be applied here as to the
#: other two rows, remainder and all. `P.TOPIC_ASSIGNMENTS` is the two-column `id, topics`
#: file §18 of that notebook validates, and it needs a corpus/year join to be usable.
#: Every source also requires its matching .analysis_window.json sidecar.
TOPIC_SOURCES = (
    P.OUTPUT / "bertopic" / "bertopic_document_topic_assignments.csv",
    P.OUTPUT / "bertopic" / "showcase_plus_id_topics.csv",
    P.TOPIC_ASSIGNMENTS,                                        # id, topics
    P.CONTENT / "bertopic_document_topic_assignments.csv",      # id, year, topic, topics
    P.CONTENT / "bertopic_topic_year_counts_selected.csv",      # year x topic counts
    P.CONTENT / "bertopic_topic_year_proportions_selected.csv",  # year x topic shares
)

#: Missing outputs must not silently trigger a model fit inside a figure builder.
TOPIC_MISSING_NOTE = (
    "BERTopic assignments not found.\n\n"
    "Run 02_content_1_bert_topic.ipynb separately to train on publications\n"
    "through 2025-12-31, then use its output/bertopic/ tables with their\n"
    ".analysis_window.json sidecars. Legacy copied outputs require the\n"
    "same provenance; filtering their rows cannot undo later training data."
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
    """`T26: vat / visceral / adipose / adipose tissue` -> `vat / visceral / adipose`.

    The topic id is dropped and the keyword list trimmed to three: the id means nothing to
    a reader and a four-keyword label does not fit inside a band. The full label stays in
    the exported table.
    """
    text = str(raw)
    if ":" in text and text.split(":", 1)[0].strip().upper().startswith("T"):
        text = text.split(":", 1)[1]
    parts = [p.strip() for p in text.split("/") if p.strip()]
    return " / ".join(parts[:3]) if parts else text.strip()


def load_topic_year_matrix(corpus: pd.DataFrame):
    """The BERTopic (paper, topic, year) table, or `None` when the run's output is absent.

    Returns a long frame with `id`, `year`, `topic_label` so the caller can put it through
    the same `flow_shares` / `top_bands` path as the other two vocabularies. The two
    matrix-shaped sources are melted back into that shape; they carry only the fourteen
    pre-selected wave topics, so `n_total` comes back as the column count and the
    remainder band is what the selection left behind rather than the true tail.
    """
    corpus = filter_analysis_window(corpus)
    for path in TOPIC_SOURCES:
        if not Path(path).exists():
            continue
        # Filtering exported rows cannot undo a model trained on later publications.
        require_topic_window_provenance(path)
        frame = pd.read_csv(path)
        columns = set(frame.columns)

        if {"id", "topics"} <= columns:
            if "year" in columns:
                frame = filter_analysis_window(frame)
            long = frame[["id", "topics"]].copy()
            long = long.merge(corpus[["id", "year"]], on="id", how="inner",
                              validate="one_to_one")
            long = long[long["topics"].astype(str).str.strip().str.lower() != "outlier"]
            long["topic_label"] = long["topics"].map(_topic_label)
            long["year"] = pd.to_numeric(long["year"], errors="coerce")
            long = long.dropna(subset=["year"])
            long["year"] = long["year"].astype(int)
            return long[["id", "year", "topic_label"]], Path(path), None

        # Matrix shaped: first column is the year, every other column a topic.
        year_col = frame.columns[0]
        frame = filter_analysis_window(frame, year_col=year_col)
        matrix = frame.set_index(year_col)
        matrix.index = pd.to_numeric(matrix.index, errors="coerce")
        matrix = matrix[matrix.index.notna()]
        matrix.index = matrix.index.astype(int)
        matrix = matrix.drop(columns=[c for c in matrix.columns if str(c) == "-1"],
                             errors="ignore")
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        matrix.columns = [_topic_label(c) for c in matrix.columns]
        return matrix, Path(path), matrix.shape[1]

    return None, None, None


def _topics_block(corpus: pd.DataFrame) -> dict:
    """The topic row's block, or a `available=False` stub carrying why it is empty."""
    loaded, source, n_total = load_topic_year_matrix(corpus)
    if loaded is None:
        return {"available": False, "note": TOPIC_MISSING_NOTE, "source": None,
                "searched": [P.raw_path(p) for p in TOPIC_SOURCES]}

    if isinstance(loaded, pd.DataFrame) and "topic_label" in loaded.columns:
        block = _flow_block(loaded, "topic_label", "topics")
        block["from_assignments"] = True
    else:
        # A pre-selected matrix: rebuild the share matrix directly, then band it the same
        # way. Its rows do not close at 100% of the corpus, so they are renormalised and
        # the panel note says the denominator is the selection, not the corpus.
        w = loaded.reindex(FLOW_YEARS).fillna(0.0)
        w.index.name = "year"
        share = w.div(w.sum(axis=1), axis=0) * 100
        band, keep, order = top_bands(share, w, BAND_RULES["topics"])
        covered = band[keep].sum(axis=1)
        block = {
            "weights": w, "share": share, "band": band, "keep": keep, "order": order,
            "n_papers": w.sum(axis=1).round().astype(int),
            "n_categories": int(n_total), "n_bands": len(keep),
            "n_other": int(n_total) - len(keep),
            "coverage_mean": float(covered.mean()),
            "coverage_min": float(covered.min()),
            "coverage_max": float(covered.max()),
            "weight": WEIGHTS["topics"], "view_note": "", "available": True,
            "from_assignments": False,
        }
    block["source"] = P.raw_path(source)
    return block


# =============================================================================
# 4. The one build
# =============================================================================
def build_panel_data(verbose: bool = True) -> dict:
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
        "topics": _topics_block(corpus),
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
# 5. Drawing primitives
# =============================================================================
# Every one of these takes an `ax` and returns it. None creates a figure, sets a figure
# size, or calls savefig: the assembler owns the page, so a panel cannot quietly impose
# its own geometry on the one it is sharing.

from contextlib import contextmanager                                   # noqa: E402

import matplotlib.patheffects as pe                                     # noqa: E402
import matplotlib.pyplot as plt                                         # noqa: E402

from utils.shared_style import (                                        # noqa: E402
    extended_palette, grid_on, panel_label, savefig,
)

#: The closing remainder is grey in every panel of the family, and it is the only grey.
#: It is deliberately LIGHT. On the RCDC row the remainder is ~38% of the page — the tail
#: of a flat 271-tag vocabulary is irreducible, not a threshold that was set badly — and a
#: mid grey that large stops reading as "everything else" and starts reading as the
#: subject of the panel. Light enough to recede, dark enough to keep its own white
#: separator visible against the top of the stack.
OTHER_COLOR = "#DCDCDC"

#: A dark halo under the white in-band labels, so a name stays readable over the pale
#: middle of the stream palette as well as over its dark ends.
_STROKE = [pe.withStroke(linewidth=2.2, foreground="#00000055")]


def _style():
    from utils import shared_style
    return shared_style._resolve(None)


@contextmanager
def _font_scale(factor):
    """Draw the enclosed figure with every font size multiplied by `factor` (as in 04)."""
    from utils import shared_style
    original = _style()
    if not factor or factor == 1:
        yield original
        return
    scaled = dict(original)
    for key in ("title_fs", "label_fs", "tick_fs", "annot_fs", "legend_fs",
                "body_fs", "panel_label_fs"):
        if key in scaled:
            scaled[key] = scaled[key] * factor
    shared_style.use_style(scaled)
    shared_style.apply_style(scaled)
    try:
        yield scaled
    finally:
        shared_style.use_style(original)
        shared_style.apply_style(original)


def _fs_scale(figure: str) -> float:
    return float((_style().get("fs_scale") or {}).get(figure, 1.0))


def _spread(ys, min_gap, lo=0.0, hi=100.0):
    """Push label positions apart until none overlaps, keeping order and staying in range.

    From §6 of the source notebook. Returns positions in the order given.
    """
    order = sorted(range(len(ys)), key=lambda k: ys[k])
    placed, prev = {}, lo - min_gap
    for k in order:
        y = max(ys[k], prev + min_gap)
        placed[k], prev = y, y
    over = prev - hi
    if over > 0:                        # ran off the top: shift down, then re-settle
        for k in placed:
            placed[k] -= over
        for k in reversed(order[:-1]):
            nxt = order[order.index(k) + 1]
            placed[k] = min(placed[k], placed[nxt] - min_gap)
    return [placed[k] for k in range(len(ys))]


def _short_label(text: str, limit: int = 40) -> str:
    """Trim a category name to something a margin callout can hold, on a word boundary.

    RCDC carries names like "Health Disparities and Racial or Ethnic Minority Health
    Research" — 63 characters, wider than the gutter, so it runs off the page and over its
    neighbours. Truncating is better than wrapping here: a two-line callout doubles the
    vertical space every leader line has to be spread by.
    """
    text = str(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut.rstrip(',;')}\u2026"


def _label_year(band, n_papers, column):
    """(index, ha) for a band's in-place label: its thickest year that carries weight.

    Two rules, both there because the window opens on 26 papers. The label goes at the
    band's OWN thickest year — a band that peaks early and thins out is the shape this
    figure exists to show, and a fixed mid-point rule leaves it unnamed — but the first
    years are excluded from the search, because a band is at its widest in 2014 for the
    same reason every share there is unstable, and a name pinned to that edge both reads
    as a claim about 26 papers and runs off the left of the axes.
    """
    values = np.asarray(band[column], dtype=float)
    counts = np.asarray([n_papers.get(y, 0) for y in band.index], dtype=float)
    eligible = counts >= 0.02 * max(counts.max(), 1)
    if not eligible.any():
        eligible = np.ones_like(values, dtype=bool)
    masked = np.where(eligible, values, -np.inf)
    i = int(masked.argmax())
    if i == 0:
        return i, "left"
    if i == len(values) - 1:
        return i, "right"
    return i, "center"


def _weight_note(weight: str) -> str:
    return {"n_papers": "whole counting",
            "n_frac": "fractional counting"}.get(weight, f"summed {weight}")


def _missing_panel(ax, title: str, note: str):
    """What a panel draws when its source is not on disk: the reason, not an empty box.

    The axes keeps its frame — dashed, and with the ticks off — so the row still reads as
    a panel this page has reserved rather than as a hole somebody left in the layout. The
    geometry is identical to the filled version, so dropping the source file in changes
    what is in this row and nothing about the shape of the page.
    """
    st = _style()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linestyle((0, (4, 4)))
        spine.set_edgecolor("#C4C4C4")
        spine.set_linewidth(0.9)
    ax.set_facecolor("#FAFAFA")
    ax.set_title(title, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.015, "source not on disk — this row is reserved, not dropped",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=st["annot_fs"] - 1.5, color="#666666")
    ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center",
            fontsize=st["annot_fs"], color="#555555", family="monospace",
            linespacing=1.7,
            bbox=dict(boxstyle="round,pad=1.0", facecolor="white",
                      edgecolor="#D6D6D6", linewidth=0.8))
    return ax


def draw_share_stream(ax, block, title, *, label_min=7.0, callout_gap=4.2,
                      show_year_n=True, xlabel=None):
    """One 100%-normalised composition stream, with every band named.

    A band thick enough at its own thickest year to hold its name is labelled in place; the
    rest are called out in the right-hand gutter with a leader line and their final-year
    share. That is why there is no legend: matching eighteen names against eighteen
    swatches is work the figure can do for the reader.

    The remainder band is grey, is always labelled in place, and says how many categories
    are inside it — an unexplained grey slab is the one thing a composition stream must
    not leave on the page.
    """
    st = _style()
    band = block["band"]
    labels = list(band.columns)
    named = [lab for lab in labels if lab != OTHER_LABEL]
    colors = list(extended_palette(len(named)))
    if OTHER_LABEL in labels:
        colors.append(OTHER_COLOR)
    years = list(band.index)
    cum = band.cumsum(axis=1)

    ax.stackplot(years, band.T.values, labels=labels, colors=colors,
                 edgecolor="white", linewidth=0.6)

    thin = []
    pad = (max(years) - min(years)) * 0.012
    for j, lab in enumerate(labels):
        i, ha = _label_year(band, block["n_papers"], lab)
        thickest = band.iloc[i, j]
        text = f"{lab} ({block['n_other']})" if lab == OTHER_LABEL else lab
        if thickest >= label_min:
            x = years[i] + (pad if ha == "left" else -pad if ha == "right" else 0)
            ax.text(x, cum.iloc[i, j] - thickest / 2, text,
                    ha=ha, va="center", fontsize=st["annot_fs"] - 1,
                    fontweight="bold", color="white", path_effects=_STROKE, zorder=5)
        else:
            thin.append(j)

    if thin:
        anchors = [cum.iloc[-1, j] - band.iloc[-1, j] / 2 for j in thin]
        targets = _spread(anchors, min_gap=callout_gap)
        outside = ax.get_yaxis_transform()
        for j, y_anchor, y_target in zip(thin, anchors, targets):
            ax.annotate(f"{_short_label(labels[j])}  {band.iloc[-1, j]:.1f}%",
                        xy=(years[-1], y_anchor), xycoords="data",
                        xytext=(1.02, y_target), textcoords=outside,
                        ha="left", va="center", fontsize=st["annot_fs"] - 1.5,
                        color=colors[j], fontweight="bold", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=colors[j], lw=0.9,
                                        shrinkA=0, shrinkB=2))
        ax.text(1.02, 1.035, f"{years[-1]} share", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=st["annot_fs"] - 2, color="#777777")

    ax.set_xlim(min(years), max(years))
    ax.set_ylim(0, 100)
    ax.set_xticks(years)
    if show_year_n:
        # The window's first years rest on 26 and 61 papers. That n belongs under the tick
        # rather than in a footnote, because it is what says how much a wiggle is worth.
        counts = block["n_papers"]
        ax.set_xticklabels([f"{y}\n{counts.get(y, 0):,}" for y in years])
    else:
        ax.set_xticklabels([str(y) for y in years])
    ax.set_yticks(range(0, 101, 25))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 25)])
    ax.set_ylabel("Share of the year's total")
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=6)
    ax.set_title(title, fontweight="bold", loc="left", pad=26)

    # The panel states its own coverage: with a remainder this large on the RCDC row, a
    # reader who cannot see what the grey is worth cannot read the panel at all.
    ax.text(0.0, 1.015,
            f"{block['n_bands']} of {block['n_categories']} drawn · they hold "
            f"{block['coverage_min']:.0f}–{block['coverage_max']:.0f}% of each year "
            f"(mean {block['coverage_mean']:.0f}%) · {_weight_note(block['weight'])}"
            + (f" · {block['view_note']}" if block["view_note"] else ""),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=st["annot_fs"] - 1.5, color="#666666")
    return ax


def draw_topic_stream(ax, D):
    """A — what the papers are about, from the text rather than from a classifier."""
    block = D["topics"]
    title = "BERTopic topics"
    if not block["available"]:
        return _missing_panel(ax, title, block["note"])
    return draw_share_stream(ax, block, title, label_min=6.0)


def draw_for_stream(ax, D):
    """B — the publisher-side classification, at the level that names a discipline."""
    return draw_share_stream(ax, D["for"], "Fields of Research 2020, Level 4",
                             label_min=7.0)


def draw_rcdc_stream(ax, D):
    """C — the funder vocabulary, where the tail is irreducible and the panel says so."""
    return draw_share_stream(ax, D["rcdc"], "RCDC categories", label_min=4.4,
                             callout_gap=3.6,
                             xlabel="Publication year, and the papers carrying "
                                    "≥1 category that year")


def draw_rank_flow(ax, block, title):
    """A bump chart of the drawn bands: who overtook whom, with no remainder to draw.

    The share stream answers "how much"; this answers "in what order", which a stack of
    eighteen bands cannot be read for. Line width is proportional to mean share so the
    two figures agree about which lines are the heavy ones.
    """
    st = _style()
    share, keep = block["share"], block["keep"]
    ranks = share[keep].rank(axis=1, ascending=False, method="first")
    years = list(share.index)
    colors = dict(zip(keep, extended_palette(len(keep))))
    heaviest = max(share[keep].mean().max(), 1e-9)
    outside = ax.get_yaxis_transform()

    for cat in keep:
        y = ranks[cat].values
        ax.plot(years, y, color=colors[cat],
                linewidth=1.0 + 3.2 * (share[cat].mean() / heaviest),
                marker="o", markersize=st["marker_size"] * 0.45,
                markeredgecolor="white", markeredgewidth=0.6,
                solid_capstyle="round", zorder=2)
        # The row IS the rank, so it is carried in the end labels and the left margin is
        # left free of a tick column the names would collide with.
        name = _short_label(cat, 34)
        ax.text(-0.012, y[0], f"#{int(y[0])}  {name}", transform=outside, ha="right",
                va="center", fontsize=st["annot_fs"] - 1.5, color=colors[cat],
                fontweight="bold", clip_on=False)
        ax.text(1.012, y[-1], f"{name}  #{int(y[-1])}", transform=outside, ha="left",
                va="center", fontsize=st["annot_fs"] - 1.5, color=colors[cat],
                fontweight="bold", clip_on=False)

    ax.set_ylim(len(keep) + 0.6, 0.4)
    ax.set_yticks(range(1, len(keep) + 1))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)
    ax.set_xticks(years)
    ax.set_xlabel("Publication year")
    grid_on(ax, axis="y")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, fontweight="bold", loc="left", pad=12)
    return ax


def draw_for_rank_flow(ax, D):
    return draw_rank_flow(ax, D["for"],
                          f"FOR Level 4 fields, rank by yearly share "
                          f"({FLOW_MIN}–{FLOW_MAX})")


def draw_rcdc_rank_flow(ax, D):
    return draw_rank_flow(ax, D["rcdc"],
                          f"RCDC categories, rank by yearly share "
                          f"({FLOW_MIN}–{FLOW_MAX})")


# =============================================================================
# 6. The pages
# =============================================================================
MAIN_CAPTION = {
    "A": "Composition of UK Biobank publications by BERTopic topic, 2014–2025. Topics "
         "are learned from title and abstract text (SPECTER embeddings, UMAP + HDBSCAN, "
         "outliers force-reassigned so every paper carries exactly one topic — D5, D6). "
         "Each year is normalised independently, so a narrowing band means a topic "
         "growing more slowly than the corpus, not shrinking.",
    "B": "The same window in the publisher's own vocabulary: FOR 2020 Level 4 fields, "
         "whole counting (a paper counts once in each of the ~1.5 fields it carries). "
         "The drawn bands hold roughly 90% of each year; the grey band is every "
         "remaining field.",
    "C": "The same window in the funder vocabulary: RCDC categories, fractional counting "
         "(each paper spreads a weight of 1 across the ~7.4 tags it carries, so a paper "
         "tagged with twenty conditions does not outvote one tagged with two). The grey "
         "remainder is large and irreducible — 271 tags at that density leave ~38% "
         "outside any readable number of bands — and is labelled with its own category "
         "count rather than left unexplained.",
}

SI_CAPTIONS = {
    "rank_flow": {
        "A": "Rank of each drawn FOR Level 4 field by its share of the year, 2014–2025. "
             "Line width is proportional to mean share, so the heavy lines here are the "
             "thick bands of the main panel's B.",
        "B": "The same for RCDC categories. Rank answers the question a stack of eighteen "
             "bands cannot be read for — which categories crossed, and when.",
    },
}

def thin_years_note(D) -> str:
    """The window's weakest years, derived rather than written down.

    Printed on the page rather than left to a caption that may not travel with it: the
    flow opens on a couple of dozen papers, where a single paper moves a share by several
    percentage points, and every panel's left-hand edge has to be read with that in hand.
    """
    counts = D["for"]["n_papers"]
    years = list(counts.index)
    first, second, last = counts.iloc[0], counts.iloc[1], counts.iloc[-1]
    fields = D["for"]["n_categories"]
    move = 100.0 / max(int(first), 1)
    return (
        f"The window opens thin: {int(first):,} papers in {years[0]} and {int(second):,} "
        f"in {years[1]}, against {int(last):,} in {years[-1]}. A share across {fields} "
        f"fields computed on {int(first):,} papers moves ~{move:.0f} percentage points "
        f"when one paper changes, so read the left-hand edge of every panel as indicative "
        f"only — each year's population is printed under its own tick, per panel, because "
        f"the three vocabularies do not cover the same papers."
    )


def _assemble(spec, nrows, ncols, figsize, D, name, save=True, slots=None,
              hspace=0.42, wspace=0.26, height_ratios=None, width_ratios=None,
              right=0.72, left=0.075, top=0.955, bottom=0.075, label_panels=True,
              letter_x=-0.058, letter_y=1.045, footer=None):
    """Draw `spec` (an ordered list of draw functions) into a letter-labelled grid.

    Same contract as the 03 and 04 assemblers, with one addition these pages need: the
    grid stops at `right`, because every stream panel calls its thin bands out into the
    margin beyond it. Leaving that gutter in the gridspec rather than in each draw
    function is what keeps the three rows' callout columns aligned with each other.
    """
    letters = "ABCDEFGHIJKL"
    kw = {}
    if height_ratios is not None:
        kw["height_ratios"] = list(height_ratios)
    if width_ratios is not None:
        kw["width_ratios"] = list(width_ratios)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols, hspace=hspace, wspace=wspace,
                          left=left, right=right, top=top, bottom=bottom, **kw)
    if slots is None:
        cells = [(row, col) for row in range(nrows) for col in range(ncols)]
        slots = cells[:len(spec)]
    elif len(slots) != len(spec):
        raise ValueError(f"{len(spec)} draw functions but {len(slots)} slots")
    axes = [fig.add_subplot(gs[rows, cols]) for rows, cols in slots]

    for ax, letter, draw in zip(axes, letters, spec):
        draw(ax, D)
        if label_panels:
            # Free text, NOT the default placement: `panel_label(ax, "B")` with no
            # position sets the axes TITLE, which would silently replace the descriptive
            # title each of these panels carries.
            panel_label(ax, letter, x=letter_x, y=letter_y)
    if footer:
        fig.text(left, bottom * 0.30, footer, ha="left", va="bottom",
                 fontsize=_style()["annot_fs"] - 1.5, color="#666666", wrap=True)
    if save:
        savefig(fig, name)
    return fig


def figure_main(D, save=True):
    """The main-paper page: three composition streams over one window, stacked.

    One column, not two. These are time series over twelve years with eighteen bands and a
    margin of callouts — side by side they would each be half as wide as they need and
    their callout columns would collide down the middle of the page. Stacked, the years
    line up vertically, which is what makes "did all three vocabularies move at once"
    answerable by eye.

    Panel letters are carried in the titles as well as stamped in the corner, because this
    page is read at full width and the corner letter is a long way from the panel it names
    by the time the page is three rows tall.
    """
    st = _style()
    with _font_scale(_fs_scale("main")):
        return _assemble(
            [draw_topic_stream,     # A
             draw_for_stream,       # B
             draw_rcdc_stream],     # C
            3, 1, st["figsize_main"], D, "02_01_figure_01_content_composition",
            save=save, hspace=0.34, right=0.76, top=0.965, bottom=0.085,
            footer=thin_years_note(D),
        )


def figure_si_rank_flow(D, save=True):
    """SI 1 — the two bump charts the source notebook drew as 06b and 07b.

    They are not on the main page because rank and share answer different questions and
    the main page is already three panels of share. Nothing is discarded: the bands are
    the same bands, drawn from the same `D`.
    """
    st = _style()
    with _font_scale(_fs_scale("si1_rank_flow")):
        return _assemble(
            [draw_for_rank_flow,    # A
             draw_rcdc_rank_flow],  # B
            2, 1, st["figsize_si"], D,
            "02_02_supplementary_figure_01_category_rank_flow",
            save=save, hspace=0.22, left=0.24, right=0.78, top=0.95, bottom=0.06,
        )
