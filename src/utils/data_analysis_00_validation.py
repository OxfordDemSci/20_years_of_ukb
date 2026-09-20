"""Cached validation analysis and explicitly requested original model evaluation.

The six-prompt inference workflow is preserved from the original validation
notebook. It runs only with ``run_inference=True`` and only after input checks.
Encoder thresholds in that original workflow are calibrated on the evaluation
labels: encoder scores are in-sample calibration results, not held-out accuracy.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import shared_paths as P
from .shared_analysis_window import filter_analysis_window


PROMPTS = (
    ("p1_conservative", "Conservative instructions"),
    ("p2_balanced", "Balanced instructions"),
    ("p3_evidence_cues", "Evidence cues"),
    ("p4_context_no_shot", "Context, no-shot"),
    ("p5_real_one_shot", "Real one-shot"),
    ("p6_real_five_shot", "Real five-shot"),
)
MODEL_NAMES = (
    "qwen2_5_7b", "llama3_8b", "mistral_7b", "phi3_5_mini", "zephyr_7b",
    "scibert_sim", "sbert_minilm_sim",
)
ENCODER_MODELS = {"scibert_sim", "sbert_minilm_sim"}
MODEL_LABELS = {
    "qwen2_5_7b": "Qwen2.5-7B", "llama3_8b": "Llama3-8B",
    "mistral_7b": "Mistral-7B", "phi3_5_mini": "Phi3.5-mini",
    "zephyr_7b": "Zephyr-7B", "scibert_sim": "SciBERT*", "sbert_minilm_sim": "MiniLM*",
}
PERFORMANCE_STEM = "00_03_figure_validation_performance"
AGREEMENT_STEM = "00_04_figure_validation_agreement"
PERFORMANCE_CAPTION = (
    "Model performance across six prompt strategies. A, Accuracy. B, Precision. "
    "C, Recall (sensitivity). D, F1 score. Rows identify models and columns identify "
    "prompt strategies; cell labels and colours show percentages on a common "
    "0–100% scale. Metrics are calculated only among each model's parsed predictions; "
    "parse coverage and confusion-matrix counts are provided in the accompanying CSV "
    "tables. A dashed rule separates instruction-tuned language models from encoder "
    "baselines (asterisks). Encoder similarity thresholds were optimised for F1 on "
    "the evaluation labels, so encoder results represent in-sample calibration, "
    "not held-out validation; encoder baselines do not use the prompts. Blank cells "
    "with a dash indicate unavailable results or no parsed predictions."
)
CAPTION = (
    "Pairwise agreement across six prompt strategies: A, Conservative instructions; "
    "B, Balanced instructions; C, Evidence cues; D, Context without examples; "
    "E, Real one-shot; F, Real five-shot. Cells give the percentage of matching "
    "predictions among papers for which both models returned parsed predictions, "
    "on a common 0–100% colour scale. Each panel reports its range of paired-paper "
    "denominators; exact cell counts are supplied in the accompanying CSV tables. "
    "Blank cells with a dash indicate unavailable comparisons or zero paired "
    "predictions. Dashed rules separate language models from encoder baselines "
    "(asterisks). Encoder thresholds were optimised on the evaluation labels: "
    "their results represent in-sample calibration, not held-out validation. "
    "Agreement measures concordance between models, not accuracy against labels."
)


def _path(value, env_name, default):
    raw = value if value is not None else os.environ.get(env_name, "").strip()
    path = Path(raw).expanduser() if raw else Path(default)
    return path if path.is_absolute() else P.ROOT / path


def _sample_size(value, env_name, *, required=False):
    raw = value if value is not None else os.environ.get(env_name, "").strip()
    if raw in (None, "") and not required:
        return None
    try:
        result = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Set {env_name} to the intended positive integer sample size.") from None
    if result <= 0 or str(raw).strip() != str(result):
        raise ValueError(f"{env_name} must be a positive integer; got {raw!r}.")
    return result


def _text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                preferred = parsed.get("preferred")
                if preferred is not None:
                    return str(preferred).strip()
                return next((str(v).strip() for v in parsed.values() if v is not None and str(v).strip()), "")
        except (ValueError, SyntaxError):
            pass
    return text


def _load_labelled(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"Missing labelled validation CSV: {path}")
    frame = pd.read_csv(path, dtype={"id": "string"})
    required = {"id", "title", "abstract", "year"}
    if required - set(frame):
        raise ValueError(f"{path.name} is missing columns: {sorted(required - set(frame))}")
    frame = filter_analysis_window(frame).copy()
    frame["id"] = frame["id"].str.strip()
    frame["title"] = frame["title"].map(_text)
    frame["abstract"] = frame["abstract"].map(_text)
    frame = frame.loc[frame["id"].notna() & frame["id"].ne("") & frame["abstract"].ne("")]
    frame = frame.drop_duplicates("id", keep="first").reset_index(drop=True)
    frame["label"] = int(label)
    return frame


def _preflight(positive, negative, n_positive, n_negative):
    pools = (_load_labelled(positive, 1), _load_labelled(negative, 0))
    overlap = set(pools[0]["id"]) & set(pools[1]["id"])
    if overlap:
        raise ValueError(f"Positive and negative validation pools overlap ({len(overlap):,} IDs).")
    for label, pool, requested, extra in zip(("Positive", "Negative"), pools, (n_positive, n_negative), (3, 2)):
        if requested is not None and len(pool) < requested + extra:
            raise ValueError(
                f"{label} pool has {len(pool):,} valid 2013–2025 papers; "
                f"{requested:,} evaluation papers require {extra} additional prompt examples."
            )
    return pools


def _bool_column(series, *, allow_missing, name):
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map({"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
    missing = series.isna() | text.eq("").fillna(False)
    invalid = mapped.isna() & ~missing
    if invalid.any() or (not allow_missing and mapped.isna().any()):
        raise ValueError(f"Invalid Boolean labels in cached validation column {name}.")
    return mapped.astype("boolean")


def _read_predictions(path):
    frame = pd.read_csv(path, dtype={"id": "string"})
    required = {"id", "year", "True_label"}
    if required - set(frame):
        raise ValueError(f"{path.name} is missing columns: {sorted(required - set(frame))}")
    if frame.empty:
        raise ValueError(f"{path.name} contains no evaluation papers.")
    frame["id"] = frame["id"].str.strip()
    if frame["id"].isna().any() or frame["id"].eq("").any() or frame["id"].duplicated().any():
        raise ValueError(f"{path.name} must contain unique, non-empty paper IDs.")
    years = pd.to_numeric(frame["year"], errors="coerce")
    if not (years.between(2013, 2025) & years.mod(1).eq(0)).all() or len(filter_analysis_window(frame)) != len(frame):
        raise ValueError(f"{path.name} is incompatible with the 2013–2025 analysis window; regenerate its predictions explicitly.")
    frame["year"] = years.astype(int)
    frame["True_label"] = _bool_column(frame["True_label"], allow_missing=False, name="True_label")
    if frame["True_label"].nunique() != 2:
        raise ValueError(f"{path.name} must contain both positive and negative evaluation papers.")
    models = [name for name in MODEL_NAMES if name in frame]
    if not models:
        raise ValueError(f"{path.name} contains no recognised model prediction columns.")
    for model in models:
        frame[model] = _bool_column(frame[model], allow_missing=True, name=model)
    return frame, models


def _metrics(truth, predictions):
    valid = predictions.notna()
    a, b = truth[valid].astype(bool), predictions[valid].astype(bool)
    tp, tn = int((a & b).sum()), int((~a & ~b).sum())
    fp, fn = int((~a & b).sum()), int((a & ~b).sum())
    n = len(a)
    return dict(accuracy=(tp + tn) / n if n else np.nan,
                precision=tp / (tp + fp) if tp + fp else (0.0 if n else np.nan),
                recall=tp / (tp + fn) if tp + fn else (0.0 if n else np.nan),
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else (0.0 if n else np.nan),
                tp=tp, tn=tn, fp=fp, fn=fn, n_parsed=n,
                parse_rate=n / len(truth))


def _agreement(frame, models):
    agreement = pd.DataFrame(np.nan, index=models, columns=models)
    counts = pd.DataFrame(0, index=models, columns=models)
    for a in models:
        for b in models:
            valid = frame[a].notna() & frame[b].notna()
            counts.loc[a, b] = int(valid.sum())
            if valid.any():
                agreement.loc[a, b] = float((frame.loc[valid, a] == frame.loc[valid, b]).mean())
    return agreement, counts


def _draw_matrix(ax, values, *, title, row_models, column_labels, model_columns=False, fontsize=9):
    from .shared_style import blue_cream_red_colormap, palette, set_title
    cmap = blue_cream_red_colormap().with_extremes(bad="white")
    array = values.to_numpy(dtype=float)
    image = ax.pcolormesh(np.ma.masked_invalid(array), cmap=cmap, vmin=0, vmax=1,
                         edgecolors=palette("navy"), linewidth=.45)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(values.shape[1]) + .5, column_labels, rotation=40, ha="right", fontsize=fontsize)
    ax.set_yticks(np.arange(values.shape[0]) + .5, [MODEL_LABELS[m] for m in row_models], fontsize=fontsize)
    ax.tick_params(length=0, pad=5)
    title_options = {"y": 1.075, "pad": 6} if model_columns else {"pad": 12}
    set_title(ax, title, fontsize=12, **title_options)
    for row, col in np.ndindex(array.shape):
        value = array[row, col]
        label = f"{100 * value:.1f}" if np.isfinite(value) else "—"
        ax.text(col + .5, row + .5, label, ha="center", va="center", fontsize=fontsize - .5)
    llm_count = sum(m not in ENCODER_MODELS for m in row_models)
    if 0 < llm_count < len(row_models):
        ax.axhline(llm_count, color=palette("navy"), lw=1.6, linestyle="--")
        if model_columns:
            ax.axvline(llm_count, color=palette("navy"), lw=1.6, linestyle="--")
    ax.grid(False)
    return image


def _save_publication_figure(fig, figure_dir, stem, *, show_figures):
    import matplotlib.pyplot as plt
    from .shared_style import PNG_DPI, finalize_figure
    finalize_figure(fig)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
        paths.append(path)
    if show_figures:
        from IPython.display import display
        display(fig)
    plt.close(fig)
    return paths


def _render_performance(summary, figure_dir, *, n_papers, n_positive, show_figures=True):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from .shared_style import apply_typography
    apply_typography()
    models = [m for m in MODEL_NAMES if m in set(summary["model"])]
    prompts = [p for p, _ in PROMPTS]
    labels = ["Conservative", "Balanced", "Evidence cues", "Context", "One-shot", "Five-shot"]
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.6))
    for index, (ax, metric, title) in enumerate(zip(axes.ravel(),
            ("accuracy", "precision", "recall", "f1"), ("Accuracy", "Precision", "Recall", "F1 score"))):
        values = summary.pivot(index="model", columns="prompt", values=metric).reindex(index=models, columns=prompts)
        image = _draw_matrix(ax, values, title=f"{chr(65 + index)}  {title}",
                             row_models=models, column_labels=labels, fontsize=10)
    fig.subplots_adjust(left=.12, right=.9, bottom=.21, top=.94, wspace=.43, hspace=.50)
    cbar = fig.colorbar(image, cax=fig.add_axes([.93, .33, .012, .50]), format=PercentFormatter(1))
    cbar.set_label("Performance among parsed predictions", fontsize=10)
    parse = pd.to_numeric(summary["parse_rate"], errors="coerce").dropna()
    coverage = f"{100 * parse.min():.1f}–{100 * parse.max():.1f}%" if len(parse) else "unavailable"
    fig.text(.12, .072,
             f"Evaluation: {n_papers:,} papers ({n_positive:,} positive; {n_papers - n_positive:,} negative), 2013–2025. "
             f"Parse coverage: {coverage}.\n"
             "* Encoder baselines: thresholds optimised on these labels (in-sample); not held-out validation.",
             ha="left", va="bottom", fontsize=10, linespacing=1.6)
    return _save_publication_figure(fig, figure_dir, PERFORMANCE_STEM, show_figures=show_figures)


def _render_agreement(agreements, figure_dir, *, comparison_counts=None, n_papers=None, show_figures=True):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from .shared_style import apply_typography
    apply_typography()
    models = [m for m in MODEL_NAMES if any(m in frame.index for frame in agreements.values())]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11.5))
    for index, (ax, (prompt, title)) in enumerate(zip(axes.ravel(), PROMPTS)):
        values = agreements[prompt].reindex(index=models, columns=models)
        if comparison_counts is not None:
            counts = comparison_counts[prompt].reindex(index=models, columns=models).fillna(0)
            values = values.where(counts.gt(0))
            available = counts.to_numpy()[counts.to_numpy() > 0]
            if len(available):
                low, high = int(available.min()), int(available.max())
                number = f"{low:,}" if low == high else f"{low:,}–{high:,}"
                suffix = f" / {n_papers:,}" if n_papers is not None else ""
                note = f"Paired predictions: n = {number}{suffix} papers"
            else:
                note = "No paired predictions available"
            ax.text(0, 1.013, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)
        image = _draw_matrix(ax, values, title=f"{chr(65 + index)}  {title}", row_models=models,
                             column_labels=[MODEL_LABELS[m] for m in models], model_columns=True, fontsize=8.5)
    fig.subplots_adjust(left=.10, right=.915, bottom=.20, top=.935, wspace=.47, hspace=.67)
    cax = fig.add_axes([.944, .31, .012, .49])
    cbar = fig.colorbar(image, cax=cax, format=PercentFormatter(1))
    cbar.set_label("Pairwise agreement")
    fig.text(.10, .07,
             "Cell percentages use papers with two parsed predictions; exact paired counts are provided in the CSV tables.\n"
             "* Encoder baselines use thresholds optimised on the evaluation labels (in-sample); not held-out validation.",
             ha="left", va="bottom", fontsize=10, linespacing=1.6)
    return _save_publication_figure(fig, figure_dir, AGREEMENT_STEM, show_figures=show_figures)


def run_validation(output_dir=None, *, positive_csv=None, negative_csv=None,
                   n_positive=None, n_negative=None, run_inference=False, figure_dir=None,
                   show_figures=True):
    """Reuse all six prediction caches, or explicitly run the original evaluation.

    Missing caches produce a SKIP by default. Partial or incompatible caches raise
    before any model import, download or inference. Remove/move incompatible caches
    explicitly before requesting a new evaluation; this function never overwrites
    them automatically.
    """
    output = _path(output_dir, "UKB_VALIDATION_OUTPUT_DIR", P.OUTPUT / "validation")
    figures = Path(figure_dir) if figure_dir is not None else P.FIG_DATA_ANALYSIS / "00_dataset" / "validation"
    if not figures.is_absolute():
        figures = P.ROOT / figures
    paths = {prompt: output / f"predictions_{prompt}.csv" for prompt, _ in PROMPTS}
    present = [path.is_file() for path in paths.values()]
    reused_predictions = all(present)
    if not any(present) and not run_inference:
        reason = ("No saved validation predictions. Supply all six predictions_*.csv files "
                  f"under {output}, or explicitly enable validation inference with labelled inputs and sample sizes.")
        print(f"[SKIP] Validation: {reason}")
        return {"status": "SKIP", "reason": reason, "files": [], "figure_files": []}
    if any(present) and not all(present):
        absent = [path.name for path in paths.values() if not path.is_file()]
        raise ValueError(f"Incomplete validation prediction cache: missing {', '.join(absent)}. No models were run.")
    n_pos = _sample_size(n_positive, "UKB_VALIDATION_N_POS", required=not all(present))
    n_neg = _sample_size(n_negative, "UKB_VALIDATION_N_NEG", required=not all(present))
    positive = _path(positive_csv, "UKB_VALIDATION_POSITIVE_CSV", P.DATA / "validation" / "ukb_ground_truth_positive_labelled.csv")
    negative = _path(negative_csv, "UKB_VALIDATION_NEGATIVE_CSV", P.DATA / "validation" / "ukb_negative_pre2014_labelled.csv")
    pools = None
    explicit_sources = (positive_csv is not None or negative_csv is not None or
                        bool(os.environ.get("UKB_VALIDATION_POSITIVE_CSV")) or bool(os.environ.get("UKB_VALIDATION_NEGATIVE_CSV")))
    if not all(present) or explicit_sources or (positive.is_file() and negative.is_file()):
        pools = _preflight(positive, negative,
                           None if all(present) else n_pos,
                           None if all(present) else n_neg)
    if not all(present):
        output.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        _run_original_inference(positive, negative, output, figures, n_pos, n_neg)
    frames, model_lists = {}, {}
    reference = None
    for prompt, path in paths.items():
        frame, models = _read_predictions(path)
        identity = frame.set_index("id")[["year", "True_label"]].sort_index()
        if reference is not None and not identity.equals(reference):
            raise ValueError(f"{path.name} does not describe the same evaluation papers/labels as the other prompt caches.")
        reference = identity
        frames[prompt], model_lists[prompt] = frame, models
    positives = int(reference["True_label"].sum())
    negatives = len(reference) - positives
    if (n_pos is not None and n_pos != positives) or (n_neg is not None and n_neg != negatives):
        raise ValueError("Cached validation sample sizes differ from the requested positive/negative sample sizes.")
    if pools is not None:
        source_rows = pd.concat(pools).set_index("id")
        labels = source_rows["label"].astype(bool)
        if not reference.index.isin(labels.index).all() or not reference["True_label"].astype(bool).equals(labels.reindex(reference.index)):
            raise ValueError("Cached validation IDs/labels do not match the configured labelled source pools.")
        cached = next(iter(frames.values())).set_index("id").sort_index()
        sources = source_rows.reindex(reference.index)
        for column in ("year", "title", "abstract"):
            if column not in cached:
                raise ValueError(f"Cached validation lacks {column}; cannot verify it against the labelled source pools.")
            if column == "year":
                if not np.array_equal(pd.to_numeric(cached[column]).to_numpy(), pd.to_numeric(sources[column]).to_numpy()):
                    raise ValueError("Cached validation year values differ from the configured labelled source pools.")
                continue
            expected = sources[column].map(_text)
            actual = cached[column].map(_text)
            if not actual.equals(expected):
                raise ValueError(f"Cached validation {column} values differ from the configured labelled source pools.")
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    files, rows, agreements, comparison_counts = [], [], {}, {}
    for prompt, frame in frames.items():
        prompt_rows = []
        old_path = output / f"results_{prompt}.csv"
        previous = pd.read_csv(old_path) if old_path.is_file() else pd.DataFrame()
        for model in model_lists[prompt]:
            row = {"model": model, "prompt": prompt, "type": "Encoder" if model in {"scibert_sim", "sbert_minilm_sim"} else "LLM"}
            if "model" in previous:
                old = previous.loc[previous["model"].eq(model)]
                if len(old) == 1:
                    row.update({key: old.iloc[0][key] for key in ("runtime_s", "items_per_sec", "threshold", "calib_f1_all_eval") if key in old})
            row.update(_metrics(frame["True_label"], frame[model]))
            if row["type"] == "Encoder":
                row["calibration_note"] = "Threshold calibrated on the evaluation labels (in-sample)."
            prompt_rows.append(row)
        rows.extend(prompt_rows)
        agreement, counts = _agreement(frame, model_lists[prompt])
        agreements[prompt] = agreement
        comparison_counts[prompt] = counts
        for name, table, index in ((f"results_{prompt}.csv", pd.DataFrame(prompt_rows), False),
                                   (f"pairwise_agreement_percent_{prompt}.csv", (agreement * 100).round(1), True),
                                   (f"pairwise_agreement_n_{prompt}.csv", counts, True)):
            path = output / name
            table.to_csv(path, index=index)
            files.append(path)
    summary = pd.DataFrame(rows)
    ranked = summary.sort_values(["accuracy", "f1", "precision", "recall"], ascending=False, na_position="last").reset_index(drop=True)
    for name, table in (("ALL_prompt_model_results_summary.csv", summary), ("ALL_prompt_model_results_ranked_by_accuracy.csv", ranked)):
        path = output / name
        table.to_csv(path, index=False)
        files.append(path)
    figure_files = _render_performance(summary, figures, n_papers=len(reference), n_positive=positives, show_figures=show_figures)
    figure_files += _render_agreement(agreements, figures, comparison_counts=comparison_counts,
                                      n_papers=len(reference), show_figures=show_figures)
    files.extend(figure_files)
    cohort_note = (f" Evaluation contains {len(reference):,} papers ({positives:,} positive and "
                   f"{negatives:,} negative), published during 2013–2025.")
    for stem, caption in ((PERFORMANCE_STEM, PERFORMANCE_CAPTION), (AGREEMENT_STEM, CAPTION)):
        caption_path = figures / f"{stem}_caption.txt"
        caption_path.write_text(caption + cohort_note + "\n")
        files.append(caption_path)
    action = "reused" if reused_predictions else "generated"
    reason = "Reused saved predictions" if reused_predictions else "Generated predictions by explicit inference"
    print(f"[PASS] Validation: {action} six prediction tables; {len(reference):,} papers, {len(summary)} model/prompt results.")
    return {"status": "PASS", "reason": reason, "files": files, "figure_files": figure_files,
            "n_papers": len(reference), "summary": summary, "ranked": ranked}


def _run_original_inference(TP_PATH, TN_PATH, OUT_DIR, FIGURE_DIR, N_POS, N_NEG):
    """Original six-prompt evaluation; called only after explicit opt-in/preflight."""
    import re, json, time, gc, random, warnings
    import torch
    from IPython.display import display
    from tqdm.auto import tqdm
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer

    SEED = 42
    USE_4BIT = True

    # ============================================================
    # Data loading
    # ============================================================

    def extract_text(x):
        if pd.isna(x):
            return ""
        if not isinstance(x, str):
            return str(x).strip()
        s = x.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, dict):
                    if "preferred" in obj and obj["preferred"] is not None:
                        return str(obj["preferred"]).strip()
                    for v in obj.values():
                        if v is not None and str(v).strip():
                            return str(v).strip()
            except Exception:
                pass
        return s

    def load_labelled_csv(path, expected_label):
        return _load_labelled(Path(path), expected_label)[
            ["id", "title", "abstract", "year", "label"]
        ].copy()

    tp_pool = load_labelled_csv(TP_PATH, 1)
    tn_pool = load_labelled_csv(TN_PATH, 0)

    print("TP pool:", len(tp_pool), "unique ids:", tp_pool["id"].nunique())
    print("TN pool:", len(tn_pool), "unique ids:", tn_pool["id"].nunique())

    if len(tp_pool) < N_POS + 3:
        raise ValueError(
            f"Positive pool has {len(tp_pool)} valid papers; UKB_VALIDATION_N_POS={N_POS} "
            "requires three additional papers for held-out prompt examples."
        )
    if len(tn_pool) < N_NEG + 2:
        raise ValueError(
            f"Negative pool has {len(tn_pool)} valid papers; UKB_VALIDATION_N_NEG={N_NEG} "
            "requires two additional papers for held-out prompt examples."
        )

    tp_eval = tp_pool.sample(n=N_POS, random_state=SEED).copy()
    tn_eval = tn_pool.sample(n=N_NEG, random_state=SEED).copy()

    eval_df = pd.concat([tp_eval, tn_eval], ignore_index=True)
    eval_df = eval_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    eval_df["True_label"] = eval_df["label"].astype(bool)

    eval_path = os.path.join(OUT_DIR, f"ukb_eval_{N_POS}pos_{N_NEG}neg.csv")
    eval_df.to_csv(eval_path, index=False)

    print("Eval set:", eval_df.shape)
    print(eval_df["label"].value_counts())
    print("Saved eval set:", eval_path)
    display(eval_df.head(10))

    # ============================================================
    # Few-shot example selection
    # ============================================================

    def contains_ukb_text(row):
        text = f"{row.get('title','')} {row.get('abstract','')}".lower()
        return any(x in text for x in [
            "uk biobank", "ukb", "ukbb", "united kingdom biobank", "uk biobank resource"
        ])

    def contains_hard_negative_cue(row):
        text = f"{row.get('title','')} {row.get('abstract','')}".lower()
        cues = [
            "unlike uk biobank", "compared with uk biobank", "such as uk biobank",
            "including uk biobank", "biobanks such as", "ethical", "governance",
            "china kadoorie", "biobank japan", "finnish biobank", "fingen", "finngen",
            "all of us", "janus serum bank", "copenhagen hospital biobank"
        ]
        return any(c in text for c in cues)

    eval_ids = set(eval_df["id"].astype(str))
    tp_demo_pool = tp_pool[~tp_pool["id"].astype(str).isin(eval_ids)].copy()
    tn_demo_pool = tn_pool[~tn_pool["id"].astype(str).isin(eval_ids)].copy()

    # Prefer one positive without explicit UKB text if available, because this is the hard recall case.
    tp_no_ukb = tp_demo_pool[~tp_demo_pool.apply(contains_ukb_text, axis=1)]
    tp_with_ukb = tp_demo_pool[tp_demo_pool.apply(contains_ukb_text, axis=1)]
    tn_hard = tn_demo_pool[tn_demo_pool.apply(contains_hard_negative_cue, axis=1)]

    one_pos = (tp_no_ukb if len(tp_no_ukb) > 0 else tp_demo_pool).sample(n=1, random_state=SEED + 1)
    one_neg = (tn_hard if len(tn_hard) > 0 else tn_demo_pool).sample(n=1, random_state=SEED + 2)

    # Five-shot: 3 positives + 2 negatives.
    tp_demo_1 = (tp_no_ukb if len(tp_no_ukb) >= 1 else tp_demo_pool).sample(n=1, random_state=SEED + 3)
    tp_demo_2 = (tp_with_ukb if len(tp_with_ukb) >= 2 else tp_demo_pool).sample(n=2, random_state=SEED + 4)
    tn_demo_1 = (tn_hard if len(tn_hard) >= 1 else tn_demo_pool).sample(n=1, random_state=SEED + 5)
    tn_demo_2 = tn_demo_pool.sample(n=1, random_state=SEED + 6)

    fewshot_examples = []
    for _, r in pd.concat([tp_demo_1, tp_demo_2]).iterrows():
        fewshot_examples.append({"title": r["title"], "abstract": r["abstract"], "label": True})
    for _, r in pd.concat([tn_demo_1, tn_demo_2]).iterrows():
        fewshot_examples.append({"title": r["title"], "abstract": r["abstract"], "label": False})

    random.Random(SEED).shuffle(fewshot_examples)

    print("\nOne-shot positive example:")
    display(one_pos[["id", "title", "abstract", "label"]].head(1))

    print("\nOne-shot negative example:")
    display(one_neg[["id", "title", "abstract", "label"]].head(1))

    print("\nFew-shot examples:")
    for ex in fewshot_examples:
        print(ex["label"], ex["title"][:120])

    # ============================================================
    # Prompts
    # ============================================================

    def truncate_text(s, max_chars=3500):
        s = str(s or "")
        return s[:max_chars]

    def json_instruction():
        return """Return strict JSON only:
    {"implies_UKB_use":true|false}
    Do not include explanation, markdown, or any extra keys."""

    def make_prompt_v1_conservative(title, abstract):
        return f"""You will be given a scientific paper title and abstract.

    Task: decide whether the paper used UK Biobank data or resources.

    Definition:
    - true: the study used UK Biobank data, participants, samples, imaging, genetics, linked health records, or another UK Biobank resource.
    - false: the paper only mentions UK Biobank, discusses biobanks generally, compares with UK Biobank, or uses other biobanks but not UK Biobank.

    Be conservative. If uncertain, return false.

    {json_instruction()}

    title: \"\"\"{truncate_text(title, 700)}\"\"\"

    abstract: \"\"\"{truncate_text(abstract, 3200)}\"\"\"

    JSON:
    """

    def make_prompt_v2_balanced(title, abstract):
        return f"""You will be given a scientific paper title and abstract.

    Task: decide whether the paper likely used UK Biobank data or resources.

    Important:
    - Some true UK Biobank-use papers do not mention "UK Biobank" in the abstract.
    - The paper may still use UK Biobank if the abstract describes a UK population-scale cohort, genetic/imaging/health-record analysis, or data-resource use that is consistent with UK Biobank.
    - Do not require explicit words "UK Biobank" if the evidence strongly suggests use.

    Return true when the title/abstract provides reasonable evidence that the paper analysed UK Biobank participants, data, samples, imaging, genetics, linked records, or a UK Biobank-derived cohort.

    Return false when the paper is only about generic biobanking, ethics/governance, reviews, comparisons, or another named biobank.

    {json_instruction()}

    title: \"\"\"{truncate_text(title, 700)}\"\"\"

    abstract: \"\"\"{truncate_text(abstract, 3200)}\"\"\"

    JSON:
    """

    def make_prompt_v3_evidence_cues(title, abstract):
        return f"""Classify whether this paper uses UK Biobank.

    Use the following cues.

    Positive evidence can include:
    - explicit UK Biobank / UKB / UKBB mention;
    - analysis of a very large UK cohort with genetic, imaging, health-record, lifestyle, biomarker, or hospital-linked data;
    - phrases such as participants, cohort, baseline assessment, imaging assessment, genotyping, exome sequencing, linked health records, Hospital Episode Statistics, or Townsend deprivation index in a UK population context;
    - a study design that clearly analyses participant-level data rather than merely discussing biobanks.

    Negative evidence can include:
    - generic discussion of biobanks;
    - ethics, governance, consent, infrastructure, sample storage, or review articles;
    - use of another biobank only;
    - mentions like "such as UK Biobank", "unlike UK Biobank", or comparison with UK Biobank.

    Prefer true if the abstract strongly looks like an original analysis using UK Biobank-style data, even if UK Biobank is not named in the abstract.
    Prefer false if the abstract is generic or only about other biobanks.

    {json_instruction()}

    title: \"\"\"{truncate_text(title, 700)}\"\"\"

    abstract: \"\"\"{truncate_text(abstract, 3200)}\"\"\"

    JSON:
    """

    def make_prompt_v4_context_no_shot(title, abstract):
        return f"""You will classify a paper using only its title and abstract.

    Context:
    All papers in this evaluation were retrieved because their full text matched at least one UK Biobank-related query. Therefore, UK Biobank may be mentioned only outside the abstract.

    Question:
    Based on the title and abstract, is it likely that the paper used UK Biobank data/resources in its own analysis?

    Label true if likely UK Biobank use.
    Label false if the paper likely only mentions UK Biobank, discusses biobanks generally, or uses other non-UKB resources.

    Do not be overly strict: if the paper is an original epidemiological, genetic, imaging, biomarker, or clinical-risk study and the abstract strongly suggests use of a UK population-scale linked cohort, return true.

    {json_instruction()}

    title: \"\"\"{truncate_text(title, 700)}\"\"\"

    abstract: \"\"\"{truncate_text(abstract, 3200)}\"\"\"

    JSON:
    """

    def make_prompt_v5_one_shot(title, abstract):
        pos = one_pos.iloc[0]
        neg = one_neg.iloc[0]

        return f"""You will be given a scientific paper title and abstract.

    Task:
    Decide whether the paper likely used UK Biobank data/resources in its own analysis.

    Context:
    All papers in this evaluation were retrieved because their full text matched a UK Biobank-related search. Some true positives may not mention UK Biobank in the abstract.

    {json_instruction()}

    Example positive:
    title: \"\"\"{truncate_text(pos["title"], 500)}\"\"\"
    abstract: \"\"\"{truncate_text(pos["abstract"], 1800)}\"\"\"
    answer: {{"implies_UKB_use":true}}

    Example negative:
    title: \"\"\"{truncate_text(neg["title"], 500)}\"\"\"
    abstract: \"\"\"{truncate_text(neg["abstract"], 1800)}\"\"\"
    answer: {{"implies_UKB_use":false}}

    Now classify this paper.

    title: \"\"\"{truncate_text(title, 700)}\"\"\"

    abstract: \"\"\"{truncate_text(abstract, 3200)}\"\"\"

    JSON:
    """

    def make_prompt_v6_five_shot(title, abstract):
        examples_txt = []
        for i, ex in enumerate(fewshot_examples, start=1):
            examples_txt.append(
                f"""Example {i}
    title: \"\"\"{truncate_text(ex["title"], 450)}\"\"\"
    abstract: \"\"\"{truncate_text(ex["abstract"], 1300)}\"\"\"
    answer: {{"implies_UKB_use":{str(ex["label"]).lower()}}}
    """
            )

        examples_block = "\n".join(examples_txt)

        return f"""You will be given a scientific paper title and abstract.

    Task:
    Decide whether the paper likely used UK Biobank data/resources in its own analysis.

    Context:
    All papers in this evaluation were retrieved because their full text matched a UK Biobank-related search.
    Some true UK Biobank-use papers may not mention UK Biobank in the abstract.
    Some false positives mention biobanks, UK cohorts, or other biobanks but do not use UK Biobank.

    Return true when the paper likely analysed UK Biobank participants, data, samples, imaging, genetics, linked health records, or other UK Biobank resources.
    Return false for generic biobank discussion, ethics/governance, reviews, comparisons, or other-biobank-only papers.

    {json_instruction()}

    Few-shot examples:
    {examples_block}

    Now classify this paper.

    title: \"\"\"{truncate_text(title, 700)}\"\"\"

    abstract: \"\"\"{truncate_text(abstract, 3200)}\"\"\"

    JSON:
    """

    PROMPT_BUILDERS = {
        "p1_conservative": make_prompt_v1_conservative,
        "p2_balanced": make_prompt_v2_balanced,
        "p3_evidence_cues": make_prompt_v3_evidence_cues,
        "p4_context_no_shot": make_prompt_v4_context_no_shot,
        "p5_real_one_shot": make_prompt_v5_one_shot,
        "p6_real_five_shot": make_prompt_v6_five_shot,
    }

    # ============================================================
    # Parsing and metrics
    # ============================================================

    def extract_first_json_obj(text):
        raw = (text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
        try:
            return json.loads(raw)
        except Exception:
            pass

        m = re.search(r"\{.*?\}", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    def parse_llm_result(obj):
        if not isinstance(obj, dict):
            return None

        val = obj.get("implies_UKB_use", None)

        if isinstance(val, bool):
            return val

        if isinstance(val, str):
            v = val.strip().lower()
            if v in {"true", "yes", "1"}:
                return True
            if v in {"false", "no", "0"}:
                return False

        return None

    def build_model_input(tokenizer, user_text):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            msgs = [{"role": "user", "content": user_text}]
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                return user_text
        return user_text

    def metric_dict(y_true, y_pred):
        y_true = np.asarray(y_true).astype(bool)
        y_pred = np.asarray(y_pred).astype(bool)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }

    def pairwise_agreement_matrix(pred_df, model_cols):
        agree = pd.DataFrame(index=model_cols, columns=model_cols, dtype=float)
        nmat = pd.DataFrame(index=model_cols, columns=model_cols, dtype=int)

        for a in model_cols:
            for b in model_cols:
                m = pred_df[[a, b]].dropna()
                n = len(m)
                nmat.loc[a, b] = n
                agree.loc[a, b] = (m[a].astype(bool).values == m[b].astype(bool).values).mean() if n else np.nan

        return agree, nmat

    # ============================================================
    # LLM loading and prediction
    # ============================================================

    def load_llm(model_id, tag):
        token = os.getenv("HF_TOKEN", None)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=token,
            trust_remote_code=True,
        )

        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = None
        dtype = torch.float16

        if USE_4BIT and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        kwargs = dict(
            device_map="auto",
            torch_dtype=dtype,
            token=token,
            trust_remote_code=True,
        )

        if quant_config is not None:
            kwargs["quantization_config"] = quant_config

        # Phi-specific fix: use Phi-3.5 and eager attention.
        # This avoids the common Phi-3 rope/config/flash-attention loading issue.
        if "phi" in tag.lower():
            kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()

        return tokenizer, model

    def run_llm_for_prompt(model_id, tag, prompt_name, prompt_fn, df, batch_size=8, max_input_tokens=4096, max_new_tokens=64):
        print("\n" + "=" * 100)
        print(f"LLM: {tag} | Prompt: {prompt_name}")
        print(f"Model: {model_id}")
        print("=" * 100)

        tokenizer, model = load_llm(model_id, tag)

        n = len(df)
        preds = np.full(n, np.nan, dtype=object)
        raw_outputs = [""] * n
        parse_ok = np.zeros(n, dtype=bool)

        t0 = time.time()

        for s in tqdm(range(0, n, batch_size), desc=f"{tag}-{prompt_name}"):
            e = min(s + batch_size, n)
            batch = df.iloc[s:e]

            prompts = [
                build_model_input(tokenizer, prompt_fn(r["title"], r["abstract"]))
                for _, r in batch.iterrows()
            ]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            ).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            gen_only = output_ids[:, prompt_len:]
            texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            for i, txt in enumerate(texts):
                idx = s + i
                raw_outputs[idx] = txt
                obj = extract_first_json_obj(txt)
                val_out = parse_llm_result(obj)

                if val_out is not None:
                    preds[idx] = bool(val_out)
                    parse_ok[idx] = True

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        runtime = time.time() - t0

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return preds, raw_outputs, {
            "model": tag,
            "prompt": prompt_name,
            "type": "LLM",
            "runtime_s": runtime,
            "items_per_sec": n / max(runtime, 1e-9),
            "parse_rate": parse_ok.mean(),
            "n_parsed": int(parse_ok.sum()),
        }

    # ============================================================
    # Encoder baselines
    # ============================================================

    def mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def best_threshold(scores, labels):
        labels = np.asarray(labels).astype(bool)
        scores = np.asarray(scores)
        thresholds = np.linspace(scores.min(), scores.max(), 200)

        best_f1 = -1
        best_t = None

        for t in thresholds:
            pred = scores >= t
            f1 = f1_score(labels, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        return float(best_t), float(best_f1)

    def run_scibert_baseline(df):
        tag = "scibert_sim"
        model_id = "allenai/scibert_scivocab_uncased"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("\n" + "=" * 100)
        print("Encoder baseline:", tag)
        print("=" * 100)

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModel.from_pretrained(model_id).to(device).eval()

        query = "This scientific paper uses UK Biobank data or resources."
        q = tok([query], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.inference_mode():
            qout = model(**q)
            qemb = mean_pool(qout.last_hidden_state, q["attention_mask"])
            qemb = torch.nn.functional.normalize(qemb, dim=1)

        scores = np.zeros(len(df))
        texts = (df["title"].fillna("") + "\n" + df["abstract"].fillna("")).tolist()

        t0 = time.time()
        batch_size = 64

        for s in tqdm(range(0, len(texts), batch_size), desc=tag):
            e = min(s + batch_size, len(texts))
            inp = tok(texts[s:e], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            with torch.inference_mode():
                out = model(**inp)
                emb = mean_pool(out.last_hidden_state, inp["attention_mask"])
                emb = torch.nn.functional.normalize(emb, dim=1)
                scores[s:e] = (emb @ qemb.T).squeeze(1).detach().cpu().numpy()

        runtime = time.time() - t0

        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        thr, calib_f1 = best_threshold(scores, df["True_label"].values)
        preds = scores >= thr

        meta = {
            "model": tag,
            "prompt": "encoder_similarity",
            "type": "Encoder",
            "runtime_s": runtime,
            "items_per_sec": len(df) / max(runtime, 1e-9),
            "threshold": thr,
            "calib_f1_all_eval": calib_f1,
        }

        return preds, scores, meta

    def run_sbert_baseline(df):
        tag = "sbert_minilm_sim"
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("\n" + "=" * 100)
        print("Encoder baseline:", tag)
        print("=" * 100)

        model = SentenceTransformer(model_id, device=device)

        query = "This scientific paper uses UK Biobank data or resources."
        q_emb = model.encode([query], normalize_embeddings=True)

        texts = (df["title"].fillna("") + "\n" + df["abstract"].fillna("")).tolist()

        t0 = time.time()
        emb = model.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
        runtime = time.time() - t0

        scores = emb @ q_emb[0]

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        thr, calib_f1 = best_threshold(scores, df["True_label"].values)
        preds = scores >= thr

        meta = {
            "model": tag,
            "prompt": "encoder_similarity",
            "type": "Encoder",
            "runtime_s": runtime,
            "items_per_sec": len(df) / max(runtime, 1e-9),
            "threshold": thr,
            "calib_f1_all_eval": calib_f1,
        }

        return preds, scores, meta

    # ============================================================
    # Models
    # ============================================================

    LLM_SPECS = [
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2_5_7b", 8),
        ("meta-llama/Meta-Llama-3-8B-Instruct", "llama3_8b", 8),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral_7b", 8),
        ("microsoft/Phi-3.5-mini-instruct", "phi3_5_mini", 16),
        ("HuggingFaceH4/zephyr-7b-beta", "zephyr_7b", 8),
    ]

    if not os.getenv("HF_TOKEN", "").strip():
        LLM_SPECS = [x for x in LLM_SPECS if "llama" not in x[1]]
        print("Skipping Llama-3 because no HF token/access was detected.")

    print("LLMs to run:")
    for model_id, tag, bs in LLM_SPECS:
        print(tag, model_id, "batch_size=", bs)

    # ============================================================
    # Run prompt-by-prompt
    # ============================================================

    all_results = []
    all_prediction_files = []

    y_true = eval_df["True_label"].astype(bool).values

    for prompt_name, prompt_fn in PROMPT_BUILDERS.items():
        print("\n\n" + "#" * 120)
        print(f"RUNNING PROMPT: {prompt_name}")
        print("#" * 120)

        preds_wide = eval_df[["id", "title", "abstract", "year", "True_label"]].copy()
        raw_df = eval_df[["id", "True_label"]].copy()
        prompt_results = []

        for model_id, tag, batch_size in LLM_SPECS:
            try:
                preds, raw_outputs, meta = run_llm_for_prompt(
                    model_id=model_id,
                    tag=tag,
                    prompt_name=prompt_name,
                    prompt_fn=prompt_fn,
                    df=eval_df,
                    batch_size=batch_size,
                    max_input_tokens=4096,
                    max_new_tokens=80,
                )

                preds_wide[tag] = preds
                raw_df[f"{tag}_raw"] = raw_outputs

                mask = pd.notna(preds)
                if mask.sum() > 0:
                    metrics = metric_dict(y_true[mask], np.asarray(preds[mask], dtype=bool))
                else:
                    metrics = {
                        "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan,
                        "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                    }

                row = {**meta, **metrics}
                prompt_results.append(row)
                all_results.append(row)

                print(f"\nResult for {tag} / {prompt_name}")
                display(pd.DataFrame([row]))

            except Exception as e:
                print(f"[ERROR] {tag} failed for {prompt_name}: {type(e).__name__}: {e}")
                row = {
                    "model": tag,
                    "prompt": prompt_name,
                    "type": "LLM",
                    "error": f"{type(e).__name__}: {e}",
                }
                prompt_results.append(row)
                all_results.append(row)

        try:
            scibert_preds, scibert_scores, meta = run_scibert_baseline(eval_df)
            preds_wide["scibert_sim"] = scibert_preds
            preds_wide["scibert_score"] = scibert_scores
            row = {**meta, **metric_dict(y_true, scibert_preds)}
            row["prompt"] = prompt_name
            prompt_results.append(row)
            all_results.append(row)
        except Exception as e:
            print("[ERROR] SciBERT failed:", e)

        try:
            sbert_preds, sbert_scores, meta = run_sbert_baseline(eval_df)
            preds_wide["sbert_minilm_sim"] = sbert_preds
            preds_wide["sbert_score"] = sbert_scores
            row = {**meta, **metric_dict(y_true, sbert_preds)}
            row["prompt"] = prompt_name
            prompt_results.append(row)
            all_results.append(row)
        except Exception as e:
            print("[ERROR] S-BERT failed:", e)

        pred_path = os.path.join(OUT_DIR, f"predictions_{prompt_name}.csv")
        raw_path = os.path.join(OUT_DIR, f"raw_outputs_{prompt_name}.csv")
        res_path = os.path.join(OUT_DIR, f"results_{prompt_name}.csv")

        preds_wide.to_csv(pred_path, index=False)
        raw_df.to_csv(raw_path, index=False)

        res_df_prompt = pd.DataFrame(prompt_results)
        res_df_prompt.to_csv(res_path, index=False)

        print("\n=== Results for prompt:", prompt_name, "===")
        display(res_df_prompt.sort_values("accuracy", ascending=False, na_position="last"))

        model_cols = [
            c for c in preds_wide.columns
            if c not in ["id", "title", "abstract", "year", "True_label", "scibert_score", "sbert_score"]
        ]

        agreement, nmat = pairwise_agreement_matrix(preds_wide, model_cols)
        agreement_pct = (agreement * 100).round(1)

        print("\n=== Pairwise agreement (%) for prompt:", prompt_name, "===")
        display(agreement_pct)

        print("\n=== Pairwise compared-row counts for prompt:", prompt_name, "===")
        display(nmat)

        agreement_path = os.path.join(OUT_DIR, f"pairwise_agreement_percent_{prompt_name}.csv")
        nmat_path = os.path.join(OUT_DIR, f"pairwise_agreement_n_{prompt_name}.csv")

        agreement_pct.to_csv(agreement_path)
        nmat.to_csv(nmat_path)


        print("Saved:")
        print(pred_path)
        print(raw_path)
        print(res_path)
        print(agreement_path)
        print(nmat_path)

        all_prediction_files.append(pred_path)

    # ============================================================
    # Combined summary and final ranking
    # ============================================================

    all_results_df = pd.DataFrame(all_results)
    all_results_path = os.path.join(OUT_DIR, "ALL_prompt_model_results_summary.csv")
    all_results_df.to_csv(all_results_path, index=False)

    print("\nSaved combined results:", all_results_path)

    metric_cols = [
        "prompt", "model", "type",
        "accuracy", "precision", "recall", "f1",
        "tp", "tn", "fp", "fn",
        "parse_rate", "items_per_sec",
        "runtime_s", "error"
    ]
    existing_cols = [c for c in metric_cols if c in all_results_df.columns]

    ranked = (
        all_results_df[existing_cols]
        .sort_values(["accuracy", "f1", "precision", "recall"], ascending=[False, False, False, False], na_position="last")
        .reset_index(drop=True)
    )

    ranked_path = os.path.join(OUT_DIR, "ALL_prompt_model_results_ranked_by_accuracy.csv")
    ranked.to_csv(ranked_path, index=False)

    print("\n=== FINAL ORDERED RESULTS: highest accuracy to lowest ===")
    display(ranked)

    print("Saved ranked results:", ranked_path)

    # Optional: show best LLM-only results
    if "type" in ranked.columns:
        print("\n=== LLM-only ranking ===")
        display(ranked[ranked["type"].eq("LLM")].reset_index(drop=True))
