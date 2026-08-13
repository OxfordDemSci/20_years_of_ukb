from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import shared_paths as P


UKB_PATTERN = re.compile(
    r"\b(uk\s*biobank|u\.?k\.?\s*biobank|united\s+kingdom\s+biobank|ukb|ukbb)\b",
    flags=re.I,
)

CATEGORY_PATTERNS = {
    "explicit UKB": r"\b(uk\s*biobank|u\.?k\.?\s*biobank|united\s+kingdom\s+biobank|ukb|ukbb)\b",
    "generic biobank": r"\b(biobank|biobanking|biobanks)\b",
    "UK population cue": r"\b(uk|british|england|scotland|wales|united kingdom)\b",
    "cohort / participants": r"\b(cohort|participants|population[- ]based|prospective|baseline assessment)\b",
    "genetics / genomics": r"\b(genetic|genomic|genotype|genotyping|gwas|polygenic|exome|sequencing)\b",
    "imaging": r"\b(imaging|mri|brain imaging|cardiac imaging|radiomics)\b",
    "linked records / EHR": r"\b(linked records|hospital episode statistics|hes|electronic health records|ehr|registry|registries)\b",
    "machine learning": r"\b(machine learning|deep learning|artificial intelligence|neural network|prediction model)\b",
    "cardiometabolic": r"\b(cardiovascular|heart|diabetes|obesity|metabolic|hypertension)\b",
    "cancer": r"\b(cancer|tumou?r|oncology|carcinoma|neoplasm)\b",
    "mental health / brain": r"\b(depression|anxiety|psychiatric|mental health|brain|cognition|dementia)\b",
    "other named biobank": r"\b(china kadoorie|biobank japan|finn?gen|all of us|million veteran|lifelines|decode|cartagene)\b",
}

MODEL_NAMES = ("qwen", "llama3_8b", "mistral_7b")


def load_publications(path: Path = P.SHOWCASE_PLUS) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"id", "title"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing required publication columns: {sorted(missing)}")
    frame = frame.copy()
    if "abstract" not in frame:
        frame["abstract"] = ""
    if "year" not in frame:
        frame["year"] = pd.NA
    frame["analysis_year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["analysis_text"] = make_analysis_text(frame)
    frame["explicit_ukb_mention"] = frame["analysis_text"].str.contains(UKB_PATTERN, na=False)
    return frame


def make_analysis_text(frame: pd.DataFrame) -> pd.Series:
    title = frame["title"].fillna("").astype(str).str.strip()
    abstract = frame["abstract"].fillna("").astype(str).str.strip()
    return (title + "\n" + abstract).str.strip()


def output_dirs(name: str) -> tuple[Path, Path]:
    table_dir = P.OUTPUT_TABLES / name
    figure_dir = P.FIG_DATA_ANALYSIS / name
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir


def save_figure(figure: plt.Figure, path: Path, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def contains_pattern(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, str(text), flags=re.I))


def normalise_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    values = series.astype(str).str.strip().str.lower()
    return values.map(
        {
            "true": True,
            "1": True,
            "1.0": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "0.0": False,
            "no": False,
            "n": False,
            "nan": False,
            "none": False,
            "<na>": False,
            "": False,
        }
    ).fillna(False).astype(bool)


def model_agreement_columns() -> set[str]:
    columns = {
        "n_true_votes",
        "consensus_group",
        "three_model_TRUE_agreement",
        "three_model_FALSE_agreement",
        "all_three_parsed",
    }
    for model in MODEL_NAMES:
        columns.update(
            {
                f"{model}_parse_ok",
                f"{model}_true",
                f"{model}_false",
                f"{model}_label",
            }
        )
    return columns


def parse_json_cell(value):
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def sample_balanced(frame: pd.DataFrame, group_column: str, maximum: int, seed: int = 42) -> pd.DataFrame:
    samples = []
    for _, group in frame.groupby(group_column, dropna=False):
        samples.append(group.sample(n=min(maximum, len(group)), random_state=seed))
    return pd.concat(samples, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
