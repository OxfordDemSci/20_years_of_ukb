import argparse
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

from common import ensure_real_path, make_text_for_analysis, savefig

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


def contains_pattern(text, pattern):
    return bool(re.search(pattern, str(text), flags=re.I))


def run(three_true_path, rest_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    three_true = pd.read_csv(three_true_path)
    rest = pd.read_csv(rest_path)
    rows = []
    for group_name, sub in [("Three-model TRUE agreement", three_true), ("Rest: not three-model TRUE agreement", rest)]:
        texts = make_text_for_analysis(sub)
        for cat, pat in CATEGORY_PATTERNS.items():
            hits = texts.apply(lambda x: contains_pattern(x, pat))
            rows.append({
                "binary_split": group_name,
                "category": cat,
                "n_group": len(sub),
                "n_with_category": int(hits.sum()),
                "percent_with_category": round(hits.mean() * 100, 4) if len(sub) else 0,
            })
    category_df = pd.DataFrame(rows)
    category_df.to_csv(os.path.join(output_dir, "keyword_category_summary.csv"), index=False)
    pivot = category_df.pivot(index="category", columns="binary_split", values="percent_with_category").fillna(0)
    true_col = "Three-model TRUE agreement"
    rest_col = "Rest: not three-model TRUE agreement"
    pivot = pivot.sort_values(true_col, ascending=True)

    plt.figure(figsize=(10, 8))
    y = range(len(pivot))
    width = 0.38
    plt.barh([v - width / 2 for v in y], pivot[true_col], height=width, label=true_col)
    plt.barh([v + width / 2 for v in y], pivot[rest_col], height=width, label=rest_col)
    plt.yticks(list(y), pivot.index)
    plt.xlabel("Percentage of papers with cue (%)")
    plt.title("Keyword/category profile")
    plt.grid(axis="x", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    savefig(fig_dir, "keyword_category_profile_by_binary_split.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--three_true_path", default="INPUT your path for three_model_TRUE_agreement.csv here")
    p.add_argument("--rest_path", default="INPUT your path for rest_NOT_three_model_TRUE_agreement.csv here")
    p.add_argument("--output_dir", default="INPUT your path for analysis output folder here")
    a = p.parse_args()
    run(
        ensure_real_path(a.three_true_path, "--three_true_path"),
        ensure_real_path(a.rest_path, "--rest_path"),
        ensure_real_path(a.output_dir, "--output_dir"),
    )


if __name__ == "__main__":
    main()
