import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import ensure_real_path, make_text_for_analysis, savefig

UKB_PATTERN = re.compile(r"\b(uk\s*biobank|u\.?k\.?\s*biobank|united\s+kingdom\s+biobank|ukb|ukbb)\b", flags=re.I)


def run(combined_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_csv(combined_path)
    df["analysis_text"] = make_text_for_analysis(df)
    df["explicit_ukb_title_or_abstract"] = df["analysis_text"].apply(lambda x: bool(UKB_PATTERN.search(str(x))))
    summary = df.assign(binary_split=np.where(df["three_model_TRUE_agreement"], "Three-model TRUE agreement", "Rest: not three-model TRUE agreement")).groupby("binary_split").agg(
        n=("id", "count"),
        explicit_ukb=("explicit_ukb_title_or_abstract", "sum"),
    ).reset_index()
    summary["explicit_ukb_percent"] = summary["explicit_ukb"] / summary["n"].clip(lower=1) * 100
    summary.to_csv(os.path.join(output_dir, "explicit_ukb_mention_summary.csv"), index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(summary["binary_split"], summary["explicit_ukb_percent"])
    plt.ylabel("Explicit UKB mention in title/abstract (%)")
    plt.title("Explicit UK Biobank mention")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    for i, v in enumerate(summary["explicit_ukb_percent"]):
        plt.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    savefig(fig_dir, "explicit_ukb_mention_by_binary_split.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined_path", default="INPUT your path for three_model_combined_labels.csv here")
    p.add_argument("--output_dir", default="INPUT your path for analysis output folder here")
    a = p.parse_args()
    run(ensure_real_path(a.combined_path, "--combined_path"), ensure_real_path(a.output_dir, "--output_dir"))


if __name__ == "__main__":
    main()
