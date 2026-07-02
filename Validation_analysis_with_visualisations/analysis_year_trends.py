import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

from common import ensure_real_path, savefig


def run(combined_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_csv(combined_path).dropna(subset=["year_int"]).copy()
    df["year_int"] = df["year_int"].astype(int)
    yearly = df.groupby("year_int").agg(
        total_candidates=("id", "count"),
        three_model_TRUE_agreement=("three_model_TRUE_agreement", "sum"),
        three_model_FALSE_agreement=("three_model_FALSE_agreement", "sum"),
        qwen_TRUE=("qwen_true", "sum"),
        llama3_8b_TRUE=("llama3_8b_true", "sum"),
        mistral_7b_TRUE=("mistral_7b_true", "sum"),
    ).reset_index()
    yearly["rest_NOT_three_model_TRUE_agreement"] = yearly["total_candidates"] - yearly["three_model_TRUE_agreement"]
    yearly["three_model_TRUE_agreement_rate_percent"] = yearly["three_model_TRUE_agreement"] / yearly["total_candidates"].clip(lower=1) * 100
    yearly.to_csv(os.path.join(output_dir, "yearly_three_model_consensus_trend.csv"), index=False)

    plt.figure(figsize=(11, 5.8))
    plt.plot(yearly["year_int"], yearly["three_model_TRUE_agreement"], marker="o", label="Three-model TRUE agreement")
    plt.plot(yearly["year_int"], yearly["rest_NOT_three_model_TRUE_agreement"], marker="o", label="Rest")
    plt.yscale("log")
    plt.xlabel("Publication year")
    plt.ylabel("Number of candidates, log scale")
    plt.title("Yearly trend")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    savefig(fig_dir, "yearly_trend_three_TRUE_vs_rest_logscale.png")

    plt.figure(figsize=(11, 5.8))
    plt.plot(yearly["year_int"], yearly["three_model_TRUE_agreement_rate_percent"], marker="o")
    plt.xlabel("Publication year")
    plt.ylabel("Three-model TRUE agreement rate (%)")
    plt.title("Yearly three-model TRUE agreement rate")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    savefig(fig_dir, "yearly_three_model_TRUE_agreement_rate.png")

    plt.figure(figsize=(11, 5.8))
    for col, label in [("qwen_TRUE", "Qwen"), ("llama3_8b_TRUE", "Llama3-8B"), ("mistral_7b_TRUE", "Mistral-7B")]:
        plt.plot(yearly["year_int"], yearly[col], marker="o", label=label)
    plt.xlabel("Publication year")
    plt.ylabel("Number of TRUE predictions")
    plt.title("Yearly positive predictions by model")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    savefig(fig_dir, "yearly_model_TRUE_counts.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined_path", default="INPUT your path for three_model_combined_labels.csv here")
    p.add_argument("--output_dir", default="INPUT your path for analysis output folder here")
    a = p.parse_args()
    run(ensure_real_path(a.combined_path, "--combined_path"), ensure_real_path(a.output_dir, "--output_dir"))


if __name__ == "__main__":
    main()
