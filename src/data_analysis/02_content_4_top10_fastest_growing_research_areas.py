#!/usr/bin/env python3
"""Recreate the 'Top 10 Fastest Growing Areas of Research' horizontal bar chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from notebook_academic_collab_helpers import apply_project_plot_style


def build_data() -> pd.DataFrame:
    """Return chart data in descending order as shown in the source figure."""
    return pd.DataFrame(
        {
            "area": [
                "Bioinformatics and\nComputational Biology",
                "Human Geography",
                "Paediatrics",
                "Medical Biotechnology",
                "Econometrics",
                "Analytical Chemistry",
                "Distributed Computing/\nSystems",
                "Electrical Engineering",
                "Information Systems",
                "Atmospheric Sciences",
            ],
            "count": [111, 18, 18, 13, 6, 5, 5, 5, 4, 4],
        }
    )


def make_plot(output_path: Path) -> None:
    apply_project_plot_style()
    df = build_data()

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)

    bars = ax.barh(
        df["area"][::-1],
        df["count"][::-1],
        color="#1b82b1",
        edgecolor="black",
        linewidth=0.7,
        height=0.68,
    )

    ax.set_title("Top 10 Fastest Growing Areas of Research", fontsize=18, weight="bold", pad=18)
    ax.set_xlabel("Growth count", fontsize=11)
    ax.set_ylabel("")

    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", alpha=0.14)
    ax.tick_params(axis="x", labelbottom=True, length=3)
    ax.tick_params(axis="y", length=0, labelsize=10)

    xmax = int(df["count"].max() * 1.06)
    ax.set_xlim(0, xmax)

    for bar, value in zip(bars, df["count"][::-1]):
        ax.text(
            bar.get_width() + xmax * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    output_dir = Path("output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        output_dir / "top10_fastest_growing_research_areas_recreated.png",
        output_dir / "top10_fastest_growing_research_areas_recreated.pdf",
        output_dir / "top10_fastest_growing_research_areas_recreated.svg",
    ]

    for output_file in output_files:
        make_plot(output_file)
        print(f"Saved figure to: {output_file}")
