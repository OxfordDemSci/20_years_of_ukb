"""Keep the geography supplement square in both layout and exported output."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from utils import data_analysis_05_author_plots as plots
from utils import shared_style


def test_geography_metrics_square_export_and_unclipped_labels(tmp_path, monkeypatch):
    names = ["United States", "China", "United Kingdom", "Australia",
             "Turkey", "Lithuania", "Nigeria", "Serbia"]
    counts = [9049, 8000, 7500, 1000, 58, 27, 25, 16]
    org_counts = [8797, 7900, 7400, 995, 48, 23, 20, 13]
    countries = pd.DataFrame({
        "country": names, "iso3": names,
        "fractional_paper_credit": counts,
        "author_basis_unique_papers": counts,
        "org_basis_unique_papers": org_counts,
    })
    years = list(range(plots.A.FIRST_YEAR, plots.A.LAST_COMPLETE_YEAR + 1))
    core = SimpleNamespace(
        country_metrics=countries,
        country_credits=pd.DataFrame([
            {"year": year, "iso3": name, "country": name, "credit": count}
            for year in years for name, count in zip(names, counts)
        ]),
        country_by_year=pd.DataFrame({"year": years, "effective_entities": 5}),
    )
    style = dict(shared_style.load_style("05_author_characteristics"),
                 save=True, savedir=str(tmp_path), formats=["png"])
    monkeypatch.setattr(shared_style, "PNG_DPI", 72)
    fig, paths = plots.plot_geography_metrics_supplement(core, style)
    try:
        assert tuple(fig.get_size_inches()) == (14.5, 14.5)
        assert len(fig.axes) == 4
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= 14.5 and bounds.y1 <= 14.5
        for ax in fig.axes:
            box = ax.get_window_extent()
            assert 0.9 <= box.width / box.height <= 1.1
        with Image.open(paths[0]) as exported:
            assert exported.size == (1044, 1044)
    finally:
        plt.close(fig)
