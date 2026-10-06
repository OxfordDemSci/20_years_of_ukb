import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import shared_patent_utils as patent
from utils import data_analysis_02_content_panels as content
from utils import data_analysis_04_non_academic_panels as non_academic


def test_collapsed_divisions_are_distinct_and_have_parent_names():
    categories = [
        {"name": "3101 Biochemistry and Cell Biology"},
        {"name": "31 Biological Sciences"},
        {"name": "3105 Genetics"},
        {"name": "4201 Allied Health and Rehabilitation Science"},
        {"name": "5110 Medical and Biological Physics"},
        {"name": "5203 Clinical and Health Psychology"},
        {"name": "No numeric classification"},
        None,
    ]
    assert patent.collapse_to_top_level(categories) == [
        ("31", "Biological Sciences"), ("42", "Health Sciences"),
        ("51", "Physical Sciences"), ("52", "Psychology"),
    ]
    assert patent.collapse_to_top_level([]) == []


def test_diversity_deduplicates_even_previously_cached_divisions():
    frame = pd.DataFrame({"top_level_topics": [
        [("31", "Child"), ("31", "Parent"), ("32", "Parent")], [], ["31", "31"],
    ]})
    assert patent.analyze_topic_diversity(frame)["topic_count"].tolist() == [2, 0, 1]


def test_absent_categories_have_no_rank_or_bridging_line():
    from utils.shared_style import load_style
    load_style("02_content")
    shares = pd.DataFrame({"One": [100., 50., 100.], "Two": [0., 50., 0.]},
                          index=[2013, 2014, 2015])
    fig, ax = plt.subplots()
    content.draw_rank_flow(ax, {"share": shares, "keep": ["One", "Two"]}, "A  Test")
    values = ax.lines[1].get_ydata()
    assert np.isnan(values[0]) and values[1] == 2 and np.isnan(values[2])
    assert not any("Two" in text.get_text() for text in ax.texts)
    plt.close(fig)


def test_levy_callout_uses_reference_year_without_changing_metadata():
    row = {"first_author": "Levy", "year": 2020,
           "doi_clean": "10.1016/j.clnu.2020.12.018"}
    assert non_academic._attention_paper_label(row) == "Levy et al. (2021)"
    assert row["year"] == 2020


def test_reach_legend_names_the_actual_news_and_company_series():
    from utils.shared_style import load_style
    load_style("04_non_academic_panels")
    data = pd.DataFrame({key: [1, 2] for key in non_academic.STREAM_ORDER},
                        index=[2024, 2025])
    fig, ax = plt.subplots()
    non_academic.draw_reach_by_year(ax, {"reach_by_year": data})
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "News mentions" in labels
    assert "Company affiliations" in labels
    assert "Non-academic collaboration" not in labels
    plt.close(fig)
