import pandas as pd

from utils import data_analysis_05_author_characteristics as authors


def _author_row(researcher_id, name, h_index, papers, citations):
    row = {
        column: 0
        for column in authors.TOP_AUTHOR_TABLE_COLUMNS
        if column != "rank"
    }
    row.update(
        {
            "researcher_id": researcher_id,
            "full_name": name,
            "orcid": None,
            "ukb_h_index": h_index,
            "n_ukb_papers": papers,
            "total_ukb_citations": citations,
            "mean_ukb_citations": 12.345,
            "home_institution": "Example University",
            "home_country": "United Kingdom",
            "modal_for_l2": "Health Sciences",
            "modal_for_l4": "Epidemiology",
        }
    )
    return row


def test_top_authors_are_ranked_deterministically_and_have_compact_view():
    metrics = pd.DataFrame(
        [
            _author_row("lower_h", "Lower H", 9, 50, 500),
            _author_row("fewer_papers", "Fewer Papers", 10, 15, 500),
            _author_row("lower_citations", "Lower Citations", 10, 20, 800),
            _author_row("zulu", "Zulu Author", 10, 20, 900),
            _author_row("alpha", "Alpha Author", 10, 20, 900),
        ]
    )

    result = authors.top_authors_by_ukb_h_index(metrics, n=3)

    assert result["researcher_id"].tolist() == [
        "alpha",
        "zulu",
        "lower_citations",
    ]
    assert result["rank"].tolist() == [1, 2, 3]
    assert tuple(result.columns) == authors.TOP_AUTHOR_TABLE_COLUMNS

    view = authors.top_authors_notebook_view(result)
    assert view.index.name == "Author"
    assert view.index.tolist() == ["Alpha Author", "Zulu Author", "Lower Citations"]
    assert view.columns[:4].tolist() == [
        "Rank",
        "UKB h-index",
        "UKB papers",
        "Total UKB citations",
    ]
    assert view.loc["Alpha Author", "Mean citations per paper"] == 12.3


def test_standalone_ranking_includes_2013_and_keeps_author_identities_separate():
    first = {"researcher_id": "a", "first_name": "Sam", "last_name": "Smith"}
    second = {"researcher_id": "b", "first_name": "Sam", "last_name": "Smith"}
    unresolved = {"first_name": "Unresolved"}
    papers = pd.DataFrame([
        {"id": "p1", "year": 2013, "times_cited": 10, "authors": [first, first, second, unresolved]},
        {"id": "p2", "year": 2014, "times_cited": 1, "authors": [first]},
        {"id": "p3", "year": 2025, "times_cited": 10, "authors": [first, second]},
        {"id": "p4", "year": 2026, "times_cited": 999, "authors": [first]},
        {"id": "p5", "year": 2012, "times_cited": 999, "authors": [first]},
    ])
    result = authors.build_top_author_publication_metrics(papers)
    assert result["researcher_id"].tolist() == ["a", "b"]
    assert result["ukb_h_index"].tolist() == [2, 2]
    assert result["ukb_i10_index"].tolist() == [2, 2]
    assert result["n_ukb_papers"].tolist() == [3, 2]
    assert result["total_ukb_citations"].tolist() == [21, 20]
    assert abs(result.loc[0, "fractional_paper_credit"] - (1/3 + 1 + 1/2)) < 1e-12
