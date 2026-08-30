import pandas as pd

from utils import shared_name_gender as gender


def test_mostly_label_is_collapsed_in_expanded_rule():
    result = gender.infer_name_gender("Sasha")

    assert result.dictionary_detail == "mostly_male"
    assert result.strict_category == "Unknown"
    assert result.dictionary_category == "Male"


def test_offline_ensemble_recovers_dictionary_unknown():
    result = gender.infer_name_gender("Wei")

    assert result.dictionary_category == "Unknown"
    assert result.category == "Male"
    assert result.method == "offline_ensemble_consensus"
    assert result.vote_count == 2


def test_unresolved_package_conflict_remains_unknown():
    result = gender.infer_name_gender("Sasha")

    assert result.category == "Unknown"
    assert result.conflict
    assert result.unknown_reason == "offline_library_conflict"


def test_identity_linkage_fills_only_unanimous_evidence():
    authorships = pd.DataFrame(
        {
            "first_name": ["Alice", "A.", "Alice", "James"],
            "researcher_id": ["same", "same", "conflict", "conflict"],
            "identity_resolved": [True, True, True, True],
        }
    )

    result = gender.classify_authorship_names(authorships)

    assert result.loc[1, "name_gender"] == "Female"
    assert result.loc[1, "name_gender_method"] == "researcher_identity_consensus"
    assert result.loc[2:3, "name_gender"].eq("Unknown").all()
    assert result.loc[2:3, "name_gender_identity_conflict"].all()


def test_researcher_list_features_preserve_nested_names_and_identity_consensus():
    researchers = pd.Series(
        [
            [
                {
                    "first_name": "Alice",
                    "last_name": "Smith",
                    "researcher_id": "same",
                },
                {
                    "first_name": "A.",
                    "last_name": "Smith",
                    "researcher_id": "same",
                },
            ],
            None,
            [{"first_name": "James", "last_name": "Jones", "id": "other"}],
        ],
        index=[10, 10, 30],
    )

    result = gender.researcher_list_features(researchers)

    assert result.iloc[0]["researcher_names"] == ["Alice Smith", "A. Smith"]
    assert result.iloc[0]["name_gender_categories"] == ["Female", "Female"]
    assert result.iloc[1]["researcher_names"] == []
    assert result.iloc[1]["name_gender_categories"] == []
    assert result.iloc[2]["name_gender_categories"] == ["Male"]


def test_nested_category_summary_keeps_denominator_explicit():
    categories = pd.Series([["Female", "Unknown"], ["Male", "mostly_female"], None])

    summary = gender.summarize_name_categories(categories)

    assert summary.total == 4
    assert summary.classified == 3
    assert summary.unknown == 1
    assert summary.coverage_percent == 75.0
    assert gender.female_name_percentage(categories, denominator="all") == 50.0
    assert round(
        gender.female_name_percentage(categories, denominator="classified"), 6
    ) == round(200 / 3, 6)


def test_researcher_list_features_can_skip_inference():
    researchers = pd.Series(
        [[{"first_name": "Alice", "last_name": "Smith"}]], index=[42]
    )

    result = gender.researcher_list_features(researchers, infer_name_categories=False)

    assert result.loc[42, "researcher_names"] == ["Alice Smith"]
    assert result.loc[42, "name_gender_categories"] == []


def test_all_pinned_libraries_are_available_offline():
    versions = gender.offline_library_versions()

    assert len(versions) == 5
    assert versions["available"].all()
    assert versions["inference_mode"].eq("offline bundled data").all()
