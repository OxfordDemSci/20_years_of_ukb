"""Patent topics are counted and labelled as distinct ANZSRC 2020 divisions (review item 1.3).

The supplementary patent figures once labelled division 31 "Biochemistry and Cell Biology"
(the first child group seen) and drew a histogram of repeated classification entries as if
it were distinct divisions per patent. These tests pin both fixes.
"""

import ast
import json
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import shared_paths as P
from utils import shared_patent_utils as patent
from utils import shared_style as S

# One patent carrying a division entry plus three of its groups, and one group of another
# division — the shape Dimensions writes.
CATEGORIES = [
    {"name": "3101 Biochemistry and Cell Biology"},
    {"name": "31 Biological Sciences"},
    {"name": "3105 Genetics"},
    {"name": "3102 Bioinformatics and Computational Biology"},
    {"name": "4203 Health Services and Systems"},
]


class DivisionLabelTests(unittest.TestCase):
    def test_every_division_label_is_the_canonical_name(self):
        codes = pd.read_csv(P.FOR_2020_CODES, dtype={"code": str})
        canonical = codes[codes["level"] == "L2_division"].set_index("code")["name"].to_dict()
        names = patent.for_division_names()
        for code, name in canonical.items():
            self.assertEqual(names[code], name)
        # Every 4-digit group collapses to its own division's name, never its own.
        groups = codes[codes["level"] == "L4_group"]
        for code, name in zip(groups["code"], groups["name"]):
            [(division, label)] = patent.collapse_to_top_level([{"name": f"{code} {name}"}])
            self.assertEqual(division, code[:2])
            self.assertEqual(label, canonical[code[:2]])

    def test_the_four_mislabelled_divisions(self):
        cases = {
            "3101 Biochemistry and Cell Biology": ("31", "Biological Sciences"),
            "4201 Allied Health and Rehabilitation Science": ("42", "Health Sciences"),
            "5110 Medical and Biological Physics": ("51", "Physical Sciences"),
            "5203 Clinical and Health Psychology": ("52", "Psychology"),
        }
        for entry, expected in cases.items():
            self.assertEqual(patent.collapse_to_top_level([{"name": entry}]), [expected])

    def test_module_defines_collapse_once(self):
        # A second, older definition earlier in the file was shadowed by the fixed one;
        # deleting the fix would have silently revived it.
        tree = ast.parse(Path(patent.__file__).read_text())
        names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual(names.count("collapse_to_top_level"), 1)


class DistinctDivisionCountTests(unittest.TestCase):
    def test_collapse_lists_each_division_once(self):
        self.assertEqual(patent.collapse_to_top_level(CATEGORIES),
                         [("31", "Biological Sciences"), ("42", "Health Sciences")])
        self.assertEqual(patent.collapse_to_top_level([]), [])

    def test_overview_histogram_draws_distinct_divisions(self):
        S.load_style("04_non_academic_02_patents")
        topics = pd.Series([CATEGORIES, [], CATEGORIES[:1]]).apply(patent.collapse_to_top_level)
        frame = pd.DataFrame({"top_level_topics": topics})
        frame["topic_count"] = frame["top_level_topics"].apply(len)
        self.assertEqual(frame["topic_count"].tolist(), [2, 0, 1])
        countries = pd.DataFrame({"iso2": ["US"], "count": [3]})
        graph, _ = patent.build_topic_cooccurrence_network(frame, min_weight=1)
        fig = patent.plot_patent_overview(frame, countries, graph, patent.for_division_names())
        histogram = fig.axes[1]
        self.assertEqual([bar.get_height() for bar in histogram.patches], [1, 1, 1])
        self.assertIn("Mean: 1.00", [t.get_text() for t in histogram.get_legend().get_texts()])
        self.assertIn("canonical ANZSRC 2020 two-digit divisions", fig._ukb_caption)
        plt.close(fig)

    def test_network_and_country_weights_already_count_a_division_once(self):
        # Weights were never inflated by the duplicates; only labels were wrong. Feeding
        # the raw, repeated codes must give the same answer as the distinct ones.
        repeated = pd.DataFrame({
            "top_level_topics": [[("31", "x"), ("31", "y"), ("32", "z")]],
            "assignee_countries": [["US", "GB"]],
        })
        _, pairs = patent.build_topic_cooccurrence_network(repeated, min_weight=1)
        self.assertEqual(dict(pairs), {("31", "32"): 1})
        rows, *_ = patent.analyze_country_topics(
            repeated, country_col="assignee_countries",
            code_to_label=patent.for_division_names())
        self.assertEqual(sorted(rows.groupby("topic_code")["weight"].sum().round(6).items()),
                         [("31", 1.0), ("32", 1.0)])
        self.assertEqual(set(rows["topic_label"]),
                         {"Biological Sciences", "Biomedical and Clinical Sciences"})


class NotebookTests(unittest.TestCase):
    def test_part_ii_labels_divisions_canonically_and_draws_no_entry_histogram(self):
        nb = json.loads((P.DATA_ANALYSIS / "04_non_academic.ipynb").read_text())
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        self.assertIn("topcode_to_label = patent.for_division_names()", code)
        self.assertNotIn("n_topics_col='n_topics'", code)
        self.assertNotIn("code not in topcode_to_label", code)


if __name__ == "__main__":
    unittest.main()
