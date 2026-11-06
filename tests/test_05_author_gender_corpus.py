"""The BMRC gender-by-field job must be 05's method, whatever shape the files are in.

The synthetic corpus mixes the shapes the VM copy may hold: authors as JSON text and as
list<struct>, FoR as JSON text (`category_for_2020`, `category_for`) and as a struct.
Everything is checked against shared_name_gender run directly on the same rows.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils import data_analysis_05_author_gender_corpus as G
from utils import shared_name_gender as NG
from utils import shared_paths as P

AFF = pa.list_(pa.struct([("name", pa.string()), ("country", pa.string())]))
AUTHOR = pa.struct([("first_name", pa.string()), ("last_name", pa.string()),
                    ("researcher_id", pa.string()), ("affiliations", AFF)])
FOR_STRUCT = pa.struct([
    ("first_level", pa.struct([("full", pa.list_(pa.struct([("code", pa.string()), ("name", pa.string())])))])),
    ("second_level", pa.struct([("full", pa.list_(pa.struct([("code", pa.string()), ("name", pa.string())])))])),
])
LABELS = {"32": "Biomedical and Clinical Sciences", "42": "Health Sciences",
          "52": "Psychology", "3202": "Clinical Sciences", "4202": "Epidemiology",
          "4206": "Public Health", "5202": "Biological Psychology"}

# (id, year, [(first_name, researcher_id)], [FoR codes])
PAPERS = [
    ("p01", 2016, [("Anna", "ur.1"), ("John", "ur.2"), ("M.", "ur.5")], ["32", "3202"]),
    ("p02", 2018, [("Mary", "ur.5"), ("J.", None)], ["4202"]),            # L2 42 only by folding
    ("p03", 2020, [("John", "ur.9"), ("", "ur.3")], ["42", "4206", "3202"]),
    ("p04", 2021, [("Anna", "ur.9"), ("Zzyzxq", None)], ["5202"]),         # ur.9: conflict
    ("p05", 2014, [("Anna", "ur.1")], ["4202"]),                           # out of 2015–2025
    ("p06", 2023, [], ["4202"]),                                           # no authors
    ("p07", 2024, [("Peter", "ur.7"), ("Lucy", "ur.8")], []),              # no FoR
    ("p08", 2025, [("Sarah", None), ("David", "ur.2"), ("Ewa", None)], ["32", "3202", "4202"]),
    ("p09", 2017, [("Hans-Peter", "ur.11"), ("José", None)], ["52", "5202"]),
    ("p10", 2019, [("Anna  Maria", "ur.12")], ["4206"]),
]
UKBB = {"p01", "p03", "p08", "p10"}


def _for_json(codes):
    return json.dumps([{"id": "8" + c, "name": f"{c} {LABELS[c]}"} for c in codes])


def _for_struct(codes):
    entry = lambda cs: {"full": [{"code": c, "name": f"{c} {LABELS[c]}"} for c in cs]}
    return {"first_level": entry([c for c in codes if len(c) == 2]),
            "second_level": entry([c for c in codes if len(c) == 4])}


def _authors_json(authors):
    return json.dumps([{"first_name": f, "last_name": "X", "researcher_id": r,
                        "affiliations": [{"name": "U"}]} for f, r in authors])


def _authors_struct(authors):
    return [{"first_name": f, "last_name": "X", "researcher_id": r,
             "affiliations": [{"name": "U", "country": "GB"}]} for f, r in authors]


def _repr_list(entries):
    """The BMRC copy's newest text files: a Python repr, not JSON (single quotes, None)."""
    return str(entries)


def write_corpus(root: Path) -> list:
    """Four readable files in four shapes, plus nothing else."""
    root.mkdir(parents=True, exist_ok=True)
    groups = [PAPERS[0:4], PAPERS[4:7], PAPERS[7:9], PAPERS[9:]]
    paths = []
    for i, group in enumerate(groups):
        base = {"id": [p[0] for p in group], "year": [p[1] for p in group],
                "date": [f"{p[1]}-06-01" for p in group], "type": ["article"] * len(group),
                "authors_count": [len(p[2]) for p in group]}
        if i == 3:      # 000000005000–5007 on BMRC: repr cells, double year, no authors_count
            authors = [[{"first_name": f, "last_name": "X", "researcher_id": r,
                         "affiliations": [{"name": "U"}]} for f, r in p[2]] for p in group]
            fors = [[{"id": "8" + c, "name": f"{c} {LABELS[c]}"} for c in p[3]] for p in group]
            table = pa.table({"id": base["id"], "year": [float(y) for y in base["year"]],
                              "date": base["date"], "type": base["type"],
                              "authors": [_repr_list(a) for a in authors],
                              "category_for": [_repr_list(f) for f in fors]})
        elif i == 0:    # the local export's shape
            table = pa.table({**base, "authors": [_authors_json(p[2]) for p in group],
                              "category_for_2020": [_for_json(p[3]) for p in group]})
        elif i == 1:    # nested authors, older FoR column name
            table = pa.table({**base,
                              "authors": pa.array([_authors_struct(p[2]) for p in group],
                                                  type=pa.list_(AUTHOR)),
                              "category_for": [_for_json(p[3]) for p in group]})
        else:           # nested authors and nested FoR
            table = pa.table({**base,
                              "authors": pa.array([_authors_struct(p[2]) for p in group],
                                                  type=pa.list_(AUTHOR)),
                              "category_for": pa.array([_for_struct(p[3]) for p in group],
                                                       type=FOR_STRUCT)})
        path = root / f"{i:012d}.parquet"
        pq.write_table(table, path)
        paths.append(str(path))
    return paths


def run_chain(files, out: Path, *, shards=1, buckets=3, arms=(("all", "none"),),
              ids=None, codes=None, first_year=2015, allow_failed=False):
    out = str(out)
    for s in range(shards):
        G.main(["names", "--files", *files, "--out", out, "--shard", str(s),
                "--num-shards", str(shards), "--buckets", str(buckets),
                "--first-year", str(first_year), "--progress", "0"])
    extra = ["--allow-failed"] if allow_failed else []
    G.main(["classify", "--out", out, "--buckets", str(buckets), "--num-shards", str(shards), *extra])
    G.main(["identity", "--out", out, "--buckets", str(buckets), "--num-shards", str(shards), *extra])
    for label, mode in arms:
        for s in range(shards):
            argv = ["count", "--files", *files, "--out", out, "--shard", str(s),
                    "--num-shards", str(shards), "--buckets", str(buckets),
                    "--first-year", str(first_year), "--label", label,
                    "--id-filter", mode, "--progress", "0"]
            if ids:
                argv += ["--ids", ids]
            if codes:
                argv += ["--codes", codes]
            G.main(argv)
    G.main(["merge", "--out", out, "--num-shards", str(shards),
            "--arms", ",".join(a for a, _ in arms), *extra])
    return pd.read_parquet(Path(out) / "gender_field_counts.parquet")


def reference_counts(first_year=2015, codes=None, ids=None):
    """What 05 would count on the synthetic papers: classify_authorship_names over every
    in-window authorship, then whole counting per folded L2 / L4 code."""
    rows = []
    for pid, year, authors, fors in PAPERS:
        if not first_year <= year <= 2025:
            continue
        for pos, (first, rid) in enumerate(authors, 1):
            rows.append({"paper": pid, "year": year, "first_name": G.normalise_text(first) or "",
                         "researcher_id": G.normalise_text(rid),
                         "identity_resolved": G.normalise_text(rid) is not None,
                         "fors": fors})
    auth = pd.DataFrame(rows)
    auth["name_gender"] = NG.classify_authorship_names(auth)["name_gender"].to_numpy()
    out = []
    for r in auth.itertuples():
        if ids is not None and r.paper not in ids:
            continue
        l4 = {c for c in r.fors if len(c) == 4}
        l2 = {c for c in r.fors if len(c) == 2} | {c[:2] for c in l4}
        if codes is not None:
            l2 &= codes["L2"]
            l4 &= codes["L4"]
        out += [(lvl, c, r.name_gender) for lvl, cs in (("L2", l2), ("L4", l4)) for c in cs]
    ref = pd.DataFrame(out, columns=["level", "code", "gender"])
    return ref.groupby(["level", "code", "gender"]).size()


def tally(counts):
    return counts.groupby(["level", "code", "gender"]).n_authorships.sum()


class NameKeyTest(unittest.TestCase):
    def test_key_classifies_exactly_like_the_raw_name(self):
        raw = ["Anna", "anna", "Anna M.", "  Anna   Maria ", "J.", "J. Robert", "", None, "   ",
               "José", "Hans-Peter", "O'Neil", "Zzyzxq", "M", "Ł. Kowalska", "Xiaoming", "Kim"]
        for value in raw:
            with self.subTest(value=value):
                direct = NG.infer_name_gender(G.normalise_text(value) or "")
                keyed = G.classify_key(G.name_key(value))
                self.assertEqual((direct.category, direct.unknown_reason),
                                 (keyed.category, keyed.unknown_reason))

    def test_sentinels(self):
        self.assertEqual(G.name_key(None), G.MISSING_KEY)
        self.assertEqual(G.name_key("J. K."), G.INITIALS_KEY)
        self.assertEqual(G.classify_key(G.INITIALS_KEY).unknown_reason, "initials_only")
        self.assertEqual(G.classify_key(G.MISSING_KEY).unknown_reason, "missing_name")

    def test_normalise_text_is_05s(self):
        from utils import data_analysis_05_author_characteristics as A
        for value in [None, float("nan"), "", "  a  b ", "x", 3]:
            self.assertEqual(A._normalise_text(value), G.normalise_text(value))

    def test_bucket_is_stable(self):
        self.assertEqual(G.stable_bucket("Anna", 32), G.stable_bucket("Anna", 32))
        self.assertTrue(0 <= G.stable_bucket("ur.1", 7) < 7)


class ReadingTest(unittest.TestCase):
    def test_struct_and_text_authors_flatten_identically(self):
        authors = [[("Anna", "ur.1"), ("J.", None)], [], [("  Mary ", " ur.5 ")]]
        text = pa.array([_authors_json(a) for a in authors])
        struct = pa.array([_authors_struct(a) for a in authors], type=pa.list_(AUTHOR))
        rows = np.array([0, 1, 2])
        a = G.flatten_authors(text, "text", rows)
        b = G.flatten_authors(struct, "struct", rows)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.first_name.tolist(), ["Anna", "J.", "Mary"])
        self.assertEqual(a.researcher_id.tolist(), ["ur.1", None, "ur.5"])

    def test_struct_projection_reads_only_name_leaves(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = write_corpus(Path(tmp) / "corpus")
            fmt, cols, problem = G.authors_layout(pq.ParquetFile(files[1]))
            self.assertEqual(fmt, "struct")
            self.assertEqual(problem, "")
            self.assertTrue(all(c.split(".")[-1] in ("first_name", "researcher_id") for c in cols))

    def test_repr_cells_parse_like_json(self):
        from utils import data_analysis_03_academic_impact_field_counts as FC
        from utils.shared_for import parse_listcol
        entries = [{"id": "80003", "name": "32 Biomedical and Clinical Sciences"},
                   {"id": "80137", "name": "4202 Epidemiology"},
                   {"id": "80999", "name": "4499 Women's Studies"},              # repr -> "…"
                   {"id": "80998", "name": "4498 Both ' and \" quotes"}]       # repr escapes
        for cell in (json.dumps(entries), str(entries)):
            with self.subTest(cell=cell[:30]):
                got = list(FC._for_pairs(cell, "text"))
                self.assertEqual([c for c, _ in got], ["32", "4202", "4499", "4498"])
                self.assertEqual(got[2][1], "Women's Studies")
                want = [d["name"].split(" ", 1)[0] for d in parse_listcol(cell)]
                self.assertEqual([c for c, _ in got], want)
        rcdc = str([{"id": "558", "name": "Prevention"}, {"id": "1", "name": "Aging"}])
        self.assertEqual(list(FC._flat_pairs(rcdc, "text", None)),
                         [("Prevention", "Prevention"), ("Aging", "Aging")])

    def test_repr_authors_flatten_like_json(self):
        authors = [{"first_name": "Anna", "researcher_id": "ur.1", "orcid": None},
                   {"first_name": "J.", "researcher_id": None, "orcid": None}]
        a = G.flatten_authors(pa.array([json.dumps(authors)]), "text", np.array([0]))
        b = G.flatten_authors(pa.array([str(authors)]), "text", np.array([0]))
        pd.testing.assert_frame_equal(a, b)

    def test_roles(self):
        auth = G.add_roles(pd.DataFrame({"row": [0, 1, 1, 1, 2, 2]}))
        self.assertEqual(auth.role.tolist(), ["Single author", "First author", "Middle author",
                                              "Last author", "First author", "Last author"])

    def test_fields_fold_and_restrict(self):
        labels = {}
        fields, any_code = G.paper_fields(_for_json(["4202", "3202"]), "text", None, labels)
        self.assertEqual(fields, {"L2": ["32", "42"], "L4": ["3202", "4202"]})
        self.assertTrue(any_code)
        codes = {"L2": {"42"}, "L4": {"4206"}}
        fields, any_code = G.paper_fields(_for_json(["4202", "3202"]), "text", codes, labels)
        self.assertEqual(fields, {"L2": ["42"], "L4": []})
        self.assertEqual(labels["4202"], "Epidemiology")


class ChainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.files = write_corpus(self.root / "corpus")
        self.ids = self.root / "ukbb_ids.txt"
        self.ids.write_text("\n".join(sorted(UKBB)) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_matches_shared_name_gender_on_every_shape(self):
        counts = run_chain(self.files, self.root / "out")
        pd.testing.assert_series_equal(tally(counts), reference_counts(), check_names=False,
                                       check_dtype=False)

    def test_identity_fill_and_conflict_are_marked(self):
        counts = run_chain(self.files, self.root / "out")
        reasons = counts.groupby(["gender", "reason"]).n_authorships.sum()
        # "M." of ur.5 is filled from "Mary"; both ur.9 authorships are blanked.
        self.assertGreater(reasons.get(("Female", G.FILL_REASON), 0), 0)
        self.assertGreater(reasons.get(("Unknown", G.CONFLICT_REASON), 0), 0)
        self.assertGreater(reasons.get(("Unknown", "initials_only"), 0), 0)

    def test_sharding_does_not_change_anything(self):
        one = run_chain(self.files, self.root / "one", shards=1, buckets=2)
        three = run_chain(self.files, self.root / "three", shards=3, buckets=5)
        key = ["year", "level", "code", "role", "gender", "reason"]
        pd.testing.assert_frame_equal(one.sort_values(key).reset_index(drop=True),
                                      three.sort_values(key).reset_index(drop=True))

    def test_arms_partition_the_corpus(self):
        arms = (("background", "exclude"), ("ukbb", "include"))
        counts = run_chain(self.files, self.root / "arms", arms=arms, ids=str(self.ids))
        both = run_chain(self.files, self.root / "none")
        pd.testing.assert_series_equal(tally(counts), tally(both), check_dtype=False)
        ukbb = counts[counts.arm == "ukbb"]
        pd.testing.assert_series_equal(tally(ukbb), reference_counts(ids=UKBB),
                                       check_names=False, check_dtype=False)
        totals = pd.read_parquet(self.root / "arms" / "gender_field_totals.parquet")
        self.assertEqual(int(totals.n_papers.sum()), 9)       # p05 is out of window

    def test_code_restriction(self):
        codes_csv = self.root / "codes.csv"
        pd.DataFrame({"level": ["L2", "L4", "L4"], "code": ["42", "4202", "3202"],
                      "label": ["", "", ""]}).to_csv(codes_csv, index=False)
        counts = run_chain(self.files, self.root / "codes", codes=str(codes_csv))
        codes = {"L2": {"42"}, "L4": {"4202", "3202"}}
        pd.testing.assert_series_equal(tally(counts), reference_counts(codes=codes),
                                       check_names=False, check_dtype=False)
        totals = pd.read_parquet(self.root / "codes" / "gender_field_totals.parquet")
        self.assertEqual(int(totals.n_no_inset_for.sum()), 2)   # p04, p09: psychology only
        self.assertEqual(int(totals.n_no_for.sum()), 1)         # p07
        self.assertEqual(int(totals.n_no_authors.sum()), 1)     # p06
        missing = G.missing_summary(counts, totals)
        paper = missing[missing.unit == "paper"].set_index("reason")
        self.assertAlmostEqual(paper.loc["n_no_inset_for", "share_pct"], 100 * 2 / 9)
        self.assertNotIn("n_filtered_arm", paper.index)
        pairs = missing[missing.unit == "authorship-field pair (L2)"]
        self.assertAlmostEqual(pairs.share_pct.sum(), 100.0)

    def test_unreadable_file_blocks_the_chain(self):
        bad = self.root / "corpus" / "zz_bad.parquet"
        pq.write_table(pa.table({"id": ["q1"], "year": [2020],
                                 "authors": pa.array([["Anna"]], type=pa.list_(pa.string()))}), bad)
        files = self.files + [str(bad)]
        G.main(["names", "--files", *files, "--out", str(self.root / "bad"), "--buckets", "2",
                "--progress", "0"])
        with self.assertRaises(SystemExit):
            G.main(["classify", "--out", str(self.root / "bad"), "--buckets", "2"])
        log = pd.read_csv(self.root / "bad" / "logs" / "filelog.names.shard00000.csv")
        self.assertIn("unsupported authors type", log.error.dropna().iloc[0])

    def test_schema_groups_every_shape(self):
        vocab = self.root / "vocab.txt"
        vocab.write_text("\n".join(c for c in LABELS if c != "5202") + "\n")
        G.main(["schema", "--files", *self.files, "--out", str(self.root / "schema"),
                "--for-vocab", str(vocab)])
        groups = pd.read_csv(self.root / "schema" / "schema_groups.csv")
        self.assertEqual(len(groups), 4)
        self.assertTrue((groups.sampled_codes > 0).all())   # the repr file's codes are read
        self.assertEqual(set(groups.authors_format), {"text", "struct"})
        self.assertGreater(groups.sampled_non2020_codes.sum(), 0)


class ComparisonTest(unittest.TestCase):
    """Hand-computed: one field, two years, UK Biobank weighted 3:1 towards 2016."""

    def counts(self):
        rows = [  # arm, year, gender, n
            ("ukbb", 2016, "Female", 15), ("ukbb", 2016, "Male", 15), ("ukbb", 2016, "Unknown", 10),
            ("ukbb", 2020, "Female", 5), ("ukbb", 2020, "Male", 5),
            ("background", 2016, "Female", 40), ("background", 2016, "Male", 60),
            ("background", 2020, "Female", 60), ("background", 2020, "Male", 40),
        ]
        df = pd.DataFrame(rows, columns=["arm", "year", "gender", "n_authorships"])
        return df.assign(level="L2", code="42", label="Health Sciences", role="First author",
                         reason="", n_credit=0.0)

    def test_field_gap_and_year_standardisation(self):
        out = G.compare_fields(self.counts()).iloc[0]
        self.assertAlmostEqual(out.ukbb_female_share, 50.0)
        self.assertAlmostEqual(out.background_female_share, 50.0)
        # (30 x 40 % + 10 x 60 %) / 40: the background at UK Biobank's year mix.
        self.assertAlmostEqual(out.background_standardised, 45.0)
        self.assertAlmostEqual(out.gap_standardised_pp, 5.0)
        self.assertAlmostEqual(out.ukbb_unknown_share, 20.0)
        self.assertAlmostEqual(out.ukbb_female_share_if_unknown_female, 60.0)
        self.assertLess(out.ukbb_ci_low, 50.0)
        self.assertGreater(out.ukbb_ci_high, 50.0)

    def test_overall_by_role(self):
        out = G.compare_overall(self.counts()).set_index("role")
        self.assertAlmostEqual(out.loc["All authorships", "background_standardised"], 45.0)
        self.assertAlmostEqual(out.loc["First author", "gap_standardised_pp"], 5.0)
        self.assertTrue(np.isnan(out.loc["Last author", "ukbb_female_share"]))

    def test_wilson_matches_05(self):
        from utils import data_analysis_05_author_characteristics as A
        low, high = G.wilson_interval([3, 50], [10, 120])
        for i, (k, n) in enumerate([(3, 10), (50, 120)]):
            ref = A._wilson_interval(k, n)
            self.assertAlmostEqual(low[i], ref[0])
            self.assertAlmostEqual(high[i], ref[1])


@unittest.skipUnless(P.SHOWCASE_PLUS.exists(), "local showcase parquet not available")
class ParityWith05Test(unittest.TestCase):
    """The chain on the real UK Biobank corpus reproduces 05's gender_by_field exactly."""

    def test_gender_by_field(self):
        from utils import data_analysis_05_author_characteristics as A

        _, papers = A.load_author_papers()
        auth, _, institutions, countries = A.build_authorship_tables(papers)
        expected = A.build_gender_tables(auth, countries, institutions)[2]
        expected = expected.set_index("for_l2")[["Female", "Male", "Unknown"]].astype(int)
        with tempfile.TemporaryDirectory() as tmp:
            counts = run_chain([str(P.SHOWCASE_PLUS)], Path(tmp), buckets=4,
                               first_year=A.FIRST_YEAR)
        got = (counts[counts.level == "L2"].groupby(["label", "gender"]).n_authorships.sum()
               .unstack(fill_value=0)[["Female", "Male", "Unknown"]])
        got.index.name = "for_l2"
        pd.testing.assert_frame_equal(got.sort_index().astype(int), expected.sort_index(),
                                      check_names=False, check_column_type=False)


if __name__ == "__main__":
    unittest.main()
