"""Rebuild the Altmetric input table from data the project already holds.

WHY THIS EXISTS
---------------
`04_non_academic_03_altmetric.ipynb` was written against an Altmetric export,
`altmetric.csv`, that is **not in the repository** and has no provenance record —
nobody wrote down where it came from or when. The notebook has been unrunnable
ever since. This module rebuilds as much of that table as the corpus and the
policy-document pull can honestly supply, so the analysis is not blocked on a
file nobody can find.

WHAT IT CAN AND CANNOT REBUILD
------------------------------
    Altmetric Attention Score  -> YES. `altmetric` in the corpus parquet
                                  (17,982 of 26,109 papers, 68.9%).
    Policy mentions            -> YES, but it is a DIFFERENT QUANTITY. Altmetric
                                  counts policy-document mentions from its own
                                  source list; this counts Dimensions policy
                                  documents citing the paper, from
                                  policy_documents.csv. Related, not equal.
    News mentions              -> NO. Nothing in the parquet, the policy pull, the
                                  trials pull or the patents pull carries news
                                  mentions; they come from Altmetric's media
                                  monitoring. The column is emitted as 0 so the
                                  downstream code runs unchanged, and 0 here means
                                  UNKNOWN, not "no news coverage".

The emitted column names deliberately match the Altmetric export's
(`DOI`, `Altmetric Attention Score`, `News mentions`, `Policy mentions`,
`Publication Date`) so a real export can be dropped in later and the notebook
switches source by changing one path — the swappable-backend rule. Because the
two are not equivalent, the difference is stated here, at the constant, and in
doc/data/altmetric.md rather than left for a reader to discover.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import shared_paths as P
from utils.shared_showcase import load_showcase, parse_listcol

# Emitted for every row; 0 means "no source for this", not "measured as zero".
NEWS_MENTIONS_UNAVAILABLE = 0


def policy_citation_counts(policy_csv: Path = None) -> pd.Series:
    """Papers -> number of Dimensions policy documents citing them.

    Indexed by Dimensions publication id. Only papers cited at least once appear.
    """
    policy_csv = Path(policy_csv or P.POLICY_CSV)
    pol = pd.read_csv(policy_csv)
    counts: dict[str, int] = {}
    for cell in pol["publication_ids"]:
        # One policy document may cite a paper more than once in its reference
        # list; a document counts once per paper.
        for pid in set(parse_listcol(cell)):
            if isinstance(pid, str) and pid:
                counts[pid] = counts.get(pid, 0) + 1
    return pd.Series(counts, dtype="int64", name="Policy mentions")


def build_altmetric_table(out_path: Path = None) -> pd.DataFrame:
    """Assemble the Altmetric-shaped table and optionally write it out."""
    corpus = load_showcase(columns=["id", "doi", "year", "times_cited", "altmetric"])
    policy = policy_citation_counts()

    out = pd.DataFrame({
        "id": corpus["id"],
        "DOI": corpus["doi"],
        "Altmetric Attention Score": corpus["altmetric"],
        "News mentions": NEWS_MENTIONS_UNAVAILABLE,
        "Policy mentions": corpus["id"].map(policy).fillna(0).astype("int64"),
        # The corpus carries a year, not a date. Downstream only ever takes
        # `.dt.year`, so a January-1 stamp is lossless for that use and would be
        # wrong for anything finer — do not read a day or month off this column.
        "Publication Date": pd.to_datetime(
            corpus["year"].astype("Int64").astype(str) + "-01-01", errors="coerce"
        ),
    })
    # Deliberately NOT carrying `times_cited`: it is not part of the Altmetric
    # export schema, and the notebook merges this table against the corpus, which
    # has its own. A duplicate column name is suffixed away by the merge
    # (times_cited_altmetric / times_cited_dim) and the annotation builder's
    # row.get("times_cited") then silently returns None -- every label reading
    # "Citations: NA". Emit only what the real export emits.

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
    return out


def coverage_report(table: pd.DataFrame) -> str:
    """One-paragraph statement of what the rebuilt table does and does not carry."""
    n = len(table)
    aas = table["Altmetric Attention Score"].notna().sum()
    pol = (table["Policy mentions"] > 0).sum()
    both = ((table["Altmetric Attention Score"] > 0) & (table["Policy mentions"] > 0)).sum()
    return (
        f"{n:,} papers | Attention Score on {aas:,} ({aas/n:.1%}) | "
        f"policy-cited {pol:,} ({pol/n:.1%}) | both positive {both:,} ({both/n:.1%}) | "
        f"News mentions UNAVAILABLE (emitted as 0 for all rows)"
    )
