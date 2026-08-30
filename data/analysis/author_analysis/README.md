# Author-characteristics analysis cache

Analysis `05_author_characteristics.ipynb` reads the Showcase+ all-endpoints-wide
publication parquet as its only source dataset. This directory is reserved for derived,
rebuildable caches and the Natural Earth 1:110m country geometry used by the geography
supplement. Generated data files are ignored by Git; this README keeps the directory and
documents provenance.

The analysis never treats a cache as an independent source. Complete-year results cover
2013-2025, while provisional 2026 records remain in the source-level quality audit only.
See `output/tables/data_analysis/05_author_characteristics/analysis_parameters.csv` and
`methods_author_characteristics.txt` after running the notebook.

## Offline name-category inference

Every active analysis uses `src/utils/shared_name_gender.py`. The classifier combines the
local datasets bundled with `gender-guesser`, `gender-detector`, `nomquamgender`,
`names-dataset`, and `gename`; it contains no API client and performs no network requests.
Install the pinned packages with `python -m pip install -r requirements-analysis.txt`.

Each package contributes at most one vote. `mostly_female` and `mostly_male` are grouped
with Female and Male. Non-conflicting available votes are accepted; conflicting votes
require at least three votes and a two-vote margin, otherwise the result remains Unknown.
For resolved Dimensions researcher IDs, an Unknown observation is linked only when every
direct classified observation agrees, and identities with conflicting direct calls are
returned to Unknown. Strict, expanded, ensemble, and identity-linked coverage are exported
alongside per-library votes, package versions, conflicts, methods, and the complete
post-ensemble unresolved-name queue.

These outputs are statistical name categories, not observed sex or self-identified gender.
Unknown remains substantive because classification error and missingness are culturally
patterned.
