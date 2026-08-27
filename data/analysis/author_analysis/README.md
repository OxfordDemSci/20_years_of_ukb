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
