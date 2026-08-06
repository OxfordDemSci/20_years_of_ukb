# Dimensions linked endpoint pipeline

This pipeline starts from a Dimensions publication endpoint parquet, collects all linked clinical trials, grants, patents, policy documents, datasets, and source titles, and creates a publication-level wide parquet.

## Inputs

- A publication parquet containing unique Dimensions publication IDs and the full publication endpoint fields
- A Dimensions API key supplied through the `DIMENSIONS_API_KEY` environment variable

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
export DIMENSIONS_API_KEY="your-api-key"

python dimensions_endpoint_pipeline.py \
  --publications showcase_plus_all_endpoint.parquet \
  --output showcase_plus_all_endpoints_wide.parquet
```

The API key is read only from the environment and is never written to code or output files.

## Outputs

The command creates:

- `showcase_plus_all_endpoints_wide.parquet`: one row per publication
- `showcase_plus_all_endpoints_wide.stats.json`: collection and merge statistics
- `showcase_plus_all_endpoints_wide_tables/`: normalized publication, endpoint, and link parquets
- `showcase_plus_all_endpoints_wide_tables/.checkpoints/`: resumable API checkpoints

The wide file keeps the publication row count unchanged. For each endpoint it adds `__linked_ids`, `__n_links`, `__n_records`, and one prefixed column per endpoint field. One-to-many values are stored as JSON arrays ordered by endpoint ID.

## Linkage rules

| Endpoint | Publication linkage |
| --- | --- |
| Clinical trials | `clinical_trials.publication_ids` plus `publications.clinical_trial_ids` |
| Grants | `publications.supporting_grant_ids` to `grants.id` |
| Patents | `patents.publication_ids` |
| Policy documents | `policy_documents.publication_ids` |
| Datasets | `datasets.associated_publication_id` |
| Source titles | `publications.source_title.id` to `source_titles.id` |

Reports are intentionally excluded.
