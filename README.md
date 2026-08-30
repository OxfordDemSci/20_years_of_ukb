# 20_years_of_ukb
A place to develop analysis and infrastructure related to the history of the UKBiobank.

## Analysis dependencies

Install the pinned offline name-category libraries used across analyses with:

```bash
python -m pip install -r requirements-analysis.txt
```

Name-category inference uses only package-bundled local data. It makes no API calls and
stores per-library votes and ensemble provenance in the analysis outputs.
