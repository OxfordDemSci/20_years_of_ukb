# Dataset agreement inputs

`matched_ukb_full_final_2013_2025_three_model_labels.csv` is the full candidate-level
three-model labels input for `src/data_analysis/00_dataset.ipynb`. The notebook
detects this file automatically; `UKB_COMBINED_LABELS_CSV` can override it.

Keep all candidate rows, including disagreements, FALSE labels and unparsed
responses. The notebook restricts publication dates to 2013–2025 inclusive and
deduplicates paper IDs before analysis. It reads the saved model labels without
rerunning the classifiers.

Derived tables and figures are written under `output/`, leaving this input intact.
