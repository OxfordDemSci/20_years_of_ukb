# BERTopic dynamic topic modelling for UKB Showcase+

This folder contains the BERTopic dynamic topic modelling code for the UKB Showcase+ corpus.

The script assumes you already have a final Showcase+ CSV, where each row is one publication. The CSV should contain at least title and publication year/date. Abstract is strongly recommended.

## Install

cd BERTopic
pip install -r requirements.txt

A GPU runtime is recommended because allenai-specter embeddings are slow on CPU.

## Set paths

Either edit config.py:

SHOWCASE_PLUS_CSV_PATH = "input your path for Showcase+ CSV"
OUTPUT_DIR = "input your path for BERTopic output folder"

or pass paths directly:

python run_bertopic_dynamic.py --input_csv "input your path for Showcase+ CSV" --output_dir "input your path for BERTopic output folder"

Column names are auto-detected. If needed, edit these in config.py:

ID_COL = None
TITLE_COL = None
ABSTRACT_COL = None
YEAR_COL = None
DATE_COL = None

## Run

Full run with hyperparameter testing:

python run_bertopic_dynamic.py

Fast run using the final selected parameters:

python run_bertopic_dynamic.py --no_grid

## What the script does

1. Loads the final Showcase+ CSV.
2. Builds topic text as title + abstract.
3. Applies light cleaning only.
4. Embeds each paper with allenai-specter.
5. Fits BERTopic using UMAP, HDBSCAN and c-TF-IDF.
6. Tests several BERTopic settings and selects the best model using coherence, topic diversity, outlier rate and topic-count penalty.
7. Saves document-topic assignments, topic information and dynamic topic tables.
8. Saves streamgraph, ridge-wave, heatmap and native BERTopic topics-over-time outputs.

## Main outputs

tables/analysis_input_with_text_year.csv
tables/yearly_paper_counts.csv
tables/bertopic_hyperparameter_leaderboard.csv
tables/bertopic_final_model_metrics.csv
tables/bertopic_document_topic_assignments.csv
tables/bertopic_topic_info.csv
tables/bertopic_topics_over_time_native.csv
tables/bertopic_topic_growth_table.csv
tables/bertopic_topic_year_counts_selected.csv
tables/bertopic_topic_year_proportions_selected.csv

figures/bertopic_dynamic_streamgraph.png
figures/bertopic_topic_ridge_waves.png
figures/bertopic_topic_prevalence_heatmap.html
figures/bertopic_native_topics_over_time.html

models/bertopic_showcase_plus/
cache/embeddings_allenai-specter_*.npy
bertopic_run_manifest.json

## Minimal interpretation

BERTopic topics are bottom-up scientific themes learned from title/abstract text.

Counts show absolute growth of a topic.

Shares show whether the topic became more or less central within annual UKB research.

The model is fitted globally across all years, so a topic ID keeps the same meaning over time.

The final incomplete year is kept in raw outputs but should be interpreted cautiously.
