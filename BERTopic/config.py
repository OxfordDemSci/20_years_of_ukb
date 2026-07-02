# Configuration for UKB Showcase+ BERTopic dynamic topic modelling.
#
# Edit the paths below before running, or pass paths by command line:
#
# python run_bertopic_dynamic.py --input_csv /path/to/showcase_plus.csv --output_dir /path/to/output

SHOWCASE_PLUS_CSV_PATH = "input your path for Showcase+ CSV"
OUTPUT_DIR = "input your path for BERTopic output folder"

# Optional column names.
# Leave as None to auto-detect common column names.
ID_COL = None
TITLE_COL = None
ABSTRACT_COL = None
YEAR_COL = None
DATE_COL = None

# Analysis settings.
MIN_YEAR = 2014
INCOMPLETE_YEARS = [2026]
RANDOM_STATE = 42
TOP_N_TOPICS_FOR_FIGURES = 14

# Fixed document embedding model.
EMBEDDING_MODEL_NAME = "allenai-specter"

# Embedding batch sizes.
BATCH_SIZE_GPU = 64
BATCH_SIZE_CPU = 32

# Hyperparameter testing.
RUN_TOPIC_GRID = True
MAX_DOCS_FOR_TOPIC_GRID = None

PARAM_GRID = [
    {"n_neighbors": 15, "min_cluster_size": 25, "min_samples": 5},
    {"n_neighbors": 25, "min_cluster_size": 35, "min_samples": 10},
    {"n_neighbors": 35, "min_cluster_size": 45, "min_samples": 10},
    {"n_neighbors": 45, "min_cluster_size": 55, "min_samples": 15},
    {"n_neighbors": 25, "min_cluster_size": 60, "min_samples": 20},
]

# Final parameters from the current Showcase+ run.
# Used when running with --no_grid.
FINAL_PARAMS = {"n_neighbors": 45, "min_cluster_size": 55, "min_samples": 15}
