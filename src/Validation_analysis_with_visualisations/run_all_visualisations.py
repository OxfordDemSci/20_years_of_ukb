import argparse

from common import ensure_real_path
from analysis_explicit_ukb_mentions import run as run_explicit
from analysis_keyword_categories import run as run_keywords
from analysis_model_agreement import run as run_agreement
from analysis_semantic_map import run as run_semantic
from analysis_tfidf_terms import run as run_tfidf
from analysis_year_trends import run as run_years


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined_path", default="INPUT your path for three_model_combined_labels.csv here")
    p.add_argument("--three_true_path", default="INPUT your path for three_model_TRUE_agreement.csv here")
    p.add_argument("--rest_path", default="INPUT your path for rest_NOT_three_model_TRUE_agreement.csv here")
    p.add_argument("--output_dir", default="INPUT your path for visualisation output folder here")
    p.add_argument("--max_tfidf_per_group", type=int, default=20000)
    p.add_argument("--max_semantic_per_group", type=int, default=3000)
    args = p.parse_args()

    combined_path = ensure_real_path(args.combined_path, "--combined_path")
    three_true_path = ensure_real_path(args.three_true_path, "--three_true_path")
    rest_path = ensure_real_path(args.rest_path, "--rest_path")
    output_dir = ensure_real_path(args.output_dir, "--output_dir")

    run_agreement(combined_path, output_dir)
    run_years(combined_path, output_dir)
    run_explicit(combined_path, output_dir)
    run_keywords(three_true_path, rest_path, output_dir)
    run_tfidf(three_true_path, rest_path, output_dir, max_per_group=args.max_tfidf_per_group)
    run_semantic(three_true_path, rest_path, output_dir, max_per_group=args.max_semantic_per_group)


if __name__ == "__main__":
    main()
