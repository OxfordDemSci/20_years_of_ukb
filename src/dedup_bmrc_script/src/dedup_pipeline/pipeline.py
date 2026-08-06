"""Pipeline orchestration for stages A–E."""

import os

from .stage_a import stage_a_build_clusters
from .stage_b import stage_b_route_ids
from .stage_c import stage_c_score_and_select
from .stage_d import stage_d_build_patches
from .stage_e import stage_e_apply


def run_pipeline(args):
    """Run all stages using an argparse namespace."""
    os.makedirs(args.out_dir, exist_ok=True)

    clusters_map_path = os.path.join(args.out_dir, "clusters_map.parquet")
    stage_a_build_clusters(args.step5_dir, clusters_map_path, args.min_group_size)

    stage_b_dir = os.path.join(args.out_dir, "stageB")
    stage_b_route_ids(
        clusters_map_path=clusters_map_path,
        raw_base_dir=args.raw_base_dir,
        out_dir=stage_b_dir,
        ext=args.ext,
        check_exists=not args.no_exists_check,
    )

    stage_c_dir = os.path.join(args.out_dir, "stageC")
    stage_c_score_and_select(
        clusters_map_path=clusters_map_path,
        wanted_ids_path=os.path.join(stage_b_dir, "wanted_ids_by_file.parquet"),
        raw_base_dir=args.raw_base_dir,
        out_dir=stage_c_dir,
        ext=args.ext,
        zero_id_weight=args.zero_id_weight,
    )

    stage_d_dir = os.path.join(args.out_dir, "stageD")
    stage_d_build_patches(
        scores_dir=os.path.join(stage_c_dir, "scores_parts"),
        keep_decisions_path=os.path.join(stage_c_dir, "keep_decisions.parquet"),
        out_dir=stage_d_dir,
        list_cap=args.list_cap,
    )

    stage_e_dir = os.path.join(args.out_dir, "stageE")
    stage_e_apply(
        raw_base_dir=args.raw_base_dir,
        ext=args.ext,
        loser_list_path=os.path.join(stage_d_dir, "loser_list.parquet"),
        merge_patches_path=os.path.join(stage_d_dir, "merge_patches.parquet"),
        out_dir=stage_e_dir,
        only_file=args.apply_only_file,
    )
