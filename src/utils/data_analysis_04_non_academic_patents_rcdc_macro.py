#!/usr/bin/env python3
"""
rcdc_macro_for_pipeline.py

Input: CSV with columns:
 - patent_id
 - rcdc_labels  (either JSON list like '["625","439"]' OR semicolon/comma separated like "625;439" or "625,439")
 - publication_year or publication_date (required to enforce the analysis cutoff)

Output:
 - rcdc_cooccurrence.csv
 - rcdc_communities_louvain.csv
 - rcdc_communities_leiden.csv (if leiden available)
 - cluster_label_summary.csv  (community -> top labels)
 - network_plot.png
"""

import argparse
import ast
import json
from hashlib import sha256
from importlib.metadata import version
from collections import Counter, defaultdict
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
from utils.shared_analysis_window import ANALYSIS_START_DATE, ANALYSIS_END_DATE, filter_analysis_window
from utils.shared_showcase import parse_listcol
from utils import shared_paths as P
from utils.shared_style import PNG_DPI, apply_typography, figure_export_formats, finalize_figure

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# community detection
import community as community_louvain   # python-louvain
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from utils.shared_style import save_figure_file
import seaborn as sns

# Optional leiden
HAS_IGRAPH = False
try:
    import igraph as ig
    import leidenalg
    HAS_IGRAPH = True
except Exception:
    HAS_IGRAPH = False


result_path = str(P.PATENT_RCDC_MACRO / f"{ANALYSIS_START_DATE.date()}_through_{ANALYSIS_END_DATE.date()}")

# The original invocation at the foot of this module used ten repeats and kept
# the last partition, so seed 9 is its reproducible canonical fit.
ANALYSIS_PARTITION_METHOD = {
    "algorithm": "python-louvain",
    "python_louvain_version": version("python-louvain"),
    "networkx_version": nx.__version__,
    "numpy_version": np.__version__,
    "normalization": "npmi",
    "threshold": 0.01,
    "resolution": 1.0,
    "random_state": 9,
}


def _partition_provenance(frame, patent_labels, method):
    id_col = "id" if "id" in frame else "patent_id"
    if frame[id_col].isna().any() or frame[id_col].duplicated().any():
        raise ValueError("RCDC partition input requires unique, nonmissing patent IDs")
    cohort = sorted(
        [[str(pid), sorted(set(labels))] for pid, labels in zip(frame[id_col], patent_labels)]
    )
    return {
        "schema_version": 2,
        "analysis_start_date": ANALYSIS_START_DATE.date().isoformat(),
        "analysis_end_date": ANALYSIS_END_DATE.date().isoformat(),
        "cohort_sha256": sha256(json.dumps(cohort, separators=(",", ":")).encode()).hexdigest(),
        "patents": len(cohort),
        "labels": len({label for _, labels in cohort for label in labels}),
        "method": method,
    }


def load_or_build_analysis_partition(frame, *, category_col="category_rcdc", cache_dir=None):
    """Fit the original Louvain method on eligible patents, or reuse its verified cache.

    Cache identity includes both window boundaries, patent IDs, label assignments, algorithm
    parameters and dependency versions. Legacy summaries without that provenance
    are never used as fitted partitions and are never overwritten.
    """
    eligible = filter_analysis_window(frame, year_col="publication_year", date_col="publication_date")
    patent_labels = []
    for cell in eligible[category_col]:
        labels = [value.get("id") if isinstance(value, dict) else value
                  for value in parse_listcol(cell)]
        patent_labels.append(sorted({str(value) for value in labels if value is not None}))
    expected = _partition_provenance(eligible, patent_labels, dict(ANALYSIS_PARTITION_METHOD))
    if not expected["labels"]:
        raise ValueError("No RCDC labels remain in the eligible patent cohort")
    cache_key = sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(result_path)
    summary = cache_dir / f"cluster_label_summary_louvain.{cache_key}.csv"
    sidecar = summary.with_suffix(".provenance.json")
    if summary.exists() and sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            signature = metadata.pop("partition_sha256")
            if metadata == expected and signature == sha256(summary.read_bytes()).hexdigest():
                return pd.read_csv(summary), summary
        except (OSError, ValueError, KeyError):
            pass

    labels, matrix, counts = build_cooccurrence_matrix(patent_labels)
    weights = compute_pmi_matrix(matrix, counts, len(eligible), normalized=True)
    weights[weights <= ANALYSIS_PARTITION_METHOD["threshold"]] = 0.0
    graph = graph_from_matrix(labels, weights, threshold=0.0)
    partition = run_louvain(
        graph, random_state=ANALYSIS_PARTITION_METHOD["random_state"],
        resolution=ANALYSIS_PARTITION_METHOD["resolution"],
    )
    result = summarize_communities(partition, counts, top_k=10)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(summary, index=False)
    expected["partition_sha256"] = sha256(summary.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Rebuilt RCDC partition from {len(eligible)} eligible patents ({len(result)} clusters)")
    return result, summary
# --------------------------
# Utilities: parsing input
# --------------------------
def parse_label_cell(cell):
    """Parses a cell that may be JSON list or delimited string into list of strings."""
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()!='']
    s = str(cell).strip()
    # Try JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()!='']
    except Exception:
        pass
    # Try Python literal
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed if str(x).strip()!='']
    except Exception:
        pass
    # Fallback: split on common delimiters
    for sep in [';', ',', '|', '/']:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()!='']
            if parts:
                return parts
    # single token
    return [s] if s!='' else []

# --------------------------
# Build co-occurrence
# --------------------------
def build_cooccurrence_matrix(patent_labels):
    """
    patent_labels: list of lists of label ids (strings)
    Returns: (labels_list, cooccurrence_matrix (numpy NxN), label_counts dict)
    """
    labels = sorted(list({lab for labs in patent_labels for lab in labs}))
    idx = {lab:i for i, lab in enumerate(labels)}
    N = len(labels)
    M = np.zeros((N, N), dtype=float)
    counts = Counter()
    for labs in patent_labels:
        unique = sorted(set(labs))
        for a in unique:
            counts[a] += 1
        for i in range(len(unique)):
            for j in range(i+1, len(unique)):
                A = idx[unique[i]]
                B = idx[unique[j]]
                M[A, B] += 1
                M[B, A] += 1
    # optionally set diagonal to counts of label occurrence
    for lab, c in counts.items():
        M[idx[lab], idx[lab]] = c
    return labels, M, counts

# --------------------------
# PMI / NPMI normalization
# --------------------------
def compute_pmi_matrix(M, label_counts, total_patents, eps=1e-12, normalized=True):
    """
    Compute PMI or normalized PMI between label pairs.
    M: raw co-occurrence matrix (with diagonal occurrences)
    label_counts: Counter mapping label->count
    total_patents: number of patents
    normalized: if True return NPMI in [-1,1], else PMI
    """
    labels = sorted(list(label_counts.keys()))
    N = len(labels)
    pmi = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            # joint probability P(i,j) = M[i,j] / total_patents (for i!=j); diagonal uses co-occurrence with itself = count
            pij = M[i,j] / float(total_patents)
            pi = label_counts[labels[i]] / float(total_patents)
            pj = label_counts[labels[j]] / float(total_patents)
            if pij <= 0 or pi <= 0 or pj <= 0:
                pmi[i,j] = 0.0
            else:
                v = math.log(pij / (pi*pj) + eps)
                if normalized:
                    # NPMI = PMI / -log(pij)
                    denom = -math.log(pij + eps)
                    pmi[i,j] = v / denom if denom != 0 else 0.0
                else:
                    pmi[i,j] = v
    return pmi

# --------------------------
# Make graph from weighted matrix
# --------------------------
def graph_from_matrix(labels, W, threshold=0.0, self_edges=False):
    G = nx.Graph()
    for lab in labels:
        G.add_node(lab)
    N = W.shape[0]
    for i in range(N):
        for j in range(i, N):
            w = float(W[i,j])
            if i==j and not self_edges:
                continue
            if w > threshold:
                if i==j:
                    G.add_edge(labels[i], labels[j], weight=w)  # self-edge; networkx will ignore but keep weight
                else:
                    G.add_edge(labels[i], labels[j], weight=w)
    return G

# --------------------------
# Louvain partition wrapper
# --------------------------
def run_louvain(G, weight='weight', random_state=0, resolution=1.0):
    # returns partition dict: label -> community_id
    partition = community_louvain.best_partition(G, weight=weight, random_state=random_state, resolution=resolution)
    return partition

# --------------------------
# Leiden wrapper (igraph)
# --------------------------
def run_leiden(labels, W, resolution=1.0):
    # build igraph Graph
    N = len(labels)
    # create igraph with weighted edges from upper triangle where weight>0
    g = ig.Graph()
    g.add_vertices(N)
    edges = []
    weights = []
    for i in range(N):
        for j in range(i+1, N):
            w = float(W[i,j])
            if w > 0:
                edges.append((i,j))
                weights.append(w)
    if len(edges) == 0:
        raise RuntimeError("No edges to run Leiden on")
    g.add_edges(edges)
    g.es['weight'] = weights
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=resolution)
    # partition is a list of vertex sets
    label_to_comm = {}
    for comm_id, vs in enumerate(partition):
        for v in vs:
            label_to_comm[labels[v]] = comm_id
    return label_to_comm

# --------------------------
# Summarize communities -> top labels
# --------------------------
def summarize_communities(partition, label_counts, top_k=10):
    # partition: dict label->comm
    comm_to_labels = defaultdict(list)
    for lab, com in partition.items():
        comm_to_labels[com].append(lab)
    rows = []
    for com, labs in comm_to_labels.items():
        labs_sorted = sorted(labs, key=lambda x: -label_counts.get(x,0))
        rows.append({"community": com, "n_labels": len(labs), "top_labels": ";".join(labs_sorted[:top_k]), "all_labels": ";".join(labs_sorted)})
    df = pd.DataFrame(rows).sort_values('community')
    return df

# --------------------------
# Plot network with communities
# --------------------------
def plot_network(G, partition, outpath=result_path + "/network_plot.png", figsize=(12,10), weight_attr='weight'):
    # partition: dict label->comm
    apply_typography()
    communities = defaultdict(list)
    for n, com in partition.items():
        communities[com].append(n)
    # color map
    ncom = len(communities)
    # layout
    pos = nx.spring_layout(G, seed=42, k=0.5)
    plt.figure(figsize=figsize)
    cmap = sns.color_palette("tab20", n_colors=max(2,ncom))
    for com_id, nodes in communities.items():
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=200, node_color=[cmap[com_id % len(cmap)]], label=f"c{com_id}")
    # edges: draw thin edges
    edges = [(u,v) for u,v in G.edges() if partition.get(u) == partition.get(v)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis('off')
    plt.legend()
    finalize_figure(plt.gcf())
    plt.tight_layout()
    output_format = figure_export_formats([Path(outpath).suffix or plt.rcParams['savefig.format']])[0]
    outpath = Path(outpath).with_suffix(f'.{output_format}')
    finalize_figure(plt.gcf())
    save_figure_file(plt.gcf(), outpath, dpi=PNG_DPI if output_format == 'png' else 300)
    plt.close()

# --------------------------
# Main pipeline
# --------------------------
def main(args):
    global result_path
    result_path = str(Path(args.output_dir))
    Path(result_path).mkdir(parents=True, exist_ok=True)
    df = filter_analysis_window(pd.read_csv(args.input), year_col="publication_year",
                                date_col="publication_date")
    if 'patent_id' not in df.columns:
        raise ValueError("Input CSV must have a 'patent_id' column")
    if 'rcdc_labels' not in df.columns:
        raise ValueError("Input CSV must have a 'rcdc_labels' column")
    # parse labels
    patent_labels = []
    for cell in df['rcdc_labels']:
        patent_labels.append(parse_label_cell(cell))
    df['parsed_labels'] = patent_labels
    total_patents = len(df)
    print(f"Read {total_patents} patents, parsed labels for each.")

    # build co-occurrence
    labels, M, counts = build_cooccurrence_matrix(patent_labels)
    print(f"Found {len(labels)} unique RCDC labels.")
    # save raw co-occurrence
    co_df = pd.DataFrame(M, index=labels, columns=labels)
    co_df.to_csv(result_path + "/rcdc_cooccurrence_raw.csv")
    print("Saved rcdc_cooccurrence_raw.csv")

    # compute PMI / NPMI
    compute = args.pmi or args.npmi
    if compute:
        total = total_patents
        label_counts = {lab: counts.get(lab, 0) for lab in labels}
        pmi_mat = compute_pmi_matrix(M, label_counts, total, normalized=args.npmi)
        pmi_df = pd.DataFrame(pmi_mat, index=labels, columns=labels)
        name = "rcdc_npmi.csv" if args.npmi else "rcdc_pmi.csv"
        pmi_df.to_csv(result_path + "/" + name)
        print(f"Saved {name}")
        W = pmi_mat
    else:
        # use raw co-occurrence but remove diagonal or keep? we'll use off-diagonal
        W = M.copy()
        np.fill_diagonal(W, 0.0)

    # threshold optionally
    if args.threshold > 0:
        W[W <= args.threshold] = 0.0

    # build graph
    G = graph_from_matrix(labels, W, threshold=0.0)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Run Louvain multiple times to check stability
    partitions = []
    nreps = args.repeats
    for seed in range(nreps):
        part = run_louvain(G, random_state=seed, resolution=args.resolution)
        partitions.append(part)
    # choose the partition from the last run as canonical
    canon_partition = partitions[-1]
    # save louvain partition
    louv_df = pd.DataFrame.from_dict(canon_partition, orient='index', columns=['community']).reset_index()
    louv_df.rename(columns={'index':'label'}, inplace=True)
    louv_df.to_csv(result_path + "/rcdc_communities_louvain.csv", index=False)
    print("Saved rcdc_communities_louvain.csv")

    # summarize
    sum_louv = summarize_communities(canon_partition, counts, top_k=10)
    sum_louv.to_csv(result_path + "/cluster_label_summary_louvain.csv", index=False)
    provenance = _partition_provenance(df, patent_labels, {
        "algorithm": "python-louvain", "python_louvain_version": version("python-louvain"),
        "networkx_version": nx.__version__, "numpy_version": np.__version__,
        "normalization": "npmi" if args.npmi else "pmi" if args.pmi else "raw",
        "threshold": args.threshold, "resolution": args.resolution,
        "random_state": nreps - 1,
    })
    partition_path = Path(result_path) / "cluster_label_summary_louvain.csv"
    provenance["partition_sha256"] = sha256(partition_path.read_bytes()).hexdigest()
    partition_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Saved cluster_label_summary_louvain.csv")

    # modularity
    mod = community_louvain.modularity(canon_partition, G, weight='weight')
    print(f"Louvain modularity: {mod:.4f}")

    # compute ARI across louvain repeats
    # create label order
    label_order = labels
    def partition_to_vector(part):
        return [part.get(lab, -1) for lab in label_order]
    vecs = [partition_to_vector(p) for p in partitions]
    ari_matrix = np.zeros((len(vecs), len(vecs)))
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            ari_matrix[i,j] = adjusted_rand_score(vecs[i], vecs[j])
    ari_df = pd.DataFrame(ari_matrix)
    ari_df.to_csv(result_path + "/louvain_repeat_ari.csv", index=False)
    print("Saved louvain_repeat_ari.csv (stability across repeats)")

    # run leiden if available
    if HAS_IGRAPH:
        try:
            leiden_part = run_leiden(labels, W, resolution=args.resolution)
            leiden_df = pd.DataFrame.from_dict(leiden_part, orient='index', columns=['community']).reset_index()
            leiden_df.rename(columns={'index':'label'}, inplace=True)
            leiden_df.to_csv(result_path + "/rcdc_communities_leiden.csv", index=False)
            print("Saved rcdc_communities_leiden.csv")
            sum_leiden = summarize_communities(leiden_part, counts, top_k=10)
            sum_leiden.to_csv(result_path + "/cluster_label_summary_leiden.csv", index=False)
            print("Saved cluster_label_summary_leiden.csv")
            # ARI between louvain canonical and leiden
            ari_ld = adjusted_rand_score(partition_to_vector(canon_partition), partition_to_vector(leiden_part))
            print(f"ARI between Louvain and Leiden: {ari_ld:.4f}")
        except Exception as e:
            print("Leiden failed:", e)
    else:
        print("Leiden not available; skip.")

    # plot network with louvain partition
    plot_network(G, canon_partition, outpath=result_path + "/network_plot_louvain.png")
    print("Saved network_plot_louvain.png")

    # produce mapping CSV: community -> labels & counts
    sum_louv['modularity'] = mod
    sum_louv.to_csv(result_path + "/rcdc_macro_candidates_louvain.csv", index=False)
    print("Saved rcdc_macro_candidates_louvain.csv (candidates for macro-FOR naming)")

    # For convenience: make cluster -> dominant patent examples
    # map label -> patents that contain it
    label_to_patents = defaultdict(list)
    for pid, labs in zip(df['patent_id'], df['parsed_labels']):
        for lab in labs:
            label_to_patents[lab].append(pid)
    rows = []
    for _, row in sum_louv.iterrows():
        comm = row['community']
        labs = row['all_labels'].split(';')
        example_pids = set()
        for lab in labs[:5]:
            example_pids.update(label_to_patents.get(lab, [])[:5])
        rows.append({'community': comm, 'top_labels': row['top_labels'], 'example_patent_ids': ";".join(list(example_pids)[:20])})
    pd.DataFrame(rows).to_csv(result_path + "/community_examples.csv", index=False)
    print("Saved community_examples.csv")

    print("Done. Inspect 'rcdc_macro_candidates_louvain.csv' and 'cluster_label_summary_louvain.csv' to assign macro-FOR names.")

# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Input CSV file with patent_id and rcdc_labels")
    p.add_argument("--output-dir", default=result_path, help="Directory for cutoff-scoped analysis outputs")
    p.add_argument("--pmi", action='store_true', help="Compute PMI matrix (not normalized)")
    p.add_argument("--npmi", action='store_true', help="Compute normalized PMI (NPMI) matrix (preferred)")
    p.add_argument("--threshold", type=float, default=0.0, help="Threshold to zero-out small weights")
    p.add_argument("--resolution", type=float, default=1.0, help="Resolution parameter for Louvain/Leiden")
    p.add_argument("--repeats", type=int, default=5, help="Number of Louvain repeats for stability")
    args = p.parse_args()
    main(args)



# python3 data_analysis_04_non_academic_patents_rcdc_macro.py --input /Users/valler/Python/RA/20_years_of_ukb/data/analysis/non_academic/patent/category/patents_for_macro_rcdc.csv --npmi --threshold 0.01 --repeats 10 --resolution 1
