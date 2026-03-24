#!/usr/bin/env python3
"""
rcdc_macro_for_pipeline.py

Input: CSV with columns:
 - patent_id
 - rcdc_labels  (either JSON list like '["625","439"]' OR semicolon/comma separated like "625;439" or "625,439")

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
from collections import Counter, defaultdict
import math
import os

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# community detection
import community as community_louvain   # python-louvain
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

# Optional leiden
HAS_IGRAPH = False
try:
    import igraph as ig
    import leidenalg
    HAS_IGRAPH = True
except Exception:
    HAS_IGRAPH = False


result_path = "/Users/valler/Python/RA/20_years_of_ukb/file/paten_rcdc_macro"
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
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# --------------------------
# Main pipeline
# --------------------------
def main(args):
    df = pd.read_csv(args.input)
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
    p.add_argument("--pmi", action='store_true', help="Compute PMI matrix (not normalized)")
    p.add_argument("--npmi", action='store_true', help="Compute normalized PMI (NPMI) matrix (preferred)")
    p.add_argument("--threshold", type=float, default=0.0, help="Threshold to zero-out small weights")
    p.add_argument("--resolution", type=float, default=1.0, help="Resolution parameter for Louvain/Leiden")
    p.add_argument("--repeats", type=int, default=5, help="Number of Louvain repeats for stability")
    args = p.parse_args()
    main(args)



# python3 patent_rcdc_macro.py --input /Users/valler/Python/RA/20_years_of_ukb/data/patent/category/patents_for_macro_rcdc.csv --npmi --threshold 0.01 --repeats 10 --resolution 1