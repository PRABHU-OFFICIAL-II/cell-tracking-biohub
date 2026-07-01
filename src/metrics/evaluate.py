"""
Local implementation of the competition metric.
Combined score = 0.5 * edge_jaccard + 0.5 * division_jaccard

Edge Jaccard:
- Per timepoint: bipartite match predicted nodes to GT nodes by centroid distance (max 7 µm)
- A predicted edge is TP when both endpoints match GT nodes connected by a GT edge
- Jaccard = TP / (TP + FP + FN), with penalty for over-predicting node count

Division Jaccard:
- Per GT division: check if predicted graph has a component covering the pre-split node
  and touching both daughter lineages
- Micro-averaged across all divisions

Physical scale: z=1.625, y=x=0.40625 µm/voxel
Max matching distance: 7.0 µm
"""

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple

VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])
MAX_MATCH_DIST = 7.0


def _phys_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b) * VOXEL_SCALE))


def _node_coords(G: nx.DiGraph, n: int) -> np.ndarray:
    d = G.nodes[n]
    return np.array([d["z"], d["y"], d["x"]], dtype=np.float64)


def match_nodes_per_timepoint(
    G_pred: nx.DiGraph,
    G_gt: nx.DiGraph,
    t: int,
    max_dist: float = MAX_MATCH_DIST,
) -> Dict[int, int]:
    """
    Optimal bipartite matching of predicted → GT nodes at timepoint t.
    Returns dict pred_node_id → gt_node_id.
    """
    pred_nodes = [n for n, d in G_pred.nodes(data=True) if d.get("t") == t]
    gt_nodes = [n for n, d in G_gt.nodes(data=True) if d.get("t") == t]

    if not pred_nodes or not gt_nodes:
        return {}

    N, M = len(pred_nodes), len(gt_nodes)
    BIG = 1e9
    cost = np.full((N, M), BIG)

    for i, pn in enumerate(pred_nodes):
        for j, gn in enumerate(gt_nodes):
            d = _phys_dist(_node_coords(G_pred, pn), _node_coords(G_gt, gn))
            if d <= max_dist:
                cost[i, j] = d

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < BIG:
            mapping[pred_nodes[r]] = gt_nodes[c]
    return mapping


def compute_edge_jaccard(
    G_pred: nx.DiGraph,
    G_gt: nx.DiGraph,
) -> Tuple[float, int, int, int]:
    """
    Compute edge Jaccard with over-prediction penalty.
    Returns (jaccard, TP, FP, FN).
    """
    T_vals = set(d["t"] for _, d in G_gt.nodes(data=True))

    # Build full pred→gt node mapping across all timepoints
    full_mapping: Dict[int, int] = {}
    for t in T_vals:
        full_mapping.update(match_nodes_per_timepoint(G_pred, G_gt, t))

    gt_edge_set = set(G_gt.edges())
    pred_edges = list(G_pred.edges())

    TP = 0
    FP = 0
    for src, tgt in pred_edges:
        gt_src = full_mapping.get(src)
        gt_tgt = full_mapping.get(tgt)
        if gt_src is not None and gt_tgt is not None and (gt_src, gt_tgt) in gt_edge_set:
            TP += 1
        else:
            FP += 1

    FN = len(gt_edge_set) - TP

    # Over-prediction penalty: if |pred_nodes| > |gt_nodes|, add extra FP
    n_pred = G_pred.number_of_nodes()
    n_gt = G_gt.number_of_nodes()
    if n_pred > n_gt:
        FP += (n_pred - n_gt)

    denom = TP + FP + FN
    jaccard = TP / denom if denom > 0 else 0.0
    return jaccard, TP, FP, FN


def compute_division_jaccard(
    G_pred: nx.DiGraph,
    G_gt: nx.DiGraph,
) -> Tuple[float, int, int, int]:
    """
    Compute division Jaccard.
    Returns (jaccard, TP, FP, FN).
    """
    T_vals = set(d["t"] for _, d in G_gt.nodes(data=True))
    full_mapping: Dict[int, int] = {}
    for t in T_vals:
        full_mapping.update(match_nodes_per_timepoint(G_pred, G_gt, t))

    reverse_mapping = {v: k for k, v in full_mapping.items()}  # gt→pred

    # GT divisions: GT nodes with 2+ outgoing edges
    gt_divisions = [n for n in G_gt.nodes if G_gt.out_degree(n) >= 2]

    TP = 0
    FN = 0
    for gt_mother in gt_divisions:
        gt_daughters = list(G_gt.successors(gt_mother))

        pred_mother = reverse_mapping.get(gt_mother)
        if pred_mother is None:
            FN += 1
            continue

        # Check if predicted graph covers pre-split + both daughter lineages
        pred_daughters = [reverse_mapping.get(d) for d in gt_daughters]
        if all(d is not None for d in pred_daughters):
            # Check connectivity in predicted graph
            covered = all(nx.has_path(G_pred, pred_mother, d) for d in pred_daughters if d is not None)
            if covered:
                TP += 1
            else:
                FN += 1
        else:
            FN += 1

    # FP: predicted divisions that don't match any GT division
    pred_divisions = [n for n in G_pred.nodes if G_pred.out_degree(n) >= 2]
    matched_pred = set()
    for gt_mother in gt_divisions:
        pred_mother = reverse_mapping.get(gt_mother)
        if pred_mother is not None:
            matched_pred.add(pred_mother)
    FP = sum(1 for n in pred_divisions if n not in matched_pred)

    denom = TP + FP + FN
    jaccard = TP / denom if denom > 0 else 0.0
    return jaccard, TP, FP, FN


def compute_combined_score(G_pred: nx.DiGraph, G_gt: nx.DiGraph) -> dict:
    """Compute the full combined competition score."""
    edge_j, e_tp, e_fp, e_fn = compute_edge_jaccard(G_pred, G_gt)
    div_j, d_tp, d_fp, d_fn = compute_division_jaccard(G_pred, G_gt)
    combined = 0.5 * edge_j + 0.5 * div_j
    return {
        "combined": combined,
        "edge_jaccard": edge_j,
        "division_jaccard": div_j,
        "edge_TP": e_tp, "edge_FP": e_fp, "edge_FN": e_fn,
        "div_TP": d_tp, "div_FP": d_fp, "div_FN": d_fn,
    }
