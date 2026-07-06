"""
Linear Assignment Problem (LAP) tracker.
Frame-to-frame linking + gap closing, same principle as TrackMate / ISBI cell tracking.

Physical scale: z=1.625, y=x=0.40625 µm/voxel.
Max link distance: 7.0 µm (per competition metric).
"""

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from typing import List, Optional

VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])  # z, y, x µm/voxel
MAX_LINK_DIST_UM = 7.0
MAX_GAP_FRAMES = 2  # allow cells to disappear for up to 2 frames


def physical_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance in micrometres between two (z, y, x) voxel coordinates."""
    diff = (a - b).astype(np.float64) * VOXEL_SCALE
    return np.sqrt(np.dot(diff, diff))


def build_cost_matrix(
    pts_src: np.ndarray,
    pts_tgt: np.ndarray,
    max_dist: float = MAX_LINK_DIST_UM,
) -> np.ndarray:
    """
    Build cost matrix for LAP assignment.
    Cost = squared physical distance; inf if > max_dist.

    Args:
        pts_src: (N, 3) [z, y, x] at frame t
        pts_tgt: (M, 3) [z, y, x] at frame t+1

    Returns:
        (N+M, N+M) augmented cost matrix (TrackMate-style with dummy nodes)
    """
    N, M = len(pts_src), len(pts_tgt)
    BIG = 1e9

    cost = np.full((N, M), BIG)
    for i, a in enumerate(pts_src):
        for j, b in enumerate(pts_tgt):
            d = physical_distance(a, b)
            if d <= max_dist:
                cost[i, j] = d ** 2

    # Standard TrackMate-style LAP augmentation:
    # Top-left (N×M):     real linking costs
    # Top-right (N×N):    diagonal — source termination (death)
    # Bottom-left (M×M):  diagonal — target initiation (birth)
    # Bottom-right (M×N): transpose of top-left with BIG→0
    aug = np.full((N + M, N + M), BIG)
    aug[:N, :M] = cost                          # top-left: real costs

    # Top-right (N×N diagonal): cost for source i to terminate
    for i in range(N):
        aug[i, M + i] = np.min(cost[i]) if np.any(cost[i] < BIG) else BIG

    # Bottom-left (M×M diagonal): cost for target j to be born
    for j in range(M):
        aug[N + j, j] = np.min(cost[:, j]) if np.any(cost[:, j] < BIG) else BIG

    # Bottom-right (M×N): transpose of top-left, BIG replaced by 0
    sub = cost.copy()
    sub[sub >= BIG] = 0
    aug[N:, M:] = sub.T

    return aug, N, M


def solve_lap_frame_pair(
    pts_src: np.ndarray,
    pts_tgt: np.ndarray,
    node_ids_src: np.ndarray,
    node_ids_tgt: np.ndarray,
    max_dist: float = MAX_LINK_DIST_UM,
) -> List[tuple]:
    """
    Solve assignment between two frames.
    Returns list of (src_node_id, tgt_node_id) edge pairs.
    """
    if len(pts_src) == 0 or len(pts_tgt) == 0:
        return []

    aug, N, M = build_cost_matrix(pts_src, pts_tgt, max_dist)
    row_ind, col_ind = linear_sum_assignment(aug)

    edges = []
    for r, c in zip(row_ind, col_ind):
        if r < N and c < M and aug[r, c] < 1e8:
            edges.append((int(node_ids_src[r]), int(node_ids_tgt[c])))
    return edges


def track(
    detections: List[np.ndarray],
    max_dist: float = MAX_LINK_DIST_UM,
    max_gap: int = MAX_GAP_FRAMES,
) -> nx.DiGraph:
    """
    Full LAP tracking pipeline.

    Args:
        detections: List[T] of (N_t, 3) arrays, each row = [z, y, x]
        max_dist: maximum link distance in µm
        max_gap: max number of frames a cell can be missing

    Returns:
        NetworkX DiGraph with node attrs (t, z, y, x) and directed edges
    """
    G = nx.DiGraph()
    node_counter = 0
    frame_nodes = []

    # Create nodes for every detection
    for t, dets in enumerate(detections):
        ids = []
        for det in dets:
            G.add_node(node_counter, t=t, z=int(det[0]), y=int(det[1]), x=int(det[2]))
            ids.append(node_counter)
            node_counter += 1
        frame_nodes.append(np.array(ids, dtype=np.int64))

    T = len(detections)

    # Frame-to-frame linking (gap = 1)
    for t in range(T - 1):
        src_ids = frame_nodes[t]
        tgt_ids = frame_nodes[t + 1]
        if len(src_ids) == 0 or len(tgt_ids) == 0:
            continue
        src_pts = np.array([[G.nodes[n]["z"], G.nodes[n]["y"], G.nodes[n]["x"]] for n in src_ids])
        tgt_pts = np.array([[G.nodes[n]["z"], G.nodes[n]["y"], G.nodes[n]["x"]] for n in tgt_ids])
        edges = solve_lap_frame_pair(src_pts, tgt_pts, src_ids, tgt_ids, max_dist)
        G.add_edges_from(edges)

    # Gap closing: link unlinked nodes across skipped frames
    if max_gap > 1:
        for gap in range(2, max_gap + 1):
            for t in range(T - gap):
                src_ids = frame_nodes[t]
                tgt_ids = frame_nodes[t + gap]
                # Only attempt gap closing for nodes without existing outgoing/incoming edges
                src_unlinked = [n for n in src_ids if G.out_degree(n) == 0]
                tgt_unlinked = [n for n in tgt_ids if G.in_degree(n) == 0]
                if not src_unlinked or not tgt_unlinked:
                    continue
                src_pts = np.array([[G.nodes[n]["z"], G.nodes[n]["y"], G.nodes[n]["x"]] for n in src_unlinked])
                tgt_pts = np.array([[G.nodes[n]["z"], G.nodes[n]["y"], G.nodes[n]["x"]] for n in tgt_unlinked])
                # Slightly relax distance for gap frames
                edges = solve_lap_frame_pair(
                    src_pts, tgt_pts,
                    np.array(src_unlinked), np.array(tgt_unlinked),
                    max_dist * gap,
                )
                G.add_edges_from(edges)

    return G
