"""
Division detection and post-processing.
A division = a node with 2+ outgoing edges in the tracking graph.
We apply biological constraints to prune false positive divisions.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple

VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])


def physical_distance(a_node: dict, b_node: dict) -> float:
    a = np.array([a_node["z"], a_node["y"], a_node["x"]]) * VOXEL_SCALE
    b = np.array([b_node["z"], b_node["y"], b_node["x"]]) * VOXEL_SCALE
    return float(np.linalg.norm(a - b))


def get_division_nodes(G: nx.DiGraph) -> List[int]:
    """Return all node IDs that have 2+ outgoing edges."""
    return [n for n in G.nodes if G.out_degree(n) >= 2]


def validate_division(G: nx.DiGraph, mother: int, max_daughter_dist: float = 15.0) -> bool:
    """
    Check if a division event is biologically plausible:
    - Exactly 2 daughters
    - Both daughters are within max_daughter_dist µm of the mother
    - Daughters are separated from each other (not the same cell)
    """
    daughters = list(G.successors(mother))
    if len(daughters) != 2:
        return False

    d0, d1 = daughters
    m_data = G.nodes[mother]
    d0_data = G.nodes[d0]
    d1_data = G.nodes[d1]

    # Both daughters must be near mother
    dist_m_d0 = physical_distance(m_data, d0_data)
    dist_m_d1 = physical_distance(m_data, d1_data)
    if dist_m_d0 > max_daughter_dist or dist_m_d1 > max_daughter_dist:
        return False

    # Daughters should be separated (not identical detections)
    dist_d0_d1 = physical_distance(d0_data, d1_data)
    if dist_d0_d1 < 1.0:
        return False

    return True


def prune_invalid_divisions(G: nx.DiGraph, max_daughter_dist: float = 15.0) -> nx.DiGraph:
    """
    Remove edges that form implausible divisions.
    When a division fails validation, remove the lower-confidence daughter edge
    (the one farther from the mother).
    """
    G = G.copy()
    for mother in get_division_nodes(G):
        if not validate_division(G, mother, max_daughter_dist):
            daughters = list(G.successors(mother))
            m_data = G.nodes[mother]
            # Keep closest daughter, remove the rest
            dists = [(physical_distance(m_data, G.nodes[d]), d) for d in daughters]
            dists.sort()
            for _, d in dists[1:]:
                G.remove_edge(mother, d)
    return G


def add_division_edges_from_proximity(
    G: nx.DiGraph,
    max_division_dist: float = 12.0,
) -> nx.DiGraph:
    """
    Heuristic: if a node has 0 outgoing edges but two nearby nodes in the next frame
    have no incoming edges, tentatively add a division.
    Conservative — only adds when evidence is strong.
    """
    T_max = max(nx.get_node_attributes(G, "t").values(), default=0)

    by_t = {}
    for n, data in G.nodes(data=True):
        by_t.setdefault(data["t"], []).append(n)

    G = G.copy()
    for t in range(T_max):
        frame_nodes = by_t.get(t, [])
        next_frame = by_t.get(t + 1, [])
        no_out = [n for n in frame_nodes if G.out_degree(n) == 0]
        no_in = [n for n in next_frame if G.in_degree(n) == 0]

        if len(no_in) < 2:
            continue

        for mother in no_out:
            m_data = G.nodes[mother]
            orphans_near = []
            for cand in no_in:
                if physical_distance(m_data, G.nodes[cand]) <= max_division_dist:
                    orphans_near.append(cand)
            if len(orphans_near) == 2:
                # Exactly 2 nearby orphans → likely division
                G.add_edge(mother, orphans_near[0])
                G.add_edge(mother, orphans_near[1])

    return G
