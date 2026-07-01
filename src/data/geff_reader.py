"""
Read .geff ground-truth tracking graphs.
.geff is a zarr v3 directory with:
  nodes/ids — node ID array
  nodes/props/{t,z,y,x}/values — integer centroid coordinates
  edges/ids — (N, 2) array of (source_id, target_id)
Annotations are sparse — not every cell is labeled.
"""

import zarr
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Union


def load_geff(geff_path: Union[str, Path]) -> nx.DiGraph:
    """Parse a .geff directory into a NetworkX DiGraph.

    Node attributes: t, z, y, x (integer voxel coordinates)
    Edges: directed, source → target (temporal order)
    """
    store = zarr.open(str(geff_path), mode="r")

    node_ids = np.array(store["nodes/ids"])
    t_vals = np.array(store["nodes/props/t/values"])
    z_vals = np.array(store["nodes/props/z/values"])
    y_vals = np.array(store["nodes/props/y/values"])
    x_vals = np.array(store["nodes/props/x/values"])

    edge_ids = np.array(store["edges/ids"])  # shape (N, 2)

    G = nx.DiGraph()
    for i, nid in enumerate(node_ids):
        G.add_node(int(nid), t=int(t_vals[i]), z=int(z_vals[i]), y=int(y_vals[i]), x=int(x_vals[i]))

    for src, tgt in edge_ids:
        G.add_edge(int(src), int(tgt))

    return G


def get_division_nodes(G: nx.DiGraph) -> list:
    """Return node IDs that have 2+ outgoing edges (cell divisions)."""
    return [n for n in G.nodes if G.out_degree(n) >= 2]


def graph_to_arrays(G: nx.DiGraph):
    """Return (nodes_df, edges_array) suitable for submission building."""
    import pandas as pd
    rows = []
    for nid, data in G.nodes(data=True):
        rows.append({"node_id": nid, "t": data["t"], "z": data["z"], "y": data["y"], "x": data["x"]})
    nodes_df = pd.DataFrame(rows)
    edges = np.array(list(G.edges()), dtype=np.int64)
    return nodes_df, edges
