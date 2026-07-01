"""
Build the final submission.csv from a dict of {dataset_name: nx.DiGraph}.
Output format:
  id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
"""

import pandas as pd
import networkx as nx
from typing import Dict
from pathlib import Path


def graph_to_submission_rows(dataset: str, G: nx.DiGraph) -> list:
    """Convert one tracking graph to submission rows for a single dataset."""
    rows = []

    # Node rows
    for node_id, data in G.nodes(data=True):
        rows.append({
            "dataset": dataset,
            "row_type": "node",
            "node_id": int(node_id),
            "t": int(data["t"]),
            "z": int(data["z"]),
            "y": int(data["y"]),
            "x": int(data["x"]),
            "source_id": -1,
            "target_id": -1,
        })

    # Edge rows
    for src, tgt in G.edges():
        rows.append({
            "dataset": dataset,
            "row_type": "edge",
            "node_id": -1,
            "t": -1,
            "z": -1,
            "y": -1,
            "x": -1,
            "source_id": int(src),
            "target_id": int(tgt),
        })

    return rows


def build_submission(
    graphs: Dict[str, nx.DiGraph],
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """
    Build submission CSV from all dataset graphs.

    Args:
        graphs: {dataset_name: tracking_graph}
        output_path: where to save the CSV

    Returns:
        DataFrame of the submission
    """
    all_rows = []
    for dataset, G in graphs.items():
        all_rows.extend(graph_to_submission_rows(dataset, G))

    df = pd.DataFrame(all_rows, columns=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
    df.insert(0, "id", range(len(df)))

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    return df


def validate_submission(df: pd.DataFrame, expected_datasets: list) -> bool:
    """Quick sanity check on a submission DataFrame."""
    ok = True

    # Check all expected datasets are present
    present = set(df["dataset"].unique())
    for ds in expected_datasets:
        if ds not in present:
            print(f"MISSING dataset: {ds}")
            ok = False

    # Check id is consecutive
    if list(df["id"]) != list(range(len(df))):
        print("id column is not consecutive integers")
        ok = False

    # Node rows should have node_id >= 0
    nodes = df[df["row_type"] == "node"]
    if (nodes["node_id"] < 0).any():
        print("Some node rows have node_id < 0")
        ok = False

    # Edge rows should have source/target >= 0
    edges = df[df["row_type"] == "edge"]
    if (edges["source_id"] < 0).any() or (edges["target_id"] < 0).any():
        print("Some edge rows have negative source/target")
        ok = False

    if ok:
        print("Submission looks valid.")
    return ok
