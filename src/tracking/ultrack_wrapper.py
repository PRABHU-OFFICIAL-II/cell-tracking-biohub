"""
Wrapper around ultrack for ILP-based hierarchical tracking.
ultrack is built by Jordão Bragantini (competition organizer, Royer Lab).
It is robust to dense cell populations and directly handles divisions.

Install: pip install ultrack
Docs: https://github.com/royerlab/ultrack

Expected input: foreground segmentation mask (or probability map) + contour map per timepoint.
Output: tracked graph exported as pandas DataFrame → converted to our NetworkX DiGraph.
"""

import numpy as np
import networkx as nx
from typing import List, Optional


def build_ultrack_config(
    max_distance: float = 7.0,
    max_neighbors: int = 5,
    n_workers: int = 4,
) -> dict:
    """
    Build ultrack MainConfig with sensible defaults for zebrafish embryo tracking.
    max_distance in µm.
    """
    from ultrack.config import MainConfig, SegmentationConfig, LinkingConfig, TrackingConfig

    cfg = MainConfig()
    cfg.segmentation_config.max_distance = max_distance
    cfg.linking_config.max_distance = max_distance
    cfg.linking_config.max_neighbors = max_neighbors
    cfg.tracking_config.n_workers = n_workers
    return cfg


def probability_map_from_detections(
    detections: np.ndarray,
    shape: tuple,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Convert (N, 3) centroid detections to a soft probability map via Gaussian blobs.
    Useful for feeding into ultrack when we only have centroid detections.
    shape: (Z, Y, X)
    """
    from scipy.ndimage import gaussian_filter
    prob = np.zeros(shape, dtype=np.float32)
    for z, y, x in detections:
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            prob[z, y, x] = 1.0
    return gaussian_filter(prob, sigma=sigma)


def track_with_ultrack(
    foreground_maps: List[np.ndarray],
    contour_maps: Optional[List[np.ndarray]] = None,
    max_distance: float = 7.0,
) -> nx.DiGraph:
    """
    Run ultrack tracking on a sequence of foreground probability maps.

    Args:
        foreground_maps: List[T] of (Z, Y, X) float32 maps in [0, 1]
        contour_maps: optional List[T] of (Z, Y, X) float32 contour/boundary maps
        max_distance: max linking distance in µm

    Returns:
        NetworkX DiGraph with node attrs (t, z, y, x)
    """
    import ultrack
    from ultrack import track, to_tracks_layer
    from ultrack.config import MainConfig

    cfg = build_ultrack_config(max_distance=max_distance)

    T = len(foreground_maps)
    if contour_maps is None:
        contour_maps = [np.zeros_like(fm) for fm in foreground_maps]

    # ultrack expects (T, Z, Y, X) stacks
    foreground = np.stack(foreground_maps, axis=0)
    contours = np.stack(contour_maps, axis=0)

    tracks_df, graph = track(
        foreground=foreground,
        contours=contours,
        config=cfg,
        scale=np.array([1.625, 0.40625, 0.40625]),
    )

    return _tracks_df_to_graph(tracks_df)


def _tracks_df_to_graph(tracks_df) -> nx.DiGraph:
    """Convert ultrack output DataFrame to our NetworkX DiGraph format."""
    G = nx.DiGraph()

    for _, row in tracks_df.iterrows():
        nid = int(row["track_id"])
        G.add_node(nid, t=int(row["t"]), z=int(row["z"]), y=int(row["y"]), x=int(row["x"]))

    # Build edges from parent_track_id column if present
    if "parent_track_id" in tracks_df.columns:
        for _, row in tracks_df.iterrows():
            parent = row.get("parent_track_id")
            if parent is not None and parent >= 0:
                G.add_edge(int(parent), int(row["track_id"]))

    return G
