"""
Full end-to-end pipeline for one dataset sample.
Orchestrates: load → segment → track → detect divisions → build graph
"""

import numpy as np
import networkx as nx
from pathlib import Path
from typing import Optional, Literal

from .data.zarr_reader import load_zarr_volume, load_timepoint, VOXEL_SCALE
from .tracking.lap_tracker import track as lap_track
from .tracking.division_detector import prune_invalid_divisions, add_division_edges_from_proximity
from .submission.build_csv import graph_to_submission_rows


SegmentationBackend = Literal["blob", "stardist", "cellpose"]


def run_segmentation(
    volume,  # zarr.Array, shape (T, Z, Y, X)
    backend: SegmentationBackend = "stardist",
    model=None,
    **kwargs,
) -> list:
    """
    Run cell detection on all timepoints.
    Returns List[T] of (N_t, 3) int arrays [z, y, x].
    """
    from tqdm import tqdm
    T = volume.shape[0]
    all_dets = []

    if backend == "blob":
        from .segmentation.blob_detector import detect_blobs_log, normalize_volume
        for t in tqdm(range(T), desc="Blob detect"):
            frame = normalize_volume(np.array(volume[t]))
            blobs = detect_blobs_log(frame, **kwargs)
            coords = blobs[:, :3].astype(np.int32) if blobs.size > 0 else np.empty((0, 3), dtype=np.int32)
            all_dets.append(coords)

    elif backend == "stardist":
        from .segmentation.stardist_detector import detect_timepoint, normalize_percentile
        for t in tqdm(range(T), desc="StarDist detect"):
            frame = np.array(volume[t])
            coords = detect_timepoint(model, frame, **kwargs)
            all_dets.append(coords)

    elif backend == "cellpose":
        from .segmentation.cellpose_detector import detect_timepoint
        for t in tqdm(range(T), desc="Cellpose detect"):
            frame = np.array(volume[t])
            coords = detect_timepoint(model, frame, **kwargs)
            all_dets.append(coords)

    return all_dets


def run_pipeline(
    zarr_path: str,
    backend: SegmentationBackend = "stardist",
    model=None,
    use_ultrack: bool = False,
    max_link_dist: float = 7.0,
    max_gap: int = 2,
    seg_kwargs: Optional[dict] = None,
) -> nx.DiGraph:
    """
    Full tracking pipeline for one zarr volume.

    Args:
        zarr_path: path to .zarr directory
        backend: segmentation method
        model: pretrained model (required for stardist/cellpose)
        use_ultrack: use ILP-based ultrack instead of LAP
        max_link_dist: max link distance in µm
        max_gap: max gap frames for gap closing
        seg_kwargs: extra kwargs passed to the segmentation function

    Returns:
        Tracking graph as NetworkX DiGraph
    """
    seg_kwargs = seg_kwargs or {}

    # Auto-load model if not provided
    if model is None:
        if backend == "stardist":
            from .segmentation.stardist_detector import load_stardist_model
            print("Loading StarDist 3D model (3D_demo)...")
            model = load_stardist_model("3D_demo")
        elif backend == "cellpose":
            from .segmentation.cellpose_detector import load_cellpose_model
            print("Loading Cellpose model (cyto3)...")
            model = load_cellpose_model("cyto3")

    volume = load_zarr_volume(zarr_path)
    print(f"Loaded {zarr_path}: shape={volume.shape}")

    print("Running segmentation...")
    detections = run_segmentation(volume, backend=backend, model=model, **seg_kwargs)
    total_dets = sum(len(d) for d in detections)
    print(f"Detected {total_dets} cells across {volume.shape[0]} timepoints")

    if use_ultrack:
        print("Running ultrack (ILP)...")
        from .tracking.ultrack_wrapper import probability_map_from_detections, track_with_ultrack
        shape = volume.shape[1:]  # Z, Y, X
        foreground_maps = [
            probability_map_from_detections(dets, shape)
            for dets in detections
        ]
        G = track_with_ultrack(foreground_maps, max_distance=max_link_dist)
    else:
        print("Running LAP tracker...")
        G = lap_track(detections, max_dist=max_link_dist, max_gap=max_gap)

    print(f"Tracking graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Post-processing divisions...")
    G = prune_invalid_divisions(G)
    G = add_division_edges_from_proximity(G)

    n_divs = sum(1 for n in G.nodes if G.out_degree(n) >= 2)
    print(f"Divisions detected: {n_divs}")

    return G
