"""
Cellpose 3D cell detection.
Uses cyto3 model in 3D mode.

Install: pip install cellpose
"""

import numpy as np
from typing import List, Optional
from skimage import measure


def load_cellpose_model(model_type: str = "cyto3", gpu: bool = True):
    """Load Cellpose model. Works with cellpose v3+ (CellposeModel) and v2 (Cellpose)."""
    from cellpose import models
    if hasattr(models, 'CellposeModel'):
        # cellpose v3+
        return models.CellposeModel(model_type=model_type, gpu=gpu)
    else:
        # cellpose v2
        return models.Cellpose(model_type=model_type, gpu=gpu)


def normalize_percentile(vol: np.ndarray, pmin: float = 1.0, pmax: float = 99.8) -> np.ndarray:
    p1, p99 = np.percentile(vol, [pmin, pmax])
    return np.clip((vol.astype(np.float32) - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)


def masks_to_centroids(masks: np.ndarray) -> np.ndarray:
    """Extract centroids from a 3D label image."""
    props = measure.regionprops(masks)
    if not props:
        return np.empty((0, 3), dtype=np.int32)
    centroids = np.array([p.centroid for p in props], dtype=np.int32)  # z, y, x
    return centroids


def detect_timepoint(
    model,
    vol: np.ndarray,
    diameter: float = 15.0,
    anisotropy: float = 4.0,
    do_3D: bool = True,
) -> np.ndarray:
    """
    Detect cells in one (Z, Y, X) volume using Cellpose 3D.
    anisotropy = z_scale / xy_scale = 1.625 / 0.40625 = 4.0

    Returns:
        centroids: (N, 3) int array [z, y, x]
    """
    vol_norm = normalize_percentile(vol)
    result = model.eval(
        vol_norm,
        diameter=diameter,
        do_3D=do_3D,
        anisotropy=anisotropy,
        channels=[0, 0],
        z_axis=0,
    )
    # v3 returns (masks, flows, styles), v2 same — unpack safely
    masks = result[0] if isinstance(result, (list, tuple)) else result
    return masks_to_centroids(masks)


def detect_all_timepoints(
    model,
    volume: np.ndarray,
    diameter: float = 15.0,
    anisotropy: float = 4.0,
) -> List[np.ndarray]:
    """
    Run Cellpose on every timepoint.

    Returns:
        List of length T, each element is (N_t, 3) int array [z, y, x]
    """
    T = volume.shape[0]
    detections = []
    for t in range(T):
        centroids = detect_timepoint(model, volume[t], diameter=diameter, anisotropy=anisotropy)
        detections.append(centroids)
    return detections
