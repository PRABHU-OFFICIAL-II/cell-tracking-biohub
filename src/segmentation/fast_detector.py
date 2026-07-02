"""
Fast 3D cell detector using 2D local maxima on Gaussian-smoothed slices.
Target: <5 seconds per (64, 256, 256) timepoint on CPU.
Strategy:
  1. Gaussian smooth each Z-slice to suppress noise
  2. Find 2D local maxima per slice (fast scipy operation)
  3. Merge nearby detections across Z via simple NMS
  4. Filter by intensity threshold
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import label as nd_label
from typing import List


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(vol, [1, 99])
    return np.clip((vol.astype(np.float32) - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)


def detect_local_maxima_2d(
    img: np.ndarray,
    min_distance: int = 5,
    threshold: float = 0.3,
) -> np.ndarray:
    """Find local maxima in a 2D image. Returns (N, 2) array of [y, x]."""
    smoothed = gaussian_filter(img, sigma=1.5)
    neighborhood = maximum_filter(smoothed, size=min_distance * 2 + 1)
    local_max = (smoothed == neighborhood) & (smoothed >= threshold)
    coords = np.argwhere(local_max)  # (N, 2): y, x
    return coords


def detect_timepoint_fast(
    vol: np.ndarray,
    min_distance: int = 5,
    threshold: float = 0.3,
    z_merge_dist: int = 3,
) -> np.ndarray:
    """
    Detect cells in one (Z, Y, X) volume using fast 2D local maxima per slice.

    Returns:
        centroids: (N, 3) int array [z, y, x]
    """
    vol_norm = normalize_volume(vol)
    Z = vol_norm.shape[0]

    all_pts = []
    for z in range(Z):
        pts_2d = detect_local_maxima_2d(vol_norm[z], min_distance, threshold)
        if len(pts_2d) > 0:
            z_col = np.full((len(pts_2d), 1), z, dtype=np.int32)
            pts_3d = np.hstack([z_col, pts_2d.astype(np.int32)])  # (N, 3): z, y, x
            all_pts.append(pts_3d)

    if not all_pts:
        return np.empty((0, 3), dtype=np.int32)

    pts = np.vstack(all_pts)

    # Merge detections that are within z_merge_dist in Z and min_distance in XY
    # Simple greedy NMS: sort by intensity, suppress nearby points
    pts = _nms_3d(pts, vol_norm, min_distance, z_merge_dist)
    return pts


def _nms_3d(
    pts: np.ndarray,
    vol: np.ndarray,
    xy_dist: int,
    z_dist: int,
) -> np.ndarray:
    """Greedy 3D NMS: keep highest-intensity point, suppress neighbors."""
    if len(pts) == 0:
        return pts

    scores = vol[pts[:, 0], pts[:, 1], pts[:, 2]]
    order = np.argsort(-scores)
    pts = pts[order]
    scores = scores[order]

    kept = []
    suppressed = np.zeros(len(pts), dtype=bool)

    for i in range(len(pts)):
        if suppressed[i]:
            continue
        kept.append(i)
        z0, y0, x0 = pts[i]
        dz = np.abs(pts[:, 0] - z0)
        dy = np.abs(pts[:, 1] - y0)
        dx = np.abs(pts[:, 2] - x0)
        suppress = (dz <= z_dist) & (dy <= xy_dist) & (dx <= xy_dist)
        suppress[i] = False
        suppressed |= suppress

    return pts[kept].astype(np.int32)


def detect_all_timepoints_fast(
    volume,  # zarr.Array or np.ndarray (T, Z, Y, X)
    threshold: float = 0.3,
    min_distance: int = 5,
) -> List[np.ndarray]:
    """
    Run fast detector on all timepoints.
    Returns List[T] of (N_t, 3) int arrays [z, y, x].
    """
    from tqdm import tqdm
    T = volume.shape[0]
    detections = []
    for t in tqdm(range(T), desc="Fast detect"):
        frame = np.array(volume[t])
        pts = detect_timepoint_fast(frame, min_distance=min_distance, threshold=threshold)
        detections.append(pts)
    return detections
