"""
Classical 3D blob detection using Laplacian-of-Gaussian (LoG).
Fast baseline that requires no GPU and no pretrained model.
Returns centroids (t, z, y, x) with confidence scores.
"""

import numpy as np
from skimage.feature import blob_log
from skimage.filters import gaussian
from typing import List, Tuple


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Normalize uint16 to float [0, 1] using 1st/99th percentile."""
    p1, p99 = np.percentile(vol, [1, 99])
    clipped = np.clip(vol.astype(np.float32), p1, p99)
    return (clipped - p1) / (p99 - p1 + 1e-8)


def detect_blobs_log(
    vol: np.ndarray,
    min_sigma: float = 2.0,
    max_sigma: float = 6.0,
    num_sigma: int = 5,
    threshold: float = 0.05,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Detect 3D cell blobs via LoG.

    Args:
        vol: (Z, Y, X) float32 normalized volume
        min_sigma, max_sigma: sigma range for LoG scale space
        threshold: minimum blob intensity threshold
        overlap: suppress blobs with IoU > overlap

    Returns:
        array of shape (N, 4): [z, y, x, sigma] per detection
    """
    blobs = blob_log(
        vol,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
    )
    return blobs  # shape (N, 4): z, y, x, sigma


def detect_all_timepoints(
    volume: np.ndarray,
    **kwargs,
) -> List[np.ndarray]:
    """
    Run blob detection on every timepoint.

    Args:
        volume: (T, Z, Y, X) uint16 array

    Returns:
        List of length T, each element is (N_t, 3) array of [z, y, x]
    """
    T = volume.shape[0]
    detections = []
    for t in range(T):
        frame = normalize_volume(volume[t])
        blobs = detect_blobs_log(frame, **kwargs)
        if blobs.size > 0:
            coords = blobs[:, :3].astype(np.int32)  # z, y, x
        else:
            coords = np.empty((0, 3), dtype=np.int32)
        detections.append(coords)
    return detections
