"""
Cell detector using connected component centroids on thresholded signal.
More robust than local maxima: gives exactly one detection per bright blob.

Strategy:
1. Gaussian smooth to suppress noise
2. Threshold at high percentile (bright cells)
3. Label connected components in 3D
4. Filter by size (remove noise)
5. Return centroid of each component
"""

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.ndimage import find_objects


def detect_timepoint_blob_centroid(
    vol: np.ndarray,
    percentile: float = 95.0,
    min_voxels: int = 10,
    max_voxels: int = 50000,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Detect cells via connected components on bright signal.

    Args:
        vol: (Z, Y, X) uint16 volume
        percentile: intensity threshold percentile (95 → top 5% of signal)
        min_voxels: minimum component size to count as a cell
        max_voxels: maximum component size (removes merged/background blobs)
        sigma: gaussian smoothing sigma in voxels

    Returns:
        (N, 3) int array of [z, y, x] centroids
    """
    vol_f = vol.astype(np.float32)
    smoothed = gaussian_filter(vol_f, sigma=sigma)

    thresh = np.percentile(smoothed, percentile)
    binary = smoothed >= thresh

    labeled, n = label(binary)
    if n == 0:
        return np.empty((0, 3), dtype=np.int32)

    centroids = []
    slices = find_objects(labeled)
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        component = labeled[sl] == (i + 1)
        size = component.sum()
        if size < min_voxels or size > max_voxels:
            continue
        # Centroid within the slice bounding box
        coords = np.argwhere(component)
        centroid = coords.mean(axis=0)
        # Offset back to full volume coordinates
        offset = np.array([sl[0].start, sl[1].start, sl[2].start])
        centroid = (centroid + offset).astype(np.int32)
        centroids.append(centroid)

    if not centroids:
        return np.empty((0, 3), dtype=np.int32)

    return np.array(centroids, dtype=np.int32)
