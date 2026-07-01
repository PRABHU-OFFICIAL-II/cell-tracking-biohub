"""
StarDist 3D cell detection.
Uses the pretrained '3D_demo' model (available offline via stardist package).
Fine-tunable on competition training data.

Install: pip install stardist tensorflow
"""

import numpy as np
from typing import List, Optional


def load_stardist_model(model_name: str = "3D_demo", basedir: Optional[str] = None):
    """Load a StarDist 3D model. Uses pretrained weights by default."""
    from stardist.models import StarDist3D
    if basedir:
        model = StarDist3D(None, name=model_name, basedir=basedir)
    else:
        model = StarDist3D.from_pretrained(model_name)
    return model


def normalize_percentile(vol: np.ndarray, pmin: float = 1.0, pmax: float = 99.8) -> np.ndarray:
    """StarDist standard normalization."""
    from csbdeep.utils import normalize
    return normalize(vol, pmin, pmax)


def detect_timepoint(model, vol: np.ndarray, prob_thresh: float = 0.5, nms_thresh: float = 0.4) -> np.ndarray:
    """
    Detect cells in one (Z, Y, X) volume.

    Returns:
        centroids: (N, 3) int array of [z, y, x]
    """
    vol_norm = normalize_percentile(vol.astype(np.float32))
    labels, details = model.predict_instances(vol_norm, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    # details['points'] gives [z, y, x] centroids of each detected instance
    centroids = details["points"].astype(np.int32)
    return centroids


def detect_all_timepoints(
    model,
    volume: np.ndarray,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.4,
) -> List[np.ndarray]:
    """
    Run StarDist detection on every timepoint.

    Args:
        model: loaded StarDist3D model
        volume: (T, Z, Y, X) uint16

    Returns:
        List of length T, each element is (N_t, 3) int array [z, y, x]
    """
    T = volume.shape[0]
    detections = []
    for t in range(T):
        centroids = detect_timepoint(model, volume[t], prob_thresh, nms_thresh)
        detections.append(centroids)
    return detections
