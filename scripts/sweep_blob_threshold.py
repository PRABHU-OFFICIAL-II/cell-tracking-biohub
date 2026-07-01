"""
Quick sweep of blob detection threshold to find the sweet spot.
Runs on 1 sample, tests multiple thresholds, reports combined score each time.
Usage: python scripts/sweep_blob_threshold.py --data_dir data/train/train
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.zarr_reader import load_zarr_volume
from src.data.geff_reader import load_geff
from src.segmentation.blob_detector import detect_all_timepoints
from src.tracking.lap_tracker import track
from src.tracking.division_detector import prune_invalid_divisions
from src.metrics.evaluate import compute_combined_score
import networkx as nx
import numpy as np


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def run_with_threshold(volume, threshold):
    import numpy as np
    from src.segmentation.blob_detector import normalize_volume, detect_blobs_log

    T = volume.shape[0]
    detections = []
    for t in range(T):
        frame = normalize_volume(np.array(volume[t]))
        blobs = detect_blobs_log(frame, threshold=threshold)
        coords = blobs[:, :3].astype(np.int32) if blobs.size > 0 else np.empty((0, 3), dtype=np.int32)
        detections.append(coords)

    total = sum(len(d) for d in detections)
    avg_per_frame = total / T

    G = track(detections)
    G = prune_invalid_divisions(G)
    return G, avg_per_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    zarr_path = sorted(data_dir.glob('*.zarr'))[0]
    sample = zarr_path.stem
    geff_path = data_dir / f'{sample}.geff'

    print(f'Sample: {sample}')
    print(f'{"Threshold":>10} {"Cells/frame":>12} {"Combined":>10} {"EdgeJ":>8} {"DivJ":>8} {"TP":>6} {"FP":>8}')
    print('-' * 70)

    volume = load_zarr_volume(str(zarr_path))
    G_gt = load_geff(str(geff_path))

    best_score = -1
    best_thresh = None
    for thresh in THRESHOLDS:
        G_pred, avg = run_with_threshold(volume, thresh)
        result = compute_combined_score(G_pred, G_gt)
        c = result['combined']
        ej = result['edge_jaccard']
        dj = result['division_jaccard']
        tp = result['edge_TP']
        fp = result['edge_FP']
        print(f'{thresh:>10.2f} {avg:>12.1f} {c:>10.4f} {ej:>8.4f} {dj:>8.4f} {tp:>6.0f} {fp:>8.0f}')
        if c > best_score:
            best_score = c
            best_thresh = thresh

    print(f'\nBest threshold: {best_thresh} (combined={best_score:.4f})')


if __name__ == '__main__':
    main()
