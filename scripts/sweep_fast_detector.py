"""
Sweep fast detector parameters (threshold, min_distance) on 1 sample.
Cell diameter in zebrafish embryo: ~10-20µm → ~25-50px at 0.40625µm/px.
min_distance should be ~cell_radius in pixels.
Usage: python scripts/sweep_fast_detector.py --data_dir data/train/train
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.zarr_reader import load_zarr_volume
from src.data.geff_reader import load_geff
from src.segmentation.fast_detector import detect_timepoint_fast
from src.tracking.lap_tracker import track
from src.tracking.division_detector import prune_invalid_divisions
from src.metrics.evaluate import compute_combined_score
import numpy as np


# min_distance in pixels (cell radius); cells are ~10-20µm, XY scale=0.40625µm/px
# 10µm radius → 24px, 7µm → 17px, 5µm → 12px
MIN_DISTANCES = [10, 15, 20, 25]
THRESHOLDS = [0.2, 0.3, 0.4, 0.5]


def run_with_params(volume, threshold, min_distance):
    T = volume.shape[0]
    detections = []
    for t in range(T):
        frame = np.array(volume[t])
        pts = detect_timepoint_fast(frame, min_distance=min_distance, threshold=threshold)
        detections.append(pts)

    total = sum(len(d) for d in detections)
    avg_per_frame = total / T

    G = track(detections)
    G = prune_invalid_divisions(G)
    return G, avg_per_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--sample_idx', type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    zarr_path = sorted(data_dir.glob('*.zarr'))[args.sample_idx]
    sample = zarr_path.stem
    geff_path = data_dir / f'{sample}.geff'

    print(f'Sample: {sample}')
    print(f'{"min_dist":>8} {"thresh":>7} {"cells/f":>8} {"combined":>10} {"edgeJ":>8} {"TP":>5} {"FP":>8}')
    print('-' * 65)

    volume = load_zarr_volume(str(zarr_path))
    G_gt = load_geff(str(geff_path))

    best_score = -1
    best_params = None

    for min_dist in MIN_DISTANCES:
        for thresh in THRESHOLDS:
            G_pred, avg = run_with_params(volume, thresh, min_dist)
            result = compute_combined_score(G_pred, G_gt)
            c = result['combined']
            ej = result['edge_jaccard']
            tp = result['edge_TP']
            fp = result['edge_FP']
            print(f'{min_dist:>8} {thresh:>7.2f} {avg:>8.1f} {c:>10.4f} {ej:>8.4f} {tp:>5.0f} {fp:>8.0f}')
            if c > best_score:
                best_score = c
                best_params = (min_dist, thresh)

    print(f'\nBest: min_distance={best_params[0]}, threshold={best_params[1]} → combined={best_score:.4f}')


if __name__ == '__main__':
    main()
