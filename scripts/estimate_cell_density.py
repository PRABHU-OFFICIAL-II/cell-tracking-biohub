"""
Estimate true cell density from the fluorescence signal.
Strategy: look at the GT-annotated cells and measure their signal properties,
then estimate how many cells of similar brightness exist per frame.
Also checks how many local maxima exist at different thresholds.

Usage: python scripts/estimate_cell_density.py --data_dir data/train/train
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, label

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.zarr_reader import load_zarr_volume
from src.data.geff_reader import load_geff


def estimate_from_gt_signal(volume, G_gt):
    """
    At each GT-annotated timepoint, measure the signal at GT cell locations
    and count how many voxels exceed a similar brightness threshold.
    """
    from collections import defaultdict
    by_t = defaultdict(list)
    for n, d in G_gt.nodes(data=True):
        by_t[d['t']].append((d['z'], d['y'], d['x']))

    gt_signals = []
    for t, cells in list(by_t.items())[:10]:
        frame = np.array(volume[t]).astype(np.float32)
        p1, p99 = np.percentile(frame, [1, 99])
        frame_norm = np.clip((frame - p1) / (p99 - p1 + 1e-8), 0, 1)
        for z, y, x in cells:
            gt_signals.append(frame_norm[z, y, x])

    if not gt_signals:
        return None

    gt_signals = np.array(gt_signals)
    print(f"GT cell signal stats (normalized): min={gt_signals.min():.3f}, "
          f"mean={gt_signals.mean():.3f}, median={np.median(gt_signals):.3f}, "
          f"p25={np.percentile(gt_signals, 25):.3f}")
    return gt_signals


def count_blobs_at_threshold(volume, t, threshold, min_distance=10):
    """Count 2D local maxima per frame at given threshold."""
    from scipy.ndimage import maximum_filter
    frame = np.array(volume[t]).astype(np.float32)
    p1, p99 = np.percentile(frame, [1, 99])
    frame_norm = np.clip((frame - p1) / (p99 - p1 + 1e-8), 0, 1)

    # Count connected components above threshold as a rough cell count
    smoothed = gaussian_filter(frame_norm, sigma=2.0)
    binary = smoothed > threshold
    labeled, n_components = label(binary)
    return n_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    zarr_dirs = sorted(data_dir.glob('*.zarr'))[:1]  # use first sample

    zarr_path = zarr_dirs[0]
    sample = zarr_path.stem
    geff_path = data_dir / f'{sample}.geff'

    print(f'Sample: {sample}')
    volume = load_zarr_volume(str(zarr_path))
    G_gt = load_geff(str(geff_path))

    print(f'Volume shape: {volume.shape}')
    print(f'GT nodes: {G_gt.number_of_nodes()}, spanning t={min(d["t"] for _,d in G_gt.nodes(data=True))}'
          f'..{max(d["t"] for _,d in G_gt.nodes(data=True))}')
    print()

    # Measure GT cell signals
    gt_signals = estimate_from_gt_signal(volume, G_gt)
    print()

    # Count blobs at different thresholds across a few timepoints
    print("Connected components above threshold (rough cell count estimate):")
    print(f"{'thresh':>8} " + " ".join(f"  t={t}" for t in [0, 10, 20, 30]))
    print("-" * 50)
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        counts = [count_blobs_at_threshold(volume, t, thresh) for t in [0, 10, 20, 30]]
        print(f"{thresh:>8.1f} " + " ".join(f"{c:6d}" for c in counts))

    # Also check a specific well-known t from the GT
    print()
    print("Checking signal distribution at t=0:")
    frame = np.array(volume[0]).astype(np.float32)
    p1, p99 = np.percentile(frame, [1, 99])
    frame_norm = np.clip((frame - p1) / (p99 - p1 + 1e-8), 0, 1)
    for pct in [50, 60, 70, 80, 90, 95, 99]:
        val = np.percentile(frame_norm, pct)
        print(f"  p{pct}: {val:.3f}")


if __name__ == '__main__':
    main()
