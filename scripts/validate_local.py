"""
Run the full pipeline on training data and compute local metric scores.
Usage: python scripts/validate_local.py --data_dir /path/to/train --backend stardist
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline
from src.data.geff_reader import load_geff
from src.metrics.evaluate import compute_combined_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to train/ directory')
    parser.add_argument('--backend', default='fast', choices=['blob', 'fast', 'stardist', 'cellpose'])
    parser.add_argument('--max_samples', type=int, default=5, help='Max samples to evaluate')
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection threshold (fast/blob backends)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    zarr_dirs = sorted(data_dir.glob('*.zarr'))[:args.max_samples]

    # Load model once, reuse across all samples
    model = None
    if args.backend == 'stardist':
        from src.segmentation.stardist_detector import load_stardist_model
        print("Loading StarDist 3D model...")
        model = load_stardist_model('3D_demo')
    elif args.backend == 'cellpose':
        from src.segmentation.cellpose_detector import load_cellpose_model
        print("Loading Cellpose model...")
        model = load_cellpose_model('cyto3')

    scores = []
    for zarr_path in zarr_dirs:
        sample_name = zarr_path.stem
        geff_path = zarr_path.parent / f'{sample_name}.geff'
        if not geff_path.exists():
            print(f'No ground truth for {sample_name}, skipping')
            continue

        print(f'\n=== {sample_name} ===')
        try:
            seg_kwargs = {'threshold': args.threshold} if args.backend in ('fast', 'blob') else {}
            G_pred = run_pipeline(str(zarr_path), backend=args.backend, model=model, seg_kwargs=seg_kwargs)
            G_gt = load_geff(str(geff_path))
            result = compute_combined_score(G_pred, G_gt)
            scores.append(result)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f'ERROR: {e}')
            import traceback; traceback.print_exc()

    if scores:
        avg = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]}
        print(f'\n=== Average over {len(scores)} samples ===')
        print(json.dumps(avg, indent=2))


if __name__ == '__main__':
    main()
