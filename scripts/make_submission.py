"""
Generate submission.csv from test data.

Usage:
  # Run on already-downloaded data
  python scripts/make_submission.py --test_dir data/test --output submission.csv

  # Limit samples for quick test
  python scripts/make_submission.py --test_dir data/test --output submission.csv --max_samples 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline
from src.submission.build_csv import build_submission, validate_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True, help='Path to test/ directory with .zarr files')
    parser.add_argument('--output', default='submission.csv', help='Output CSV path')
    parser.add_argument('--backend', default='fast', choices=['blob', 'fast', 'stardist', 'cellpose'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold (fast/blob)')
    parser.add_argument('--min_distance', type=int, default=10, help='Min distance in px (fast backend)')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit number of samples (for testing)')
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    zarr_dirs = sorted(test_dir.glob('*.zarr'))
    if args.max_samples:
        zarr_dirs = zarr_dirs[:args.max_samples]

    print(f"Found {len(zarr_dirs)} test samples")
    if not zarr_dirs:
        print("No .zarr files found. Check --test_dir path.")
        return

    # Load model once if needed
    model = None
    if args.backend == 'stardist':
        from src.segmentation.stardist_detector import load_stardist_model
        print("Loading StarDist model...")
        model = load_stardist_model('3D_demo')
    elif args.backend == 'cellpose':
        from src.segmentation.cellpose_detector import load_cellpose_model
        print("Loading Cellpose model...")
        model = load_cellpose_model('cyto3')

    seg_kwargs = {}
    if args.backend in ('fast', 'blob'):
        seg_kwargs['threshold'] = args.threshold
    if args.backend == 'fast':
        seg_kwargs['min_distance'] = args.min_distance

    graphs = {}
    for i, zarr_path in enumerate(zarr_dirs):
        dataset_name = zarr_path.stem
        print(f"\n[{i+1}/{len(zarr_dirs)}] {dataset_name}")
        try:
            G = run_pipeline(str(zarr_path), backend=args.backend, model=model, seg_kwargs=seg_kwargs)
            graphs[dataset_name] = G
            print(f"  → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\nBuilding submission for {len(graphs)} datasets...")
    df = build_submission(graphs, output_path=args.output)

    expected = [p.stem for p in zarr_dirs if p.stem in graphs]
    validate_submission(df, expected)


if __name__ == '__main__':
    main()
