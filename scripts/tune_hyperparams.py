"""
Hyperparameter tuning with Optuna.
Optimizes over segmentation thresholds and tracking distances.
Usage: python scripts/tune_hyperparams.py --data_dir /path/to/train --n_trials 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def objective(trial, zarr_paths, geff_paths):
    from src.pipeline import run_pipeline
    from src.data.geff_reader import load_geff
    from src.metrics.evaluate import compute_combined_score

    # Sample hyperparameters
    max_link = trial.suggest_float('max_link_dist', 3.0, 12.0)
    max_gap = trial.suggest_int('max_gap', 1, 3)
    prob_thresh = trial.suggest_float('stardist_prob_thresh', 0.3, 0.7)

    scores = []
    for zarr_path, geff_path in zip(zarr_paths, geff_paths):
        try:
            G_pred = run_pipeline(
                str(zarr_path),
                backend='stardist',
                max_link_dist=max_link,
                max_gap=max_gap,
                seg_kwargs={'prob_thresh': prob_thresh},
            )
            G_gt = load_geff(str(geff_path))
            result = compute_combined_score(G_pred, G_gt)
            scores.append(result['combined'])
        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=10)
    args = parser.parse_args()

    import optuna
    data_dir = Path(args.data_dir)
    zarr_paths = sorted(data_dir.glob('*.zarr'))[:args.max_samples]
    geff_paths = [p.parent / f'{p.stem}.geff' for p in zarr_paths]
    valid = [(z, g) for z, g in zip(zarr_paths, geff_paths) if g.exists()]
    zarr_paths, geff_paths = zip(*valid) if valid else ([], [])

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, zarr_paths, geff_paths), n_trials=args.n_trials)

    print('\nBest params:', study.best_params)
    print('Best score:', study.best_value)
    optuna.visualization.plot_optimization_history(study).show()


if __name__ == '__main__':
    main()
