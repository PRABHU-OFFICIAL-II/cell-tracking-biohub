# Biohub Cell Tracking — Competition Solution

Kaggle: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)  
Deadline: September 29, 2026

---

## Setup (run once on any machine)

```bash
git clone <your-repo-url>
cd cell-tracking-biohub

# Install dependencies
pip install -r requirements.txt

# Download 3 training samples (requires kaggle.json at ~/.kaggle/kaggle.json)
python scripts/download_samples.py

# Create the missing root zarr.json metadata files
python scripts/create_zarr_metadata.py
```

---

## Run baseline (CPU, no GPU needed)

```bash
python scripts/validate_local.py --data_dir data/train/train --backend blob --max_samples 3
```

---

## Run StarDist 3D (GPU recommended)

```bash
pip install stardist tensorflow

python scripts/validate_local.py --data_dir data/train/train --backend stardist --max_samples 3
```

---

## Run Cellpose 3D (GPU recommended)

```bash
pip install cellpose

python scripts/validate_local.py --data_dir data/train/train --backend cellpose --max_samples 3
```

---

## Hyperparameter tuning

```bash
pip install optuna

python scripts/tune_hyperparams.py --data_dir data/train/train --n_trials 50 --max_samples 10
```

---

## Build submission CSV locally (for testing)

```bash
python -c "
from src.pipeline import run_pipeline
from src.submission.build_csv import build_submission
import os

data_dir = 'data/train/train'
graphs = {}
for name in os.listdir(data_dir):
    if name.endswith('.zarr'):
        dataset = name.replace('.zarr', '')
        graphs[dataset] = run_pipeline(f'{data_dir}/{name}', backend='stardist')

build_submission(graphs, 'submission.csv')
"
```

---

## Project structure

```
src/
  data/           zarr + geff readers
  segmentation/   blob, stardist, cellpose detectors
  tracking/       LAP tracker, ultrack wrapper, division detector
  metrics/        local metric implementation (edge + division Jaccard)
  submission/     CSV builder
  pipeline.py     end-to-end orchestrator

scripts/
  download_samples.py     download training data from Kaggle
  create_zarr_metadata.py create missing zarr root metadata
  validate_local.py       run pipeline + score on local training data
  tune_hyperparams.py     Optuna hyperparameter search

kaggle_notebook/
  solution.ipynb          final Kaggle submission notebook

configs/
  default.yaml            hyperparameter config
```

---

## GPU machine checklist

1. `git pull` to get latest code
2. `pip install -r requirements.txt`
3. `python scripts/download_samples.py` (first time only)
4. `python scripts/create_zarr_metadata.py` (first time only)
5. Run the command Claude gives you for the current experiment
6. Paste the output back to Claude
