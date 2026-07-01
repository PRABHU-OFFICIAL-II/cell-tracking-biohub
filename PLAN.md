# Biohub Cell Tracking — Competition Plan

## Competition Summary
- **Task**: Detect cells in 3D+time fluorescence microscopy, link them across frames, detect cell divisions
- **Data**: Zarr v3 volumes, shape (T=100, Z=64, Y=256, X=256), uint16, zebrafish embryo
- **Metric**: Combined score = 0.5 * Edge Jaccard + 0.5 * Division Jaccard
- **Deadline**: September 29, 2026 final submission
- **Format**: Kaggle notebook, no internet, ≤12h GPU runtime

---

## Architecture Overview

```
[Zarr Volume] → [Preprocessing] → [Cell Detection/Segmentation] → [Tracking/Linking] → [Division Detection] → [Submission CSV]
```

### Three-stage pipeline:
1. **Segmentation**: Detect cell instances per timepoint (3D blobs → centroids)
2. **Tracking**: Link detections across time via optimal assignment (LAP / graph matching)
3. **Division**: Identify mother cells that split into two daughters

---

## Phase 1 — Data Loading & EDA (Week 1)

### Goals
- Read and visualize zarr v3 volumes
- Parse .geff ground-truth graphs
- Understand cell density, motion speed, division frequency
- Compute ground-truth statistics: how many cells/frame, avg displacement, division rate

### Key decisions
- Physical scale: z=1.625, y=x=0.40625 µm/voxel → normalize distances accordingly
- Sparse labels: ground truth only covers a fraction of cells → metric accounts for this

### Files
- `src/data/zarr_reader.py` — load zarr volume as numpy/dask array
- `src/data/geff_reader.py` — parse geff graph into networkx DiGraph
- `notebooks/01_eda.ipynb` — visualization and statistics

---

## Phase 2 — Segmentation (Weeks 2–3)

### Strategy A (Baseline): Classical blob detection
- 3D Difference-of-Gaussians (DoG) or Laplacian-of-Gaussian (LoG) blob detection
- `skimage.feature.blob_log` applied per timepoint
- Output: list of (z, y, x, sigma) per timepoint
- Fast, no GPU needed, easy to tune

### Strategy B (Better): 3D U-Net instance segmentation
- Use **StarDist 3D** with pretrained fluorescence weights (`'3D_demo'` or `'3D_fluo'`)
- Predicts star-convex polyhedra → extract centroids + instance masks
- Fine-tune last layers on competition training data if time permits
- `pip install stardist tensorflow`

### Strategy C (Best if time allows): **Cellpose 3D**
- `cellpose` model with `model_type='cyto3'`, use 3D mode
- Returns instance segmentation masks
- Extract centroids from masks
- Works well on dense populations

### Fallback ensemble
- Run A + B, merge detections via NMS (non-max suppression) in physical space
- Keep detections with consensus between methods

### Files
- `src/segmentation/blob_detector.py`
- `src/segmentation/stardist_detector.py`
- `src/segmentation/cellpose_detector.py`
- `src/segmentation/nms.py`

---

## Phase 3 — Tracking (Weeks 3–5)

### Core approach: Linear Assignment Problem (LAP) — same as TrackMate, ISBI standard
Two-step LAP:
1. **Frame-to-frame linking**: For each pair of consecutive frames, solve assignment minimizing cost = squared distance (in physical space). Max gap = 7.0 µm per metric spec.
2. **Gap closing**: Allow cells to disappear for 1–2 frames and be re-linked (handles detection failures)

### Library: **laptrack** or **scipy.optimize.linear_sum_assignment**
- For each pair of frames: build cost matrix (N×M), cap at 7 µm, solve assignment
- Physical distance: `sqrt((dz*1.625)^2 + (dy*0.40625)^2 + (dx*0.40625)^2)`

### Advanced: **ultrack** (by competition organizers Jordão Bragantini / Royer Lab)
- Hierarchical ILP-based tracking that is robust to dense populations
- Directly handles divisions as bifurcations in the tracking graph
- `pip install ultrack`
- Strong candidate for best score given it's the tool the organizers likely benchmarked against

### Tracking graph output
- NetworkX DiGraph: nodes = (dataset, node_id, t, z, y, x), edges = (source_id, target_id)
- Division = node with out-degree ≥ 2

### Files
- `src/tracking/lap_tracker.py` — frame-to-frame LAP + gap closing
- `src/tracking/ultrack_wrapper.py` — ultrack interface
- `src/tracking/graph_utils.py` — build/export networkx graph

---

## Phase 4 — Division Detection (Week 5–6)

### Division criteria
- Morphological: mother cell → two daughters in next frame
- Tracking-based: a node with 2 outgoing edges in the tracking graph
- Signal-based: fluorescence patterns change before division (optional)

### Strategy
1. From tracking graph, any node with 2+ outgoing edges is a division
2. Post-process: enforce biologically realistic constraints
   - Daughters should be roughly half the volume of mother
   - Daughter centroids should be within expected distance of mother
   - Division should not repeat too frequently

### Files
- `src/tracking/division_detector.py`

---

## Phase 5 — Metric Implementation & Local Validation (Week 4, parallel)

### Implement the competition metric locally
- Edge Jaccard: bipartite matching per timepoint, count TP/FP/FN edges
- Division Jaccard: find connected components covering pre-split + both daughters
- This is critical: iterate quickly without submitting to Kaggle

### Files
- `src/metrics/edge_jaccard.py`
- `src/metrics/division_jaccard.py`
- `src/metrics/evaluate.py`

---

## Phase 6 — Optimization & Hyperparameter Tuning (Weeks 6–8)

### Parameters to tune
- Detection: blob sigma range, threshold, min distance
- Tracking: max link distance, gap penalty, division cost
- NMS: IoU threshold, score threshold

### Tools
- Optuna for hyperparameter optimization on training set
- Cross-validation by embryo ID (train/val split must be embryo-disjoint)

---

## Phase 7 — Kaggle Notebook Packaging (Week 8–9)

### Requirements
- No internet access during rerun
- All models saved as Kaggle Dataset artifacts
- Runtime ≤ 12h GPU
- Output: `submission.csv`

### Strategy
- Package pretrained model weights as a Kaggle Dataset
- Single notebook: install deps → load model → segment → track → write CSV
- Profile runtime on train set to stay under 12h limit
- Optimize: batch processing, skip already-done samples, use GPU efficiently

### Files
- `kaggle_notebook/solution.ipynb` — the final submission notebook

---

## Winning Strategy Summary

| Component | Approach | Why |
|-----------|----------|-----|
| Segmentation | StarDist 3D + Cellpose 3D ensemble | Proven SOTA on 3D fluorescence |
| Tracking | ultrack (ILP) + LAP fallback | Built by organizers, robust to dense scenes |
| Division | Graph bifurcation + spatial constraints | Direct metric alignment |
| Validation | Local metric implementation | Fast iteration |
| Tuning | Optuna, embryo-disjoint CV | Avoid leakage |

**Key insight**: The evaluation metric specifically rewards correct *edges* not just detections — a tracker that links cells confidently and avoids spurious edges will outscore one with more detections. Prioritize precision over recall in edge predictions.

**Division insight**: Division Jaccard is a binary metric per event — missing a division is costly. Use conservative thresholds (better to predict too many divisions than miss them).

---

## Tech Stack
- **Python 3.10+**
- **zarr, zarr-developers/zarr-python** (v3 support)
- **numpy, scipy, scikit-image** (classical processing)
- **stardist** (3D cell segmentation)
- **cellpose** (3D instance segmentation)
- **ultrack** (ILP tracking)
- **laptrack** (LAP tracking)
- **networkx** (graph construction)
- **optuna** (hyperparameter tuning)
- **torch / tensorflow** (model inference)
- **pandas** (CSV output)

---

## Risk Log
| Risk | Mitigation |
|------|-----------|
| ultrack too slow on GPU | Fall back to LAP, profile on 1 sample |
| StarDist doesn't generalize | Fine-tune on training data |
| Division detection poor | Add feature-based division classifier |
| 12h runtime exceeded | Profile each stage, parallelize or simplify detection |
| Metric score < 1.0 (sparse labels inflate it) | Trust local metric, aim for > 0.8 on labeled nodes |
