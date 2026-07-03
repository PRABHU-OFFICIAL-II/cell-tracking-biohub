# Kaggle Notebook: Biohub Cell Tracking
# No internet required — reads zarr chunks directly without the zarr library.
# Uses only: numpy, scipy, networkx, numcodecs (all pre-installed on Kaggle)

import numpy as np
import networkx as nx
from pathlib import Path
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment

# ── Constants ────────────────────────────────────────────────────────────────

VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])  # z, y, x µm/voxel
MAX_LINK_DIST_UM = 7.0
CHUNK_SHAPE = (64, 256, 256)  # Z, Y, X per timepoint chunk
INPUT_DIR = Path("/kaggle/input/biohub-cell-tracking-during-development/test")
OUTPUT_CSV = Path("/kaggle/working/submission.csv")

TEST_SAMPLES = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
]

# ── Zarr chunk reader (no zarr library) ──────────────────────────────────────

def read_timepoint(zarr_dir: Path, t: int) -> np.ndarray:
    """
    Read one (Z, Y, X) uint16 volume directly from a zarr v3 chunk file.
    Chunk codec: bytes(endian=little) + blosc(zstd, bitshuffle).
    numcodecs.Blosc().decode() handles any blosc-compressed data.
    """
    chunk_path = zarr_dir / "0" / "c" / str(t) / "0" / "0" / "0"
    data = chunk_path.read_bytes()

    try:
        from numcodecs import Blosc
        decoded = Blosc().decode(data)
        return np.frombuffer(decoded, dtype='<u2').reshape(CHUNK_SHAPE).copy()
    except Exception:
        pass

    # Fallback: try zarr if somehow installed
    try:
        import zarr
        store = zarr.open(str(zarr_dir), mode="r")
        return np.array(store["0"][t])
    except Exception:
        pass

    raise RuntimeError(f"Cannot read chunk {chunk_path} — numcodecs and zarr both failed")


# ── Segmentation ──────────────────────────────────────────────────────────────

def normalize_volume(vol):
    p1, p99 = np.percentile(vol, [1, 99])
    return np.clip((vol.astype(np.float32) - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)


def detect_local_maxima_2d(img, min_distance=10, threshold=0.5):
    smoothed = gaussian_filter(img, sigma=2.5)
    neighborhood = maximum_filter(smoothed, size=min_distance * 2 + 1)
    local_max = (smoothed == neighborhood) & (smoothed >= threshold)
    return np.argwhere(local_max)  # (N, 2): y, x


def nms_3d(pts, vol, xy_dist, z_dist):
    if len(pts) == 0:
        return pts
    scores = vol[pts[:, 0], pts[:, 1], pts[:, 2]]
    order = np.argsort(-scores)
    pts = pts[order]
    kept = []
    suppressed = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if suppressed[i]:
            continue
        kept.append(i)
        z0, y0, x0 = pts[i]
        suppress = (
            (np.abs(pts[:, 0] - z0) <= z_dist) &
            (np.abs(pts[:, 1] - y0) <= xy_dist) &
            (np.abs(pts[:, 2] - x0) <= xy_dist)
        )
        suppress[i] = False
        suppressed |= suppress
    return pts[kept].astype(np.int32)


def detect_timepoint(vol, min_distance=10, threshold=0.5, z_merge_dist=5):
    vol_norm = normalize_volume(vol)
    Z = vol_norm.shape[0]
    all_pts = []
    for z in range(Z):
        pts_2d = detect_local_maxima_2d(vol_norm[z], min_distance, threshold)
        if len(pts_2d) > 0:
            z_col = np.full((len(pts_2d), 1), z, dtype=np.int32)
            all_pts.append(np.hstack([z_col, pts_2d.astype(np.int32)]))
    if not all_pts:
        return np.empty((0, 3), dtype=np.int32)
    pts = np.vstack(all_pts)
    return nms_3d(pts, vol_norm, min_distance, z_merge_dist)


# ── Tracking ──────────────────────────────────────────────────────────────────

def physical_dist(a, b):
    diff = (a - b).astype(np.float64) * VOXEL_SCALE
    return np.sqrt(np.dot(diff, diff))


def solve_lap(pts_src, pts_tgt, ids_src, ids_tgt, max_dist):
    N, M = len(pts_src), len(pts_tgt)
    BIG = 1e9
    cost = np.full((N, M), BIG)
    for i, a in enumerate(pts_src):
        for j, b in enumerate(pts_tgt):
            d = physical_dist(a, b)
            if d <= max_dist:
                cost[i, j] = d ** 2

    aug = np.full((N + M, N + M), BIG)
    aug[:N, :M] = cost
    for i in range(N):
        aug[i, M + i] = np.min(cost[i]) if np.any(cost[i] < BIG) else BIG
    for j in range(M):
        aug[N + j, j] = np.min(cost[:, j]) if np.any(cost[:, j] < BIG) else BIG
    sub = cost.copy(); sub[sub >= BIG] = 0
    aug[N:, M:] = sub.T

    row_ind, col_ind = linear_sum_assignment(aug)
    return [
        (int(ids_src[r]), int(ids_tgt[c]))
        for r, c in zip(row_ind, col_ind)
        if r < N and c < M and aug[r, c] < 1e8
    ]


def track(detections, max_dist=MAX_LINK_DIST_UM, max_gap=2):
    G = nx.DiGraph()
    node_counter = 0
    frame_nodes = []

    for t, dets in enumerate(detections):
        ids = []
        for det in dets:
            G.add_node(node_counter, t=t, z=int(det[0]), y=int(det[1]), x=int(det[2]))
            ids.append(node_counter)
            node_counter += 1
        frame_nodes.append(np.array(ids, dtype=np.int64))

    T = len(detections)

    def pts(ids):
        return np.array([[G.nodes[n]["z"], G.nodes[n]["y"], G.nodes[n]["x"]] for n in ids])

    for t in range(T - 1):
        s, tgt = frame_nodes[t], frame_nodes[t + 1]
        if len(s) and len(tgt):
            G.add_edges_from(solve_lap(pts(s), pts(tgt), s, tgt, max_dist))

    for gap in range(2, max_gap + 1):
        for t in range(T - gap):
            s = [n for n in frame_nodes[t] if G.out_degree(n) == 0]
            tgt = [n for n in frame_nodes[t + gap] if G.in_degree(n) == 0]
            if s and tgt:
                G.add_edges_from(solve_lap(
                    pts(s), pts(tgt), np.array(s), np.array(tgt), max_dist * gap))

    return G


def prune_invalid_divisions(G):
    to_remove = []
    for n in list(G.nodes):
        succs = list(G.successors(n))
        if len(succs) > 2:
            nc = np.array([G.nodes[n]["z"], G.nodes[n]["y"], G.nodes[n]["x"]], dtype=np.float64)
            dists = [physical_dist(nc, np.array([G.nodes[s]["z"], G.nodes[s]["y"], G.nodes[s]["x"]], dtype=np.float64))
                     for s in succs]
            for idx in np.argsort(dists)[2:]:
                to_remove.append((n, succs[idx]))
    G.remove_edges_from(to_remove)
    return G


# ── Submission builder ────────────────────────────────────────────────────────

def build_submission(graphs):
    import csv
    rows = []
    row_id = 0
    for dataset, G in graphs.items():
        for node_id, data in G.nodes(data=True):
            rows.append([row_id, dataset, "node", int(node_id),
                         int(data["t"]), int(data["z"]), int(data["y"]), int(data["x"]), -1, -1])
            row_id += 1
        for src, tgt in G.edges():
            rows.append([row_id, dataset, "edge", -1, -1, -1, -1, -1, int(src), int(tgt)])
            row_id += 1

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

graphs = {}
for sample in TEST_SAMPLES:
    zarr_dir = INPUT_DIR / f"{sample}.zarr"
    print(f"\n=== {sample} ===")

    detections = []
    for t in range(100):
        vol = read_timepoint(zarr_dir, t)
        pts = detect_timepoint(vol, min_distance=10, threshold=0.5)
        detections.append(pts)
        if (t + 1) % 25 == 0:
            print(f"  t={t+1}/100 — {sum(len(d) for d in detections)} cells so far")

    total = sum(len(d) for d in detections)
    print(f"Segmented: {total} cells across 100 frames")

    G = track(detections)
    G = prune_invalid_divisions(G)
    n_divs = sum(1 for n in G.nodes if G.out_degree(n) >= 2)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {n_divs} divisions")
    graphs[sample] = G

build_submission(graphs)
print("\nDone.")
