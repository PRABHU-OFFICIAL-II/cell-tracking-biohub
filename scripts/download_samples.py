"""
Download train samples from Kaggle using Bearer token + GCS redirect.
The API returns a 302 redirect to a signed GCS URL — follow it without auth headers.
"""

import json
import time
import requests
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
OUT_DIR = Path(__file__).parent.parent / "data" / "train" / "train"

SAMPLES = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "44b6_0c582fdc",
]

BASE = "https://www.kaggle.com/api/v1"
N_TIMEPOINTS = 100


def get_token():
    return json.loads(KAGGLE_JSON.read_text())["key"]


def download_file(token, kaggle_path, out_dir, retries=3):
    """Download one file: get 302 redirect from Kaggle, follow to GCS.
    kaggle_path is like 'train/sample.zarr/0/c/5/0/0/0'.
    We strip the leading 'train/' so files land at out_dir/sample.zarr/...
    """
    relative = kaggle_path.removeprefix("train/")
    dest = out_dir / relative
    if dest.exists() and dest.stat().st_size > 0:
        return True  # already done

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Slashes in path must be URL-encoded — plain slashes in URL segments return 404
    encoded_path = kaggle_path.replace("/", "%2F")
    url = f"{BASE}/competitions/data/download/{COMPETITION}/{encoded_path}"

    for attempt in range(retries):
        try:
            # Step 1: get redirect (don't follow, GCS doesn't want Kaggle auth headers)
            r1 = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                              timeout=30, allow_redirects=False)
            if r1.status_code == 404:
                return False  # file doesn't exist
            if r1.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            if r1.status_code not in (302, 301, 303, 307, 308, 200):
                print(f"  Unexpected {r1.status_code} for {kaggle_path}")
                return False

            # Step 2: follow redirect to GCS (no auth header)
            if r1.status_code == 200:
                content = r1.content
            else:
                gcs_url = r1.headers["Location"]
                r2 = requests.get(gcs_url, timeout=60, stream=True)
                r2.raise_for_status()
                content = r2.content

            dest.write_bytes(content)
            return True

        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED {kaggle_path}: {e}")
    return False


def zarr_paths(sample):
    """All downloadable paths inside a .zarr volume."""
    paths = [f"train/{sample}.zarr/0/zarr.json"]
    for t in range(N_TIMEPOINTS):
        paths.append(f"train/{sample}.zarr/0/c/{t}/0/0/0")
    return paths


def geff_paths(sample):
    """All downloadable paths inside a .geff graph.
    Exact paths confirmed by scanning the Kaggle file listing.
    Node/prop chunks use c/0 (single chunk, 1D array).
    Edge chunk uses c/0/0 (2D array).
    """
    return [
        # Group metadata
        f"train/{sample}.geff/zarr.json",
        f"train/{sample}.geff/nodes/zarr.json",
        f"train/{sample}.geff/nodes/props/zarr.json",
        f"train/{sample}.geff/nodes/props/t/zarr.json",
        f"train/{sample}.geff/nodes/props/x/zarr.json",
        f"train/{sample}.geff/nodes/props/y/zarr.json",
        f"train/{sample}.geff/nodes/props/z/zarr.json",
        f"train/{sample}.geff/edges/zarr.json",
        f"train/{sample}.geff/edges/props/zarr.json",
        # Array zarr.json metadata
        f"train/{sample}.geff/nodes/ids/zarr.json",
        f"train/{sample}.geff/nodes/props/t/values/zarr.json",
        f"train/{sample}.geff/nodes/props/x/values/zarr.json",
        f"train/{sample}.geff/nodes/props/y/values/zarr.json",
        f"train/{sample}.geff/nodes/props/z/values/zarr.json",
        f"train/{sample}.geff/edges/ids/zarr.json",
        # Data chunks (c/0 for 1D, c/0/0 for 2D)
        f"train/{sample}.geff/nodes/ids/c/0",
        f"train/{sample}.geff/nodes/props/t/values/c/0",
        f"train/{sample}.geff/nodes/props/x/values/c/0",
        f"train/{sample}.geff/nodes/props/y/values/c/0",
        f"train/{sample}.geff/nodes/props/z/values/c/0",
        f"train/{sample}.geff/edges/ids/c/0/0",
    ]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = get_token()

    for sample in SAMPLES:
        print(f"\n=== {sample} ===")

        # zarr volume
        print(f"Downloading zarr (T={N_TIMEPOINTS} chunks + metadata)...")
        ok = 0
        for path in zarr_paths(sample):
            if download_file(token, path, OUT_DIR):
                ok += 1
        print(f"  {ok}/{len(zarr_paths(sample))} files downloaded")

        # geff ground truth
        print("Downloading geff (ground truth)...")
        paths = geff_paths(sample)
        ok = sum(1 for p in paths if download_file(token, p, OUT_DIR))
        print(f"  {ok}/{len(paths)} files downloaded")

    print("\nVerifying zarr opens...")
    import zarr
    for sample in SAMPLES:
        zarr_dir = OUT_DIR / f"{sample}.zarr"
        try:
            store = zarr.open(str(zarr_dir), mode="r")
            arr = store["0"]
            t0 = arr[0]
            print(f"  {sample}.zarr: shape={arr.shape}, t0 range=[{t0.min()},{t0.max()}]")
        except Exception as e:
            print(f"  {sample}.zarr: ERROR {e}")


if __name__ == "__main__":
    main()
