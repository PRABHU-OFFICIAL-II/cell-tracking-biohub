"""
Download a few train samples from Kaggle using Bearer token auth.
Constructs file paths directly from zarr v3 spec instead of listing all files.
zarr layout: <sample>.zarr/0/c/{t}/0/0/0 for T=0..99, plus zarr.json metadata
geff layout:  <sample>.geff/zarr.json + nodes/ids + nodes/props/{t,z,y,x}/values + edges/ids
"""

import json
import time
import requests
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
OUT_DIR = Path(__file__).parent.parent / "data" / "train"

SAMPLES = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "44b6_0c582fdc",
]

BASE = "https://www.kaggle.com/api/v1"
N_TIMEPOINTS = 100  # T dimension per volume


def get_token():
    return json.loads(KAGGLE_JSON.read_text())["key"]


def zarr_files(sample):
    """All file paths inside <sample>.zarr that we need."""
    paths = [
        f"train/{sample}.zarr/zarr.json",
        f"train/{sample}.zarr/0/zarr.json",
    ]
    for t in range(N_TIMEPOINTS):
        paths.append(f"train/{sample}.zarr/0/c/{t}/0/0/0")
    return paths


def geff_files(sample):
    """All file paths inside <sample>.geff that we need."""
    return [
        f"train/{sample}.geff/zarr.json",
        f"train/{sample}.geff/nodes/ids",
        f"train/{sample}.geff/nodes/ids/zarr.json",
        f"train/{sample}.geff/nodes/props/t/values",
        f"train/{sample}.geff/nodes/props/t/zarr.json",
        f"train/{sample}.geff/nodes/props/z/values",
        f"train/{sample}.geff/nodes/props/z/zarr.json",
        f"train/{sample}.geff/nodes/props/y/values",
        f"train/{sample}.geff/nodes/props/y/zarr.json",
        f"train/{sample}.geff/nodes/props/x/values",
        f"train/{sample}.geff/nodes/props/x/zarr.json",
        f"train/{sample}.geff/edges/ids",
        f"train/{sample}.geff/edges/ids/zarr.json",
    ]


def download_file(session, token, file_name, out_dir, retries=3):
    url = f"{BASE}/competitions/data/download/{COMPETITION}/{file_name}"
    dest = out_dir / file_name
    if dest.exists():
        return  # already downloaded
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        try:
            r = session.get(url, stream=True, timeout=60)
            if r.status_code == 404:
                return  # file simply doesn't exist (sparse geff structure)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    f.write(chunk)
            return
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"  FAILED {file_name}: {e}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = get_token()

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    for sample in SAMPLES:
        print(f"\n=== {sample} ===")

        # zarr volume: metadata files first, then all timepoint chunks
        meta_files = [
            f"train/{sample}.zarr/zarr.json",
            f"train/{sample}.zarr/0/zarr.json",
        ]
        print(f"Downloading zarr metadata ({len(meta_files)} files)...")
        for f in meta_files:
            download_file(session, token, f, OUT_DIR)

        print(f"Downloading zarr chunks (T=0..{N_TIMEPOINTS-1})...")
        for t in range(N_TIMEPOINTS):
            fpath = f"train/{sample}.zarr/0/c/{t}/0/0/0"
            download_file(session, token, fpath, OUT_DIR)
            if (t + 1) % 10 == 0:
                print(f"  {t+1}/{N_TIMEPOINTS} timepoints done")

        # geff ground truth
        gf = geff_files(sample)
        print(f"Downloading geff ({len(gf)} files)...")
        for f in gf:
            download_file(session, token, f, OUT_DIR)
        print(f"  Done.")

    print(f"\nAll downloads complete.")
    print(f"Zarr dirs: {list(OUT_DIR.glob('*.zarr'))}")
    print(f"Geff dirs: {list(OUT_DIR.glob('*.geff'))}")


if __name__ == "__main__":
    main()
