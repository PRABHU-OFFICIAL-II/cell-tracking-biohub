"""
Download the 4 test zarr samples from Kaggle for submission generation.

Two samples (44b6_*) are shared with training data, so we can reuse those.
New samples (6bba_*) must be downloaded fresh.

Usage: python scripts/download_test.py
"""

import json
import time
import requests
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
OUT_DIR = Path(__file__).parent.parent / "data" / "test"
BASE = "https://www.kaggle.com/api/v1"
N_TIMEPOINTS = 100

# All 4 test datasets confirmed from sample_submission.csv
TEST_SAMPLES = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
]


def get_token():
    return json.loads(KAGGLE_JSON.read_text())["key"]


def download_file(token, kaggle_path, out_dir, retries=3):
    """Download kaggle_path (e.g. 'test/sample.zarr/0/c/5/0/0/0') to out_dir."""
    relative = kaggle_path.removeprefix("test/")
    dest = out_dir / relative
    if dest.exists() and dest.stat().st_size > 0:
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    encoded_path = kaggle_path.replace("/", "%2F")
    url = f"{BASE}/competitions/data/download/{COMPETITION}/{encoded_path}"

    for attempt in range(retries):
        try:
            r1 = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                              timeout=30, allow_redirects=False)
            if r1.status_code == 404:
                return False
            if r1.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            if r1.status_code not in (302, 301, 303, 307, 308, 200):
                return False
            if r1.status_code == 200:
                content = r1.content
            else:
                r2 = requests.get(r1.headers["Location"], timeout=60)
                r2.raise_for_status()
                content = r2.content
            dest.write_bytes(content)
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED {kaggle_path}: {e}")
    return False


def create_root_zarr_json(sample_dir: Path):
    """Create root zarr.json — Kaggle doesn't serve it."""
    root_json = sample_dir / "zarr.json"
    if not root_json.exists():
        root_json.write_text(json.dumps({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {}
        }, indent=2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = get_token()

    print(f"Downloading {len(TEST_SAMPLES)} test samples to {OUT_DIR}")

    for i, sample in enumerate(TEST_SAMPLES):
        print(f"\n[{i+1}/{len(TEST_SAMPLES)}] {sample}")

        # Download zarr.json for the array metadata
        paths = [f"test/{sample}.zarr/0/zarr.json"]
        paths += [f"test/{sample}.zarr/0/c/{t}/0/0/0" for t in range(N_TIMEPOINTS)]

        ok = sum(1 for p in paths if download_file(token, p, OUT_DIR))
        print(f"  {ok}/{len(paths)} files downloaded")

        create_root_zarr_json(OUT_DIR / f"{sample}.zarr")

    # Verify all volumes open correctly
    print("\nVerifying zarr volumes...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.zarr_reader import load_zarr_volume

    all_ok = True
    for sample in TEST_SAMPLES:
        zarr_dir = OUT_DIR / f"{sample}.zarr"
        try:
            v = load_zarr_volume(str(zarr_dir))
            print(f"  {sample}: shape={v.shape} OK")
        except Exception as e:
            print(f"  {sample}: ERROR {e}")
            all_ok = False

    if all_ok:
        print("\nAll test samples ready. Run:")
        print("  python scripts/make_submission.py --test_dir data/test --output submission.csv")


if __name__ == "__main__":
    main()
