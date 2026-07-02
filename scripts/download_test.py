"""
Download test samples from Kaggle for submission.
Lists all test zarr directories, then downloads each one.
Usage: python scripts/download_test.py [--max_samples N]
"""

import json
import time
import argparse
import requests
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
OUT_DIR = Path(__file__).parent.parent / "data" / "test"
BASE = "https://www.kaggle.com/api/v1"
N_TIMEPOINTS = 100


def get_token():
    return json.loads(KAGGLE_JSON.read_text())["key"]


def list_test_samples(token):
    """List all test zarr directories via Kaggle file listing API."""
    print("Listing test files from Kaggle API...")
    page = 1
    samples = set()
    while True:
        url = f"{BASE}/competitions/data/list/{COMPETITION}?page={page}&pageSize=200&search=test"
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        r.raise_for_status()
        data = r.json()
        files = data if isinstance(data, list) else data.get("files", [])
        if not files:
            break
        for f in files:
            name = f.get("name", "")
            # name looks like "test/44b6_xxxx.zarr/0/zarr.json" or similar
            if name.startswith("test/") and ".zarr" in name:
                parts = name.split("/")
                if len(parts) >= 2:
                    zarr_name = parts[1]  # e.g. "44b6_xxxx.zarr"
                    if zarr_name.endswith(".zarr"):
                        samples.add(zarr_name[:-5])  # strip .zarr
        if len(files) < 200:
            break
        page += 1

    return sorted(samples)


def download_file(token, kaggle_path, out_dir, retries=3):
    """Download one file. kaggle_path like 'test/sample.zarr/0/c/5/0/0/0'."""
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
    paths = [f"test/{sample}.zarr/0/zarr.json"]
    for t in range(N_TIMEPOINTS):
        paths.append(f"test/{sample}.zarr/0/c/{t}/0/0/0")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=None, help='Limit number of test samples to download')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = get_token()

    samples = list_test_samples(token)
    print(f"Found {len(samples)} test samples")

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limiting to {len(samples)} samples")

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] {sample}")
        paths = zarr_paths(sample)
        ok = sum(1 for p in paths if download_file(token, p, OUT_DIR))
        print(f"  {ok}/{len(paths)} files downloaded")

    print("\nVerifying zarr opens...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.zarr_reader import load_zarr_volume
    for sample in samples[:3]:
        zarr_dir = OUT_DIR / f"{sample}.zarr"
        try:
            v = load_zarr_volume(str(zarr_dir))
            print(f"  {sample}: shape={v.shape}")
        except Exception as e:
            print(f"  {sample}: ERROR {e}")

    # Save the list of test samples for use by make_submission.py
    sample_list_path = OUT_DIR / "test_samples.txt"
    sample_list_path.write_text("\n".join(samples))
    print(f"\nSaved sample list to {sample_list_path}")


if __name__ == "__main__":
    main()
