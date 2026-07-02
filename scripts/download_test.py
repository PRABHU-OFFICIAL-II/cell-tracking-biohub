"""
Download test samples from Kaggle for submission.

Kaggle doesn't serve zarr.json at the root level — we generate it locally.
Test zarr structure mirrors train: test/SAMPLE.zarr/0/c/T/0/0/0

Usage:
  python scripts/download_test.py
  python scripts/download_test.py --max_samples 5  # quick test with 5 samples
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


def list_all_files(token, prefix="test"):
    """List all competition files matching prefix. Kaggle API returns all at once."""
    url = f"{BASE}/competitions/data/list/{COMPETITION}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    print(f"  List API status: {r.status_code}")
    if r.status_code != 200:
        print(f"  Response: {r.text[:300]}")
        return []
    data = r.json()
    all_items = data if isinstance(data, list) else data.get("files", [])
    return [f.get("name", "") for f in all_items if f.get("name", "").startswith(prefix + "/")]


def extract_test_samples(file_list):
    """Extract unique sample names from the file listing."""
    samples = set()
    for name in file_list:
        # name: "test/44b6_xxxx.zarr/0/zarr.json" etc.
        parts = name.split("/")
        if len(parts) >= 2 and parts[1].endswith(".zarr"):
            samples.add(parts[1][:-5])  # strip .zarr
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
                r2 = requests.get(gcs_url, timeout=60)
                r2.raise_for_status()
                content = r2.content

            dest.write_bytes(content)
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAILED {kaggle_path}: {e}")
    return False


def zarr_file_paths(sample):
    """Paths to download for one test .zarr sample."""
    paths = [f"test/{sample}.zarr/0/zarr.json"]
    for t in range(N_TIMEPOINTS):
        paths.append(f"test/{sample}.zarr/0/c/{t}/0/0/0")
    return paths


def create_root_zarr_json(sample_dir: Path):
    """Create the missing root zarr.json (Kaggle doesn't serve it)."""
    root_json = sample_dir / "zarr.json"
    if not root_json.exists():
        root_json.write_text(json.dumps({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {}
        }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of test samples (for quick test)')
    parser.add_argument('--list_only', action='store_true',
                        help='Only list samples, do not download')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = get_token()

    print("Listing test files from Kaggle...")
    all_files = list_all_files(token, prefix="test")
    samples = extract_test_samples(all_files)
    print(f"Found {len(samples)} test samples")

    if not samples:
        # Fallback: scan for known naming pattern without listing
        print("Listing returned nothing — trying direct probe for sample names from submission_file_list...")
        # Check sample_submission.csv which usually lists expected dataset names
        sub_files = [f for f in all_files if "sample_submission" in f.lower()]
        print(f"  submission-related files: {sub_files}")
        return

    # Save sample list
    list_path = OUT_DIR / "test_samples.txt"
    list_path.write_text("\n".join(samples))
    print(f"Sample list saved to {list_path}")

    if args.list_only:
        for s in samples:
            print(f"  {s}")
        return

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Downloading {len(samples)} samples")

    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] {sample}")
        paths = zarr_file_paths(sample)
        ok = 0
        for p in paths:
            if download_file(token, p, OUT_DIR):
                ok += 1
        print(f"  {ok}/{len(paths)} files")

        # Generate root zarr.json
        create_root_zarr_json(OUT_DIR / f"{sample}.zarr")

    # Verify a few
    print("\nVerifying zarr volumes...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.zarr_reader import load_zarr_volume
    for sample in samples[:3]:
        zarr_dir = OUT_DIR / f"{sample}.zarr"
        try:
            v = load_zarr_volume(str(zarr_dir))
            print(f"  {sample}: shape={v.shape} OK")
        except Exception as e:
            print(f"  {sample}: ERROR {e}")


if __name__ == "__main__":
    main()
