"""
Download sample_submission.csv from Kaggle to see all expected test dataset names.
Usage: python scripts/get_sample_submission.py
"""

import json
import requests
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
BASE = "https://www.kaggle.com/api/v1"
OUT = Path(__file__).parent.parent / "data" / "sample_submission.csv"


def get_token():
    return json.loads(KAGGLE_JSON.read_text())["key"]


def list_all_files(token):
    # Kaggle listing API: no pagination params, returns all files at once
    url = f"{BASE}/competitions/data/list/{COMPETITION}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    print(f"  List API status: {r.status_code}")
    if r.status_code != 200:
        print(f"  Response: {r.text[:500]}")
        return []
    data = r.json()
    files = data if isinstance(data, list) else data.get("files", [])
    print(f"  Got {len(files)} files")
    return files


def download_file(token, path, dest):
    encoded = path.replace("/", "%2F")
    url = f"{BASE}/competitions/data/download/{COMPETITION}/{encoded}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                     timeout=30, allow_redirects=False)
    if r.status_code in (302, 301, 303):
        r2 = requests.get(r.headers["Location"], timeout=60)
        r2.raise_for_status()
        dest.write_bytes(r2.content)
    elif r.status_code == 200:
        dest.write_bytes(r.content)
    else:
        print(f"Status {r.status_code} for {path}")
        return False
    return True


def main():
    token = get_token()
    print("Listing all competition files...")
    files = list_all_files(token)
    print(f"\nAll files ({len(files)} total):")
    for f in files:
        print(f"  {f.get('name', '?')}  ({f.get('totalBytes', '?')} bytes)")

    # Try to download sample_submission.csv
    sample_sub = [f for f in files if "sample_submission" in f.get("name", "").lower()]
    for f in sample_sub:
        name = f["name"]
        print(f"\nDownloading: {name}")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        if download_file(token, name, OUT):
            print(f"Saved to {OUT}")
            # Show first few lines
            lines = OUT.read_text().splitlines()
            print(f"\nFirst 5 lines ({len(lines)} total rows):")
            for line in lines[:5]:
                print(f"  {line}")
            datasets = set()
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) > 1:
                    datasets.add(parts[1])
            print(f"\nUnique datasets: {len(datasets)}")
            for d in sorted(datasets)[:10]:
                print(f"  {d}")


if __name__ == "__main__":
    main()
