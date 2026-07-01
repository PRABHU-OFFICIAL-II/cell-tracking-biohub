"""
Create the missing zarr.json metadata files for downloaded zarr v3 volumes.
The competition spec tells us exactly:
  - shape: (100, 64, 256, 256), dtype: uint16
  - chunks: (1, 64, 256, 256) — one timepoint per chunk
  - compression: blosc/zstd
  - chunk path format: 0/c/{t}/0/0/0
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "train" / "train"

# zarr v3 root metadata — marks this as a zarr group
ZARR_ROOT_META = {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {}
}

# zarr v3 array metadata for the image array at path "0/"
ZARR_ARRAY_META = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [100, 64, 256, 256],
    "data_type": "uint16",
    "chunk_grid": {
        "name": "regular",
        "configuration": {
            "chunk_shape": [1, 64, 256, 256]
        }
    },
    "chunk_key_encoding": {
        "name": "default",
        "configuration": {
            "separator": "/"
        }
    },
    "fill_value": 0,
    "codecs": [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {
            "name": "blosc",
            "configuration": {
                "cname": "zstd",
                "clevel": 5,
                "shuffle": "bitshuffle",
                "blocksize": 0
            }
        }
    ],
    "attributes": {},
    "dimension_names": ["t", "z", "y", "x"]
}


def create_zarr_metadata(zarr_dir: Path):
    # Root zarr.json
    root_meta = zarr_dir / "zarr.json"
    if not root_meta.exists():
        root_meta.write_text(json.dumps(ZARR_ROOT_META, indent=2))
        print(f"  Created {root_meta.name}")

    # Array zarr.json at 0/
    array_dir = zarr_dir / "0"
    array_dir.mkdir(exist_ok=True)
    array_meta = array_dir / "zarr.json"
    if not array_meta.exists():
        array_meta.write_text(json.dumps(ZARR_ARRAY_META, indent=2))
        print(f"  Created 0/zarr.json")


def main():
    zarr_dirs = sorted(DATA_DIR.glob("*.zarr"))
    if not zarr_dirs:
        print(f"No .zarr directories found in {DATA_DIR}")
        return

    for zarr_dir in zarr_dirs:
        print(f"{zarr_dir.name}")
        create_zarr_metadata(zarr_dir)

    print("\nDone. Testing zarr open...")
    import zarr
    for zarr_dir in zarr_dirs[:1]:
        store = zarr.open(str(zarr_dir), mode="r")
        arr = store["0"]
        print(f"  {zarr_dir.name}: shape={arr.shape}, dtype={arr.dtype}")
        t0 = arr[0]
        print(f"  Timepoint 0: shape={t0.shape}, min={t0.min()}, max={t0.max()}")


if __name__ == "__main__":
    main()
