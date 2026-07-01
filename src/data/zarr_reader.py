"""
Load zarr v3 volumes and .geff ground-truth graphs.
Data format: (T, Z, Y, X) uint16, chunks of one timepoint each.
Physical scale: z=1.625, y=x=0.40625 µm/voxel.
"""

import zarr
import numpy as np
from pathlib import Path
from typing import Union

VOXEL_SCALE = np.array([1.625, 0.40625, 0.40625])  # z, y, x in µm/voxel


def load_zarr_volume(zarr_path: Union[str, Path]) -> zarr.Array:
    """Open a zarr v3 volume. Returns the array at path '0/'."""
    store = zarr.open(str(zarr_path), mode="r")
    return store["0"]


def load_timepoint(arr: zarr.Array, t: int) -> np.ndarray:
    """Load a single timepoint (Z, Y, X) as uint16 numpy array."""
    return np.array(arr[t])


def load_all_timepoints(arr: zarr.Array) -> np.ndarray:
    """Load full volume (T, Z, Y, X). Use only for small datasets."""
    return np.array(arr[:])


def get_volume_info(zarr_path: Union[str, Path]) -> dict:
    arr = load_zarr_volume(zarr_path)
    T, Z, Y, X = arr.shape
    return {
        "path": str(zarr_path),
        "shape": arr.shape,
        "dtype": arr.dtype,
        "n_timepoints": T,
        "spatial_shape": (Z, Y, X),
        "physical_size_um": (Z * VOXEL_SCALE[0], Y * VOXEL_SCALE[1], X * VOXEL_SCALE[2]),
    }
