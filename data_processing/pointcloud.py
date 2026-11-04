# Portions adapted from Street Crafter.
# Copyright 2024-2025 3D Vision Group, Zhejiang University.
# See licenses/StreetCrafter.txt for the upstream terms.

"""PLY I/O adapted from Street Crafter. See NOTICE."""

from typing import NamedTuple
import numpy as np
from plyfile import PlyData, PlyElement


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    mask: np.array
    camera_mask: np.array = None


def fetchPly(path):
    """Load a PLY file with points, colors, and mask (and optionally camera_id)."""
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    mask = vertices["mask"].astype(np.bool_)
    if "camera_mask" in vertices:
        camera_mask = vertices["camera_mask"].astype(np.uint8)
    elif "camera_id" in vertices:
        camera_id = vertices["camera_id"].astype(np.int32)
        camera_mask = np.zeros(len(camera_id), dtype=np.uint8)
        valid = camera_id >= 0
        camera_mask[valid] = 1 << camera_id[valid]
    else:
        camera_mask = None
    return BasicPointCloud(
        points=positions, colors=colors, mask=mask, camera_mask=camera_mask
    )


def storePly(path, xyz, rgb, mask, camera_mask=None, camera_id=None):
    """Store a PLY file with points, colors, mask, and optionally camera_mask.

    Args:
        camera_mask: Bitmask where bit i indicates visibility in camera i (preferred)
        camera_id: Legacy single camera ID (for backwards compatibility)
    """
    if rgb.max() <= 1.0 and rgb.min() >= 0:
        rgb = np.clip(rgb * 255, 0.0, 255.0)
    mask = mask.astype(np.bool_)
    if camera_id is not None and camera_mask is None:
        camera_mask = np.zeros(len(camera_id), dtype=np.uint8)
        valid = camera_id >= 0
        camera_mask[valid] = 1 << camera_id[valid]
    if camera_mask is not None:
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("mask", "?"),
            ("camera_mask", "u1"),
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate(
            (xyz, rgb, mask, camera_mask.reshape(-1, 1)), axis=1
        )
    else:
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("mask", "?"),
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, rgb, mask), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
