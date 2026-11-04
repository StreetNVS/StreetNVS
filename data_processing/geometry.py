# Portions adapted from Street Crafter.
# Copyright 2024-2025 3D Vision Group, Zhejiang University.
# See licenses/StreetCrafter.txt for the upstream terms.

"""Camera, track and box geometry adapted from Street Crafter. See NOTICE."""

import os
import pickle
import cv2
import numpy as np


def image_filename_to_cam(x):
    return int(x.split(".")[0][-1])


def image_filename_to_frame(x):
    return int(x.split(".")[0][:6])


image_heights = [1280, 1280, 1280, 886, 886]
image_widths = [1920, 1920, 1920, 1920, 1920]


def get_object(object_list, name):
    """Search for an object by name in an object list."""
    object_list = [obj for obj in object_list if obj.name == name]
    return object_list[0]


def load_track(seq_save_dir):
    """Load tracking data from processed directory."""
    track_dir = os.path.join(seq_save_dir, "track")
    assert os.path.exists(track_dir), f"Track directory {track_dir} does not exist."
    track_info_path = os.path.join(track_dir, "track_info.pkl")
    with open(track_info_path, "rb") as f:
        track_info = pickle.load(f)
    track_camera_visible_path = os.path.join(track_dir, "track_camera_visible.pkl")
    with open(track_camera_visible_path, "rb") as f:
        track_camera_visible = pickle.load(f)
    trajectory_path = os.path.join(track_dir, "trajectory.pkl")
    with open(trajectory_path, "rb") as f:
        trajectory = pickle.load(f)
    return (track_info, track_camera_visible, trajectory)


def load_ego_poses(datadir):
    """Load ego poses from processed directory."""
    ego_pose_dir = os.path.join(datadir, "ego_pose")
    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(5)]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:
        if "_" not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = image_filename_to_cam(ego_pose_path)
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_cam_poses[cam].append(ego_cam_pose)
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point
    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(5)]
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point
    return (ego_frame_poses, ego_cam_poses)


def load_calibration(datadir):
    """Load camera calibration from processed directory."""
    extrinsics_dir = os.path.join(datadir, "extrinsics")
    assert os.path.exists(extrinsics_dir), f"{extrinsics_dir} does not exist"
    intrinsics_dir = os.path.join(datadir, "intrinsics")
    assert os.path.exists(intrinsics_dir), f"{intrinsics_dir} does not exist"
    intrinsics = []
    extrinsics = []
    for i in range(5):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = (intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)
    for i in range(5):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(cam_to_ego)
    return (extrinsics, intrinsics)


opencv2camera = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def get_extrinsic(camera_calibration):
    """Get camera extrinsic from Waymo calibration."""
    camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4)
    extrinsic = np.matmul(camera_extrinsic, opencv2camera)
    return extrinsic


def get_intrinsic(camera_calibration):
    """Get camera intrinsic from Waymo calibration."""
    camera_intrinsic = camera_calibration.intrinsic
    fx = camera_intrinsic[0]
    fy = camera_intrinsic[1]
    cx = camera_intrinsic[2]
    cy = camera_intrinsic[3]
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic


def project_label_to_image(dim, obj_pose, calibration):
    """Project 3D bounding box label to image."""
    from .geometry import project_numpy
    from .geometry import bbox_to_corner3d

    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = (calibration.width, calibration.height)
    points_uv, valid = project_numpy(
        xyz=points_vehicle[..., :3],
        K=intrinsic,
        RT=np.linalg.inv(extrinsic),
        H=height,
        W=width,
    )
    return (points_uv, valid)


def project_label_to_mask(dim, obj_pose, calibration, calibration_dict=None):
    """Project 3D bounding box label to mask."""
    from .geometry import bbox_to_corner3d, get_bound_2d_mask

    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T
    if calibration_dict is not None:
        extrinsic = calibration_dict["extrinsic"]
        intrinsic = calibration_dict["intrinsic"]
        width = calibration_dict["width"]
        height = calibration_dict["height"]
    else:
        extrinsic = get_extrinsic(calibration)
        intrinsic = get_intrinsic(calibration)
        width, height = (calibration.width, calibration.height)
    mask = get_bound_2d_mask(
        corners_3d=points_vehicle[..., :3],
        K=intrinsic,
        pose=np.linalg.inv(extrinsic),
        H=height,
        W=width,
    )
    return mask


def get_lane_shift_direction(ego_frame_poses, frame):
    """Get lane-shift direction for trajectory modification.

    Returns a horizontal unit vector perpendicular to the ego's forward
    direction (rotated 90° clockwise about world +Z, matching the original
    ``[dy, -dx]`` convention so the per-scene LANE_SHIFT_SIGN overrides
    keep their meaning).

    The forward direction is taken from, in order:
      1. the local frame-to-frame ego-position delta (original behaviour),
      2. the whole-clip displacement (start → end), used when the ego is
         briefly stationary (e.g. waiting at a light) and the local delta
         underflows,
      3. the ego body's +X (forward) axis in world frame, used when the
         ego is parked the entire clip.

    Without these fallbacks the local delta normalises to NaN / amplified
    noise on stationary segments, which makes the lane-shift trajectory
    sweep along an arbitrary world direction.
    """
    assert frame >= 0 and frame < len(ego_frame_poses)
    EPS = 0.001
    if frame == 0:
        delta = ego_frame_poses[1][:3, 3] - ego_frame_poses[0][:3, 3]
    else:
        delta = ego_frame_poses[frame][:3, 3] - ego_frame_poses[frame - 1][:3, 3]
    if np.linalg.norm(delta[:2]) < EPS:
        total = ego_frame_poses[-1][:3, 3] - ego_frame_poses[0][:3, 3]
        if np.linalg.norm(total[:2]) >= EPS:
            delta = total
        else:
            delta = ego_frame_poses[frame][:3, 0]
    d = delta[:2]
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    d = d / n
    direction = np.array([d[1], -d[0], 0.0])
    return direction


def project_numpy(xyz, K, RT, H, W):
    """Project 3D points to 2D image plane."""
    xyz_cam = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = np.dot(xyz_cam, K.T)
    xyz_pixel = xyz_pixel[:, :2] / xyz_pixel[:, 2:]
    valid_x = np.logical_and(xyz_pixel[:, 0] >= 0, xyz_pixel[:, 0] < W)
    valid_y = np.logical_and(xyz_pixel[:, 1] >= 0, xyz_pixel[:, 1] < H)
    valid_pixel = np.logical_and(valid_x, valid_y)
    mask = np.logical_and(valid_depth, valid_pixel)
    return (xyz_pixel, mask)


def get_bound_2d_mask(corners_3d, K, pose, H, W):
    """Get 2D mask from 3D bounding box corners."""
    corners_3d = np.dot(corners_3d, pose[:3, :3].T) + pose[:3, 3:].T
    corners_3d[..., 2] = np.clip(corners_3d[..., 2], a_min=0.001, a_max=None)
    corners_3d = np.dot(corners_3d, K.T)
    corners_2d = corners_3d[:, :2] / corners_3d[:, 2:]
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def bbox_to_corner3d(bbox):
    """Convert bounding box to 8 corner points."""
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]
    corner3d = np.array(
        [
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ]
    )
    return corner3d


def inbbox_points(points, corner3d):
    """Check if points are inside bounding box."""
    min_xyz = corner3d[0]
    max_xyz = corner3d[-1]
    return np.logical_and(
        np.all(points >= min_xyz, axis=-1), np.all(points <= max_xyz, axis=-1)
    )
