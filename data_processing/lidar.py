# Portions adapted from Street Crafter.
# Copyright 2024-2025 3D Vision Group, Zhejiang University.
# See licenses/StreetCrafter.txt for the upstream terms.

"""Extract colored background and actor clouds (Street Crafter-derived)."""

import os
import gc
from copy import deepcopy
import cv2
import numpy as np
import tensorflow as tf
import transforms3d
from tqdm import tqdm
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception import lidar as _v2_lidar, context as _v2_context
from waymo_open_dataset.v2.perception.utils.lidar_utils import (
    convert_range_image_to_cartesian,
)
from .geometry import load_track, load_calibration, bbox_to_corner3d, inbbox_points
from .pointcloud import storePly

DUMMY_DISTANCE_VALUE = 2000.0


def convert_range_image_to_point_cloud(
    range_image: _v2_lidar.RangeImage,
    calibration: _v2_context.LiDARCalibrationComponent,
    pixel_pose=None,
    frame_pose=None,
    keep_polar_features=False,
):
    """Converts one range image from polar coordinates to point cloud.

    Same as in WOD API, but returns the mask in addition plus channel id.

    Args:
        range_image: One range image return captured by a LiDAR sensor.
        calibration: Parameters for calibration of a LiDAR sensor.
        pixel_pose: If not none, it sets pose for each range image pixel.
        frame_pose: This must be set when `pose` is set.
        keep_polar_features: If true, keep the features from the polar range image.

    Returns:
        A tuple of:
            1. Lidar points-cloud
            2. Missing points points-cloud
            3. Range image mask above dummy distance.
    """
    val_clone = deepcopy(range_image.tensor.numpy())
    no_return = val_clone[..., 0] == -1
    val_clone[..., 0][no_return] = DUMMY_DISTANCE_VALUE
    object.__setattr__(range_image, "values", val_clone.flatten())
    if pixel_pose is not None:
        assert frame_pose is not None
        pixel_pose_clone = deepcopy(pixel_pose.tensor.numpy())
        pixel_pose_mask = pixel_pose_clone[..., 0] == 0
        tr_orig = frame_pose.world_from_vehicle.transform.reshape(4, 4)
        rot = tr_orig[:3, :3]
        x, y, z = tr_orig[:3, 3]
        yaw, pitch, roll = transforms3d.euler.mat2euler(rot, "szyx")
        pixel_pose_clone[..., 0][pixel_pose_mask] = roll
        pixel_pose_clone[..., 1][pixel_pose_mask] = pitch
        pixel_pose_clone[..., 2][pixel_pose_mask] = yaw
        pixel_pose_clone[..., 3][pixel_pose_mask] = x
        pixel_pose_clone[..., 4][pixel_pose_mask] = y
        pixel_pose_clone[..., 5][pixel_pose_mask] = z
        object.__setattr__(pixel_pose, "values", pixel_pose_clone.flatten())
    range_image_cartesian = convert_range_image_to_cartesian(
        range_image=range_image,
        calibration=calibration,
        pixel_pose=pixel_pose,
        frame_pose=frame_pose,
        keep_polar_features=keep_polar_features,
    )
    range_image_tensor = range_image.tensor
    range_image_mask = DUMMY_DISTANCE_VALUE / 2 > range_image_tensor[..., 0]
    points_tensor = tf.gather_nd(
        range_image_cartesian, tf.compat.v1.where(range_image_mask)
    )
    missing_points_tensor = tf.gather_nd(
        range_image_cartesian, tf.compat.v1.where(~range_image_mask)
    )
    return (points_tensor, missing_points_tensor, range_image_mask)


def extract_pointwise_camera_projection(
    range_image: _v2_lidar.RangeImage,
    camera_projection: _v2_lidar.CameraProjectionRangeImage,
):
    """Extracts information about where in camera images each point is projected.

    Args:
        range_image: One range image return captured by a LiDAR sensor.
        camera_projection: LiDAR point to camera image projections.

    Returns:
        A tuple of:
            1. Camera projection per point
            2. Missing points camera projection
            3. Range image mask
    """
    range_image_tensor = range_image.tensor
    range_image_mask = DUMMY_DISTANCE_VALUE / 2 > range_image_tensor[..., 0]
    camera_project_tensor = camera_projection.tensor
    pointwise_camera_projection_tensor = tf.gather_nd(
        camera_project_tensor, tf.compat.v1.where(range_image_mask)
    )
    missing_points_camera_projection_tensor = tf.gather_nd(
        camera_project_tensor, tf.compat.v1.where(~range_image_mask)
    )
    return (
        pointwise_camera_projection_tensor,
        missing_points_camera_projection_tensor,
        range_image_mask,
    )


def save_lidar(root_dir, seq_path, seq_save_dir):
    """Process and save LiDAR data for a single sequence."""
    track_info, track_camera_visible, trajectory = load_track(seq_save_dir)
    extrinsics, intrinsics = load_calibration(seq_save_dir)
    print(f"Processing sequence {seq_path}...")
    print(f"Saving to {seq_save_dir}")
    base_seq_name = os.path.basename(seq_path).split(".")[0]
    seq_context = base_seq_name[8:-19]
    import pyarrow.parquet as pq

    lidar_calib = pq.read_table(
        os.path.join(root_dir, "lidar_calibration", seq_context + ".parquet")
    ).to_pandas()
    vehicle_pose_df = pq.read_table(
        os.path.join(root_dir, "vehicle_pose", seq_context + ".parquet")
    ).to_pandas()
    import pyarrow.parquet as _pq

    _pdir = root_dir
    _ctx = seq_context

    def _read_frame(tag, ts):
        path = os.path.join(_pdir, tag, f"{_ctx}.parquet")
        return _pq.read_table(
            path, filters=[("key.frame_timestamp_micros", "==", ts)]
        ).to_pandas()

    os.makedirs(seq_save_dir, exist_ok=True)
    image_dir = os.path.join(seq_save_dir, "images")
    lidar_dir = os.path.join(seq_save_dir, "lidar")
    os.makedirs(lidar_dir, exist_ok=True)
    lidar_dir_background = os.path.join(lidar_dir, "background")
    os.makedirs(lidar_dir_background, exist_ok=True)
    lidar_dir_actor = os.path.join(lidar_dir, "actor")
    os.makedirs(lidar_dir_actor, exist_ok=True)
    pointcloud_actor = dict()
    for track_id, traj in trajectory.items():
        dynamic = not traj["stationary"]
        if dynamic and traj["label"] != "sign":
            os.makedirs(os.path.join(lidar_dir_actor, track_id), exist_ok=True)
            pointcloud_actor[track_id] = dict()
            pointcloud_actor[track_id]["xyz"] = []
            pointcloud_actor[track_id]["rgb"] = []
            pointcloud_actor[track_id]["mask"] = []
            pointcloud_actor[track_id]["camera_mask"] = []
    print("Processing LiDAR data...")
    for frame_id, (_, v) in tqdm(enumerate(vehicle_pose_df.iterrows())):
        xyzs = []
        camera_projections = []
        missing_xyzs = []
        VehiclePoseCom = v2.VehiclePoseComponent.from_dict(v)
        ts = VehiclePoseCom.key.frame_timestamp_micros
        lidar_df_frame = _read_frame("lidar", ts)
        lidar_pose_frame = _read_frame("lidar_pose", ts)
        lidar_proj_frame = _read_frame("lidar_camera_projection", ts)
        for _, r in lidar_df_frame.iterrows():
            LidarComp = v2.LiDARComponent.from_dict(r)
            laser = LidarComp.key.laser_name
            lidar_pose_df_ = lidar_pose_frame[
                lidar_pose_frame["key.laser_name"] == laser
            ]
            if lidar_pose_df_ is None or len(lidar_pose_df_) == 0:
                pixel_pose = None
                frame_pose = None
            else:
                LidarPoseComp = v2.LiDARPoseComponent.from_dict(lidar_pose_df_.iloc[0])
                pixel_pose = LidarPoseComp.range_image_return1
                frame_pose = VehiclePoseCom
            lidar_proj_df_ = lidar_proj_frame[
                lidar_proj_frame["key.laser_name"] == laser
            ]
            if lidar_proj_df_ is None or len(lidar_proj_df_) == 0:
                continue
            LidarProjComp = v2.LiDARCameraProjectionComponent.from_dict(
                lidar_proj_df_.iloc[0]
            )
            lidar_calib_ = lidar_calib[
                lidar_calib["key.laser_name"] == LidarComp.key.laser_name
            ]
            LidarCalibComp = v2.LiDARCalibrationComponent.from_dict(
                lidar_calib_.iloc[0]
            )
            pts_lidar, missing_pts, _ = convert_range_image_to_point_cloud(
                LidarComp.range_image_return1,
                LidarCalibComp,
                pixel_pose=pixel_pose,
                frame_pose=frame_pose,
                keep_polar_features=False,
            )
            missing_pts = missing_pts.numpy()
            pts_projection, missing_projection, _ = extract_pointwise_camera_projection(
                LidarComp.range_image_return1, LidarProjComp.range_image_return1
            )
            xyzs.append(pts_lidar.numpy())
            camera_projections.append(pts_projection.numpy())
            missing_xyzs.append(missing_pts)
        xyzs = np.concatenate(xyzs, axis=0)
        camera_projections = np.concatenate(camera_projections, axis=0)
        missing_xyzs = np.concatenate(missing_xyzs, axis=0)
        rgbs = np.zeros((xyzs.shape[0], 3), dtype=np.uint8)
        num_pts = xyzs.shape[0]
        point_camera_masks = np.zeros(num_pts, dtype=np.uint8)
        camera_id = camera_projections[:, 0]
        masks = camera_id > 0
        xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1)
        w2cs = [np.linalg.inv(extrinsics[i]) for i in range(5)]
        for i in range(5):
            image_filename = os.path.join(image_dir, f"{frame_id:06d}_{i}.png")
            image = cv2.imread(image_filename)[..., [2, 1, 0]].astype(np.uint8)
            h, w = image.shape[:2]
            w2c = w2cs[i]
            ixt = intrinsics[i]
            xyzs_cam = xyzs_homo @ w2c.T
            xyzs_depth = xyzs_cam[..., 2]
            uvz = xyzs_cam[..., :3] @ ixt.T
            uv = uvz[..., :2] / (uvz[..., 2:3] + 1e-08)
            u_all = uv[..., 0]
            v_all = uv[..., 1]
            del uvz
            visible_mask = (
                (u_all >= 0)
                & (u_all < w)
                & (v_all >= 0)
                & (v_all < h)
                & (xyzs_depth > 0.1)
            )
            point_camera_masks[visible_mask] |= 1 << i
            mask_rgb = camera_id == i + 1
            if mask_rgb.sum() != 0:
                u_rgb, v_rgb = (
                    camera_projections[mask_rgb, 1],
                    camera_projections[mask_rgb, 2],
                )
                u_rgb = np.clip(u_rgb, 0, w - 1).astype(np.int32)
                v_rgb = np.clip(v_rgb, 0, h - 1).astype(np.int32)
                rgb = image[v_rgb, u_rgb]
                rgbs[mask_rgb] = rgb
            del xyzs_cam, xyzs_depth, uv, u_all, v_all, visible_mask, image
        del xyzs_homo, w2cs
        actor_mask = np.zeros(xyzs.shape[0], dtype=np.bool_)
        track_info_frame = track_info[f"{frame_id:06d}"]
        for track_id, track_info_actor in track_info_frame.items():
            if track_id not in pointcloud_actor.keys():
                continue
            lidar_box = track_info_actor["lidar_box"]
            height = lidar_box["height"]
            width = lidar_box["width"]
            length = lidar_box["length"]
            pose_idx = trajectory[track_id]["frames"].index(frame_id)
            pose_vehicle = trajectory[track_id]["poses_vehicle"][pose_idx]
            xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1)
            xyzs_actor = xyzs_homo @ np.linalg.inv(pose_vehicle).T
            xyzs_actor = xyzs_actor[..., :3]
            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
            corners3d = bbox_to_corner3d(bbox)
            inbbox_mask = inbbox_points(xyzs_actor, corners3d)
            actor_mask = np.logical_or(actor_mask, inbbox_mask)
            xyzs_inbbox = xyzs_actor[inbbox_mask]
            rgbs_inbbox = rgbs[inbbox_mask]
            masks_inbbox = masks[inbbox_mask]
            camera_masks_inbbox = point_camera_masks[inbbox_mask]
            pointcloud_actor[track_id]["xyz"].append(xyzs_inbbox)
            pointcloud_actor[track_id]["rgb"].append(rgbs_inbbox)
            pointcloud_actor[track_id]["mask"].append(masks_inbbox)
            pointcloud_actor[track_id]["camera_mask"].append(camera_masks_inbbox)
            masks_inbbox = masks_inbbox[..., None]
            ply_actor_path = os.path.join(
                lidar_dir_actor, track_id, f"{frame_id:06d}.ply"
            )
            if len(xyzs_inbbox):
                storePly(
                    ply_actor_path,
                    xyzs_inbbox,
                    rgbs_inbbox,
                    masks_inbbox,
                    camera_mask=camera_masks_inbbox,
                )
        xyzs_background = xyzs[~actor_mask]
        rgbs_background = rgbs[~actor_mask]
        masks_background = masks[~actor_mask]
        camera_masks_background = point_camera_masks[~actor_mask]
        masks_background = masks_background[..., None]
        ply_background_path = os.path.join(lidar_dir_background, f"{frame_id:06d}.ply")
        storePly(
            ply_background_path,
            xyzs_background,
            rgbs_background,
            masks_background,
            camera_mask=camera_masks_background,
        )
        del xyzs, rgbs, camera_projections, missing_xyzs
        del xyzs_background, rgbs_background, masks_background, camera_masks_background
        del point_camera_masks
        del lidar_df_frame, lidar_pose_frame, lidar_proj_frame
        gc.collect()
    for track_id, pointcloud in pointcloud_actor.items():
        if len(pointcloud["xyz"]) == 0:
            continue
        xyzs = np.concatenate(pointcloud["xyz"], axis=0)
        rgbs = np.concatenate(pointcloud["rgb"], axis=0)
        masks = np.concatenate(pointcloud["mask"], axis=0)
        camera_masks = np.concatenate(pointcloud["camera_mask"], axis=0)
        masks = masks[..., None]
        ply_actor_path_full = os.path.join(lidar_dir_actor, track_id, "full.ply")
        if len(xyzs):
            storePly(ply_actor_path_full, xyzs, rgbs, masks, camera_mask=camera_masks)
