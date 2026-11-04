# Portions adapted from Street Crafter.
# Copyright 2024-2025 3D Vision Group, Zhejiang University.
# See licenses/StreetCrafter.txt for the upstream terms.


"""Extract raw images, poses, tracks and dynamic masks (Street Crafter-derived)."""

import os
import math
import json
import pickle
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2, label_pb2
from .geometry import (
    load_track,
    load_calibration,
    load_ego_poses,
    get_object,
    project_label_to_image,
    project_label_to_mask,
    opencv2camera,
    image_heights,
    image_widths,
)

camera_names_dict = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}


def parse_seq_rawdata(seq_path, seq_save_dir):
    """Process a single Waymo sequence."""
    os.makedirs(seq_save_dir, exist_ok=True)
    print(f"Processing sequence {seq_path}...")
    print(f"Saving to {seq_save_dir}")
    ego_pose_save_dir = os.path.join(seq_save_dir, "ego_pose")
    os.makedirs(ego_pose_save_dir, exist_ok=True)
    print("Processing ego pose...")
    timestamp = dict()
    timestamp["FRAME"] = dict()
    for camera_name in camera_names_dict.values():
        timestamp[camera_name] = dict()
    datafile = tf.data.TFRecordDataset(seq_path, compression_type="")
    for frame_id, data in tqdm(enumerate(datafile)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            os.path.join(ego_pose_save_dir, f"{str(frame_id).zfill(6)}.txt"), pose
        )
        timestamp["FRAME"][str(frame_id).zfill(6)] = frame.timestamp_micros / 1000000.0
        camera_calibrations = frame.context.camera_calibrations
        for i, camera in enumerate(camera_calibrations):
            camera_name = camera.name
            camera_name_str = camera_names_dict[camera_name]
            camera = get_object(frame.images, camera_name)
            camera_timestamp = camera.pose_timestamp
            timestamp[camera_name_str][str(frame_id).zfill(6)] = camera_timestamp
            camera_pose = np.array(camera.pose.transform).reshape(4, 4)
            np.savetxt(
                os.path.join(
                    ego_pose_save_dir, f"{str(frame_id).zfill(6)}_{camera_name - 1}.txt"
                ),
                camera_pose,
            )
    timestamp_save_path = os.path.join(seq_save_dir, "timestamps.json")
    with open(timestamp_save_path, "w") as f:
        json.dump(timestamp, f, indent=1)
    intrinsic_save_dir = os.path.join(seq_save_dir, "intrinsics")
    extrinsic_save_dir = os.path.join(seq_save_dir, "extrinsics")
    os.makedirs(intrinsic_save_dir, exist_ok=True)
    os.makedirs(extrinsic_save_dir, exist_ok=True)
    print("Processing camera calibration...")
    datafile = tf.data.TFRecordDataset(seq_path, compression_type="")
    for frame_id, data in tqdm(enumerate(datafile)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        camera_calibrations = frame.context.camera_calibrations
        break
    extrinsics = []
    intrinsics = []
    camera_names = []
    for camera in camera_calibrations:
        extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
        extrinsic = np.matmul(extrinsic, opencv2camera)
        intrinsic = list(camera.intrinsic)
        extrinsics.append(extrinsic)
        intrinsics.append(intrinsic)
        camera_names.append(camera.name)
    for i in range(5):
        np.savetxt(
            os.path.join(extrinsic_save_dir, f"{str(camera_names[i] - 1)}.txt"),
            extrinsics[i],
        )
        np.savetxt(
            os.path.join(intrinsic_save_dir, f"{str(camera_names[i] - 1)}.txt"),
            intrinsics[i],
        )
    image_save_dir = os.path.join(seq_save_dir, "images")
    os.makedirs(image_save_dir, exist_ok=True)
    print("Processing image data...")
    camera_timestamp = dict()
    datafile = tf.data.TFRecordDataset(seq_path, compression_type="")
    for frame_id, data in tqdm(enumerate(datafile)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        for img in frame.images:
            img_path = os.path.join(
                image_save_dir, f"{frame_id:06d}_{str(img.name - 1)}.png"
            )
            with open(img_path, "wb") as fp:
                fp.write(img.image)
            camera_timestamp[f"{frame_id:06d}_{str(img.name - 1)}"] = img.pose_timestamp
    camera_timestamp_save_path = os.path.join(seq_save_dir, "images", "timestamps.json")
    with open(camera_timestamp_save_path, "w") as f:
        json.dump(camera_timestamp, f, indent=1)
    print("Processing image data done...")
    track_dir = os.path.join(seq_save_dir, "track")
    os.makedirs(track_dir, exist_ok=True)
    print("Processing tracking data...")
    track_info = dict()
    track_camera_visible = dict()
    trajectory_info = dict()
    object_ids = dict()
    datafile = tf.data.TFRecordDataset(seq_path, compression_type="")
    for frame_id, data in tqdm(enumerate(datafile)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        track_info_cur_frame = dict()
        track_camera_visible_cur_frame = dict()
        timestamp = frame.timestamp_micros / 1000000.0
        for camera_name in camera_names_dict.keys():
            track_camera_visible_cur_frame[camera_name - 1] = []
        for label in frame.laser_labels:
            if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                obj_class = "vehicle"
            elif label.type == label_pb2.Label.Type.TYPE_PEDESTRIAN:
                obj_class = "pedestrian"
            elif label.type == label_pb2.Label.Type.TYPE_SIGN:
                obj_class = "sign"
            elif label.type == label_pb2.Label.Type.TYPE_CYCLIST:
                obj_class = "cyclist"
            else:
                obj_class = "misc"
            meta = label.metadata
            speed = np.linalg.norm([meta.speed_x, meta.speed_y])
            label_id = label.id
            if label_id not in trajectory_info.keys():
                trajectory_info[label_id] = dict()
            if label.id not in object_ids:
                object_ids[label.id] = len(object_ids)
            track_info_cur_frame[label_id] = dict()
            lidar_synced_box = dict()
            lidar_synced_box["height"] = label.box.height
            lidar_synced_box["width"] = label.box.width
            lidar_synced_box["length"] = label.box.length
            lidar_synced_box["center_x"] = label.box.center_x
            lidar_synced_box["center_y"] = label.box.center_y
            lidar_synced_box["center_z"] = label.box.center_z
            lidar_synced_box["heading"] = label.box.heading
            lidar_synced_box["label"] = obj_class
            lidar_synced_box["speed"] = speed
            lidar_synced_box["timestamp"] = timestamp
            track_info_cur_frame[label_id]["lidar_box"] = lidar_synced_box
            trajectory_info[label_id][f"{frame_id:06d}"] = lidar_synced_box
            if label.camera_synced_box.ByteSize():
                camera_synced_box = dict()
                camera_synced_box["height"] = label.camera_synced_box.height
                camera_synced_box["width"] = label.camera_synced_box.width
                camera_synced_box["length"] = label.camera_synced_box.length
                camera_synced_box["center_x"] = label.camera_synced_box.center_x
                camera_synced_box["center_y"] = label.camera_synced_box.center_y
                camera_synced_box["center_z"] = label.camera_synced_box.center_z
                camera_synced_box["heading"] = label.camera_synced_box.heading
                camera_synced_box["label"] = obj_class
                camera_synced_box["speed"] = speed
                track_info_cur_frame[label_id]["camera_box"] = camera_synced_box
                c = math.cos(camera_synced_box["heading"])
                s = math.sin(camera_synced_box["heading"])
                rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = rotz_matrix
                obj_pose_vehicle[:3, 3] = np.array(
                    [
                        camera_synced_box["center_x"],
                        camera_synced_box["center_y"],
                        camera_synced_box["center_z"],
                    ]
                )
                camera_visible = []
                for camera_name in camera_names_dict.keys():
                    camera_calibration = [
                        obj
                        for obj in frame.context.camera_calibrations
                        if obj.name == camera_name
                    ][0]
                    vertices, valid = project_label_to_image(
                        dim=[
                            camera_synced_box["length"],
                            camera_synced_box["width"],
                            camera_synced_box["height"],
                        ],
                        obj_pose=obj_pose_vehicle,
                        calibration=camera_calibration,
                    )
                    if valid.any():
                        camera_visible.append(camera_name - 1)
                        track_camera_visible_cur_frame[camera_name - 1].append(label_id)
            else:
                track_info_cur_frame[label.id]["camera_box"] = None
        track_info[f"{frame_id:06d}"] = track_info_cur_frame
        track_camera_visible[f"{frame_id:06d}"] = track_camera_visible_cur_frame
    ego_frame_poses, _ = load_ego_poses(seq_save_dir)
    for label_id in trajectory_info.keys():
        new_trajectory = dict()
        trajectory = trajectory_info[label_id]
        trajectory = dict(sorted(trajectory.items(), key=lambda item: item[0]))
        dims = []
        frames = []
        timestamps = []
        poses_vehicle = []
        poses_world = []
        speeds = []
        for frame_id, bbox in trajectory.items():
            label = bbox["label"]
            dims.append([bbox["height"], bbox["width"], bbox["length"]])
            frames.append(int(frame_id))
            timestamps.append(bbox["timestamp"])
            speeds.append(bbox["speed"])
            pose_vehicle = np.eye(4)
            pose_vehicle[:3, :3] = np.array(
                [
                    [math.cos(bbox["heading"]), -math.sin(bbox["heading"]), 0],
                    [math.sin(bbox["heading"]), math.cos(bbox["heading"]), 0],
                    [0, 0, 1],
                ]
            )
            pose_vehicle[:3, 3] = np.array(
                [bbox["center_x"], bbox["center_y"], bbox["center_z"]]
            )
            ego_pose = ego_frame_poses[int(frame_id)]
            pose_world = np.matmul(ego_pose, pose_vehicle)
            poses_vehicle.append(pose_vehicle.astype(np.float32))
            poses_world.append(pose_world.astype(np.float32))
        dims = np.array(dims).astype(np.float32)
        dim = np.max(dims, axis=0)
        poses_vehicle = np.array(poses_vehicle).astype(np.float32)
        poses_world = np.array(poses_world).astype(np.float32)
        actor_world_postions = poses_world[:, :3, 3]
        distance = np.linalg.norm(actor_world_postions[0] - actor_world_postions[-1])
        dynamic = np.any(np.std(actor_world_postions, axis=0) > 0.5) or distance > 2
        new_trajectory["label"] = label
        new_trajectory["height"], new_trajectory["width"], new_trajectory["length"] = (
            dim[0],
            dim[1],
            dim[2],
        )
        new_trajectory["poses_vehicle"] = poses_vehicle
        new_trajectory["timestamps"] = timestamps
        new_trajectory["frames"] = frames
        new_trajectory["speeds"] = speeds
        new_trajectory["symmetric"] = label != "pedestrian"
        new_trajectory["deformable"] = label == "pedestrian"
        new_trajectory["stationary"] = not dynamic
        trajectory_info[label_id] = new_trajectory
    with open(os.path.join(track_dir, "track_info.pkl"), "wb") as f:
        pickle.dump(track_info, f)
    with open(os.path.join(track_dir, "track_camera_visible.pkl"), "wb") as f:
        pickle.dump(track_camera_visible, f)
    with open(os.path.join(track_dir, "trajectory.pkl"), "wb") as f:
        pickle.dump(trajectory_info, f)
    with open(os.path.join(track_dir, "track_ids.json"), "w") as f:
        json.dump(object_ids, f, indent=2)
    print("Processing tracking data done...")
    dynamic_mask_dir = os.path.join(seq_save_dir, "dynamic_mask")
    os.makedirs(dynamic_mask_dir, exist_ok=True)
    print("Processing dynamic mask...")
    track_info, track_camera_visible, trajectory = load_track(seq_save_dir)
    extrinsics, intrinsics = load_calibration(seq_save_dir)
    for frame, track_info_frame in track_info.items():
        track_camera_visible_cur_frame = track_camera_visible[frame]
        for cam, track_ids in track_camera_visible_cur_frame.items():
            dynamic_mask_name = f"{frame}_{cam}.png"
            dynamic_mask = np.zeros(
                (image_heights[cam], image_widths[cam]), dtype=np.uint8
            ).astype(np.bool_)
            calibration_dict = dict()
            calibration_dict["extrinsic"] = extrinsics[cam]
            calibration_dict["intrinsic"] = intrinsics[cam]
            calibration_dict["height"] = image_heights[cam]
            calibration_dict["width"] = image_widths[cam]
            for track_id in track_ids:
                object_tracklet = trajectory[track_id]
                if object_tracklet["stationary"]:
                    continue
                pose_idx = trajectory[track_id]["frames"].index(int(frame))
                pose_vehicle = trajectory[track_id]["poses_vehicle"][pose_idx]
                height, width, length = (
                    trajectory[track_id]["height"],
                    trajectory[track_id]["width"],
                    trajectory[track_id]["length"],
                )
                box_mask = project_label_to_mask(
                    dim=[length, width, height],
                    obj_pose=pose_vehicle,
                    calibration=None,
                    calibration_dict=calibration_dict,
                )
                dynamic_mask = np.logical_or(dynamic_mask, box_mask)
            dynamic_mask_path = os.path.join(dynamic_mask_dir, dynamic_mask_name)
            cv2.imwrite(dynamic_mask_path, dynamic_mask.astype(np.uint8) * 255)
