# Portions adapted from Street Crafter.
# Copyright 2024-2025 3D Vision Group, Zhejiang University.
# See licenses/StreetCrafter.txt for the upstream terms.


"""Render front-camera, BEV, and extreme lane-shift conditioning."""

import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from .geometry import (
    image_filename_to_frame,
    get_lane_shift_direction,
    image_heights,
    image_widths,
)
from .pointcloud import fetchPly
from .rasterizer import render_pointcloud_diff_point_rasterization


class _FFmpegVideoWriter:
    """Write H.264 MP4 via imageio/ffmpeg.

    Produces yuv420p H.264 files that are playable in VS Code, browsers, and
    all standard media players.  Provides the same .write() / .release()
    interface as cv2.VideoWriter so callers need no changes.

    Args:
        path:    Output file path.
        fps:     Frames per second.
        is_gray: If True, accepts HxW uint8 frames (converted to RGB internally).
                 If False, accepts HxWx3 BGR uint8 frames (cv2 convention).
    """

    def __init__(self, path, fps, is_gray=False):
        import imageio

        self._writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            output_params=["-pix_fmt", "yuv420p", "-crf", "18"],
        )
        self._is_gray = is_gray

    def write(self, frame):
        if self._is_gray:
            frame_rgb = np.stack([frame, frame, frame], axis=-1)
        else:
            frame_rgb = frame[:, :, ::-1]
        self._writer.append_data(frame_rgb)

    def release(self):
        self._writer.close()


def create_bev_camera_pose(ego_pose, height, forward_offset, pitch_deg):
    """Create bird's eye view camera pose.

    Args:
        ego_pose: Vehicle pose (4x4 matrix)
        height: Camera height above vehicle (meters)
        forward_offset: Forward offset from vehicle center (meters)
        pitch_deg: Pitch angle (degrees, 0=horizontal, 90=straight down)

    Returns:
        Camera-to-world matrix (4x4)
    """
    local_offset = np.array([forward_offset, 0, height, 1])
    world_pos = ego_pose @ local_offset
    vehicle_forward = ego_pose[:3, 0]
    vehicle_left = ego_pose[:3, 1]
    vehicle_up = ego_pose[:3, 2]
    pitch_rad = np.deg2rad(pitch_deg)
    cam_forward = np.cos(pitch_rad) * vehicle_forward - np.sin(pitch_rad) * vehicle_up
    cam_forward = cam_forward / np.linalg.norm(cam_forward)
    cam_right = -vehicle_left
    cam_right = cam_right / np.linalg.norm(cam_right)
    cam_down = np.cross(cam_forward, cam_right)
    cam_down = cam_down / np.linalg.norm(cam_down)
    c2w = np.eye(4)
    c2w[:3, 0] = cam_right
    c2w[:3, 1] = cam_down
    c2w[:3, 2] = cam_forward
    c2w[:3, 3] = world_pos[:3]
    return c2w


def create_elevate_camera_pose(
    ego_pose, cam_to_ego, progress, target_height, target_pitch_deg, target_backward
):
    """Create camera pose that elevates from front camera to BEV.

    Interpolates between the front camera position/orientation and a BEV position
    based on progress (0.0 = front cam, 1.0 = BEV).

    Args:
        ego_pose: Vehicle pose (4x4 matrix)
        cam_to_ego: Original camera-to-ego transformation (4x4 matrix)
        progress: Interpolation progress from 0.0 (front cam) to 1.0 (BEV)
        target_height: Target BEV height in meters
        target_pitch_deg: Target BEV pitch angle in degrees
        target_backward: Target backward offset in meters (negative = behind car)

    Returns:
        Camera-to-world matrix (4x4)
    """
    start_c2w = ego_pose @ cam_to_ego
    end_c2w = create_bev_camera_pose(
        ego_pose, target_height, target_backward, target_pitch_deg
    )
    interp_pos = (1 - progress) * start_c2w[:3, 3] + progress * end_c2w[:3, 3]
    interp_right = (1 - progress) * start_c2w[:3, 0] + progress * end_c2w[:3, 0]
    interp_down = (1 - progress) * start_c2w[:3, 1] + progress * end_c2w[:3, 1]
    interp_forward = (1 - progress) * start_c2w[:3, 2] + progress * end_c2w[:3, 2]
    interp_forward = interp_forward / np.linalg.norm(interp_forward)
    interp_right = interp_right - np.dot(interp_right, interp_forward) * interp_forward
    interp_right = interp_right / np.linalg.norm(interp_right)
    interp_down = np.cross(interp_forward, interp_right)
    c2w = np.eye(4)
    c2w[:3, 0] = interp_right
    c2w[:3, 1] = interp_down
    c2w[:3, 2] = interp_forward
    c2w[:3, 3] = interp_pos
    return c2w


class Renderer:
    def __init__(self, args):
        self.data_dir = str(args.processed)
        self.output = args.output
        self.num_frames = args.num_frames
        self.trajectories = args.trajectories
        self.point_ratio = args.ratio
        self.point_scale = args.point_scale
        self.stride = args.stride
        self.height = 576
        self.gpu = args.gpu
        self.flip_axis = 1
        self.bev_height = args.bev_height
        self.bev_pitch = args.bev_pitch
        self.bev_backward = args.bev_backward
        self.lane_shift = args.lane_shift

    def _pose_for_frame(self, traj, frame, count, ego_pose, ego_poses, front):
        if traj == "0":
            return ego_pose @ front
        t = frame / max(count - 1, 1)
        if traj == "bev":
            return create_elevate_camera_pose(
                ego_pose,
                front,
                0.5 * (1 - np.cos(np.pi * t)),
                self.bev_height,
                self.bev_pitch,
                self.bev_backward,
            )
        if traj == "lane_shift_extreme":
            direction = get_lane_shift_direction(ego_poses, frame)
            pose = ego_pose.copy()
            pose[:3, 3] += direction * self.lane_shift * np.sin(2 * np.pi * t)
            return pose @ front
        raise ValueError(f"Unknown trajectory: {traj}")

    def read_lidar_ply(self, lidar_dir, ego_poses, trajectory, frame_stride=1):
        """Read LiDAR PLY files and transform to world coordinates.

        Args:
            frame_stride: Only load frames where frame_index % frame_stride == 0.
                          Use _STRIDE_STEP (10) for stride_global mode to skip
                          the 9/10 frames that are never used.
        """
        ply_dict = dict()
        lidar_background_dir = os.path.join(lidar_dir, "background")
        ply_dict_background = dict()
        bkgd_ply_list = sorted(
            [
                os.path.join(lidar_background_dir, f)
                for f in os.listdir(lidar_background_dir)
                if f.endswith(".ply") and f != "full.ply"
            ]
        )
        for i, bkgd_ply_path in enumerate(
            tqdm(bkgd_ply_list, desc="Reading background ply")
        ):
            frame = image_filename_to_frame(os.path.basename(bkgd_ply_path))
            if frame_stride > 1 and frame % frame_stride != 0:
                continue
            bkgd_ply = fetchPly(bkgd_ply_path)
            mask = bkgd_ply.mask
            xyz_vehicle = bkgd_ply.points[mask]
            xyz_vehicle_homo = np.concatenate(
                [xyz_vehicle, np.ones_like(xyz_vehicle[..., :1])], axis=-1
            )
            xyz_world = xyz_vehicle_homo @ ego_poses[frame].T
            xyz_world = xyz_world[..., :3]
            rgb = bkgd_ply.colors[mask]
            ply_dict_background[frame] = np.concatenate([xyz_world, rgb], axis=-1)
        ply_dict["background"] = ply_dict_background
        lidar_actor_dir = os.path.join(lidar_dir, "actor")
        if os.path.exists(lidar_actor_dir):
            for track_id in os.listdir(lidar_actor_dir):
                ply_dict_actor = dict()
                lidar_actor_dir_ = os.path.join(lidar_actor_dir, track_id)
                actor_ply_list = sorted(
                    [
                        os.path.join(lidar_actor_dir_, f)
                        for f in os.listdir(lidar_actor_dir_)
                        if f.endswith(".ply") and f != "full.ply"
                    ]
                )
                for actor_ply_path in actor_ply_list:
                    frame = image_filename_to_frame(os.path.basename(actor_ply_path))
                    if frame_stride > 1 and frame % frame_stride != 0:
                        continue
                    actor_ply = fetchPly(actor_ply_path)
                    mask = actor_ply.mask
                    if mask.sum() == 0:
                        continue
                    xyz = actor_ply.points[mask]
                    rgb = actor_ply.colors[mask]
                    if xyz.shape[0] == 0:
                        continue
                    rigid = not trajectory[track_id]["deformable"]
                    if rigid:
                        num_pointcloud_1 = (xyz[:, self.flip_axis] > 0).sum()
                        num_pointcloud_2 = (xyz[:, self.flip_axis] < 0).sum()
                        if num_pointcloud_1 >= num_pointcloud_2:
                            xyz_part = xyz[xyz[:, self.flip_axis] > 0]
                            rgb_part = rgb[xyz[:, self.flip_axis] > 0]
                        else:
                            xyz_part = xyz[xyz[:, self.flip_axis] < 0]
                            rgb_part = rgb[xyz[:, self.flip_axis] < 0]
                        xyz_flip = xyz_part.copy()
                        xyz_flip[:, self.flip_axis] *= -1
                        rgb_flip = rgb_part.copy()
                        xyz = np.concatenate([xyz, xyz_flip], axis=0)
                        rgb = np.concatenate([rgb, rgb_flip], axis=0)
                    ply_dict_actor[frame] = np.concatenate([xyz, rgb], axis=-1)
                ply_dict[track_id] = ply_dict_actor
        return ply_dict

    def make_lidar_ply_frames(self, ply_dict, frames):
        """Aggregate LiDAR point clouds from a specific list of frame indices."""
        ply_frame_dict = dict()
        bkgd_ply = []
        for frame in frames:
            if frame in ply_dict["background"]:
                bkgd_ply.append(ply_dict["background"][frame])
        if len(bkgd_ply) > 0:
            bkgd_ply = np.concatenate(bkgd_ply, axis=0)
            ply_frame_dict["background"] = bkgd_ply
        for track_id in ply_dict.keys():
            if track_id == "background":
                continue
            actor_ply = []
            for frame in frames:
                if frame not in ply_dict[track_id]:
                    continue
                actor_ply.append(ply_dict[track_id][frame])
            if len(actor_ply) == 0:
                continue
            actor_ply = np.concatenate(actor_ply, axis=0)
            ply_frame_dict[track_id] = actor_ply
        return ply_frame_dict

    def transform_lidar_ply(self, lidar_ply, pose):
        """Transform LiDAR point cloud by pose."""
        xyz, rgb = (lidar_ply[..., :3], lidar_ply[..., 3:])
        xyz_homo = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)
        xyz = (xyz_homo @ pose.T)[..., :3]
        lidar_ply = np.concatenate([xyz, rgb], axis=-1)
        return lidar_ply

    def render_one(self, scene_idx):
        scene_name = str(scene_idx).zfill(3)
        save_scene_dir = os.path.join(self.data_dir, scene_name)
        scene_dir = save_scene_dir
        render_save_dir = str(self.output)
        os.makedirs(render_save_dir, exist_ok=True)
        from .geometry import load_ego_poses, load_calibration, load_track

        ego_frame_poses, ego_cam_poses = load_ego_poses(datadir=scene_dir)
        extrinsics, intrinsics = load_calibration(datadir=scene_dir)
        track_info, _, trajectory = load_track(scene_dir)
        lidar_dir = os.path.join(scene_dir, "lidar")
        num_frames = len(ego_frame_poses)
        if num_frames < self.num_frames:
            raise ValueError(
                f"Need {self.num_frames} frames; scene contains {num_frames}"
            )
        gpu_id = self.gpu
        h_out, w_out, ixt_render, front_cam_to_ego = self._front_cam_geometry(
            extrinsics, intrinsics
        )
        render_num_frames = self.num_frames
        print(
            f"Scene {scene_idx:03d}: rendering {len(self.trajectories)} camera paths on GPU {gpu_id}"
        )
        global_stride_ply = None
        STRIDE_LIDAR_STEP = self.stride
        ply_dict = self.read_lidar_ply(
            lidar_dir=lidar_dir,
            ego_poses=ego_frame_poses,
            trajectory=trajectory,
            frame_stride=STRIDE_LIDAR_STEP,
        )
        if 0 < self.point_ratio < 1:
            ply_dict = self._apply_point_ratio(ply_dict, scene_idx)
        global_stride_frames = list(range(0, num_frames, STRIDE_LIDAR_STEP))
        global_stride_ply = self.make_lidar_ply_frames(ply_dict, global_stride_frames)
        counts = {key: len(points) for key, points in global_stride_ply.items()}
        (self.output / "sampling.json").write_text(json.dumps(counts, indent=2) + "\n")
        camera_data_per_traj = {}
        for traj in self.trajectories:
            traj_c2w_per_frame = []
            for frame in range(render_num_frames):
                ego_pose = ego_cam_poses[0, frame]
                c2w = self._pose_for_frame(
                    traj,
                    frame,
                    render_num_frames,
                    ego_pose,
                    ego_frame_poses,
                    front_cam_to_ego,
                )
                traj_c2w_per_frame.append(np.asarray(c2w, dtype=np.float64).tolist())
            traj_entry = {
                "camera_id": traj,
                "camera_name": "CAM_FRONT" if traj == "0" else f"NOVEL_{traj.upper()}",
                "reference_camera": "CAM_FRONT",
                "reference_camera_idx": 0,
                "intrinsics": np.asarray(ixt_render, dtype=np.float64).tolist(),
                "image_width": int(w_out),
                "image_height": int(h_out),
                "num_frames": int(render_num_frames),
                "c2w_per_frame": traj_c2w_per_frame,
            }
            camera_data_per_traj[traj] = traj_entry
            print(
                f"  [{traj}] rendering {render_num_frames} frames (scene has {num_frames})"
            )
            video_writer_rgb = _FFmpegVideoWriter(
                os.path.join(render_save_dir, f"render_rgb_{traj}.mp4"), fps=10
            )
            video_writer_mask = _FFmpegVideoWriter(
                os.path.join(render_save_dir, f"render_mask_{traj}.mp4"),
                fps=10,
                is_gray=True,
            )
            depth_maps = []
            depth_min_global = float("inf")
            depth_max_global = float("-inf")
            for frame in tqdm(range(render_num_frames), desc=f"{traj}"):
                ego_pose = ego_cam_poses[0, frame]
                track_info_frame = track_info.get(f"{frame:06d}", {})
                c2w = np.asarray(traj_c2w_per_frame[frame], dtype=np.float64)
                ply_frame_dict = dict(global_stride_ply)
                if "background" not in ply_frame_dict:
                    rgb = np.zeros((h_out, w_out, 3), dtype=np.float32)
                    mask_img = np.zeros((h_out, w_out), dtype=np.float32)
                    depth = np.zeros((h_out, w_out), dtype=np.float32)
                else:
                    ply_parts = [ply_frame_dict.pop("background")]
                    for track_id, ply_actor in ply_frame_dict.items():
                        if track_id not in track_info_frame:
                            continue
                        camera_box = track_info_frame[track_id].get("camera_box")
                        lidar_box = track_info_frame[track_id]["lidar_box"]
                        box = camera_box if camera_box is not None else lidar_box
                        pose_v = np.eye(4)
                        pose_v[:3, :3] = np.array(
                            [
                                [np.cos(box["heading"]), -np.sin(box["heading"]), 0],
                                [np.sin(box["heading"]), np.cos(box["heading"]), 0],
                                [0, 0, 1],
                            ]
                        )
                        pose_v[:3, 3] = np.array(
                            [box["center_x"], box["center_y"], box["center_z"]]
                        )
                        pose_v = ego_pose @ pose_v
                        ply_parts.append(self.transform_lidar_ply(ply_actor, pose_v))
                    ply_frame = np.concatenate(ply_parts, axis=0)
                    ply_xyz, ply_rgb = (ply_frame[..., :3], ply_frame[..., 3:])
                    w2c = np.linalg.inv(c2w)
                    device = torch.device(f"cuda:{gpu_id}")
                    ply_xyz_t = torch.as_tensor(
                        ply_xyz, dtype=torch.float32, device=device
                    )
                    ply_rgb_t = torch.as_tensor(
                        ply_rgb, dtype=torch.float32, device=device
                    )
                    w2c_t = torch.as_tensor(w2c, dtype=torch.float32, device=device)
                    ixt_t = torch.as_tensor(
                        ixt_render, dtype=torch.float32, device=device
                    )
                    cam_xyz = ply_xyz_t @ w2c_t[:3, :3].T + w2c_t[:3, 3:].T
                    depth_v = cam_xyz[:, 2]
                    pix = cam_xyz @ ixt_t.T
                    pix = pix[:, :2] / pix[:, 2:]
                    valid = (
                        (depth_v > 0.001)
                        & (pix[:, 0] >= 0)
                        & (pix[:, 0] < w_out)
                        & (pix[:, 1] >= 0)
                        & (pix[:, 1] < h_out)
                    )
                    valid_xyz = ply_xyz_t[valid]
                    valid_rgb = ply_rgb_t[valid]
                    valid_depth = depth_v[valid]
                    valid_mask = torch.ones_like(valid_depth)
                    valid_feat = torch.cat(
                        [valid_rgb, valid_depth[:, None], valid_mask[:, None]], dim=-1
                    )
                    rgb, mask_img, depth = self._render_frame(
                        c2w, ixt_render, valid_xyz, valid_feat, h_out, w_out, gpu_id
                    )
                video_writer_rgb.write(
                    cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                video_writer_mask.write((mask_img * 255).astype(np.uint8))
                depth_maps.append(depth)
                d_valid = depth[depth > 0]
                if len(d_valid) > 0:
                    depth_min_global = min(depth_min_global, float(d_valid.min()))
                    depth_max_global = max(depth_max_global, float(d_valid.max()))
            video_writer_rgb.release()
            video_writer_mask.release()
            self._save_depth_outputs(
                depth_maps, render_save_dir, traj, depth_min_global, depth_max_global
            )
        camera_data = {
            "scene_id": int(scene_idx),
            "render_num_frames": int(render_num_frames),
            "ego_frame_poses": [
                np.asarray(p, dtype=np.float64).tolist()
                for p in ego_frame_poses[:render_num_frames]
            ],
            "trajectories": camera_data_per_traj,
        }
        with open(os.path.join(render_save_dir, "camera_data.json"), "w") as f:
            json.dump(camera_data, f, indent=2)

    def _scaled_dims(self, cam):
        """Return (h_out, w_out, scale) for output.

        If the original height is already <= self.height, no scaling is done.
        Width is kept proportional and rounded to an even number.
        """
        h_orig = image_heights[cam]
        w_orig = image_widths[cam]
        if h_orig > self.height:
            scale = self.height / h_orig
            h_out = self.height
            w_out = int(round(w_orig * scale / 2)) * 2
        else:
            h_out, w_out, scale = (h_orig, w_orig, 1.0)
        return (h_out, w_out, scale)

    def _scale_intrinsics(self, ixt, scale):
        """Return a copy of the 3×3 intrinsics matrix scaled by `scale`."""
        ixt_s = ixt.copy()
        ixt_s[0, :] *= scale
        ixt_s[1, :] *= scale
        return ixt_s

    def _apply_point_ratio(self, ply_dict, scene_idx):
        """Subsample every per-frame point cloud with a deterministic, stable mask.

        Uses a per-(scene, key, frame) seed so the same points are always
        selected for a given scene, regardless of render order.  This prevents
        temporal jitter.
        """
        if not 0 < self.point_ratio < 1:
            return ply_dict
        result = {}
        for key in ply_dict:
            result[key] = {}
            for frame, pts in ply_dict[key].items():
                n = len(pts)
                if n == 0:
                    result[key][frame] = pts
                    continue
                seed = hash((scene_idx, str(key), int(frame))) % 2**31
                rng = np.random.RandomState(seed)
                k = max(1, int(n * self.point_ratio))
                idx = rng.choice(n, k, replace=False)
                idx.sort()
                result[key][frame] = pts[idx]
        return result

    def _render_frame(
        self, c2w, ixt_render, valid_ply_xyz, valid_ply_feature, h_out, w_out, gpu_id
    ):
        """Render a single frame. Returns (rgb, mask, depth) as float32 numpy arrays.

        Override in subclasses to swap the rasterization backend.
        """
        ply_render = render_pointcloud_diff_point_rasterization(
            c2w,
            ixt_render,
            valid_ply_xyz,
            valid_ply_feature,
            h_out,
            w_out,
            gpu_id,
            scale=self.point_scale,
        )
        rgb = ply_render[0, ..., :3].detach().cpu().numpy()
        mask = ply_render[0, ..., 3].detach().cpu().numpy()
        depth = ply_render[0, ..., 4].detach().cpu().numpy()
        return (rgb, mask, depth)

    def _normalize_depth(self, depth_map, scale, offset):
        """Normalize a single depth map to [0, 1]."""
        depth_normalized = np.zeros_like(depth_map, dtype=np.float32)
        if scale > 0:
            depth_normalized[depth_map > 0] = (
                depth_map[depth_map > 0] - offset
            ) / scale
        return np.clip(depth_normalized, 0, 1)

    def _save_depth_outputs(
        self, depth_maps, render_save_dir, cam, depth_min_global, depth_max_global
    ):
        """Save sequence-normalized depth and its normalization metadata."""
        if depth_max_global > depth_min_global:
            scale = depth_max_global - depth_min_global
            offset = depth_min_global
        else:
            scale, offset = (1.0, 0.0)
        depth_norm_metadata = {
            "depth_min": float(depth_min_global),
            "depth_max": float(depth_max_global),
            "scale": float(scale),
            "offset": float(offset),
            "normalization_mode": "per_sequence",
        }
        norm_path = os.path.join(render_save_dir, f"depth_normalization_{cam}.json")
        with open(norm_path, "w") as f:
            json.dump(depth_norm_metadata, f, indent=2)
        video_writer_depth = _FFmpegVideoWriter(
            os.path.join(render_save_dir, f"render_depth_{cam}.mp4"),
            fps=10,
            is_gray=False,
        )
        for depth_map in depth_maps:
            depth_normalized = self._normalize_depth(depth_map, scale, offset)
            depth_color = cv2.applyColorMap(
                (depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            depth_color[depth_map == 0] = 0
            video_writer_depth.write(depth_color)
        video_writer_depth.release()

    def _front_cam_geometry(self, extrinsics, intrinsics):
        """Return (h_out, w_out, ixt_scaled, cam_to_ego) for CAM_FRONT."""
        h_out, w_out, scale = self._scaled_dims(0)
        ixt = self._scale_intrinsics(intrinsics[0], scale)
        return (h_out, w_out, ixt, extrinsics[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--scene", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--trajectories",
        nargs="+",
        choices=["0", "bev", "lane_shift_extreme"],
        default=["bev", "lane_shift_extreme"],
        help="Camera 0 follows the recorded FRONT poses; the other paths are novel views",
    )
    p.add_argument("--num-frames", type=int, choices=[49, 81], default=81)
    p.add_argument("--ratio", type=float, default=0.01)
    p.add_argument("--point-scale", type=float, default=0.02)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bev-height", type=float, default=60)
    p.add_argument("--bev-pitch", type=float, default=60)
    p.add_argument("--bev-backward", type=float, default=-5)
    p.add_argument("--lane-shift", type=float, default=7)
    args = p.parse_args()
    if not 0 < args.ratio <= 1 or args.stride < 1 or args.point_scale <= 0:
        p.error("Require 0 < ratio <= 1, stride >= 1, point-scale > 0")
    if os.environ.get("PYTHONHASHSEED") != "0":
        p.error("Run with PYTHONHASHSEED=0 to reproduce the validated point sample")
    if not torch.cuda.is_available():
        p.error("Rendering requires the native CUDA point rasterizer and a GPU")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use a new output directory for each render configuration")
    report = json.loads(
        (args.processed / f"{args.scene:03d}" / "processing.json").read_text()
    )
    if report["status"] != "complete":
        p.error("Raw extraction has not completed")
    args.output.mkdir(parents=True, exist_ok=True)
    Renderer(args).render_one(args.scene)
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    settings.update(context=report["context"], pythonhashseed=0, status="complete")
    (args.output / "render.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
