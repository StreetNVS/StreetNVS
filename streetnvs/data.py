"""Strict loader for StreetNVS evaluation and novel-view clips."""

import csv
import json
from pathlib import Path
import av
import torch
from PIL import Image


def read_video(path):
    with av.open(str(path)) as c:
        frames = [f.to_image() for f in c.decode(video=0)]
    if not frames:
        raise ValueError(f"Empty video: {path}")
    return frames


def load_sample(
    data,
    scene="0",
    camera="2",
    start=0,
    height=480,
    width=832,
    refs=("0", "1", "2"),
    num_frames=49,
):
    if num_frames not in (49, 81):
        raise ValueError("Supported clip lengths are 49 and 81 frames")
    data = Path(data)
    with (data / "metadata_cond.csv").open() as metadata:
        rows = list(csv.DictReader(metadata))
    matches = [
        r
        for r in rows
        if r["scene_id"] == str(scene)
        and r["camera_id"] == str(camera)
        and int(r["start_frame"]) == start
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one clip for scene={scene}, camera={camera}, start={start}; got {len(matches)}"
        )
    row = matches[0]
    cameras = json.loads((data / "camera_data.json").read_text())[str(scene)]

    def resize(im):
        return im.resize((width, height), Image.Resampling.BILINEAR)

    def clip(key):
        if not row.get(key):
            raise ValueError(f"Missing required input: {key}")
        path = (data / row[key]).resolve()
        if not path.is_relative_to(data.resolve()):
            raise ValueError(f"Input escapes dataset directory: {key}")
        frames = read_video(path)
        if len(frames) != num_frames:
            raise ValueError(f"{key}: expected {num_frames} frames, got {len(frames)}")
        return [resize(f) for f in frames]

    def camera_params(cid):
        c = cameras[str(cid)]
        if "c2w_per_frame" in c:
            poses = torch.tensor(c["c2w_per_frame"], dtype=torch.float32)
        else:
            poses = torch.tensor(c["ego_poses"], dtype=torch.float32) @ torch.tensor(
                c["extrinsics"], dtype=torch.float32
            )
        first = 0 if len(poses) == num_frames else start
        poses = poses[first : first + num_frames].clone()
        if len(poses) != num_frames:
            raise ValueError(f"Camera {cid}: not enough poses")
        poses[:, :3, 3] *= 0.1
        k = torch.eye(4, dtype=torch.float32)
        intr = torch.tensor(c["intrinsics"], dtype=torch.float32)
        orig_w = c.get("image_width", 1920)
        orig_h = c.get("image_height", 886 if str(cid) in ("3", "4") else 1280)
        k[0, 0] = intr[0, 0] / orig_w
        k[0, 2] = intr[0, 2] / orig_w
        k[1, 1] = intr[1, 1] / orig_h
        k[1, 2] = intr[1, 2] / orig_h
        return poses, k

    poses, k = camera_params(camera)
    anchor = torch.linalg.inv(poses[0:1])
    target = clip("video") if row.get("video") else None
    if target:
        # Match the original I2V entrypoint's PIL resize of the first RGB frame.
        input_image = read_video(data / row["video"])[0].resize((width, height))
    else:
        input_image = (
            Image.open(data / row["cond_reference_image"])
            .convert("RGB")
            .resize((width, height))
        )
    sample = dict(
        prompt=row["prompt"],
        input_image=input_image,
        cond_video=clip("cond_video"),
        cond_video_mask=clip("cond_video_mask"),
        cond_depth_video=clip("cond_video_depth"),
        camera_w2cs=torch.linalg.inv(anchor @ poses),
        camera_Ks=k.expand(num_frames, -1, -1).clone(),
    )
    ref_images = []
    ref_poses = []
    ref_ks = []
    for cid in refs:
        if str(cid) == str(camera):
            continue
        ref_rows = [
            r
            for r in rows
            if r["scene_id"] == str(scene)
            and r["camera_id"] == str(cid)
            and int(r["start_frame"]) == start
        ]
        if len(ref_rows) != 1:
            raise ValueError(f"Missing reference camera {cid}")
        im = read_video(data / ref_rows[0]["video"])[0]
        cp, ck = camera_params(cid)
        ref_images.append(resize(im))
        ref_poses.append(torch.linalg.inv(anchor @ cp[:1])[0])
        ref_ks.append(ck)
    if not ref_images:
        raise ValueError(
            "Select at least one reference camera other than the target camera"
        )
    sample.update(
        camref_images=ref_images,
        camref_images_w2cs=torch.stack(ref_poses),
        camref_images_Ks=torch.stack(ref_ks),
        camref_temporal_ids=torch.zeros(len(ref_images), dtype=torch.long),
    )
    return sample, row
