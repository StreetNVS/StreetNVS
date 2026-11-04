"""Export rendered conditioning and five raw camera references for inference."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import av
import imageio_ffmpeg

from .geometry import load_calibration, load_ego_poses


def run_ffmpeg(arguments, output, count):
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        *map(str, arguments),
        "-vframes",
        str(count),
        "-r",
        "10",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.decode(errors="replace")) from exc
    with av.open(str(output)) as video:
        actual = sum(1 for _ in video.decode(video=0))
    if actual != count:
        raise ValueError(f"{output}: expected {count} frames, got {actual}")


def encode_reference(images, output):
    # Match the reference exporter's concat timing and scale filter.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as listing:
        for path in images:
            if not path.is_file():
                raise FileNotFoundError(path)
            escaped = str(path.resolve()).replace("'", "'\\''")
            listing.write(f"file '{escaped}'\nduration 0.1\n")
        listing.flush()
        run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", listing.name, "-vf", "scale=-2:576"],
            output,
            len(images),
        )


def export(processed, rendered, output, scene_id):
    settings = json.loads((rendered / "render.json").read_text())
    processing = json.loads(
        (processed / f"{scene_id:03d}" / "processing.json").read_text()
    )
    if (
        settings["status"] != "complete"
        or processing["status"] != "complete"
        or settings["context"] != processing["context"]
        or settings["scene"] != scene_id
    ):
        raise ValueError(
            "Extraction and rendering must be complete and refer to the same scene"
        )
    count = settings["num_frames"]
    camera_data = json.loads((rendered / "camera_data.json").read_text())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new export directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scene = processed / f"{scene_id:03d}"
    (output / "references").mkdir()
    # Copy the original JPEG bytes; keep every inference input inside the export.
    reference = output / "references/front.jpg"
    shutil.copyfile(scene / "images/000000_0.png", reference)
    fields = [
        "video",
        "prompt",
        "cond_video",
        "cond_video_mask",
        "cond_video_depth",
        "cond_reference_image",
        "scene_id",
        "camera_id",
        "start_frame",
        "end_frame",
    ]
    rows, cameras = {}, {}

    def row(camera):
        result = dict.fromkeys(fields, "")
        result.update(
            scene_id=scene_id,
            camera_id=camera,
            start_frame=0,
            end_frame=count,
            prompt="The video depicts a view of a street",
        )
        return result

    for trajectory, camera in camera_data["trajectories"].items():
        destination = output / trajectory
        destination.mkdir()
        entry = row(trajectory)
        entry["cond_reference_image"] = "references/front.jpg"
        for kind, field in [
            ("rgb", "cond_video"),
            ("mask", "cond_video_mask"),
            ("depth", "cond_video_depth"),
        ]:
            source = rendered / f"render_{kind}_{trajectory}.mp4"
            arguments = ["-i", source, "-ss", "0.0"]
            if kind == "mask":
                arguments += ["-vf", "negate"]
            target = destination / f"{kind}.mp4"
            run_ffmpeg(arguments, target, count)
            entry[field] = str(target.relative_to(output))
        shutil.copyfile(
            rendered / f"depth_normalization_{trajectory}.json",
            destination / "depth_normalization.json",
        )
        rows[trajectory] = entry
        cameras[trajectory] = dict(
            camera, scene_id=scene_id, ego_poses=camera_data["ego_frame_poses"]
        )
    ego, ego_cam = load_ego_poses(str(scene))
    extrinsics, intrinsics = load_calibration(str(scene))
    for camera in range(5):
        video = output / "references" / f"camera_{camera}.mp4"
        images = [
            scene / "images" / f"{frame:06d}_{camera}.png" for frame in range(count)
        ]
        encode_reference(images, video)
        camera_id = str(camera)
        entry = rows.setdefault(camera_id, row(camera))
        entry["video"] = str(video.relative_to(output))
        # A recorded target keeps the same exposure poses and intrinsics used to render it.
        if camera_id not in cameras:
            cameras[camera_id] = dict(
                scene_id=scene_id,
                camera_id=camera,
                intrinsics=intrinsics[camera].tolist(),
                extrinsics=extrinsics[camera].tolist(),
                ego_poses=ego.tolist(),
                ego_cam_poses=ego_cam[camera].tolist(),
            )
    (output / "camera_data.json").write_text(
        json.dumps({str(scene_id): cameras}, indent=2) + "\n"
    )
    with (output / "metadata_cond.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows.values())
    (output / "preparation.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Inference inputs: {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", type=Path, required=True)
    p.add_argument("--rendered", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scene", type=int, default=10)
    args = p.parse_args()
    export(args.processed, args.rendered, args.output, args.scene)


if __name__ == "__main__":
    main()
