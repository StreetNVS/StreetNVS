"""Extract one matching Waymo TFRecord / Parquet scene on CPU."""

import argparse
import json
import os
from pathlib import Path


def extract(raw, output, scene_id):
    import pyarrow.parquet as pq
    from waymo_open_dataset import dataset_pb2
    from .tfrecord import records
    from .cameras import parse_seq_rawdata
    from .lidar import save_lidar
    from .geometry import load_track
    from .parquet import repack

    tfrecords = list((raw / "v1").glob("*.tfrecord"))
    if len(tfrecords) != 1:
        raise ValueError("--raw must contain exactly one v1/*.tfrecord")
    tfrecord = tfrecords[0]
    timestamps, contexts = [], set()
    for payload in records(tfrecord):
        frame = dataset_pb2.Frame.FromString(payload)
        contexts.add(frame.context.name)
        timestamps.append(frame.timestamp_micros)
    if len(contexts) != 1 or not timestamps:
        raise ValueError("Expected one nonempty Waymo scene")
    context = contexts.pop()
    if tfrecord.name != f"segment-{context}_with_camera_labels.tfrecord":
        raise ValueError("TFRecord filename and embedded context disagree")
    vehicle = pq.read_table(raw / "v2/vehicle_pose" / f"{context}.parquet").to_pandas()
    if vehicle["key.frame_timestamp_micros"].tolist() != timestamps or set(
        vehicle["key.segment_context_name"]
    ) != {context}:
        raise ValueError(
            "TFRecord and v2 vehicle poses differ in context or timestamp order"
        )
    scene = output / f"{scene_id:03d}"
    scene.mkdir(parents=True, exist_ok=True)
    report_path = scene / "processing.json"
    # Avoid resuming a partial extraction with incompatible inputs. Reuse is
    # explicit: callers may point render/export at a previously completed scene.
    if any(scene.iterdir()):
        raise FileExistsError(f"Use a new output directory; {scene} is not empty")
    report = dict(
        context=context,
        scene_id=scene_id,
        frame_count=len(timestamps),
        source="official_raw_waymo",
        source_directory=str(raw.resolve()),
        completed_stages=[],
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    parse_seq_rawdata(str(tfrecord), str(scene))
    for folder, pattern, expected in [
        ("images", "*.png", 5 * len(timestamps)),
        ("ego_pose", "*.txt", 6 * len(timestamps)),
        ("dynamic_mask", "*.png", 5 * len(timestamps)),
    ]:
        count = len(list((scene / folder).glob(pattern)))
        if count != expected:
            raise RuntimeError(f"Incomplete {folder}: {count} != {expected}")
        report[folder] = count
    report["completed_stages"].append("cameras_tracks_masks")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    parquet = repack(raw / "v2", output / ".parquet" / context, context)
    save_lidar(str(parquet), str(tfrecord), str(scene))
    if len(list((scene / "lidar/background").glob("*.ply"))) != len(timestamps):
        raise RuntimeError("Incomplete background point clouds")
    _, _, trajectories = load_track(str(scene))
    moving = [
        key
        for key, item in trajectories.items()
        if not item["stationary"] and item["label"] != "sign"
    ]
    actors = list((scene / "lidar/actor").glob("*/full.ply"))
    report.update(
        background_frames=len(timestamps),
        actor_pointclouds=len(actors),
        tracked_objects=len(trajectories),
        moving_objects=len(moving),
        actors_without_points=[
            k for k in moving if not (scene / "lidar/actor" / k / "full.ply").exists()
        ],
        status="complete",
    )
    report["completed_stages"].append("lidar_background_and_actors")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Extracted scene {scene_id:03d}: {scene}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scene", type=int, default=10)
    args = p.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    if args.scene < 0:
        p.error("--scene must be nonnegative")
    extract(args.raw, args.output, args.scene)


if __name__ == "__main__":
    main()
