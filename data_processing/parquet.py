"""Repack official Parquet into one row group per frame without changing values."""

import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def repack(source, target, context):
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / "repack.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for component in (
        "lidar_calibration",
        "vehicle_pose",
        "lidar",
        "lidar_pose",
        "lidar_camera_projection",
    ):
        original = source / component / f"{context}.parquet"
        destination = target / component / original.name
        destination.parent.mkdir(exist_ok=True)
        if component in ("lidar_calibration", "vehicle_pose"):
            if not destination.exists():
                shutil.copyfile(original, destination)
            continue
        if report.get(component, {}).get("verified_equal") and destination.exists():
            continue
        reader = pq.ParquetFile(original)
        timestamps = (
            pq.read_table(original, columns=["key.frame_timestamp_micros"])
            .column(0)
            .to_numpy()
        )
        assert np.all(timestamps[1:] >= timestamps[:-1]), (
            "Raw timestamps must be sorted"
        )
        partial = destination.with_suffix(".part")
        groups = 0
        with pq.ParquetWriter(
            partial, reader.schema_arrow, compression="snappy"
        ) as writer:
            # A single-row group is enough for predicate pushdown, and bounded
            # batches keep the large nested LiDAR arrays out of pandas.
            for batch in reader.iter_batches(batch_size=5):
                table = pa.Table.from_batches([batch])
                keys = table["key.frame_timestamp_micros"].to_numpy()
                starts = np.r_[0, np.nonzero(keys[1:] != keys[:-1])[0] + 1, len(keys)]
                for left, right in zip(starts[:-1], starts[1:]):
                    part = table.slice(int(left), int(right - left))
                    writer.write_table(part, row_group_size=len(part))
                    groups += 1
        # Compare complete nested values in bounded, aligned batches before use.
        old = reader.iter_batches(batch_size=5)
        new_reader = pq.ParquetFile(partial)
        new = new_reader.iter_batches(batch_size=5)
        checked = 0
        for a, b in zip(old, new, strict=True):
            assert a.equals(b), (
                f"Parquet value mismatch in {component}, batch {checked}"
            )
            checked += 1
        assert new_reader.metadata.num_rows == reader.metadata.num_rows
        partial.replace(destination)
        report[component] = dict(
            original_row_groups=reader.metadata.num_row_groups,
            repacked_row_groups=groups,
            rows=reader.metadata.num_rows,
            batches_compared=checked,
            verified_equal=True,
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("Repacked and verified:", component, report[component], flush=True)
    return target
