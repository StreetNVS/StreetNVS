"""Download one authorized raw Waymo scene and its five matching v2 components."""

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote


def get_json(session, url, **params):
    response = session.get(url, params=params, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Waymo storage returned HTTP {response.status_code}: {url}")
    return response.json()


def metadata(session, bucket, name):
    return get_json(
        session,
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/"
        + quote(name, safe=""),
    )


def verified_download(session, meta, dest):
    import google_crc32c

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    size = int(meta["size"])
    if not dest.exists():
        offset = part.stat().st_size if part.exists() else 0
        if offset > size:
            raise RuntimeError(f"Oversized partial file: {part}")
        if offset < size:
            url = (
                "https://storage.googleapis.com/download/storage/v1/b/"
                + meta["bucket"]
                + "/o/"
                + quote(meta["name"], safe="")
            )
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session.get(
                url,
                params={"alt": "media", "generation": meta["generation"]},
                headers=headers,
                stream=True,
                timeout=120,
            ) as response:
                expected_status = 206 if offset else 200
                if response.status_code != expected_status:
                    raise RuntimeError(
                        f"Download HTTP {response.status_code} for {meta['name']}"
                    )
                if offset and (
                    not response.headers.get("Content-Range", "").startswith(
                        f"bytes {offset}-"
                    )
                ):
                    raise RuntimeError("Server did not honor resume offset")
                with part.open("ab" if offset else "wb") as output:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        output.write(chunk)
        path = part
    else:
        path = dest
    if path.stat().st_size != size:
        raise RuntimeError(f"Incomplete file: {path}")
    checksum, sha = (google_crc32c.Checksum(), hashlib.sha256())
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            checksum.update(chunk)
            sha.update(chunk)
    if base64.b64encode(checksum.digest()).decode() != meta["crc32c"]:
        raise RuntimeError(f"CRC32C mismatch: {path}")
    if path == part:
        part.replace(dest)
    return dict(
        uri=f"gs://{meta['bucket']}/{meta['name']}",
        generation=meta["generation"],
        bytes=size,
        crc32c=meta["crc32c"],
        sha256=sha.hexdigest(),
        local_path=str(dest),
    )


SCENE_010 = "10868756386479184868_3000_000_3020_000"
COMPONENTS = (
    "lidar_calibration",
    "vehicle_pose",
    "lidar",
    "lidar_pose",
    "lidar_camera_projection",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", default=SCENE_010, help="One exact Waymo context ID")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+", args.context):
        p.error("--context must be one Waymo context ID, without paths or wildcards")
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/devstorage.read_only"]
    )
    session = AuthorizedSession(credentials)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "download.json"
    report = dict(context=args.context, status="downloading", files=[])
    if (
        manifest.exists()
        and json.loads(manifest.read_text())["context"] != args.context
    ):
        raise ValueError("Output directory already belongs to a different context")
    filename = f"segment-{args.context}_with_camera_labels.tfrecord"
    objects = [
        (
            "waymo_open_dataset_v_1_4_3",
            f"individual_files/validation/{filename}",
            args.output / "v1" / filename,
        )
    ]
    objects.extend(
        (
            "waymo_open_dataset_v_2_0_1",
            f"validation/{component}/{args.context}.parquet",
            args.output / "v2" / component / f"{args.context}.parquet",
        )
        for component in COMPONENTS
    )
    for bucket, name, path in objects:
        meta = metadata(session, bucket, name)
        report["files"].append(verified_download(session, meta, path))
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Verified {name}", flush=True)
    report["status"] = "complete"
    manifest.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
