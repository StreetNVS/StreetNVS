# Data Preparation

This guide prepares scene 010 from raw Waymo data for front-camera inference,
followed by bird's-eye view and extreme lane-shift inference. There are four steps:

```text
raw TFRecord + matching Parquet
  → images, camera poses, tracks, dynamic masks
  → background and actor point clouds
  → sparse RGB / alpha / depth and target camera trajectories
  → metadata_cond.csv + camera_data.json + conditioning/reference videos
```

All commands below run from the repository root. Dataset access is required
through the [Waymo Open Dataset](https://waymo.com/open/). Keep raw and derived
data under the ignored `data/` directory and follow your dataset agreement.

## 1. Environment Setup

Complete the main [environment setup](../README.md#environment-setup) first.
Rendering additionally needs a **CUDA 12.8 toolkit with `nvcc`**, a compatible
C++ compiler, and an NVIDIA GPU. Set `CUDA_HOME` to your toolkit if necessary.

```bash
bash data_processing/setup.sh
source scripts/env.sh
source .venv-data/bin/activate
```

The setup script uses **uv**, just like the main environment setup. It creates
`.venv-data/` with Python 3.11 for Waymo downloading and CPU extraction, and
installs the CUDA point rasterizer in the existing `.venv/` for rendering and
inference. This keeps the TensorFlow-based Waymo tools separate from the PyTorch
inference environment. Official Waymo readers and protobufs are built during setup.

Caches, downloads, builds, and credentials stay in `.runtime/` inside the
repository. In a new terminal, run `source scripts/env.sh` and activate the
environment for the step you are running. If you build the rasterizer without
a visible GPU, set `TORCH_CUDA_ARCH_LIST=9.0` for H100 before running setup,
or use the architecture of your target GPU.

## 2. Download One Raw Scene

Use the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) with an
account authorized to access Waymo. Run this after sourcing `scripts/env.sh`
so credentials are saved inside the checkout:

```bash
gcloud auth application-default login --no-launch-browser --disable-quota-project \
  --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/devstorage.read_only

python -m data_processing.download --output data/raw/scene010
```

The default context is `10868756386479184868_3000_000_3020_000` (scene 010).
The downloader fetches **one TFRecord from v1.4.3 and five matching v2.0.1
Parquet components**: `lidar_calibration`, `vehicle_pose`, `lidar`, `lidar_pose`,
and `lidar_camera_projection`. This scene totals about 1.12 GB. The downloader
resumes interrupted downloads and checks that each file is complete.

To use another validation scene, pass its `--context` ID. If you already have
the raw files, place them in the same `v1/` and `v2/<component>/` layout.

## 3. Extract Cameras and LiDAR

```bash
python -m data_processing.extract \
  --raw data/raw/scene010 --output data/processed --scene 10
```

This checks the raw context and timestamp order before associating LiDAR with
images. It extracts the complete scene, including camera exposure poses,
tracked boxes, dynamic masks, colored background clouds, and actor clouds in
object coordinates. Moving actors are separated from the background and aligned
to their target-frame boxes during rendering. Tracks without LiDAR returns are
listed in `processing.json`.

The large Parquet tables are repacked into frame-sized row groups under
`data/processed/.parquet/` for efficient reads. Every nested value is compared
with the original before use; the downloaded files remain unchanged.

```text
data/processed/010/
├── images/             # Five cameras, original JPEG bytes (.png filenames)
├── intrinsics/
├── extrinsics/         # OpenCV camera axes: right, down, forward
├── ego_pose/           # Vehicle poses and camera exposure poses
├── track/
├── dynamic_mask/       # Projected moving-object box masks
├── lidar/
│   ├── background/     # Per-frame clouds in vehicle coordinates
│   └── actor/          # Per-track, per-frame clouds in object coordinates
└── processing.json
```

Scene 010 has 199 frames: 995 images, 995 dynamic masks, 199 background clouds,
and 57 actor clouds. One additional moving vehicle has no LiDAR returns.
Use a new output directory when rerunning extraction; partial directories are
not treated as complete.

## 4. Render and Export

Switch to the main environment for rendering and export. Rendering needs a GPU;
export runs on CPU:

```bash
source scripts/env.sh
source .venv/bin/activate
```

### Front Camera (Camera 0)

Render the recorded FRONT exposure poses and export the inputs for the first
[inference example](../README.md#1-front-camera-camera-0):

```bash
PYTHONHASHSEED=0 python -m data_processing.render \
  --processed data/processed --scene 10 \
  --trajectories 0 --num-frames 81 \
  --ratio 0.01 --point-scale 0.02 --stride 10 \
  --output data/rendered/scene010_cam0

python -m data_processing.export \
  --processed data/processed --scene 10 \
  --rendered data/rendered/scene010_cam0 --output data/scene010_cam0
```

Use `--camera 0 --reference-cameras 1 2 3 4` for inference. The other four
cameras supply the reference images, and camera 0 supplies the initial frame.

### Extreme Novel Views

To run the [extreme NVS example](../README.md#2-extreme-novel-view-synthesis),
render and export the two novel trajectories to a separate directory:

```bash
PYTHONHASHSEED=0 python -m data_processing.render \
  --processed data/processed --scene 10 \
  --trajectories bev lane_shift_extreme --num-frames 81 \
  --ratio 0.01 --point-scale 0.02 --stride 10 \
  --output data/rendered/scene010_ratio0.01

python -m data_processing.export \
  --processed data/processed --scene 10 \
  --rendered data/rendered/scene010_ratio0.01 --output data/scene010
```

The renderer aggregates the full scene at a stride of 10, applies validation-time
symmetry to rigid actors, and uses the native CUDA point rasterizer at 864 × 576.
Set `PYTHONHASHSEED=0` for repeatable point sampling across runs.

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--processed` | Required | Root directory containing extracted scenes, such as `data/processed/010/`. |
| `--scene` | `10` | Scene ID to render. |
| `--output` | Required | New directory for rendered conditioning videos and camera metadata. |
| `--trajectories` | `bev lane_shift_extreme` | Target paths: `0` for the recorded front camera, or `bev` and `lane_shift_extreme` for novel views. |
| `--ratio` | `0.01` | Fraction of points to retain; `0.1`, `0.01`, `0.001`, or another value in `(0, 1]`. |
| `--point-scale` | `0.02` | Point footprint scale used by the rasterizer. |
| `--stride` | `10` | Frame stride for full-scene LiDAR aggregation. |
| `--num-frames` | `81` | Output clip length; also accepts `49`. |
| `--gpu` | `0` | CUDA device index used for rendering. |
| `--bev-height` / `--bev-pitch` | `60` / `60` | Final elevation in metres and pitch in degrees. |
| `--bev-backward` | `-5` | Final backward offset in metres. |
| `--lane-shift` | `7` | Signed sinusoidal lateral amplitude in metres. |

Use a fresh render/export directory when changing parameters. The default
trajectory settings start at the raw FRONT exposure pose. Both trajectories
use five first-frame surround-view references. The exporter verifies every
video's frame count before writing the final CSV.

## Inference Input Format

```text
data/scene010_cam0/
├── metadata_cond.csv
├── camera_data.json
├── preparation.json
├── references/             # Original FRONT image + five reference videos
└── 0/                      # rgb.mp4, mask.mp4, depth.mp4, depth_normalization.json
```

For extreme novel views, the export is `data/scene010/` with `bev/` and
`lane_shift_extreme/` conditioning directories in place of `0/`.

`metadata_cond.csv` contains:

```csv
video,prompt,cond_video,cond_video_mask,cond_video_depth,cond_reference_image,scene_id,camera_id,start_frame,end_frame
```

Paths are relative to the export directory. The front-camera target row uses
`0` as `camera_id`; novel-view target rows use the trajectory name. Camera
reference rows use `0`–`4`, with the target camera excluded during inference.
`end_frame` is exclusive. The original FRONT image supplies the initial frame
for novel trajectories.

- **Conditioning mask:** black is observed geometry; white is unknown. The renderer's
  alpha is inverted during export. Dynamic object masks are a separate extraction
  product and are not substituted for the conditioning mask.
- **Depth:** sequence-normalized JET RGB visualization, with empty pixels black.
  Normalization values are saved alongside each trajectory.
- **Cameras:** `camera_data.json` maps scene → camera → intrinsics, extrinsics,
  vehicle poses, or explicit `c2w_per_frame`. Novel-view intrinsics include the
  render dimensions. Inference resizes to 832 × 480, scales translation by 0.1,
  and normalizes poses relative to the first target pose.

Continue with the main [inference examples](../README.md#inference): camera 0
uses `--data data/scene010_cam0 --camera 0`, and extreme lane shift uses
`--data data/scene010 --camera lane_shift_extreme`.

## Attribution

The extraction code builds on Street Crafter, and camera conversion builds on
EasyVolcap. See [NOTICE](../NOTICE) and the [third-party licenses](licenses/).
