# StreetNVS: Effective Multi-sensor Conditioning for Street-view Novel-view Synthesis

<p align="center">
  <a href="https://zhengfeikuang.com/">Zhengfei Kuang</a><sup>1</sup> ·
  <a href="https://adamsunn.github.io/">Adam Sun</a><sup>1</sup> ·
  <a href="https://www.zhuliyuan.net/">Liyuan Zhu</a><sup>1</sup> ·
  <a href="https://wutong16.github.io/">Tong Wu</a><sup>1</sup> ·
  <a href="https://primecai.github.io/">Shengqu Cai</a><sup>1</sup> ·
  <a href="https://research.nvidia.com/person/jonathan-tremblay">Jonathan Tremblay</a><sup>2</sup> ·
  <a href="https://ir0.github.io/">Iro Armeni</a><sup>1</sup> ·
  <a href="https://stanford.edu/~eadeli/">Ehsan Adeli</a><sup>1</sup> ·
  <a href="https://lioryariv.github.io/">Lior Yariv</a><sup>1</sup> ·
  <a href="https://web.stanford.edu/~gordonwz/">Gordon Wetzstein</a><sup>1</sup>
</p>
<p align="center"><sup>1</sup> Stanford University &nbsp;&nbsp; <sup>2</sup> NVIDIA</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.01590"><img src="https://img.shields.io/badge/arXiv-2606.01590-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv paper"></a>
  <a href="https://arxiv.org/pdf/2606.01590"><img src="https://img.shields.io/badge/Paper-PDF-4051b5?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Read the paper PDF"></a>
  <a href="https://streetnvs.github.io/website/"><img src="https://img.shields.io/badge/Project-Page-00897b?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project page"></a>
  <a href="https://huggingface.co/zfkuang/StreetNVS"><img src="https://img.shields.io/badge/Hugging_Face-Models-ffbd45?style=for-the-badge&logo=huggingface&logoColor=black" alt="Pretrained models"></a>
</p>

This repository is the official implementation of **StreetNVS: [Effective Multi-sensor Conditioning for Street-view Novel-view Synthesis](https://arxiv.org/abs/2606.01590)**.

Modern vehicle platforms are equipped with a rich sensor suite, including LiDAR, calibrated multi-camera rigs, and accurate ego-motion, that in principle offers strong signal for re-rendering a driving scene from novel viewpoints. A growing line of recent work leverages video diffusion models for this task, using their generative priors to synthesize plausible novel views from sparse vehicle observations. In practice, however, existing methods exploit only a fragment of this signal, and their quality tends to degrade as the target trajectory departs from the recorded driving path. We argue that this is fundamentally a multi-sensor fusion problem: sparse LiDAR reprojections supply accurate but incomplete metric geometry, surround-view reference imagery supplies dense appearance but no metric depth, and camera poses tie the two together across views. We introduce StreetNVS, a video diffusion framework that jointly conditions on all three signals through a Reference-Enhanced Camera Attention module based on a relative ray-level positional encoding. We develop a two-stage curriculum training strategy that gradually exposes the model to increasingly sparse LiDAR. On the Waymo Open Dataset, StreetNVS substantially outperforms state-of-the-art baselines under sparse LiDAR conditioning, matches methods that rely on 10–100× denser point clouds. We further show capabilities of synthesizing coherent videos along extreme out-of-trajectory paths such as elevation, lane-shift, pullback, and rotation.

<p align="center">
  <a href="https://streetnvs.github.io/website/"><img src="https://arxiv.org/html/2606.01590v2/teaser.png" alt="StreetNVS overview: multi-sensor conditioning and novel-view synthesis along elevation and lane-shift trajectories" width="100%"></a>
</p>

## File Structure

```text
StreetNVS/
├── streetnvs/          # Model and inference code, built on DiffSynth-Studio
├── data_processing/   # Waymo preparation scripts and their README
├── scripts/           # Environment setup and checkpoint download
├── assets/            # Result videos and animated previews
├── models/            # Downloaded weights
├── data/              # Your raw data and prepared inference inputs
└── outputs/           # Generated videos
```

`models/`, `data/`, and `outputs/` are created as you follow the steps below.

## Environment Setup

We recommend **Python 3.11, PyTorch 2.9.1, CUDA 12.8, and an Nvidia GPU with BF16 support and at least 64 GB of GPU memory** for the 81-frame example below. We use **Nvidia H100 80 GB GPUs**. To set up the environment on Linux, run:

```bash
git clone https://github.com/StreetNVS/StreetNVS.git
cd StreetNVS
bash scripts/setup.sh
source scripts/env.sh
source .venv/bin/activate
```

The setup script installs DiffSynth-Studio and our dependencies in a uv/venv environment. It keeps caches and temporary files in `.runtime/` inside the repository. Run the two `source` commands again when you open a new terminal.

## Download Checkpoints

Run the following to download our [StreetNVS checkpoint](https://huggingface.co/zfkuang/StreetNVS), along with the text encoder, VAE, and tokenizer from Wan2.2:

```bash
python scripts/download_models.py --output models
```

The weights are public and can be downloaded without logging in. They will be saved here:

```text
models/
├── Wan2.2-TI2V-5B/       # Text encoder, VAE, and tokenizer
└── StreetNVS/
    └── streetnvs.safetensors
```

`streetnvs.safetensors` contains the complete StreetNVS diffusion model and loads in one step.

## Data Preparation

Please follow [data_processing/README.md](data_processing/README.md) to prepare a Waymo scene. The guide walks through downloading the raw data, extracting cameras and LiDAR, and rendering the inputs for inference.

The example uses scene 010 with two camera paths: bird's-eye view (`bev`) and extreme lane shift (`lane_shift_extreme`). It saves the inputs to `data/scene010/`. You can change the point sampling ratio to `0.1`, `0.01`, or `0.001` when rendering.

You will need your own [Waymo Open Dataset](https://waymo.com/open/) access. We do not include Waymo data in this repository.

## Inference

After preparing the data, run:

```bash
python -m streetnvs.infer \
  --models models/Wan2.2-TI2V-5B \
  --checkpoint models/StreetNVS/streetnvs.safetensors \
  --data data/scene010 --scene 10 --camera bev \
  --reference-cameras 0 1 2 3 4 --num-frames 81 \
  --output outputs/scene010_bev.mp4
```

For extreme lane shift, change `--camera` to `lane_shift_extreme` and choose a different output filename. The defaults are 20 denoising steps, guidance scale 5, and seed 42. The video is saved at **832 × 480 and 10 FPS**, along with a JSON file containing the settings.

You can also use `--num-frames 49` for a 49-frame input. Camera IDs `0`–`4` refer to front, front-left, front-right, side-left, and side-right. To render a recorded camera view, use its numeric ID for `--camera`; it will be left out of the reference images automatically.

For the recorded front camera, prepare the inputs with `--trajectories 0` as described in the [data preparation guide](data_processing/README.md). Then use `--data data/scene010_cam0 --camera 0 --reference-cameras 1 2 3 4` in the command above.

Run `python -m streetnvs.infer --help` for all options.

### Results

The videos below are from the paper. Results may vary with the input point sampling and random seed. See the [project page](https://streetnvs.github.io/website/) for more videos and comparisons.

| Bird's-eye view | Extreme lane shift |
| :---: | :---: |
| [![Bird's-eye view](assets/scene010_bev.gif)](assets/scene010_bev.mp4) | [![Extreme lane shift](assets/scene010_lane_shift_extreme.gif)](assets/scene010_lane_shift_extreme.mp4) |
| [Full video](assets/scene010_bev.mp4) · [LiDAR conditioning](https://streetnvs.github.io/website/assets/novel_view/scene010/bev/cond.mp4) | [Full video](assets/scene010_lane_shift_extreme.mp4) · [LiDAR conditioning](https://streetnvs.github.io/website/assets/novel_view/scene010/lane_shift_extreme/cond.mp4) |

## Citation

```bibtex
@article{kuang2026streetnvs,
  title={Effective Multi-sensor Conditioning for Street-view Novel-view Synthesis},
  author={Kuang, Zhengfei and Sun, Adam and Zhu, Liyuan and Wu, Tong and Cai, Shengqu and Tremblay, Jonathan and Armeni, Iro and Adeli, Ehsan and Yariv, Lior and Wetzstein, Gordon},
  journal={arXiv preprint arXiv:2606.01590},
  year={2026}
}
```

## Acknowledgements

We build on [Wan2.2](https://github.com/Wan-Video/Wan2.2), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), and PRoPE. Thank you to the authors for sharing their code and models. We also thank Astra for helping prepare this repository.

StreetNVS inference extensions are released under the [Apache License 2.0](LICENSE). Third-party preprocessing and rasterizer components retain their own license terms, including research/non-commercial restrictions; see [NOTICE](NOTICE). Model and dataset licenses remain those of their respective providers.
