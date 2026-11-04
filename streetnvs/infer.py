"""Run the StreetNVS inference model."""

import argparse
import json
import time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from peft import LoraConfig, inject_adapter_in_model
from diffsynth.core import ModelConfig
from diffsynth.utils.data import save_video
from .data import load_sample


def load_checkpoint(model, path):
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        if (checkpoint.metadata() or {}).get("streetnvs_format") != "2":
            raise ValueError(
                "Expected the combined StreetNVS checkpoint. "
                "Download it with scripts/download_models.py."
            )
    state = load_file(str(path))
    model.load_state_dict(state, strict=True)
    return {"path": str(path), "loaded": len(state)}


def build_pipeline(models, checkpoint):
    from .pipeline import StreetNVSPipeline

    models = Path(models)
    pipe = StreetNVSPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(models / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=str(models / "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=str(models / "google/umt5-xxl")),
        image_height=480,
        image_width=832,
    )
    pipe.geo_dit = inject_adapter_in_model(
        LoraConfig(r=1024, lora_alpha=1024, target_modules=["ffn.0", "ffn.2"]),
        pipe.geo_dit,
    )
    report = load_checkpoint(pipe.geo_dit, checkpoint)
    pipe.eval()
    pipe.scheduler.training = False
    return pipe, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scene", default="0")
    p.add_argument("--camera", default="2")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--num-frames", type=int, choices=(49, 81), default=49)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reference-cameras", nargs="+", default=["0", "1", "2"])
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample, row = load_sample(
        args.data,
        args.scene,
        args.camera,
        args.start,
        refs=args.reference_cameras,
        num_frames=args.num_frames,
    )
    torch.manual_seed(args.seed)
    start = time.monotonic()
    pipe, report = build_pipeline(args.models, args.checkpoint)
    with torch.inference_mode():
        frames = pipe(
            **sample,
            height=480,
            width=832,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg,
            seed=args.seed,
            tiled=False,
            rand_device="cuda",
            cond_scale=1,
        )
    save_video(frames, str(out), fps=10, quality=9)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "sample": row,
                "checkpoint_load": report,
                "seconds": time.monotonic() - start,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
