"""Download the Wan text encoder, VAE, tokenizer, and StreetNVS checkpoint."""

import argparse, json
from pathlib import Path
from huggingface_hub import hf_hub_download

BASE_REPO = "Wan-AI/Wan2.2-TI2V-5B"
BASE_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
CHECKPOINT_REPO = "zfkuang/StreetNVS"
CHECKPOINT_REVISION = "d5e01da928cde3e0ba025abee050c66a0030bbeb"
BASE_FILES = [
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2_VAE.pth",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("models"))
    p.add_argument(
        "--base-only",
        action="store_true",
        help="Download the Wan text encoder, VAE and tokenizer; use a local StreetNVS checkpoint",
    )
    a = p.parse_args()
    for file in BASE_FILES:
        hf_hub_download(
            BASE_REPO,
            file,
            revision=BASE_REVISION,
            local_dir=a.output / "Wan2.2-TI2V-5B",
        )
    if not a.base_only:
        hf_hub_download(
            CHECKPOINT_REPO,
            "streetnvs.safetensors",
            revision=CHECKPOINT_REVISION,
            local_dir=a.output / "StreetNVS",
        )
    (a.output / "revisions.json").write_text(
        json.dumps(
            {
                "base": BASE_REVISION,
                "checkpoint": None if a.base_only else CHECKPOINT_REVISION,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
