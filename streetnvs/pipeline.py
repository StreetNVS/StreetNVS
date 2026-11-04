import torch
from PIL import Image
from tqdm import tqdm
from diffsynth.core import ModelConfig
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.diffusion.base_pipeline import BasePipeline
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
from diffsynth.pipelines.wan_video import (
    WanVideoUnit_ShapeChecker,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_PromptEmbedder,
)
from einops import rearrange
from .camera import precompute_prope_apply_fns
from .model import StreetNVSModel
from .pipeline_units import (
    StreetNVSUnit_COND,
    StreetNVSUnit_ImageEmbedderFused,
    StreetNVSUnit_ReferenceLatents,
    StreetNVSUnit_CameraControl,
)


class StreetNVSPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=32,
            width_division_factor=32,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer = None
        self.text_encoder = None
        self.geo_dit = None
        self.vae = None
        self.in_iteration_models = ("geo_dit",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            StreetNVSUnit_ImageEmbedderFused(),
            StreetNVSUnit_ReferenceLatents(),
            StreetNVSUnit_COND(),
            StreetNVSUnit_CameraControl(),
        ]

    @staticmethod
    def from_pretrained(
        model_configs: list[ModelConfig],
        tokenizer_config: ModelConfig,
        torch_dtype=torch.bfloat16,
        device="cuda",
        image_height=480,
        image_width=832,
        vram_limit=None,
    ):
        pipe = StreetNVSPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        if pipe.vae is None or pipe.vae.z_dim != 48 or pipe.vae.upsampling_factor != 16:
            raise ValueError("StreetNVS requires the Wan2.2-TI2V-5B VAE")
        pipe.image_height, pipe.image_width = image_height, image_width
        pipe.geo_dit = StreetNVSModel(image_height, image_width).to(
            dtype=torch_dtype, device=device
        )
        tokenizer_config.download_if_necessary()
        pipe.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path, seq_len=512, clean="whitespace"
        )
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        input_image: Image.Image,
        camera_w2cs,
        camera_Ks,
        camref_images,
        camref_images_w2cs,
        camref_images_Ks,
        cond_video,
        cond_video_mask,
        cond_depth_video,
        camref_temporal_ids=None,
        negative_prompt="",
        denoising_strength=1.0,
        cond_scale=1.0,
        seed=None,
        rand_device="cuda",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=20,
        sigma_shift=5.0,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
    ):
        if (height, width) != (
            self.image_height,
            self.image_width,
        ) or num_frames not in (49, 81):
            raise ValueError(
                "Use the model's configured image size and 49 or 81 frames"
            )
        if input_image is None or any(
            v is None or len(v) != num_frames
            for v in (cond_video, cond_video_mask, cond_depth_video)
        ):
            raise ValueError(
                "An initial image and full-length RGB, mask and depth conditioning are required"
            )
        if not camref_images or camref_images_w2cs is None or camref_images_Ks is None:
            raise ValueError(
                "At least one reference image with camera calibration is required"
            )
        n_ref = len(camref_images)
        if len(camref_images_w2cs) != n_ref or len(camref_images_Ks) != n_ref:
            raise ValueError(
                "Reference images and camera calibration must have equal lengths"
            )
        if (
            camera_w2cs is None
            or camera_Ks is None
            or len(camera_w2cs) < num_frames
            or len(camera_Ks) < num_frames
        ):
            raise ValueError(
                "Camera poses and intrinsics are required for every target frame"
            )
        if camref_temporal_ids is None:
            camref_temporal_ids = torch.zeros(n_ref, dtype=torch.long)
        if len(camref_temporal_ids) != n_ref:
            raise ValueError(
                "Reference temporal IDs must match the number of reference images"
            )
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )
        positive = {"prompt": prompt}
        negative = {"negative_prompt": negative_prompt}
        shared = dict(
            camera_w2cs=camera_w2cs,
            camera_Ks=camera_Ks,
            camref_images=camref_images,
            camref_images_w2cs=camref_images_w2cs,
            camref_images_Ks=camref_images_Ks,
            camref_temporal_ids=camref_temporal_ids,
            input_image=input_image,
            cond_video=cond_video,
            cond_video_mask=cond_video_mask,
            cond_depth_video=cond_depth_video,
            cond_scale=cond_scale,
            seed=seed,
            rand_device=rand_device,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        for unit in self.units:
            shared, positive, negative = self.unit_runner(
                unit, self, shared, positive, negative
            )
        self.load_models_to_device(self.in_iteration_models)
        for index, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )
            noise_positive = model_fn_wan_video(
                self.geo_dit, **shared, **positive, timestep=timestep
            )
            if cfg_scale != 1.0:
                noise_negative = model_fn_wan_video(
                    self.geo_dit, **shared, **negative, timestep=timestep
                )
                noise = noise_negative + cfg_scale * (noise_positive - noise_negative)
            else:
                noise = noise_positive
            shared["latents"] = self.scheduler.step(
                noise, self.scheduler.timesteps[index], shared["latents"]
            )
            shared["latents"][:, :, :n_ref] = shared["camref_latents"]
            shared["latents"][:, :, n_ref : n_ref + 1] = shared["first_frame_latents"]
        latents = shared["latents"][:, :, n_ref:]
        self.load_models_to_device(["vae"])
        video = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


def model_fn_wan_video(
    geo_dit,
    latents,
    timestep,
    context,
    cond_context,
    control_camera_dit_input,
    camref_temporal_ids,
    cond_scale=1.0,
    **kwargs,
):
    n_clean_prefix = len(camref_temporal_ids) + 1
    hw = latents.shape[3] * latents.shape[4] // 4
    total = latents.shape[2]
    timestep_tok = torch.concat(
        [
            torch.zeros(
                (n_clean_prefix, hw), dtype=latents.dtype, device=latents.device
            ),
            torch.ones(
                (total - n_clean_prefix, hw), dtype=latents.dtype, device=latents.device
            )
            * timestep,
        ]
    ).flatten()
    t = geo_dit.time_embedding(
        sinusoidal_embedding_1d(geo_dit.freq_dim, timestep_tok).unsqueeze(0)
    )
    t_mod = geo_dit.time_projection(t).unflatten(2, (6, geo_dit.dim))
    context = geo_dit.text_embedding(context)
    x, cond_x = geo_dit.patchify(latents, cond_context)
    x = x + cond_x * cond_scale
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    N_ref_frames = len(camref_temporal_ids)
    cam_tids = camref_temporal_ids.to(geo_dit.freqs[0].device)
    cam_tids = cam_tids + 64
    ref_freqs = (
        torch.cat(
            [
                geo_dit.freqs[0][cam_tids]
                .view(N_ref_frames, 1, 1, -1)
                .expand(N_ref_frames, h, w, -1),
                geo_dit.freqs[1][:h].view(1, h, 1, -1).expand(N_ref_frames, h, w, -1),
                geo_dit.freqs[2][:w].view(1, 1, w, -1).expand(N_ref_frames, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(N_ref_frames * h * w, 1, -1)
        .to(x.device)
    )
    T_target = f - N_ref_frames
    target_freqs = (
        torch.cat(
            [
                geo_dit.freqs[0][:T_target]
                .view(T_target, 1, 1, -1)
                .expand(T_target, h, w, -1),
                geo_dit.freqs[1][:h].view(1, h, 1, -1).expand(T_target, h, w, -1),
                geo_dit.freqs[2][:w].view(1, 1, w, -1).expand(T_target, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(T_target * h * w, 1, -1)
        .to(x.device)
    )
    freqs = torch.cat([ref_freqs, target_freqs], dim=0)
    precompute_prope_apply_fns(geo_dit, control_camera_dit_input)
    for block in geo_dit.geo_dit_blocks:
        x = block(x, context, t_mod, freqs, control_camera_dit_input)
    x = geo_dit.head(x, t)
    return geo_dit.unpatchify(x, (f, h, w))
