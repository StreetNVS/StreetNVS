import torch
from diffsynth.diffusion.base_pipeline import PipelineUnit
from .camera import prepare_camera_dit_input, prepend_frames_to_dit_input


def _encode_neg1_cached(pipe, num_frames, height, width, tiled, tile_size, tile_stride):
    """Encode the constant reference padding once per VAE and input shape."""
    cache = getattr(pipe, "_neg1_lat_cache", None)
    if cache is None:
        cache = pipe._neg1_lat_cache = {}
    key = (
        id(pipe.vae),
        num_frames,
        height,
        width,
        tiled,
        tuple(tile_size),
        tuple(tile_stride),
    )
    if key not in cache:
        padding = torch.full(
            (1, 3, num_frames, height, width),
            -1.0,
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        cache[key] = pipe.vae.encode(
            padding,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
    return cache[key]


class StreetNVSUnit_COND(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "cond_video",
                "cond_video_mask",
                "cond_depth_video",
                "camref_latents",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("cond_context",),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        cond_video,
        cond_video_mask,
        cond_depth_video,
        camref_latents,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        video = pipe.preprocess_video(cond_video)
        mask = pipe.preprocess_video(cond_video_mask, min_value=-1.0, max_value=1)
        pipe.load_models_to_device(self.onload_model_names)
        options = dict(
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video_latents = pipe.vae.encode(video, **options).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        mask_latents = pipe.vae.encode(mask, **options).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        context = torch.cat([video_latents, mask_latents], dim=1)
        n_ref = camref_latents.shape[2]
        padding = _encode_neg1_cached(
            pipe, 1, height, width, tiled, tile_size, tile_stride
        )
        mask_ref = padding.expand(-1, -1, n_ref, -1, -1).contiguous()
        ref_context = torch.cat([camref_latents, mask_ref], dim=1)
        context = torch.cat([ref_context, context], dim=2)
        depth = pipe.preprocess_video(cond_depth_video)
        depth_latents = pipe.vae.encode(depth, **options).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        depth_ref = padding.expand(-1, -1, n_ref, -1, -1).contiguous()
        depth_latents = torch.cat([depth_ref, depth_latents], dim=2)
        return {"cond_context": torch.cat([context, depth_latents], dim=1)}


class StreetNVSUnit_ImageEmbedderFused(PipelineUnit):
    """Replace the initial noise frame with the encoded input image."""

    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "noise",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("latents", "first_frame_latents"),
            onload_model_names=("vae",),
        )

    def process(
        self, pipe, input_image, noise, height, width, tiled, tile_size, tile_stride
    ):
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(
            0, 1
        )
        z = pipe.vae.encode(
            [image],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        noise[:, :, 0:1] = z
        return {"latents": noise, "first_frame_latents": z}


class StreetNVSUnit_ReferenceLatents(PipelineUnit):
    """Prepend clean reference images to the target latent sequence."""

    def __init__(self):
        super().__init__(
            input_params=(
                "camref_images",
                "latents",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("latents", "camref_latents"),
            onload_model_names=("vae",),
        )

    def process(
        self, pipe, camref_images, latents, height, width, tiled, tile_size, tile_stride
    ):
        pipe.load_models_to_device(self.onload_model_names)
        references = []
        for img in camref_images:
            image = pipe.preprocess_image(img.resize((width, height))).transpose(0, 1)
            z = pipe.vae.encode(
                [image],
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            references.append(z)
        camref_latents = torch.cat(references, dim=2)
        return {
            "latents": torch.cat([camref_latents, latents], dim=2),
            "camref_latents": camref_latents,
        }


class StreetNVSUnit_CameraControl(PipelineUnit):
    """Build one camera sequence for reference images and target frames."""

    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "num_frames",
                "camera_w2cs",
                "camera_Ks",
                "camref_images_w2cs",
                "camref_images_Ks",
                "camref_temporal_ids",
            ),
            output_params=("control_camera_dit_input",),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        height,
        width,
        num_frames,
        camera_w2cs,
        camera_Ks,
        camref_images_w2cs,
        camref_images_Ks,
        camref_temporal_ids,
    ):
        pipe.load_models_to_device(self.onload_model_names)
        attention = pipe.geo_dit.geo_dit_blocks[0].cam_self_attn
        options = dict(
            height=height,
            width=width,
            patches_y=attention.patches_y,
            patches_x=attention.patches_x,
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
        frames = 1 + (num_frames - 1) // 4
        target = prepare_camera_dit_input(
            camera_w2cs=camera_w2cs[:num_frames],
            camera_Ks=camera_Ks[:num_frames],
            frame_ids=torch.arange(frames, dtype=torch.long),
            is_video=True,
            **options,
        )
        reference = prepare_camera_dit_input(
            camera_w2cs=camref_images_w2cs,
            camera_Ks=camref_images_Ks,
            frame_ids=camref_temporal_ids,
            is_video=False,
            **options,
        )
        prepend_frames_to_dit_input(target, reference)
        return {"control_camera_dit_input": target}
