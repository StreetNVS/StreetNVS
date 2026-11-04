import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from .prope import camera_attention_transforms


def pinhole_unproject_grid(
    K: torch.Tensor,
    height: int,
    width: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute ray directions for pinhole cameras at a given resolution.

    Args:
        K: (3, 3) or (B, 3, 3) pixel-coordinate intrinsics.
        height, width: target resolution for the ray grid.

    Returns:
        d_cam: [H, W, 3] or [B, H, W, 3] unit ray directions in camera frame.
    """
    batched = K.dim() == 3
    if not batched:
        K = K.unsqueeze(0)
    K = K.to(device=device, dtype=dtype)
    j = torch.arange(height, device=device, dtype=dtype) + 0.5
    i = torch.arange(width, device=device, dtype=dtype) + 0.5
    jv, iv = torch.meshgrid(j, i, indexing="ij")
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    x = (iv.unsqueeze(0) - cx[:, None, None]) / fx[:, None, None]
    y = (jv.unsqueeze(0) - cy[:, None, None]) / fy[:, None, None]
    z = torch.ones_like(x)
    d_cam = torch.stack([x, y, z], dim=-1)
    d_cam = F.normalize(d_cam, dim=-1, eps=1e-06)
    if not batched:
        d_cam = d_cam[0]
    return d_cam


def world_to_ray_mats(d_cam: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    if d_cam.dim() == 3:
        d_cam = d_cam.unsqueeze(0)
    B, H, W, _ = d_cam.shape
    T = c2w.shape[1]
    device = d_cam.device
    dtype = d_cam.dtype
    d_cam = repeat(d_cam, "b h w c -> b t h w c", t=T)
    R_cam = c2w[..., :3, :3]
    t_cam = c2w[..., :3, 3]
    d_world = torch.einsum("btij,bthwj->bthwi", R_cam, d_cam)
    cam_y = R_cam[..., :, 1]
    cam_y = repeat(cam_y, "b t c -> b t h w c", h=H, w=W)
    z_ray = F.normalize(d_world, dim=-1, eps=1e-06)
    x_ray = torch.cross(cam_y, z_ray, dim=-1)
    x_ray = F.normalize(x_ray, dim=-1, eps=1e-06)
    y_ray = torch.cross(z_ray, x_ray, dim=-1)
    y_ray = F.normalize(y_ray, dim=-1, eps=1e-06)
    R_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)
    R_w2l = rearrange(R_l2w, "b t h w i j -> b t h w j i")
    t_world = repeat(t_cam, "b t c -> b t h w c", h=H, w=W)
    t_w2l = -torch.einsum("bthwij,bthwj->bthwi", R_w2l, t_world)
    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w2l
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0
    mask = torch.isnan(d_world).any(-1)
    raymats[mask] = torch.eye(4, device=device, dtype=dtype)
    return raymats


def compute_up_lat_map(
    R_c2w: torch.Tensor,
    K_pixel: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
    delta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute up_map and lat_map for *absmap* camera encoding.

    For each pixel the function computes:
        * **lat_map** — the latitude (elevation above the XZ ground-plane) of
          the world-space ray direction.
        * **up_map** — a 2-D unit vector in image space indicating the local
          "up" direction, obtained by applying a small rotation toward world-up
          and projecting the displacement.

    Args:
        R_c2w:   [B, T, 3, 3] c2w rotation matrices.
        K_pixel: [3, 3] or [B, 3, 3] intrinsics in pixel coords at
                 (height, width) resolution.
        height, width: spatial grid resolution (patch grid).
        device:  torch device.
        delta:   small rotation angle (radians) for up_map computation.

    Returns:
        up_map:  [B, T, H, W, 2] normalised displacement vectors.
        lat_map: [B, T, H, W, 1] latitude values (radians).
    """
    B, T = R_c2w.shape[:2]
    out_dtype = R_c2w.dtype
    R = R_c2w.float()
    if K_pixel.dim() == 2:
        K_pixel = K_pixel.unsqueeze(0)
    K_pixel = K_pixel.float().to(device)
    d_cam = pinhole_unproject_grid(K_pixel, height, width, device, torch.float32)
    if d_cam.dim() == 3:
        d_cam = d_cam.unsqueeze(0)
    d_cam_exp = repeat(d_cam, "B H W C -> B T H W C", T=T)
    d_world = torch.einsum("btij,bthwj->bthwi", R, d_cam_exp)
    d_world = d_world / torch.clamp_min(d_world.norm(dim=-1, keepdim=True), 1e-08)
    Xw = d_world[..., 0]
    Yw = d_world[..., 1]
    Zw = d_world[..., 2]
    lat_map = torch.atan2(-Yw, torch.sqrt(Xw**2 + Zw**2)).unsqueeze(-1)
    v = d_world
    up_world = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)
    up_exp = up_world.view(1, 1, 1, 1, 3).expand_as(v)
    k = torch.cross(v, up_exp, dim=-1)
    k = k / torch.clamp_min(k.norm(dim=-1, keepdim=True), 1e-08)
    cos_d = torch.cos(torch.tensor(delta, device=device, dtype=torch.float32))
    sin_d = torch.sin(torch.tensor(delta, device=device, dtype=torch.float32))
    kdotv = (k * v).sum(dim=-1, keepdim=True)
    v_rot = v * cos_d + torch.cross(k, v, dim=-1) * sin_d + k * kdotv * (1 - cos_d)
    R_inv = R.transpose(-1, -2)
    dirs_cam = torch.einsum("btij,bthwj->bthwi", R_inv, v_rot)
    Xs = dirs_cam[..., 0]
    Ys = dirs_cam[..., 1]
    Zs = dirs_cam[..., 2]
    Zs_safe = torch.clamp_min(Zs.abs(), 1e-08) * Zs.sign()
    fx = K_pixel[:, 0, 0].view(-1, 1, 1, 1)
    fy = K_pixel[:, 1, 1].view(-1, 1, 1, 1)
    cx = K_pixel[:, 0, 2].view(-1, 1, 1, 1)
    cy = K_pixel[:, 1, 2].view(-1, 1, 1, 1)
    du = fx * (Xs / Zs_safe) + cx
    dv = fy * (Ys / Zs_safe) + cy
    j = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    i = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    jv, iv = torch.meshgrid(j, i, indexing="ij")
    grid_x = iv.view(1, 1, height, width)
    grid_y = jv.view(1, 1, height, width)
    up_map = torch.stack((du - grid_x, dv - grid_y), dim=-1)
    up_map = up_map / torch.clamp_min(up_map.norm(dim=-1, keepdim=True), 1e-08)
    return (up_map.to(out_dtype), lat_map.to(out_dtype))


def prepare_camera_dit_input(
    camera_w2cs,
    camera_Ks,
    height,
    width,
    patches_y,
    patches_x,
    device,
    dtype,
    frame_ids,
    is_video=True,
):
    if is_video:
        indices = [0] + [1 + i * 4 for i in range((len(camera_w2cs) - 1) // 4)]
        w2cs_sub, Ks_sub = camera_w2cs[indices], camera_Ks[indices]
    else:
        w2cs_sub, Ks_sub = camera_w2cs, camera_Ks
    w2cs_lat = w2cs_sub.to(device=device, dtype=torch.float32)
    c2ws_batch = torch.linalg.inv(w2cs_lat).to(dtype).unsqueeze(0)
    K_pixel = _K_norm_to_pixel(Ks_sub[:1], height, width)
    K_patch = _K_to_patch_resolution(K_pixel[0], height, width, patches_y, patches_x)
    d_cam = pinhole_unproject_grid(
        K_patch.unsqueeze(0), patches_y, patches_x, device, dtype
    )
    raymats = world_to_ray_mats(d_cam, c2ws_batch)
    up_map, lat_map = compute_up_lat_map(
        R_c2w=c2ws_batch[..., :3, :3],
        K_pixel=K_patch,
        height=patches_y,
        width=patches_x,
        device=device,
    )
    return {
        "viewmats": rearrange(raymats, "B T H W ... -> B (T H W) ..."),
        "cam_emb": rearrange(
            torch.cat([up_map, lat_map], dim=-1), "B T H W C -> B (T H W) C"
        ).to(dtype=dtype),
        "frame_ids": frame_ids.to(device=device),
    }


def precompute_prope_apply_fns(model, camera):
    attention = model.geo_dit_blocks[0].cam_self_attn
    q, kv, out = camera_attention_transforms(
        camera["viewmats"],
        camera["frame_ids"],
        attention.head_dim,
        attention.patches_x,
        attention.patches_y,
    )
    camera.update(apply_fn_q=q, apply_fn_kv=kv, apply_fn_o=out)


def prepend_frames_to_dit_input(camera, reference):
    for key in ("viewmats", "cam_emb"):
        camera[key] = torch.cat([reference[key], camera[key]], dim=1)
    camera["frame_ids"] = torch.cat(
        [reference["frame_ids"], camera["frame_ids"]], dim=0
    )


def _K_norm_to_pixel(K_44: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert normalised [N, 4, 4] intrinsics to pixel-coord [N, 3, 3]."""
    N = K_44.shape[0]
    K33 = torch.zeros(N, 3, 3, device=K_44.device, dtype=K_44.dtype)
    K33[:, 0, 0] = K_44[:, 0, 0] * W
    K33[:, 1, 1] = K_44[:, 1, 1] * H
    K33[:, 0, 2] = K_44[:, 0, 2] * W
    K33[:, 1, 2] = K_44[:, 1, 2] * H
    K33[:, 2, 2] = 1.0
    return K33


def _K_to_patch_resolution(
    K: torch.Tensor, img_h: int, img_w: int, patch_h: int, patch_w: int
) -> torch.Tensor:
    """Scale intrinsics from image resolution to patch grid resolution."""
    K_out = K.clone()
    K_out[0, 0] *= patch_w / img_w
    K_out[1, 1] *= patch_h / img_h
    K_out[0, 2] *= patch_w / img_w
    K_out[1, 2] *= patch_h / img_h
    return K_out
