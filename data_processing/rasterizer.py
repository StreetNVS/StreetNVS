# Camera conversion adapted from EasyVolcap (MIT).
# Copyright (c) 2024 3D Vision Group, Zhejiang University.
# See licenses/EasyVolcap.txt.

"""Native point rasterizer with camera conversion adapted from EasyVolcap.

Third-party attribution and license terms are listed in NOTICE.
"""

from types import SimpleNamespace as dotdict
import numpy as np
import torch


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


def getWorld2View(R: torch.Tensor, t: torch.Tensor):
    """
    R: ..., 3, 3
    T: ..., 3, 1
    """
    sh = R.shape[:-2]
    T = torch.eye(4, dtype=R.dtype, device=R.device)
    for i in range(len(sh)):
        T = T.unsqueeze(0)
    T = T.expand(sh + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3:] = t
    return T


def getProjectionMatrix(
    K: torch.Tensor,
    H: torch.Tensor,
    W: torch.Tensor,
    znear: torch.Tensor,
    zfar: torch.Tensor,
):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]
    one = K[2, 2]
    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)
    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)
    P[2, 2] = -(zfar + znear) / (znear - zfar)
    P[2, 3] = 2 * zfar * znear / (znear - zfar)
    P[3, 2] = one
    return P


def convert_to_gaussian_camera(
    K: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    H: torch.Tensor,
    W: torch.Tensor,
    n: torch.Tensor,
    f: torch.Tensor,
    cpu_K: torch.Tensor,
    cpu_R: torch.Tensor,
    cpu_T: torch.Tensor,
    cpu_H: int,
    cpu_W: int,
    cpu_n: float = 0.01,
    cpu_f: float = 100.0,
):
    output = dotdict()
    output.K = K
    output.R = R
    output.T = T
    output.H = H.item()
    output.W = W.item()
    output.n = n.item()
    output.f = f.item()
    output.image_height = cpu_H.item()
    output.image_width = cpu_W.item()
    FoVx = focal2fov(cpu_K[0, 0].item(), cpu_W.item())
    FoVy = focal2fov(cpu_K[1, 1].item(), cpu_H.item())
    output.world_view_transform = getWorld2View(R, T).transpose(0, 1).float()
    output.projection_matrix = (
        getProjectionMatrix(K, H, W, n, f).transpose(0, 1).float()
    )
    output.full_proj_transform = torch.matmul(
        output.world_view_transform, output.projection_matrix
    ).float()
    output.camera_center = (-R.mT @ T)[..., 0].float()
    output.tanfovx = np.tan(FoVx * 0.5)
    output.tanfovy = np.tan(FoVy * 0.5)
    return output


def affine_inverse(mat):
    """Compute inverse of affine transformation matrix."""
    import torch

    inv = torch.eye(4, device=mat.device, dtype=mat.dtype)
    inv[:3, :3] = mat[:3, :3].T
    inv[:3, 3:] = -mat[:3, :3].T @ mat[:3, 3:]
    return inv


def render_pointcloud_diff_point_rasterization(
    c2w,
    ixt,
    points,
    features,
    H,
    W,
    gpu_id=0,
    scale=0.035,
):
    """Render point cloud using diff_point_rasterization (much faster than PyTorch3D).

    Args:
        c2w: Camera-to-world matrix
        ixt: Camera intrinsics
        points: Point cloud xyz coordinates
        features: Point features (RGB, depth, mask)
        H, W: Image height and width
        gpu_id: GPU device ID
        scale: Point scale for NDC-based adaptive scaling
    """
    import torch
    from diff_point_rasterization import PointRasterizationSettings, PointRasterizer

    device = torch.device(f"cuda:{gpu_id}")
    c2w = torch.as_tensor(c2w).to(device, non_blocking=True).float()
    w2c = affine_inverse(c2w)
    ixt = torch.as_tensor(ixt).to(device, non_blocking=True).float()
    cpu_ixt = ixt.cpu()
    xyz3 = torch.as_tensor(points).to(device, non_blocking=True).float()
    rgb3 = torch.as_tensor(features[..., :3]).to(device, non_blocking=True).float()
    occ1 = torch.full_like(xyz3[..., :1], 1.0)
    scales = torch.full_like(xyz3[..., :1], scale)
    views = xyz3 @ w2c[:3, :3].mT + w2c[:3, 3]
    x, y, z = views.chunk(3, dim=-1)
    scales = scales * z / ixt[0, 0]
    scales = scales * 0.5 * H if H <= W else scales * 0.5 * W
    gaussian_camera = convert_to_gaussian_camera(
        ixt,
        w2c[:3, :3],
        w2c[:3, 3:],
        torch.as_tensor(H).to(xyz3.device, non_blocking=True),
        torch.as_tensor(W).to(xyz3.device, non_blocking=True),
        torch.as_tensor(1.0).to(xyz3.device, non_blocking=True),
        torch.as_tensor(100.0).to(xyz3.device, non_blocking=True),
        cpu_ixt,
        None,
        None,
        torch.as_tensor(H),
        torch.as_tensor(W),
        torch.as_tensor(1.0),
        torch.as_tensor(100.0),
    )
    raster_settings = PointRasterizationSettings(
        image_height=gaussian_camera.image_height,
        image_width=gaussian_camera.image_width,
        tanfovx=gaussian_camera.tanfovx,
        tanfovy=gaussian_camera.tanfovy,
        bg=torch.full([3], 0.0, device=xyz3.device),
        scale_modifier=1.0,
        viewmatrix=gaussian_camera.world_view_transform,
        projmatrix=gaussian_camera.full_proj_transform,
        sh_degree=0,
        max_hit=10,
        campos=gaussian_camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    scr = torch.zeros_like(xyz3)
    with torch.no_grad():
        rasterizer = PointRasterizer(raster_settings=raster_settings)
        rendered_image, rendered_depth, rendered_alpha, radii = rasterizer(
            means3D=xyz3,
            means2D=scr,
            colors_precomp=rgb3,
            opacities=occ1,
            radius=scales,
        )
    rgb = rendered_image.permute(1, 2, 0)[None]
    alpha = rendered_alpha.permute(1, 2, 0)[None]
    depth = rendered_depth.permute(1, 2, 0)[None]
    return torch.cat([rgb, alpha, depth], dim=-1)
