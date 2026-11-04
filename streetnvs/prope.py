# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Adapted from UCPE's thirdparty/prope/torch.py for integration into diffsynth.

from functools import partial
from typing import Callable, List, Tuple

import torch


def camera_attention_transforms(viewmats, frame_ids, head_dim, patches_x, patches_y):
    """Build the relative-ray projection and temporal RoPE transforms."""
    assert head_dim % 4 == 0
    proj_dim = head_dim * 3 // 4
    time_dim = head_dim - proj_dim
    positions = frame_ids.repeat_interleave(patches_x * patches_y).to(viewmats.device)
    assert positions.numel() == viewmats.shape[1]
    coeffs = _rope_precompute_coeffs(
        positions.float(),
        freq_base=100.0,
        freq_scale=2.0,
        feat_dim=time_dim,
        dtype=viewmats.dtype,
    )
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=viewmats.transpose(-1, -2)), proj_dim),
        (partial(_rope_apply_coeffs, coeffs=coeffs), time_dim),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=_invert_SE3(viewmats)), proj_dim),
        (partial(_rope_apply_coeffs, coeffs=coeffs), time_dim),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=viewmats), proj_dim),
        (partial(_rope_apply_coeffs, coeffs=coeffs, inverse=True), time_dim),
    ]
    return (
        partial(_apply_block_diagonal, func_size_pairs=transforms_q),
        partial(_apply_block_diagonal, func_size_pairs=transforms_kv),
        partial(_apply_block_diagonal, func_size_pairs=transforms_o),
    )


def _apply_tiled_projmat(feats, matrix):
    batch, num_heads, seqlen, feat_dim = feats.shape
    assert matrix.shape == (batch, seqlen, 4, 4)
    assert feat_dim % 4 == 0
    feats_ = feats.view(batch, num_heads, seqlen, feat_dim // 4, 4)
    out = torch.einsum("btij,bntpj->bntpi", matrix, feats_)
    return out.reshape(feats.shape).to(feats.dtype)


def _rope_precompute_coeffs(
    positions: torch.Tensor,
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[None, None, :, None] * freqs
    assert angles.shape == (1, 1, positions.shape[0], num_freqs)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _rope_apply_coeffs(
    feats: torch.Tensor,
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    cos, sin = coeffs[0].to(feats.dtype), coeffs[1].to(feats.dtype)
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out
