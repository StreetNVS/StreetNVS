"""StreetNVS model architecture for the released checkpoint."""

import torch
from torch import nn
from einops import rearrange
from diffsynth.models.wan_video_dit import (
    SelfAttention,
    CrossAttention,
    GateModule,
    modulate,
    flash_attention,
    Head,
    MLP,
    precompute_freqs_cis_3d,
)


class StreetNVSSelfAttention(nn.Module):
    """Relative-ray camera attention with an absolute orientation embedding."""

    def __init__(self, dim, num_heads, patches_x, patches_y):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.cam_encoder = nn.Linear(3, dim)

    def forward(self, x, camera):
        batch, tokens, dim = x.shape
        assert camera["viewmats"].shape[1] == tokens
        x = x + self.cam_encoder(camera["cam_emb"])
        heads, head_dim = self.num_heads, self.head_dim
        q = self.q_proj(x).view(batch, tokens, heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, tokens, heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, tokens, heads, head_dim).transpose(1, 2)
        q = camera["apply_fn_q"](q)
        k = camera["apply_fn_kv"](k)
        v = camera["apply_fn_kv"](v)
        out = flash_attention(
            rearrange(q, "b h t d -> b t (h d)"),
            rearrange(k, "b h t d -> b t (h d)"),
            rearrange(v, "b h t d -> b t (h d)"),
            num_heads=heads,
        )
        out = rearrange(out, "b t (h d) -> b h t d", h=heads)
        out = camera["apply_fn_o"](out)
        return self.out_proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class GeoDiTBlock(nn.Module):
    """Wan block with a camera-conditioned attention residual."""

    def __init__(self, dim, num_heads, ffn_dim, patches_x, patches_y, eps=1e-6):
        super().__init__()
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=False)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.cam_self_attn = StreetNVSSelfAttention(
            dim, num_heads, patches_x, patches_y
        )

    def forward(self, x, context, t_mod, freqs, control_camera_dit_input):
        values = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=2)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            value.squeeze(2) for value in values
        )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        residual = self.self_attn(input_x, freqs)
        residual = residual + self.cam_self_attn(input_x, control_camera_dit_input)
        x = self.gate(x, gate_msa, residual)
        x = x + self.cross_attn(self.norm3(x), context)
        input_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return self.gate(x, gate_mlp, self.ffn(input_mlp))


class StreetNVSModel(nn.Module):
    def __init__(self, image_height=480, image_width=832):
        super().__init__()
        self.dim = dim = 3072
        self.freq_dim = 256
        self.patch_size = patch_size = (1, 2, 2)
        num_heads, ffn_dim, eps = 24, 14336, 1e-6
        self.patch_embedding = nn.Conv3d(
            48, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(4096, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.head = Head(dim, 48, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(dim // num_heads)
        self.geo_dit_blocks = nn.ModuleList(
            [
                GeoDiTBlock(
                    dim, num_heads, ffn_dim, image_width // 32, image_height // 32, eps
                )
                for _ in range(30)
            ]
        )
        self.patch_embedding_cond = nn.Sequential(
            nn.Conv3d(144, dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(dim, dim, kernel_size=patch_size, stride=patch_size),
        )
        self.mlp_cond = MLP(dim, dim)

    def patchify(self, x, cond_context):
        x = self.patch_embedding(x)
        cond_x = self.patch_embedding_cond(cond_context)
        batch, channels, frames, height, width = cond_x.shape
        cond_x = self.mlp_cond(rearrange(cond_x, "b c f h w -> b (f h w) c"))
        cond_x = rearrange(
            cond_x, "b (f h w) c -> b c f h w", f=frames, h=height, w=width
        )
        return x, cond_x

    def unpatchify(self, x, grid_size):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )
