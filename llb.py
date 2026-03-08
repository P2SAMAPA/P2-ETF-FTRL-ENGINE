# llb.py — Latent Linkage Block
# Discovers hidden inter-asset correlations directly from price data
# Uses hierarchical inner/outer token attention with ECA module
# This is the key differentiator of FTRL from standard transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg


class EfficientChannelAttention(nn.Module):
    """
    ECA module — lightweight channel attention via 1D conv.
    Addresses insufficient training data for full self-attention.
    Kernel size k is adaptive based on channel count.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Adaptive kernel size from paper formula
        k = int(abs(math.log2(channels) + 1) / 2)
        k = k if k % 2 == 1 else k + 1
        k = max(k, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k,
                              padding=k // 2, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, T, C)
        # Compute mean and max along T dim
        mean_x = x.mean(dim=1, keepdim=True)    # (B, 1, C)
        max_x  = x.max(dim=1, keepdim=True)[0]  # (B, 1, C)

        # Apply conv to each
        w_mean = self.conv(mean_x)               # (B, 1, C)
        w_max  = self.conv(max_x)                # (B, 1, C)

        # Weight tensor
        w = torch.sigmoid(w_mean + w_max)        # (B, 1, C)

        return self.norm(x * w)


class InnerAttention(nn.Module):
    """
    Intra-patch attention — captures short-term price dynamics
    within each time segment across assets.
    """
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B*n_patches, patch_len, D)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class OuterAttention(nn.Module):
    """
    Inter-patch attention — captures long-range cross-asset relationships.
    This is where latent linkages are discovered.
    ECA replaces full channel attention for data efficiency.
    """
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )
        self.eca   = EfficientChannelAttention(dim)

    def forward(self, x):
        # x: (B, W+1, D)  — W assets + 1 class token
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        x = self.eca(x)
        return x


class AttentionBlock(nn.Module):
    """
    Combined inner + outer attention block.
    Inner: intra-segment (short-term dynamics)
    Outer: inter-segment with ECA (long-range cross-asset linkages)
    """
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.inner = InnerAttention(dim, n_heads, dropout)
        self.outer = OuterAttention(dim, n_heads, dropout)
        self.proj  = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, inner_tokens, outer_tokens):
        """
        inner_tokens: (B*W, n_segments, seg_len, D) — reshaped for inner attn
        outer_tokens: (B, W+1, D)

        Returns updated inner_tokens, outer_tokens
        """
        B_W, n_seg, seg_len, D = inner_tokens.shape
        # Flatten segments for inner attention
        flat = inner_tokens.reshape(B_W * n_seg, seg_len, D)
        flat = self.inner(flat)
        inner_tokens = flat.reshape(B_W, n_seg, seg_len, D)

        # Aggregate inner info → inject into outer tokens (skip class token)
        # Mean over segments and seg_len for each asset
        B = outer_tokens.size(0)
        inner_agg = inner_tokens.reshape(
            B, cfg.W, n_seg * seg_len, D
        ).mean(dim=2)                                   # (B, W, D)
        inner_agg = self.proj(inner_agg)

        # Add to outer tokens (asset positions only, not class token)
        outer_tokens[:, 1:, :] = outer_tokens[:, 1:, :] + inner_agg

        # Outer attention across assets + class token
        outer_tokens = self.outer(outer_tokens)

        return inner_tokens, outer_tokens


class LLB(nn.Module):
    """
    Latent Linkage Block.

    Segments the price time series using an Unfold-like sliding window,
    then applies hierarchical inner/outer attention to discover
    latent cross-asset correlations.

    Input:  (B, C, H, W)
    Output: (B, H) — linkage feature vector (H = D_MODEL here)
    """

    def __init__(self):
        super().__init__()
        D  = cfg.D_MODEL
        W  = cfg.W           # 6 assets
        C  = cfg.C           # 4 features
        H  = cfg.H           # 40 lookback
        nh = cfg.N_HEADS
        dr = cfg.DROPOUT

        # Segment parameters
        self.seg_len   = 8     # days per segment
        self.n_segs    = H // self.seg_len   # 40 // 8 = 5 segments

        # Project each segment into D dimensions
        self.seg_proj = nn.Sequential(
            nn.Linear(C * self.seg_len, D),
            nn.LayerNorm(D),
        )

        # Class token and positional embeddings for outer attention
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, W + 1, D)
        )

        # Stack of attention blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(D, nh, dr)
            for _ in range(cfg.ATTENTION_BLOCKS)
        ])

        self.norm = nn.LayerNorm(D)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, D_MODEL) — latent linkage feature
        """
        B, C, H, W = x.shape

        # Reshape to (B, W, H, C) for per-asset processing
        x_t = x.permute(0, 3, 2, 1)     # (B, W, H, C)

        # Segment: split H into n_segs of seg_len
        # (B, W, n_segs, seg_len, C)
        x_seg = x_t.reshape(
            B, W, self.n_segs, self.seg_len, C
        )

        # Project each segment to D: (B, W, n_segs, seg_len*C) → (B, W, n_segs, D)
        x_seg_flat = x_seg.reshape(B, W, self.n_segs, self.seg_len * C)
        inner_tokens = self.seg_proj(x_seg_flat)         # (B, W, n_segs, D)

        # Reshape for inner attention: (B*W, n_segs, 1, D)
        # seg_len dimension collapsed to 1 after projection
        inner_tokens = inner_tokens.reshape(
            B * W, self.n_segs, 1, cfg.D_MODEL
        )

        # Build outer tokens: class token + one token per asset
        cls = self.cls_token.expand(B, 1, cfg.D_MODEL)
        # Asset tokens: mean of their segments
        asset_tokens = inner_tokens.reshape(
            B, W, self.n_segs, cfg.D_MODEL
        ).mean(dim=2)                                    # (B, W, D)

        outer_tokens = torch.cat([cls, asset_tokens], dim=1)  # (B, W+1, D)
        outer_tokens = outer_tokens + self.pos_embed

        # Run attention blocks
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)

        # Extract and normalise class token
        cls_out = self.norm(outer_tokens[:, 0, :])       # (B, D)

        return cls_out
