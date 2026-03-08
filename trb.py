# trb.py — Temporal Relationship Block
# Extracts temporal features from price fluctuations at daily and weekly granularity
# Adapted from FTRL paper for W=6 assets, H=40 lookback, CPU training

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg


class PatchEmbedding(nn.Module):
    """
    Converts price matrix into patch tokens.
    Each patch = one trading day (or 5 days for weekly).
    """
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=(patch_size, 1),
            stride=(patch_size, 1)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        out = self.proj(x)          # (B, embed_dim, H/patch, W)
        B, D, T, W = out.shape
        out = out.permute(0, 3, 2, 1)  # (B, W, T, D)
        out = out.reshape(B * W, T, D) # (BW, T, D)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


class MLP(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TemporalBlock(nn.Module):
    """Single transformer block for temporal feature extraction."""
    def __init__(self, embed_dim: int, n_heads: int,
                 ff_dim: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.mlp  = MLP(embed_dim, ff_dim, dropout)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


class TRB(nn.Module):
    """
    Temporal Relationship Block.

    Two parallel patch streams:
      Stream 1: daily patches  (patch_size=1)  — day-level features
      Stream 2: weekly patches (patch_size=5)  — week-level distribution

    Both streams use class-token transformer attention.
    Outputs are averaged and calibrated via a learned weight tensor.

    Input:  (B, C, H, W) price matrix
    Output: (B, W) class vector — temporal feature per asset
    """

    def __init__(self):
        super().__init__()
        D  = cfg.D_MODEL
        H  = cfg.H           # 40
        W  = cfg.W           # 6
        C  = cfg.C           # 4
        nh = cfg.N_HEADS
        df = cfg.D_FF
        dr = cfg.DROPOUT

        # ── Stream 1: daily patches (patch_size=1) ──────────────────────────
        self.patch_daily   = PatchEmbedding(1, C, D)    # tokens per asset = H
        self.cls_daily     = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_daily     = nn.Parameter(torch.zeros(1, H + 1, D))
        self.blocks_daily  = nn.ModuleList([
            TemporalBlock(D, nh, df, dr)
            for _ in range(cfg.ATTENTION_BLOCKS)
        ])
        self.norm_daily    = nn.LayerNorm(D)

        # ── Stream 2: weekly patches (patch_size=5) ──────────────────────────
        # H=40, patch=5 → 8 tokens per asset
        n_weekly = H // 5
        self.patch_weekly  = PatchEmbedding(5, C, D)
        self.cls_weekly    = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_weekly    = nn.Parameter(torch.zeros(1, n_weekly + 1, D))
        self.blocks_weekly = nn.ModuleList([
            TemporalBlock(D, nh, df, dr)
            for _ in range(cfg.ATTENTION_BLOCKS)
        ])
        self.norm_weekly   = nn.LayerNorm(D)

        # ── Calibration weight extraction ────────────────────────────────────
        # CNN over raw input to produce a scalar weight per asset
        self.calib = nn.Sequential(
            nn.Conv2d(C, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, W)),
            nn.Flatten(),
            nn.Linear(8 * 4 * W, W),
            nn.Tanh(),
        )

        # ── Final projection ─────────────────────────────────────────────────
        # Merge two D-dim class tokens → W-dim output
        self.merge = nn.Linear(D * 2, W)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_daily,  std=0.02)
        nn.init.trunc_normal_(self.cls_weekly, std=0.02)
        nn.init.trunc_normal_(self.pos_daily,  std=0.02)
        nn.init.trunc_normal_(self.pos_weekly, std=0.02)

    def _run_stream(self, patches, cls_token, pos_embed, blocks, norm):
        """
        Run one transformer stream.
        patches: (BW, T, D)
        Returns class token: (B, W, D)
        """
        BW, T, D = patches.shape
        B = BW // cfg.W

        # Prepend class token
        cls = cls_token.expand(BW, 1, D)
        x   = torch.cat([cls, patches], dim=1)    # (BW, T+1, D)

        # Add positional embedding
        x = x + pos_embed[:, :x.size(1), :]

        # Transformer blocks
        for blk in blocks:
            x = blk(x)

        x = norm(x)

        # Extract class token and reshape to (B, W, D)
        cls_out = x[:, 0, :]                      # (BW, D)
        cls_out = cls_out.reshape(B, cfg.W, D)    # (B, W, D)
        return cls_out

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, W)
        """
        B = x.size(0)

        # Stream 1: daily
        p_daily  = self.patch_daily(x)                          # (BW, H, D)
        c_daily  = self._run_stream(
            p_daily, self.cls_daily, self.pos_daily,
            self.blocks_daily, self.norm_daily
        )                                                        # (B, W, D)

        # Stream 2: weekly
        p_weekly = self.patch_weekly(x)                         # (BW, H/5, D)
        c_weekly = self._run_stream(
            p_weekly, self.cls_weekly, self.pos_weekly,
            self.blocks_weekly, self.norm_weekly
        )                                                        # (B, W, D)

        # Merge: concatenate along D, project to W
        merged = torch.cat([c_daily, c_weekly], dim=-1)         # (B, W, 2D)
        # Mean over asset dim to get per-batch class vector
        class_vec = self.merge(merged).mean(dim=1)              # (B, W)

        # Calibration weight from raw input
        calib_w = self.calib(x)                                 # (B, W)

        # Hadamard product
        out = class_vec * calib_w                               # (B, W)
        return out
