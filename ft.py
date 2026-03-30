# ft.py — Financial Transformer (Actor network)
# Combines TRB + LLB outputs into portfolio weight vector
# Actor in the DDPG framework

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg
from trb import TRB
from llb import LLB


class FinancialTransformer(nn.Module):
    """
    Financial Transformer — Actor network.

    Architecture:
      1. TRB → temporal class vector (B, W)
      2. LLB → latent linkage vector  (B, D_MODEL)
      3. Stack + linear merge
      4. Hadamard product with calibration weights
      5. Softmax → portfolio weights (B, W)

    Input:  price matrix (B, C, H, W) where W = number of assets (now 18)
    Output: portfolio weights (B, W) — sums to 1, all >= 0
    """

    def __init__(self):
        super().__init__()
        W = cfg.W
        D = cfg.D_MODEL

        self.trb = TRB()
        self.llb = LLB()

        # Project LLB output (D) → (W) to match TRB output dimension
        self.llb_proj = nn.Linear(D, W)

        # Merge layer: concatenate TRB (W) + LLB_proj (W) → W
        self.merge = nn.Sequential(
            nn.Linear(W * 2, W * 4),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(W * 4, W),
        )

        # Layer norm before softmax for training stability
        self.output_norm = nn.LayerNorm(W)

    def forward(self, x: torch.Tensor,
                prev_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x            : (B, C, H, W) price matrix
            prev_weights : (B, W) previous portfolio weights (optional)
                           Not used in current formulation but available
                           for future state-augmented variants

        Returns:
            weights : (B, W) portfolio weights summing to 1
        """
        # Temporal features — per-asset signal
        trb_out = self.trb(x)                    # (B, W)

        # Latent linkage features — cross-asset signal
        llb_out = self.llb(x)                    # (B, D_MODEL)
        llb_proj = self.llb_proj(llb_out)        # (B, W)

        # Merge both streams
        combined = torch.cat([trb_out, llb_proj], dim=-1)  # (B, 2W)
        merged   = self.merge(combined)                     # (B, W)

        # Normalise and apply softmax
        normed  = self.output_norm(merged)
        weights = F.softmax(normed, dim=-1)      # (B, W) — sums to 1

        return weights

    def get_action(self, state_dict: dict,
                   noise_sigma: float = 0.0) -> torch.Tensor:
        """
        Inference method: get portfolio weights with optional exploration noise.

        Args:
            state_dict   : {'matrix': (C,H,W) tensor, 'weights': (W,) tensor}
            noise_sigma  : exploration noise std (0 for pure exploitation)

        Returns:
            weights : (W,) numpy array
        """
        self.eval()
        with torch.no_grad():
            mat = state_dict['matrix'].unsqueeze(0)      # (1, C, H, W)
            wts = state_dict['weights'].unsqueeze(0)     # (1, W)

            weights = self.forward(mat, wts)             # (1, W)

            if noise_sigma > 0:
                noise   = torch.randn_like(weights) * noise_sigma
                weights = F.softmax(weights + noise, dim=-1)

        return weights.squeeze(0).numpy()


class Actor(FinancialTransformer):
    """Alias for clarity in DDPG context."""
    pass


if __name__ == "__main__":
    # Shape test
    B, C, H, W = 4, cfg.C, cfg.H, cfg.W
    x    = torch.randn(B, C, H, W)
    wts  = torch.ones(B, W) / W

    model = FinancialTransformer()
    out   = model(x, wts)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}   — expected ({B}, {W})")
    print(f"Weight sums:  {out.sum(dim=-1)}")   # should all be ~1.0
    print(f"Min weight:   {out.min().item():.6f}")   # should be > 0
    print(f"Max weight:   {out.max().item():.6f}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {n_params:,}")
