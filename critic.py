# critic.py — Critic network for DDPG
# Evaluates (state, action) pairs to produce Q-value
# Uses CNN for state processing + concatenates portfolio weights

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg

class Critic(nn.Module):
    """
    Critic network — estimates Q(state, action).

    State  : price matrix (B, C, H, W)
    Action : portfolio weights (B, W)
    Output : Q-value scalar (B, 1)

    Architecture follows paper Figure 4:
      state → 2x Conv + MaxPool + Flatten → 7x Linear → cat(action) → 2x Linear → Q
    """

    def __init__(self):
        super().__init__()
        C, H, W = cfg.C, cfg.H, cfg.W

        # ── State encoder (CNN) ───────────────────────────────────────────────
        self.conv = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, W)),
            nn.Flatten(),                               # → 32 * 4 * W
        )
        conv_out_dim = 32 * 4 * W                      # = 32 * 4 * W

        # ── State projection (7 linear layers as per paper) ──────────────────
        # Gradually compress to W dimensions
        dims = self._make_dims(conv_out_dim, W, n_layers=7)
        state_layers = []
        for i in range(len(dims) - 1):
            state_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU() if i < len(dims) - 2 else nn.Identity(),
            ])
        self.state_proj = nn.Sequential(*state_layers)  # → (B, W)

        # ── Q-value head (after concatenating action) ────────────────────────
        self.q_head = nn.Sequential(
            nn.Linear(W * 2, W * 4),
            nn.ReLU(),
            nn.Linear(W * 4, W),
            nn.Tanh(),
            nn.Linear(W, 1),
        )

        self._init_weights()

    @staticmethod
    def _make_dims(in_dim: int, out_dim: int, n_layers: int):
        """Logarithmically space intermediate dimensions."""
        dims = [in_dim]
        if n_layers > 0:
            # Create intermediate dims using logspace, excluding the first and last
            inter = np.logspace(np.log10(in_dim), np.log10(out_dim), n_layers + 2).astype(int).tolist()
            # Ensure first is exactly in_dim
            inter[0] = in_dim
            inter[-1] = out_dim
            dims.extend(inter[1:-1])
            dims.append(out_dim)
        else:
            dims = [in_dim, out_dim]
        return dims

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, state_matrix: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_matrix : (B, C, H, W)
            action       : (B, W) portfolio weights

        Returns:
            q_value : (B, 1)
        """
        # Encode state
        state_feat = self.conv(state_matrix)         # (B, conv_out_dim)
        state_proj = self.state_proj(state_feat)     # (B, W)

        # Concatenate with action
        combined = torch.cat([state_proj, action], dim=-1)  # (B, 2W)

        # Q-value
        q = self.q_head(combined)                    # (B, 1)
        return q


class TargetCritic(Critic):
    """
    Target critic — identical architecture, updated via soft update.
    Provides stable Q-value targets during training.
    """
    pass


if __name__ == "__main__":
    B, C, H, W = 4, cfg.C, cfg.H, cfg.W
    state  = torch.randn(B, C, H, W)
    action = torch.softmax(torch.randn(B, W), dim=-1)

    critic = Critic()
    q      = critic(state, action)

    print(f"State shape:  {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Q-value shape: {q.shape}   — expected ({B}, 1)")
    print(f"Q-values: {q.squeeze().detach().numpy()}")

    n_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"Critic parameters: {n_params:,}")
