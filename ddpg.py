# ddpg.py — DDPG training loop
# Deep Deterministic Policy Gradient with experience replay and soft target updates

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import copy
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg
from ft import Actor
from critic import Critic
from environment import PortfolioEnv, ReplayBuffer


class OUNoise:
    """
    Ornstein-Uhlenbeck noise for temporally correlated exploration.
    Better than pure Gaussian for portfolio weight exploration.
    """
    def __init__(self, size: int, sigma: float = cfg.NOISE_SIGMA,
                 theta: float = 0.15, dt: float = 1e-2):
        self.size  = size
        self.sigma = sigma
        self.theta = theta
        self.dt    = dt
        self.reset()

    def reset(self):
        self.state = np.zeros(self.size, dtype=np.float32)

    def sample(self) -> np.ndarray:
        dx = (self.theta * (-self.state) * self.dt +
              self.sigma * np.sqrt(self.dt) *
              np.random.randn(self.size).astype(np.float32))
        self.state = self.state + dx
        return self.state

    def decay(self, factor: float = cfg.NOISE_DECAY):
        self.sigma *= factor


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft update: θ_target = τ*θ_source + (1-τ)*θ_target"""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def hard_update(target: nn.Module, source: nn.Module):
    """Hard update: copy all parameters."""
    target.load_state_dict(source.state_dict())


class DDPGTrainer:
    """
    DDPG trainer for the Financial Transformer.

    Manages:
      - Actor (FT) and Critic networks + their targets
      - Experience replay buffer
      - Training loop with early stopping
      - Checkpoint saving
    """

    def __init__(self, window_id: int):
        self.window_id = window_id
        self.device    = torch.device('cpu')  # GitHub Actions CPU

        # Networks
        self.actor        = Actor().to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic        = Critic().to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # Freeze target networks
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimisers
        self.actor_opt  = torch.optim.Adam(
            self.actor.parameters(),  lr=cfg.LR_ACTOR
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.LR_CRITIC
        )

        # Replay buffer
        self.buffer = ReplayBuffer(cfg.BUFFER_SIZE)

        # Exploration noise
        self.noise = OUNoise(cfg.W)

        # Training log
        self.log = {
            'window_id':    window_id,
            'actor_losses':  [],
            'critic_losses': [],
            'episode_returns': [],
            'best_return':  -np.inf,
            'best_epoch':   0,
        }

    def _state_to_tensors(self, state: dict):
        """Convert state dict to tensors on device."""
        if state is None:
            return None, None
        mat = torch.FloatTensor(state['matrix']).unsqueeze(0).to(self.device)
        wts = torch.FloatTensor(state['weights']).unsqueeze(0).to(self.device)
        return mat, wts

    def _select_action(self, state: dict, sigma: float) -> np.ndarray:
        """Select action with exploration noise."""
        self.actor.eval()
        with torch.no_grad():
            mat, wts = self._state_to_tensors(state)
            action   = self.actor(mat, wts).squeeze(0).cpu().numpy()

        if sigma > 0:
            noise  = self.noise.sample()
            action = F.softmax(
                torch.FloatTensor(action + noise), dim=-1
            ).numpy()

        return action.astype(np.float32)

    def _update_networks(self) -> tuple:
        """Sample from buffer and update actor + critic."""
        if not self.buffer.ready:
            return 0.0, 0.0

        batch = self.buffer.sample(cfg.BATCH_SIZE)

        mat    = torch.FloatTensor(batch['mat']).to(self.device)
        wts    = torch.FloatTensor(batch['wts']).to(self.device)
        acts   = torch.FloatTensor(batch['actions']).to(self.device)
        rews   = torch.FloatTensor(batch['rewards']).to(self.device)
        nmat   = torch.FloatTensor(batch['nmat']).to(self.device)
        nwts   = torch.FloatTensor(batch['nwts']).to(self.device)
        dones  = torch.FloatTensor(batch['dones']).to(self.device)

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            next_acts   = self.actor_target(nmat, nwts)
            target_q    = self.critic_target(nmat, next_acts)
            target_val  = (rews.unsqueeze(1) +
                           cfg.GAMMA * target_q * (1 - dones.unsqueeze(1)))

        current_q   = self.critic(mat, acts)

        # Huber loss (smooth L1) — matches paper's loss formulation
        critic_loss = F.smooth_l1_loss(current_q, target_val)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor update ──────────────────────────────────────────────────────
        self.actor.train()
        pred_acts   = self.actor(mat, wts)
        actor_loss  = -self.critic(mat, pred_acts).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Soft target updates ───────────────────────────────────────────────
        soft_update(self.actor_target,  self.actor,  cfg.TAU)
        soft_update(self.critic_target, self.critic, cfg.TAU)

        return actor_loss.item(), critic_loss.item()

    def train_epoch(self, env: PortfolioEnv,
                    sigma: float) -> float:
        """
        Run one full episode through the environment.

        Args:
            env   : PortfolioEnv instance (reset before calling)
            sigma : current exploration noise sigma

        Returns:
            episode_return : total log return for this episode
        """
        state  = env.reset()
        done   = False
        ep_ret = 0.0
        a_losses, c_losses = [], []

        while not done:
            action = self._select_action(state, sigma)
            next_state, reward, done, _ = env.step(action)

            self.buffer.push(state, action, reward, next_state, done)

            a_loss, c_loss = self._update_networks()
            if a_loss != 0:
                a_losses.append(a_loss)
                c_losses.append(c_loss)

            ep_ret += reward
            state   = next_state

        return ep_ret, (np.mean(a_losses) if a_losses else 0.0,
                        np.mean(c_losses) if c_losses else 0.0)

    def train(self, train_env: PortfolioEnv,
              checkpoint_dir: str) -> dict:
        """
        Full training loop with early stopping.

        Args:
            train_env      : environment built from training data
            checkpoint_dir : local path to save best model

        Returns:
            training log dict
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_path  = os.path.join(
            checkpoint_dir, f"window_{self.window_id:02d}_best.pt"
        )

        patience   = 0
        sigma      = cfg.NOISE_SIGMA

        print(f"\n[DDPG] Window {self.window_id:02d} — "
              f"starting training for up to {cfg.MAX_EPOCHS} epochs")

        for epoch in range(1, cfg.MAX_EPOCHS + 1):
            ep_ret, (a_loss, c_loss) = self.train_epoch(train_env, sigma)

            self.log['episode_returns'].append(ep_ret)
            self.log['actor_losses'].append(a_loss)
            self.log['critic_losses'].append(c_loss)

            # Decay noise
            sigma = max(sigma * cfg.NOISE_DECAY, 0.001)
            self.noise.sigma = sigma

            print(f"  Epoch {epoch:3d}/{cfg.MAX_EPOCHS} | "
                  f"Return: {ep_ret:.4f} | "
                  f"A-Loss: {a_loss:.5f} | "
                  f"C-Loss: {c_loss:.5f} | "
                  f"Noise: {sigma:.4f}")

            # Save best
            if ep_ret > self.log['best_return']:
                self.log['best_return'] = ep_ret
                self.log['best_epoch']  = epoch
                torch.save({
                    'actor_state':  self.actor.state_dict(),
                    'critic_state': self.critic.state_dict(),
                    'epoch':        epoch,
                    'return':       ep_ret,
                    'window_id':    self.window_id,
                }, best_path)
                patience = 0
                print(f"  ✓ New best saved (epoch {epoch})")
            else:
                patience += 1

            # Early stopping
            if patience >= cfg.EARLY_STOP_PAT:
                print(f"\n[DDPG] Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg.EARLY_STOP_PAT} epochs)")
                break

        print(f"\n[DDPG] Training complete. "
              f"Best return: {self.log['best_return']:.4f} "
              f"at epoch {self.log['best_epoch']}")

        self.log['best_model_path'] = best_path
        return self.log

    def load_best(self, checkpoint_dir: str):
        """Load best saved weights for inference."""
        path = os.path.join(
            checkpoint_dir, f"window_{self.window_id:02d}_best.pt"
        )
        ckpt = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(ckpt['actor_state'])
        self.actor.eval()
        print(f"[DDPG] Loaded best model from epoch {ckpt['epoch']} "
              f"(return={ckpt['return']:.4f})")
        return ckpt
