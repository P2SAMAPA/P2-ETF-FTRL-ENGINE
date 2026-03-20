# environment.py — portfolio simulation environment for DDPG
#
# Reward function upgraded to risk-aware composite (inspired by arxiv 2403.16667):
#
#   reward = W_RETURN  × log_return
#          - W_DRAWDOWN × drawdown_penalty   (only when below running peak)
#          - W_TURNOVER × turnover_penalty   (replaces flat transaction cost deduction)
#
# Weights in config.py: W_RETURN=0.70, W_DRAWDOWN=0.20, W_TURNOVER=0.10
# Transaction cost is now embedded in the turnover penalty term (not double-counted).
#
# All three components are kept in the same numerical range (roughly ±0.01 to ±0.05
# per step for a diversified fixed-income ETF portfolio) so the weights behave as
# true relative priorities rather than arbitrary scalars.

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg


class PortfolioEnv:
    """
    Portfolio environment compatible with DDPG.

    State  : price matrix of shape (C, H, W) + previous weights (W,)
    Action : portfolio weight vector (W,) — softmax output from actor
    Reward : composite risk-aware reward (see module docstring)
    """

    def __init__(self,
                 price_matrices: np.ndarray,
                 daily_returns:  np.ndarray,
                 transaction_cost: float = cfg.TRANSACTION_COST):
        """
        Args:
            price_matrices  : (N, C, H, W) — sliding window feature matrices
            daily_returns   : (N+1, W)     — actual daily returns for each step
            transaction_cost: kept for backward compatibility but now embedded
                              in W_TURNOVER term rather than subtracted directly
        """
        self.matrices = price_matrices   # (N, C, H, W)
        self.returns  = daily_returns    # (N+1, W)
        self.tc       = transaction_cost
        self.N        = len(price_matrices)
        self.W        = cfg.W

        # Reward weights — single source of truth in config.py
        self.w_return   = cfg.W_RETURN
        self.w_drawdown = cfg.W_DRAWDOWN
        self.w_turnover = cfg.W_TURNOVER

        self.reset()

    def reset(self):
        """Reset environment to start. Returns initial state."""
        self.t              = 0
        self.portfolio_val  = cfg.INITIAL_CAPITAL
        self.weights        = np.ones(self.W, dtype=np.float32) / self.W
        self.done           = False
        self.peak_val       = cfg.INITIAL_CAPITAL   # for drawdown tracking
        self.portfolio_history = [self.portfolio_val]
        self.weight_history    = [self.weights.copy()]
        return self._get_state()

    def _get_state(self):
        """
        State = (price_matrix, prev_weights)
            price_matrix : (C, H, W)
            prev_weights : (W,)
        """
        return {
            'matrix':  self.matrices[self.t],  # (C, H, W)
            'weights': self.weights.copy(),     # (W,)
        }

    def step(self, action: np.ndarray):
        """
        Execute one trading step.

        Args:
            action : new portfolio weights (W,) — softmax-normalised from actor

        Returns:
            next_state : dict
            reward     : float  composite risk-aware reward
            done       : bool
            info       : dict   (portfolio_value, weights, raw_return, components)
        """
        assert not self.done, "Environment is done — call reset()."

        # ── Normalise action ─────────────────────────────────────────────────
        new_weights = np.clip(action, 0.0, 1.0)
        new_weights = new_weights / (new_weights.sum() + 1e-8)

        # ── Turnover (absolute weight change, sum across assets) ──────────────
        # Range: 0 (no rebalance) to 2.0 (full reversal). Typical: 0.1–0.5.
        turnover    = np.abs(new_weights - self.weights).sum()

        # ── Portfolio return ──────────────────────────────────────────────────
        # returns[t] is the return vector FROM day t TO day t+1
        period_returns  = self.returns[self.t]                    # (W,)
        gross_ret       = float(np.dot(new_weights, period_returns))

        # Apply transaction cost to portfolio value (still simulates real cost)
        cost            = self.tc * turnover
        net_ret         = gross_ret - cost

        # Log return for reward — numerically stable, additive over time
        log_ret         = float(np.log1p(np.clip(net_ret, -0.999, None)))

        # ── Update portfolio value and peak ───────────────────────────────────
        self.portfolio_val *= (1.0 + net_ret)
        self.peak_val       = max(self.peak_val, self.portfolio_val)

        # ── Drawdown penalty ──────────────────────────────────────────────────
        # Range: 0 (at peak) to negative (below peak).
        # Expressed as fraction of peak so it's comparable to log_ret magnitude.
        if self.portfolio_val < self.peak_val:
            drawdown_penalty = (self.portfolio_val - self.peak_val) / self.peak_val
        else:
            drawdown_penalty = 0.0

        # ── Turnover penalty ─────────────────────────────────────────────────
        # Scale turnover to log-return magnitude. Full portfolio rotation (~2.0)
        # should cost roughly the same order as a typical daily return (~0.003).
        # Scaling by tc brings it to that range (tc=0.001, turnover=2 → 0.002).
        turnover_penalty = self.tc * turnover

        # ── Composite reward ──────────────────────────────────────────────────
        reward = (  self.w_return   * log_ret
                  - self.w_drawdown * abs(drawdown_penalty)
                  - self.w_turnover * turnover_penalty      )

        # ── Advance state ─────────────────────────────────────────────────────
        self.weights = new_weights
        self.t      += 1
        self.done    = (self.t >= self.N)

        self.portfolio_history.append(self.portfolio_val)
        self.weight_history.append(self.weights.copy())

        next_state = self._get_state() if not self.done else None

        info = {
            'portfolio_value':  self.portfolio_val,
            'weights':          self.weights.copy(),
            'gross_return':     gross_ret,
            'net_return':       net_ret,
            'log_return':       log_ret,
            'drawdown_penalty': drawdown_penalty,
            'turnover':         turnover,
            'turnover_penalty': turnover_penalty,
            'reward':           reward,
        }

        return next_state, reward, self.done, info

    # ── Class-level utility (used by train.py / predict.py) ──────────────────

    @staticmethod
    def compute_daily_returns(prices: pd.DataFrame) -> np.ndarray:
        """
        Compute daily returns matrix from price DataFrame.
        Returns (T, W) float32 array — same column order as cfg.ASSETS.
        """
        rets = prices.pct_change().fillna(0).values.astype(np.float32)
        return rets

    def get_portfolio_history(self) -> np.ndarray:
        return np.array(self.portfolio_history)

    def get_weight_history(self) -> np.ndarray:
        return np.array(self.weight_history)


class ReplayBuffer:
    """Fixed-size circular replay buffer for DDPG experience replay."""

    def __init__(self, capacity: int = cfg.BUFFER_SIZE):
        self.capacity = capacity
        self.buffer   = []
        self.pos      = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch   = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
