# environment.py — portfolio simulation environment for DDPG
# Handles state construction, action execution, reward computation
#
# Reward upgraded to risk-aware composite (arxiv 2403.16667):
#   reward = W_RETURN  x  log_return
#          - W_DRAWDOWN x |drawdown_from_peak|
#          - W_TURNOVER x (tc x turnover)
# Weights defined in config.py. Transaction cost no longer double-counted.

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
    Reward : composite risk-aware reward (return / drawdown / turnover)
    """

    def __init__(self,
                 price_matrices: np.ndarray,
                 daily_returns: np.ndarray,
                 transaction_cost: float = cfg.TRANSACTION_COST):
        """
        Args:
            price_matrices   : (N, C, H, W) — sliding window feature matrices
            daily_returns    : (N+1, W)     — actual daily returns for each step
            transaction_cost : fraction of rebalanced value — now embedded in
                               W_TURNOVER reward term, not double-counted
        """
        self.matrices = price_matrices          # (N, C, H, W)
        self.returns  = daily_returns           # (N+1, W)
        self.tc       = transaction_cost
        self.N        = len(price_matrices)
        self.W        = cfg.W
        self.reset()

    def reset(self):
        """Reset environment to start. Returns initial state."""
        self.t               = 0
        self.portfolio_val   = cfg.INITIAL_CAPITAL
        self.weights         = np.ones(self.W, dtype=np.float32) / self.W
        self.done            = False
        self.peak_val        = cfg.INITIAL_CAPITAL   # tracks running peak for drawdown
        self.portfolio_history = [self.portfolio_val]
        self.weight_history    = [self.weights.copy()]
        return self._get_state()

    def _get_state(self):
        """
        State = (price_matrix, prev_weights)
        price_matrix shape: (C, H, W)
        prev_weights shape: (W,)
        """
        return {
            'matrix':  self.matrices[self.t],   # (C, H, W)
            'weights': self.weights.copy(),     # (W,)
        }

    def step(self, action: np.ndarray):
        """
        Execute one trading step.
        Args:
            action : new portfolio weights (W,) — softmax from actor
        Returns:
            next_state : dict
            reward     : float  composite risk-aware reward
            done       : bool
            info       : dict
        """
        assert not self.done, "Environment is done. Call reset()."

        # ── Normalise action ──────────────────────────────────────────────────
        new_weights = np.clip(action, 0.0, 1.0)
        new_weights = new_weights / (new_weights.sum() + 1e-8)

        # ── Turnover ──────────────────────────────────────────────────────────
        weight_change = np.abs(new_weights - self.weights).sum()
        cost          = self.tc * weight_change

        # ── Portfolio return ──────────────────────────────────────────────────
        period_returns = self.returns[self.t]           # (W,)
        port_return    = np.dot(new_weights, period_returns)
        net_return     = port_return - cost
        log_return     = float(np.log1p(np.clip(net_return, -0.999, None)))

        # ── Update portfolio value and peak ───────────────────────────────────
        self.portfolio_val *= (1.0 + net_return)
        self.peak_val       = max(self.peak_val, self.portfolio_val)

        # ── Drawdown penalty (0 when at peak, negative when below) ───────────
        if self.portfolio_val < self.peak_val:
            drawdown_penalty = (self.portfolio_val - self.peak_val) / self.peak_val
        else:
            drawdown_penalty = 0.0

        # ── Turnover penalty ──────────────────────────────────────────────────
        turnover_penalty = self.tc * weight_change

        # ── Composite reward ──────────────────────────────────────────────────
        reward = (  cfg.W_RETURN   * log_return
                  - cfg.W_DRAWDOWN * abs(drawdown_penalty)
                  - cfg.W_TURNOVER * turnover_penalty      )

        # ── Weight drift with market ──────────────────────────────────────────
        drifted     = new_weights * (1.0 + period_returns)
        drifted_sum = drifted.sum()
        if drifted_sum > 1e-8:
            self.weights = drifted / drifted_sum
        else:
            self.weights = new_weights.copy()

        # ── Advance time ──────────────────────────────────────────────────────
        self.t   += 1
        self.done = (self.t >= self.N - 1)

        self.portfolio_history.append(self.portfolio_val)
        self.weight_history.append(new_weights.copy())

        next_state = self._get_state() if not self.done else None

        info = {
            'portfolio_val':     self.portfolio_val,
            'weights':           new_weights.copy(),
            'port_return':       port_return,
            'cost':              cost,
            'net_return':        net_return,
            'drawdown_penalty':  drawdown_penalty,
            'turnover_penalty':  turnover_penalty,
            'reward':            reward,
        }

        return next_state, reward, self.done, info

    def get_portfolio_series(self) -> pd.Series:
        """Return portfolio value series as pandas Series."""
        return pd.Series(self.portfolio_history, name='FTRL_Portfolio')

    def get_weight_history(self) -> pd.DataFrame:
        """Return weight history as DataFrame."""
        return pd.DataFrame(self.weight_history, columns=cfg.ASSETS)

    @staticmethod
    def compute_daily_returns(prices: pd.DataFrame) -> np.ndarray:
        """
        Compute daily simple returns from price DataFrame.
        Args:
            prices  : (T, W) price DataFrame
        Returns:
            returns : (T-1, W) daily returns array
        """
        ret = prices.pct_change().dropna().values
        return ret.astype(np.float32)


class ReplayBuffer:
    """
    Fixed-size experience replay buffer for DDPG.
    Stores (state_matrix, state_weights, action, reward,
             next_state_matrix, next_state_weights, done) tuples.
    """

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
        idxs  = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]

        states      = [b[0] for b in batch]
        actions     = np.array([b[1] for b in batch], dtype=np.float32)
        rewards     = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = [b[3] for b in batch]
        dones       = np.array([b[4] for b in batch], dtype=np.float32)

        # Stack matrix and weight components separately
        mat  = np.stack([s['matrix']  for s in states])
        wts  = np.stack([s['weights'] for s in states])

        # Handle terminal states (next_state = None)
        nmat = np.stack([
            s['matrix']  if s is not None else np.zeros_like(mat[0])
            for s in next_states
        ])
        nwts = np.stack([
            s['weights'] if s is not None else np.zeros_like(wts[0])
            for s in next_states
        ])

        return {
            'mat':     mat,     # (B, C, H, W)
            'wts':     wts,     # (B, W)
            'actions': actions, # (B, W)
            'rewards': rewards, # (B,)
            'nmat':    nmat,    # (B, C, H, W)
            'nwts':    nwts,    # (B, W)
            'dones':   dones,   # (B,)
        }

    def __len__(self):
        return len(self.buffer)

    @property
    def ready(self) -> bool:
        return len(self.buffer) >= cfg.BATCH_SIZE
