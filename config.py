# config.py — single source of truth for all hyperparameters
# ─────────────────────────────────────────────────────────────
# Architecture sized for GitHub Actions 2-core CPU within 6-hour free tier.

import os

# ── Data ──────────────────────────────────────────────────────────────────────
ASSETS  = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV']
W       = len(ASSETS)          # portfolio width = 6

# ── Feature matrix dimensions ─────────────────────────────────────────────────
H = 40    # lookback days (rows per matrix)
C = 4     # features per asset: daily_ret, rvol_5d, vol_ratio, momentum_20d

# ── Transformer architecture ─────────────────────────────────────────────────
ATTENTION_BLOCKS = 3           # reduced from paper's 6 for CPU
N_HEADS          = 4
D_MODEL          = 64

# ── DDPG hyperparameters ──────────────────────────────────────────────────────
BATCH_SIZE  = 16
LR          = 1e-5
GAMMA       = 0.99
TAU         = 0.005
BUFFER_SIZE = 2000
MAX_EPOCHS  = 50               # capped at 30 for daily predict runs
EARLY_STOP_PAT = 10

# ── Environment ───────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 1.0
TRANSACTION_COST = 0.001       # 0.1% of rebalanced value
                               # now embedded in W_TURNOVER reward term
                               # (not double-counted as a separate deduction)

# ── Composite reward weights (must sum to 1.0) ────────────────────────────────
# Inspired by arxiv 2403.16667 additive utility framework.
# Return stays primary; drawdown and turnover act as soft regularisers.
#
#   reward = W_RETURN  × log_return
#          - W_DRAWDOWN × |drawdown_from_peak|
#          - W_TURNOVER × (tc × sum_abs_weight_change)
#
W_RETURN   = 0.70   # 70% — return is the primary objective
W_DRAWDOWN = 0.20   # 20% — penalise drawdowns from running peak
W_TURNOVER = 0.10   # 10% — penalise excessive rebalancing (replaces flat tc deduction)

# ── HuggingFace repos ────────────────────────────────────────────────────────
HF_TOKEN        = os.getenv("HF_TOKEN", "")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "P2SAMAPA/p2-etf-ftrl-engine")
HF_SOURCE_REPO  = os.getenv("HF_SOURCE_REPO",  "P2SAMAPA/p2-etf-deepwave-dl")
