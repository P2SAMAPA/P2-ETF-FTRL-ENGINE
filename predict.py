# predict.py — Daily live signal generation
# 1. Scores yesterday's signal against actual returns
# 2. Trains actor on full dataset (2008 → today)
# 3. Runs inference on latest 40-day window
# 4. Winner-takes-all: highest softmax weight = tomorrow's ETF signal
# 5. Saves latest_signal.json and appends to signal_history.json on HF

import os
import sys
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, date
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg
from dataset import load_etf_prices, load_benchmark_prices, align_dates
from features import compute_features, normalise_features
from environment import PortfolioEnv, ReplayBuffer
from ddpg import DDPGTrainer


# ── Helpers ───────────────────────────────────────────────────────────────────

def push_to_hf(local_path: str, repo_path: str):
    api = HfApi(token=cfg.HF_TOKEN)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
    )
    print(f"[HF] Pushed {repo_path}")


def load_latest_signal() -> dict:
    """Load yesterday's signal from HF. Returns None if not found."""
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename="results/latest_signal.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_signal_history() -> list:
    """Load existing signal history from HF. Returns empty list if not found."""
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename="results/signal_history.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def score_yesterday(prev_signal: dict, prices: pd.DataFrame,
                    bench: pd.Series) -> dict:
    """
    Score yesterday's signal against actual price returns.
    Returns scored record to append to history.
    """
    if prev_signal is None:
        return None

    signal_date = prev_signal.get('date')
    signal_etf  = prev_signal.get('signal')

    if not signal_date or not signal_etf:
        return None

    try:
        # Find the trading day after the signal date
        signal_dt   = pd.Timestamp(signal_date)
        future_days = prices.index[prices.index > signal_dt]

        if len(future_days) == 0:
            print(f"[Score] No trading day after {signal_date} yet — skipping")
            return None

        next_day = future_days[0]

        # ETF return on the signal day
        if signal_etf not in prices.columns:
            return None

        etf_prev  = prices[signal_etf].get(signal_dt)
        etf_next  = prices[signal_etf].get(next_day)
        agg_prev  = bench.get(signal_dt)
        agg_next  = bench.get(next_day)

        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in [etf_prev, etf_next, agg_prev, agg_next]):
            return None

        etf_return = float(etf_next / etf_prev - 1)
        agg_return = float(agg_next / agg_prev - 1)
        beats      = etf_return > agg_return

        scored = {
            **prev_signal,
            'actual_return':  round(etf_return, 6),
            'agg_return':     round(agg_return, 6),
            'excess_return':  round(etf_return - agg_return, 6),
            'beats_agg':      beats,
            'result_date':    next_day.strftime('%Y-%m-%d'),
        }

        print(f"[Score] {signal_date} signal={signal_etf} "
              f"actual={etf_return:.2%} AGG={agg_return:.2%} "
              f"{'✓' if beats else '✗'}")
        return scored

    except Exception as e:
        print(f"[Score] Error scoring yesterday: {e}")
        return None


# ── Training on full dataset ──────────────────────────────────────────────────

def train_full(prices: pd.DataFrame) -> DDPGTrainer:
    """
    Train DDPG actor on full price history (2008 → today).
    Returns trained DDPGTrainer ready for inference.
    """
    print(f"\n[Train] Full dataset: {prices.index[0].date()} → "
          f"{prices.index[-1].date()} ({len(prices)} days)")

    from features import compute_features, build_price_matrices, normalise_features

    # Features
    feat = compute_features(prices)                        # (T, C, W)

    # Normalise using full dataset stats (no test split here)
    mean = feat.mean(axis=(0, 2), keepdims=True)
    std  = feat.std(axis=(0, 2),  keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    feat = ((feat - mean) / std).astype(np.float32)

    from features import build_price_matrices
    matrices = build_price_matrices(feat)                  # (N, C, H, W)
    returns  = PortfolioEnv.compute_daily_returns(prices)  # (T-1, W)

    n = min(len(matrices), len(returns))
    matrices = matrices[:n]
    returns  = returns[:n]

    print(f"[Train] matrices={matrices.shape} returns={returns.shape}")

    env = PortfolioEnv(matrices, returns)

    # Cap predict runs at 30 epochs — full 50 is too slow on 4500+ day dataset.
    # Early stopping patience (10) unchanged so model can still bail early.
    cfg.MAX_EPOCHS = 30

    trainer = DDPGTrainer(window_id=0)   # window_id=0 = full dataset run

    checkpoint_dir = "/tmp/ftrl_predict"
    trainer.train(env, checkpoint_dir)
    trainer.load_best(checkpoint_dir)
    trainer.actor.eval()

    return trainer, feat


# ── Inference ─────────────────────────────────────────────────────────────────

def get_signal(trainer: DDPGTrainer, feat: np.ndarray) -> dict:
    """
    Run actor on the latest H-day window.
    Returns winner-takes-all signal dict.
    """
    H = cfg.H
    if feat.shape[0] < H:
        raise ValueError(f"Not enough data for lookback: {feat.shape[0]} < {H}")

    latest_window = feat[-H:]                            # (H, C, W)
    matrix        = latest_window.transpose(1, 0, 2)    # (C, H, W)
    matrix_t      = torch.FloatTensor(matrix).unsqueeze(0)  # (1, C, H, W)

    # Equal weight starting point
    prev_weights = torch.ones(1, cfg.W) / cfg.W

    with torch.no_grad():
        weights = trainer.actor(matrix_t, prev_weights).squeeze(0).numpy()

    # Winner-takes-all
    best_idx   = int(np.argmax(weights))
    signal     = cfg.ASSETS[best_idx]
    confidence = float(weights[best_idx])

    raw_weights = {cfg.ASSETS[i]: round(float(weights[i]), 6)
                   for i in range(cfg.W)}

    result = {
        'date':         date.today().strftime('%Y-%m-%d'),
        'signal':       signal,
        'confidence':   round(confidence, 4),
        'raw_weights':  raw_weights,
        'trained_on':   'full dataset',
        'generated_at': datetime.utcnow().isoformat() + 'Z',
    }

    print(f"\n[Signal] {result['date']} → {signal} "
          f"(confidence: {confidence:.1%})")
    print(f"[Weights] " +
          " | ".join(f"{k}:{v:.1%}" for k, v in raw_weights.items()))

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("FTRL Daily Signal Generator")
    print(f"Run date: {date.today()}")
    print("="*60)

    # 1. Load data
    prices = load_etf_prices()
    bench  = load_benchmark_prices()
    prices, bench = align_dates(prices, bench)

    # 2. Load yesterday's signal and history
    prev_signal    = load_latest_signal()
    signal_history = load_signal_history()

    print(f"\n[History] {len(signal_history)} previous signals loaded")
    if prev_signal:
        print(f"[Yesterday] signal={prev_signal.get('signal')} "
              f"date={prev_signal.get('date')}")

    # 3. Score yesterday's signal
    scored = score_yesterday(prev_signal, prices, bench)
    if scored:
        signal_history.append(scored)
        print(f"[History] Appended scored record. Total: {len(signal_history)}")

    # Keep only last 90 records in history file (UI shows 15, keep 90 for buffer)
    signal_history = signal_history[-90:]

    # 4. Train on full dataset
    trainer, feat = train_full(prices)

    # 5. Generate today's signal
    signal = get_signal(trainer, feat)

    # 6. Save files locally
    os.makedirs("/tmp/ftrl_predict", exist_ok=True)

    signal_path  = "/tmp/ftrl_predict/latest_signal.json"
    history_path = "/tmp/ftrl_predict/signal_history.json"

    with open(signal_path, 'w') as f:
        json.dump(signal, f, indent=2)

    with open(history_path, 'w') as f:
        json.dump(signal_history, f, indent=2)

    # 7. Push to HF
    push_to_hf(signal_path,  "results/latest_signal.json")
    push_to_hf(history_path, "results/signal_history.json")

    print(f"\n[Done] Signal for {signal['date']}: "
          f"{signal['signal']} ({signal['confidence']:.1%} confidence)")
    print(f"[Done] History: {len(signal_history)} records saved")


if __name__ == "__main__":
    main()
