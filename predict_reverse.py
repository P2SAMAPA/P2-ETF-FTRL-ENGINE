# predict_reverse.py — Daily live signal from reverse expanding windows
# Finds the best-performing reverse window, trains on that window's date range,
# generates tomorrow's winner-takes-all ETF signal.
# Scores yesterday's reverse signal and appends to reverse_signal_history.json

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, date
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg
from dataset import load_etf_prices, load_benchmark_prices, align_dates
from features import compute_features, build_price_matrices, normalise_features
from environment import PortfolioEnv
from ddpg import DDPGTrainer


# ── Reverse window definitions (must match train_reverse.py) ──────────────────
REVERSE_WINDOWS = [
    {
        'id':          i + 1,
        'train_start': f"{2008 + i}-01-01",
        'train_end':   '2024-12-31',
        'test_start':  '2025-01-01',
        'label':       f"R{i+1}: {2008+i}–2024",
    }
    for i in range(14)
]


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


def load_reverse_summaries() -> pd.DataFrame:
    """Load all available reverse window summaries from HF."""
    frames = []
    for w_id in range(1, 15):
        try:
            path = hf_hub_download(
                repo_id=cfg.HF_DATASET_REPO,
                filename=f"results/reverse_window_{w_id:02d}_summary.json",
                repo_type="dataset",
                token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
                force_download=True,
            )
            with open(path) as f:
                data = json.load(f)
            frames.append(data)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


def find_best_window(summaries: pd.DataFrame) -> dict:
    """
    Find the best reverse window by excess return over AGG.
    Falls back to window R1 (full history) if no summaries available.
    """
    if summaries.empty:
        print("[Best Window] No summaries found — defaulting to R1 (2008–2024)")
        return REVERSE_WINDOWS[0]

    best_idx = summaries['excess_return'].idxmax()
    best_row = summaries.loc[best_idx]
    wid      = int(best_row['window_id'])
    window   = REVERSE_WINDOWS[wid - 1]

    print(f"[Best Window] R{wid:02d} — {window['label']} "
          f"(excess={best_row['excess_return']:.2%}, "
          f"sharpe={best_row['ftrl_sharpe']:.3f})")
    return window


def load_latest_reverse_signal() -> dict:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename="results/latest_reverse_signal.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_reverse_signal_history() -> list:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename="results/reverse_signal_history.json",
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
    """Score yesterday's reverse signal against actual returns."""
    if not prev_signal:
        return None

    signal_date = prev_signal.get('date')
    signal_etf  = prev_signal.get('signal')

    if not signal_date or not signal_etf:
        return None

    try:
        signal_dt   = pd.Timestamp(signal_date)
        future_days = prices.index[prices.index > signal_dt]

        if len(future_days) == 0:
            return None

        next_day   = future_days[0]
        etf_prev   = prices[signal_etf].get(signal_dt)
        etf_next   = prices[signal_etf].get(next_day)
        agg_prev   = bench.get(signal_dt)
        agg_next   = bench.get(next_day)

        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in [etf_prev, etf_next, agg_prev, agg_next]):
            return None

        etf_return = float(etf_next / etf_prev - 1)
        agg_return = float(agg_next / agg_prev - 1)
        beats      = etf_return > agg_return

        scored = {
            **prev_signal,
            'actual_return': round(etf_return, 6),
            'agg_return':    round(agg_return, 6),
            'excess_return': round(etf_return - agg_return, 6),
            'beats_agg':     beats,
            'result_date':   next_day.strftime('%Y-%m-%d'),
        }

        print(f"[Score] {signal_date} reverse signal={signal_etf} "
              f"actual={etf_return:.2%} AGG={agg_return:.2%} "
              f"{'✓' if beats else '✗'}")
        return scored

    except Exception as e:
        print(f"[Score] Error: {e}")
        return None


def train_on_window(window: dict, prices: pd.DataFrame) -> tuple:
    """Train DDPG on the given reverse window's date range."""
    train_prices = prices[
        (prices.index >= window['train_start']) &
        (prices.index <= window['train_end'])
    ].copy()

    # For inference we also need features up to today
    full_prices = prices[
        prices.index >= window['train_start']
    ].copy()

    print(f"\n[Train] {window['label']} — "
          f"{len(train_prices)} training days")

    # Features on training set for normalisation stats
    train_feat = compute_features(train_prices)
    full_feat  = compute_features(full_prices)

    # Normalise using training stats only
    mean = train_feat.mean(axis=(0, 2), keepdims=True)
    std  = train_feat.std(axis=(0, 2),  keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)

    train_feat_norm = ((train_feat - mean) / std).astype(np.float32)
    full_feat_norm  = ((full_feat  - mean) / std).astype(np.float32)

    train_mat = build_price_matrices(train_feat_norm)
    train_ret = PortfolioEnv.compute_daily_returns(train_prices)

    n = min(len(train_mat), len(train_ret))
    train_mat = train_mat[:n]
    train_ret = train_ret[:n]

    env = PortfolioEnv(train_mat, train_ret)

    # Cap predict runs at 30 epochs — full 50 is too slow on large datasets.
    # Early stopping patience (10) unchanged so model can still bail early.
    cfg.MAX_EPOCHS = 30

    trainer = DDPGTrainer(window_id=200)   # 200 = reverse predict run
    trainer.train(env, "/tmp/ftrl_reverse_predict")
    trainer.load_best("/tmp/ftrl_reverse_predict")
    trainer.actor.eval()

    return trainer, full_feat_norm


def get_signal(trainer: DDPGTrainer, feat: np.ndarray,
               best_window: dict) -> dict:
    """Run actor on latest H-day window, return winner-takes-all signal."""
    H = cfg.H
    if feat.shape[0] < H:
        raise ValueError(f"Not enough data: {feat.shape[0]} < {H}")

    latest  = feat[-H:].transpose(1, 0, 2)           # (C, H, W)
    mat_t   = torch.FloatTensor(latest).unsqueeze(0)  # (1, C, H, W)
    wts_t   = torch.ones(1, cfg.W) / cfg.W

    with torch.no_grad():
        weights = trainer.actor(mat_t, wts_t).squeeze(0).numpy()

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
        'trained_on':   best_window['label'],
        'train_start':  best_window['train_start'],
        'train_end':    best_window['train_end'],
        'generated_at': datetime.utcnow().isoformat() + 'Z',
    }

    print(f"\n[Reverse Signal] {result['date']} → {signal} "
          f"(confidence: {confidence:.1%})")
    print(f"[Weights] " +
          " | ".join(f"{k}:{v:.1%}" for k, v in raw_weights.items()))
    print(f"[Trained on] {best_window['label']}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("FTRL Reverse Signal Generator")
    print(f"Run date: {date.today()}")
    print("="*60)

    # 1. Load data
    prices = load_etf_prices()
    bench  = load_benchmark_prices()
    prices, bench = align_dates(prices, bench)

    # 2. Find best reverse window from summaries
    summaries   = load_reverse_summaries()
    best_window = find_best_window(summaries)

    # 3. Load history and score yesterday
    prev_signal    = load_latest_reverse_signal()
    signal_history = load_reverse_signal_history()

    print(f"\n[History] {len(signal_history)} previous reverse signals")

    scored = score_yesterday(prev_signal, prices, bench)
    if scored:
        signal_history.append(scored)

    signal_history = signal_history[-90:]

    # 4. Train on best window
    trainer, feat = train_on_window(best_window, prices)

    # 5. Generate signal
    signal = get_signal(trainer, feat, best_window)

    # 6. Save and push
    os.makedirs("/tmp/ftrl_reverse_predict", exist_ok=True)

    signal_path  = "/tmp/ftrl_reverse_predict/latest_reverse_signal.json"
    history_path = "/tmp/ftrl_reverse_predict/reverse_signal_history.json"

    with open(signal_path, 'w') as f:
        json.dump(signal, f, indent=2)
    with open(history_path, 'w') as f:
        json.dump(signal_history, f, indent=2)

    push_to_hf(signal_path,  "results/latest_reverse_signal.json")
    push_to_hf(history_path, "results/reverse_signal_history.json")

    print(f"\n[Done] Reverse signal: {signal['signal']} "
          f"({signal['confidence']:.1%}) | "
          f"Trained on: {best_window['label']}")


if __name__ == "__main__":
    main()
