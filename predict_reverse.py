# predict_reverse.py — Daily live signal from reverse expanding windows
# Now supports ASSET_GROUP env variable to differentiate FI and Equity outputs.

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import pandas_market_calendars as mcal
from datetime import datetime, date, timedelta
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

def next_trading_day(dt: pd.Timestamp) -> pd.Timestamp:
    # unchanged
    ...

def push_to_hf(local_path: str, repo_path: str):
    # unchanged
    ...

def load_reverse_summaries() -> pd.DataFrame:
    """Load all available reverse window summaries from HF (group-aware)."""
    frames = []
    for w_id in range(1, 15):
        try:
            path = hf_hub_download(
                repo_id=cfg.HF_DATASET_REPO,
                filename=f"results/reverse_window_{w_id:02d}{cfg.OUTPUT_SUFFIX}_summary.json",
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
    # unchanged logic
    ...

def load_latest_reverse_signal() -> dict:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=f"results/latest_reverse_signal{cfg.OUTPUT_SUFFIX}.json",
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
            filename=f"results/reverse_signal_history{cfg.OUTPUT_SUFFIX}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
            force_download=True,
        )
        with open(path) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def score_signal(record: dict, prices: pd.DataFrame, bench: pd.Series) -> dict:
    # unchanged
    ...

def score_unscored_signals(history: list, prices: pd.DataFrame,
                           bench: pd.Series) -> tuple[list, bool]:
    # unchanged
    ...

def train_on_window(window: dict, prices: pd.DataFrame) -> tuple:
    # unchanged
    ...

def get_signal(trainer: DDPGTrainer, feat: np.ndarray,
               best_window: dict, signal_date: pd.Timestamp) -> dict:
    # unchanged
    ...

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print(f"FTRL Reverse Signal Generator — {cfg.ASSET_GROUP} group")
    print(f"Run date: {date.today()}")
    print("="*60)

    # 1. Load data
    prices = load_etf_prices()
    bench  = load_benchmark_prices()
    prices, bench = align_dates(prices, bench)

    # 2. Signal date
    last_data_date = prices.index[-1]
    signal_date    = next_trading_day(last_data_date)
    print(f"[Signal Date] Last data: {last_data_date.date()} → Next trading: {signal_date.date()}")

    # 3. Find best reverse window
    summaries   = load_reverse_summaries()
    best_window = find_best_window(summaries)

    # 4. Load history and score unscored signals
    prev_signal    = load_latest_reverse_signal()
    signal_history = load_reverse_signal_history()

    print(f"\n[History] {len(signal_history)} previous reverse signals")
    if prev_signal:
        print(f"[Yesterday] signal={prev_signal.get('signal')} date={prev_signal.get('date')}")

    updated_history, changed = score_unscored_signals(signal_history, prices, bench)
    if changed:
        signal_history = updated_history
        print(f"[History] Scored some previously unscored records. Total: {len(signal_history)}")

    # 5. Train on best window
    trainer, feat = train_on_window(best_window, prices)

    # 6. Generate signal
    signal = get_signal(trainer, feat, best_window, signal_date)

    # 7. Append to history
    existing_dates = {rec.get('date') for rec in signal_history}
    if signal['date'] not in existing_dates:
        signal_history.append(signal)
        print(f"[History] Appended today's reverse signal ({signal['date']})")

    # 8. Save and push (group-aware)
    os.makedirs("/tmp/ftrl_reverse_predict", exist_ok=True)

    signal_path  = f"/tmp/ftrl_reverse_predict/latest_reverse_signal{cfg.OUTPUT_SUFFIX}.json"
    history_path = f"/tmp/ftrl_reverse_predict/reverse_signal_history{cfg.OUTPUT_SUFFIX}.json"

    with open(signal_path, 'w') as f:
        json.dump(signal, f, indent=2)
    with open(history_path, 'w') as f:
        json.dump(signal_history, f, indent=2)

    push_to_hf(signal_path,  f"results/latest_reverse_signal{cfg.OUTPUT_SUFFIX}.json")
    push_to_hf(history_path, f"results/reverse_signal_history{cfg.OUTPUT_SUFFIX}.json")

    print(f"\n[Done] Reverse signal: {signal['signal']} ({signal['confidence']:.1%}) | "
          f"Trained on: {best_window['label']}")
    print(f"[Done] History: {len(signal_history)} records saved")


if __name__ == "__main__":
    main()
