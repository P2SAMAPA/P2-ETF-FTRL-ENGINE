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
    """Return the next NYSE trading day after dt."""
    nyse = mcal.get_calendar('NYSE')
    # Look ahead up to 14 calendar days to find the next valid session
    start = dt + pd.Timedelta(days=1)
    end   = dt + pd.Timedelta(days=14)
    schedule = nyse.schedule(
        start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d'),
    )
    if schedule.empty:
        # Fallback: skip weekends only (shouldn't normally happen)
        next_day = dt + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=1)
        return next_day
    return schedule.index[0]


def push_to_hf(local_path: str, repo_path: str):
    """Push a local file to the HF dataset repo."""
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
        token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
    )
    print(f"[HF] Pushed {repo_path}")


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
    """Select best reverse window by live_excess_return, fall back to excess_return."""
    if summaries.empty:
        print("[Best Window] No summaries found — defaulting to R1 (2008–2024)")
        return REVERSE_WINDOWS[0]

    # Prefer live_excess_return if populated
    if 'live_excess_return' in summaries.columns:
        live = summaries.dropna(subset=['live_excess_return'])
        if not live.empty:
            best = live.loc[live['live_excess_return'].idxmax()]
            basis = 'live_excess_return'
        else:
            best  = summaries.loc[summaries['excess_return'].idxmax()]
            basis = 'excess_return (fallback)'
    else:
        best  = summaries.loc[summaries['excess_return'].idxmax()]
        basis = 'excess_return (fallback)'

    w_id = int(best['window_id'])
    window = next((w for w in REVERSE_WINDOWS if w['id'] == w_id), REVERSE_WINDOWS[0])
    print(f"[Best Window] R{w_id} ({window['label']}) selected by {basis}")
    return window


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
    """Score a previous signal record with actual return vs benchmark."""
    signal_date = record.get('date')
    etf         = record.get('signal')
    if not signal_date or not etf:
        return record

    try:
        sig_ts  = pd.Timestamp(signal_date)
        # Find the actual return on signal_date
        if sig_ts not in prices.index or sig_ts not in bench.index:
            return record  # data not available yet

        idx = prices.index.get_loc(sig_ts)
        if idx == 0:
            return record

        prev_ts = prices.index[idx - 1]
        etf_ret  = (prices.loc[sig_ts, etf]  / prices.loc[prev_ts, etf])  - 1
        agg_ret  = (bench.loc[sig_ts]         / bench.loc[prev_ts])        - 1

        record = record.copy()
        record['actual_return']   = round(float(etf_ret), 6)
        record['benchmark_return']= round(float(agg_ret), 6)
        record['excess_return']   = round(float(etf_ret - agg_ret), 6)
        record['scored']          = True
    except Exception as e:
        print(f"[Score] Could not score {signal_date}: {e}")

    return record


def score_unscored_signals(history: list, prices: pd.DataFrame,
                           bench: pd.Series) -> tuple[list, bool]:
    """Score any records that haven't been scored yet."""
    changed = False
    updated = []
    for rec in history:
        if not rec.get('scored', False):
            new_rec = score_signal(rec, prices, bench)
            if new_rec.get('scored'):
                changed = True
            updated.append(new_rec)
        else:
            updated.append(rec)
    return updated, changed


def train_on_window(window: dict, prices: pd.DataFrame) -> tuple:
    """Train DDPG on the given reverse window and return (trainer, latest_features)."""
    train_prices = prices[
        (prices.index >= window['train_start']) &
        (prices.index <= window['train_end'])
    ].copy()

    print(f"\n[Train] Window {window['label']} | {len(train_prices)} days")

    feat_raw  = compute_features(train_prices)
    feat_norm, mu, sigma = normalise_features(feat_raw)
    matrices  = build_price_matrices(feat_norm, cfg.H)

    env     = PortfolioEnv(matrices, train_prices.values[cfg.H:])
    trainer = DDPGTrainer(env)
    trainer.train()

    # Build inference feature from full price history (latest H days)
    all_feat_raw  = compute_features(prices)
    all_feat_norm = (all_feat_raw - mu) / (sigma + 1e-8)
    all_matrices  = build_price_matrices(all_feat_norm, cfg.H)
    latest_feat   = all_matrices[-1]  # shape (C, H, W)

    return trainer, latest_feat


def get_signal(trainer: DDPGTrainer, feat: np.ndarray,
               best_window: dict, signal_date: pd.Timestamp) -> dict:
    """Run inference and return signal dict."""
    feat_tensor = torch.FloatTensor(feat).unsqueeze(0)  # (1, C, H, W)

    with torch.no_grad():
        weights = trainer.actor(feat_tensor).squeeze(0).numpy()

    best_idx    = int(np.argmax(weights))
    signal_etf  = cfg.ASSETS[best_idx]
    confidence  = float(weights[best_idx])

    signal = {
        'date':         signal_date.strftime('%Y-%m-%d'),
        'signal':       signal_etf,
        'confidence':   round(confidence, 6),
        'weights':      {cfg.ASSETS[i]: round(float(w), 6) for i, w in enumerate(weights)},
        'trained_on':   best_window['label'],
        'train_start':  best_window['train_start'],
        'train_end':    best_window['train_end'],
        'window_id':    best_window['id'],
        'scored':       False,
        'asset_group':  cfg.ASSET_GROUP,
    }

    print(f"[Signal] {signal_etf} ({confidence:.1%}) for {signal_date.date()} "
          f"| trained on {best_window['label']}")
    return signal


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
