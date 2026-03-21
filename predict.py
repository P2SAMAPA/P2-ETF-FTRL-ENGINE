# predict.py — Daily live signal generation
# 1. Scores yesterday's signal against actual returns
# 2. Finds best walk-forward window by LIVE 2025+ excess return
#    (falls back to historical excess if live metrics not yet available)
# 3. Trains actor on that window's exact training date range
# 4. Winner-takes-all: highest softmax weight = tomorrow's ETF signal
# 5. Saves latest_signal.json and appends to signal_history.json on HF

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
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def load_window_summaries() -> pd.DataFrame:
    """Load all available walk-forward window summaries from HF."""
    frames = []
    for w_id in range(1, 15):
        try:
            path = hf_hub_download(
                repo_id=cfg.HF_DATASET_REPO,
                filename=f"results/window_{w_id:02d}_summary.json",
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
    df = pd.DataFrame(frames)
    df = df.rename(columns={
        'ftrl_total_return': 'ftrl_return',
        'agg_total_return':  'agg_return',
        'ftrl_max_drawdown': 'ftrl_max_dd',
    })
    return df


def find_best_window(summaries: pd.DataFrame) -> dict:
    """Find the best walk-forward window for the live signal."""
    if summaries.empty or 'excess_return' not in summaries.columns:
        print("[Best Window] No summaries — defaulting to full dataset")
        return {
            'window_id':   0,
            'label':       'full dataset',
            'train_start': '2008-01-01',
            'train_end':   date.today().strftime('%Y-%m-%d'),
        }

    has_live = (
        'live_excess_return' in summaries.columns and
        summaries['live_excess_return'].notna().any()
    )

    if has_live:
        pool     = summaries[summaries['live_excess_return'].notna()].copy()
        best_idx = pool['live_excess_return'].idxmax()
        best_row = pool.loc[best_idx]
        metric_val   = float(best_row['live_excess_return'])
        sharpe_val   = float(best_row.get('live_sharpe') or 0)
        n_days       = int(best_row.get('live_n_days') or 0)
        metric_label = f"live_excess={metric_val:.2%} ({n_days} days 2025+)"
        basis        = "live 2025+"
    else:
        best_idx = summaries['excess_return'].idxmax()
        best_row = summaries.loc[best_idx]
        metric_val   = float(best_row['excess_return'])
        sharpe_val   = float(best_row.get('ftrl_sharpe') or 0)
        metric_label = f"hist_excess={metric_val:.2%} (re-run training for live metrics)"
        basis        = f"historical test {int(best_row.get('test_year', '?'))}"

    wid         = int(best_row['window_id'])
    train_end   = str(best_row.get('train_end', f"{int(best_row.get('test_year', 2024))-1}-12-31"))
    train_start = str(best_row.get('train_start', '2008-01-01'))
    label       = f"W{wid:02d}: {train_start[:4]}–{train_end[:4]} (best {basis})"

    print(f"[Best Window] W{wid:02d} — {metric_label}, sharpe={sharpe_val:.3f}")
    print(f"[Best Window] Training on: {train_start} → {train_end}")

    return {
        'window_id':          wid,
        'label':              label,
        'train_start':        train_start,
        'train_end':          train_end,
        'live_excess_return': metric_val if has_live else None,
        'live_sharpe':        sharpe_val if has_live else None,
        'excess_return':      float(best_row.get('excess_return', 0)),
        'ftrl_sharpe':        float(best_row.get('ftrl_sharpe', 0)),
        'basis':              basis,
    }


def score_yesterday(prev_signal: dict, prices: pd.DataFrame,
                    bench: pd.Series) -> dict:
    """
    Score yesterday's signal against actual price returns.
    Returns scored record to append to history.

    FIX: Uses .loc[] instead of .get() — .get() silently returns None
    on any timestamp mismatch (timezone differences, weekend/holiday dates).
    """
    if prev_signal is None:
        return None

    signal_date = prev_signal.get('date')
    signal_etf  = prev_signal.get('signal')

    if not signal_date or not signal_etf:
        return None

    try:
        signal_dt   = pd.Timestamp(signal_date)
        future_days = prices.index[prices.index > signal_dt]

        if len(future_days) == 0:
            print(f"[Score] No trading day after {signal_date} yet — skipping")
            return None

        next_day = future_days[0]

        # If signal_dt is not in the price index (weekend/holiday),
        # fall back to the closest prior trading day
        if signal_dt not in prices.index:
            prior_days = prices.index[prices.index <= signal_dt]
            if len(prior_days) == 0:
                print(f"[Score] {signal_dt} before price history — skipping")
                return None
            signal_dt = prior_days[-1]
            print(f"[Score] Adjusted to nearest trading day: {signal_dt.date()}")

        if signal_etf not in prices.columns:
            print(f"[Score] ETF {signal_etf} not in price columns — skipping")
            return None

        etf_prev = float(prices[signal_etf].loc[signal_dt])
        etf_next = float(prices[signal_etf].loc[next_day])
        agg_prev = float(bench.loc[signal_dt])
        agg_next = float(bench.loc[next_day])

        if any(np.isnan(v) for v in [etf_prev, etf_next, agg_prev, agg_next]):
            print(f"[Score] NaN prices detected — skipping")
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


# ── Training on best window ────────────────────────────────────────────────────

def train_on_window(best_window: dict, prices: pd.DataFrame) -> tuple:
    """Train DDPG on the best walk-forward window's date range."""
    train_prices = prices[
        (prices.index >= best_window['train_start']) &
        (prices.index <= best_window['train_end'])
    ].copy()

    full_prices = prices[
        prices.index >= best_window['train_start']
    ].copy()

    print(f"\n[Train] {best_window['label']}")
    print(f"[Train] Training set: {len(train_prices)} days "
          f"({train_prices.index[0].date()} → {train_prices.index[-1].date()})")
    print(f"[Train] Inference set: {len(full_prices)} days "
          f"(up to {full_prices.index[-1].date()})")

    train_feat = compute_features(train_prices)
    full_feat  = compute_features(full_prices)

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

    print(f"[Train] matrices={train_mat.shape} returns={train_ret.shape}")

    env = PortfolioEnv(train_mat, train_ret)

    cfg.MAX_EPOCHS = 30

    trainer = DDPGTrainer(window_id=0)
    checkpoint_dir = "/tmp/ftrl_predict"
    trainer.train(env, checkpoint_dir)
    trainer.load_best(checkpoint_dir)
    trainer.actor.eval()

    return trainer, full_feat_norm


# ── Inference ─────────────────────────────────────────────────────────────────

def get_signal(trainer: DDPGTrainer, feat: np.ndarray,
               best_window: dict) -> dict:
    """Run actor on the latest H-day window. Returns winner-takes-all signal."""
    H = cfg.H
    if feat.shape[0] < H:
        raise ValueError(f"Not enough data for lookback: {feat.shape[0]} < {H}")

    latest_window = feat[-H:]
    matrix        = latest_window.transpose(1, 0, 2)
    matrix_t      = torch.FloatTensor(matrix).unsqueeze(0)
    prev_weights  = torch.ones(1, cfg.W) / cfg.W

    with torch.no_grad():
        weights = trainer.actor(matrix_t, prev_weights).squeeze(0).numpy()

    best_idx   = int(np.argmax(weights))
    signal     = cfg.ASSETS[best_idx]
    confidence = float(weights[best_idx])

    raw_weights = {cfg.ASSETS[i]: round(float(weights[i]), 6)
                   for i in range(cfg.W)}

    result = {
        'date':             date.today().strftime('%Y-%m-%d'),
        'signal':           signal,
        'confidence':       round(confidence, 4),
        'raw_weights':      raw_weights,
        'trained_on':       best_window['label'],
        'train_start':      best_window['train_start'],
        'train_end':        best_window['train_end'],
        'best_window_id':   best_window['window_id'],
        'generated_at':     datetime.utcnow().isoformat() + 'Z',
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

    # 1. Load price data
    prices = load_etf_prices()
    bench  = load_benchmark_prices()
    prices, bench = align_dates(prices, bench)

    # 2. Load yesterday's signal and history
    prev_signal    = load_latest_signal()
    signal_history = load_signal_history()

    print(f"\n[History] {len(signal_history)} previous signals loaded")
    if prev_signal:
        print(f"[Yesterday] signal={prev_signal.get('signal')} "
              f"date={prev_signal.get('date')} "
              f"trained_on={prev_signal.get('trained_on', '?')}")

    # 3. Score yesterday's signal
    scored = score_yesterday(prev_signal, prices, bench)
    if scored:
        # Avoid duplicate entries for the same signal date
        existing_dates = {r.get('date') for r in signal_history}
        if scored.get('date') not in existing_dates:
            signal_history.append(scored)
            print(f"[History] Appended scored record. Total: {len(signal_history)}")
        else:
            print(f"[History] Duplicate {scored.get('date')} — skipping")
    else:
        print("[History] No scored record this run — not yet scoreable")

    signal_history = signal_history[-90:]

    # 4. Find best walk-forward window
    summaries   = load_window_summaries()
    best_window = find_best_window(summaries)

    # 5. Train on best window
    trainer, feat = train_on_window(best_window, prices)

    # 6. Generate today's signal
    signal = get_signal(trainer, feat, best_window)

    # 7. Save and push
    os.makedirs("/tmp/ftrl_predict", exist_ok=True)

    signal_path  = "/tmp/ftrl_predict/latest_signal.json"
    history_path = "/tmp/ftrl_predict/signal_history.json"

    with open(signal_path, 'w') as f:
        json.dump(signal, f, indent=2)
    with open(history_path, 'w') as f:
        json.dump(signal_history, f, indent=2)

    push_to_hf(signal_path,  "results/latest_signal.json")
    push_to_hf(history_path, "results/signal_history.json")

    print(f"\n[Done] Signal for {signal['date']}: "
          f"{signal['signal']} ({signal['confidence']:.1%} confidence)")
    print(f"[Done] Trained on: {best_window['label']}")
    print(f"[Done] History: {len(signal_history)} records saved")


if __name__ == "__main__":
    main()
