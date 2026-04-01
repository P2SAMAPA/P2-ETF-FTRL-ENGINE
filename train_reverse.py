# train_reverse.py — Reverse expanding window training entry point
# Phase 2: drops oldest year each window, always tests on 2025+2026YTD
# Usage: python train_reverse.py --window <1-14>
#
# Window R1:  Train 2008–2024 → Test 2025+2026YTD
# Window R2:  Train 2009–2024 → Test 2025+2026YTD
# ...
# Window R14: Train 2021–2024 → Test 2025+2026YTD

import argparse
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg
from dataset import load_etf_prices, load_benchmark_prices, align_dates
from features import compute_features, build_price_matrices, normalise_features
from environment import PortfolioEnv
from ddpg import DDPGTrainer

# ── Reverse window definitions ────────────────────────────────────────────────
# Always test on 2025-01-01 → today
# Train start drops one year per window: 2008, 2009, ..., 2021
# Train end is always 2024-12-31

REVERSE_WINDOWS = [
    {
        'id':          i + 1,
        'train_start': f"{2008 + i}-01-01",
        'train_end':   '2024-12-31',
        'test_start':  '2025-01-01',
        'test_end':    None,   # None = use all available data up to today
        'label':       f"R{i+1}: {2008+i}–2024 → 2025+",
    }
    for i in range(14)
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, required=True,
                        help='Reverse window id (1-14)')
    return parser.parse_args()


def push_to_hf(local_path: str, repo_path: str):
    api = HfApi(token=cfg.HF_TOKEN)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
    )
    print(f"[HF] Pushed {repo_path}")


def json_safe(v):
    """Convert numpy scalars to plain Python types for json.dump."""
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def sharpe(rets):
    excess = rets - 0.0
    return float((excess.mean() / (excess.std() + 1e-8)) * np.sqrt(252))


def max_drawdown(vals):
    peak = np.maximum.accumulate(vals)
    dd   = (vals - peak) / (peak + 1e-8)
    return float(dd.min())


def main():
    args = parse_args()
    wid  = args.window
    assert 1 <= wid <= 14, "Window must be 1-14"

    window = REVERSE_WINDOWS[wid - 1]
    print(f"\n{'='*60}")
    print(f"FTRL Reverse Training — {cfg.ASSET_GROUP} group — Window R{wid:02d}")
    print(f"Train: {window['train_start']} → {window['train_end']}")
    print(f"Test:  {window['test_start']} → present")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    prices = load_etf_prices()
    bench  = load_benchmark_prices()
    prices, bench = align_dates(prices, bench)

    # ── 2. Split train / test ─────────────────────────────────────────────────
    train_prices = prices[
        (prices.index >= window['train_start']) &
        (prices.index <= window['train_end'])
    ].copy()

    test_prices = prices[
        prices.index >= window['test_start']
    ].copy()

    test_bench = bench[
        bench.index >= window['test_start']
    ].copy()

    print(f"[dataset] Train: {len(train_prices)} days | "
          f"Test: {len(test_prices)} days "
          f"({test_prices.index[0].date()} → {test_prices.index[-1].date()})")

    if len(train_prices) < cfg.H + 50:
        print(f"[Error] Not enough training data: {len(train_prices)} days")
        return

    has_test = len(test_prices) >= cfg.H + 5
    if not has_test:
        print(f"[Warning] Very little test data: {len(test_prices)} days — skipping backtest")

    # ── 3. Compute features ───────────────────────────────────────────────────
    train_feat_raw = compute_features(train_prices)

    if has_test:
        test_feat_raw = compute_features(test_prices)
        # FIX: normalise_features(train, test) → returns both normalised arrays
        train_feat, test_feat = normalise_features(train_feat_raw, test_feat_raw)
    else:
        mean = train_feat_raw.mean(axis=(0, 2), keepdims=True)
        std  = train_feat_raw.std(axis=(0, 2),  keepdims=True)
        std  = np.where(std < 1e-8, 1.0, std)
        train_feat = ((train_feat_raw - mean) / std).astype(np.float32)
        test_feat  = np.array([])

    # FIX: pass cfg.H explicitly
    train_mat = build_price_matrices(train_feat, cfg.H)
    train_ret = PortfolioEnv.compute_daily_returns(train_prices)

    # FIX: align returns to matrices using H-1 offset (consistent with train.py)
    n_train   = len(train_mat)
    train_ret = train_ret[cfg.H - 1: cfg.H - 1 + n_train]
    n_train   = min(len(train_mat), len(train_ret))
    train_mat = train_mat[:n_train]
    train_ret = train_ret[:n_train]

    print(f"[train] matrices={train_mat.shape} returns={train_ret.shape}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    local_model_dir = f"/tmp/ftrl_reverse_models{cfg.OUTPUT_SUFFIX}"
    os.makedirs(local_model_dir, exist_ok=True)

    train_env = PortfolioEnv(train_mat, train_ret)
    ddpg_wid  = 100 + wid   # offset to avoid collision with forward windows
    trainer   = DDPGTrainer(window_id=ddpg_wid)
    train_log = trainer.train(train_env, local_model_dir)

    # FIX: ddpg.py saves window_{ddpg_wid:02d}_best.pt (no suffix).
    # Copy to the suffixed name so the push path matches.
    raw_ckpt      = os.path.join(local_model_dir, f"window_{ddpg_wid:02d}_best.pt")
    suffixed_ckpt = os.path.join(local_model_dir,
                                 f"window_{ddpg_wid:02d}{cfg.OUTPUT_SUFFIX}_best.pt")
    if os.path.exists(raw_ckpt) and raw_ckpt != suffixed_ckpt:
        shutil.copy2(raw_ckpt, suffixed_ckpt)

    # ── 5. Backtest on test set ───────────────────────────────────────────────
    summary = {
        'window_id':         wid,
        'label':             window['label'],
        'train_start':       window['train_start'],
        'train_end':         window['train_end'],
        'test_start':        window['test_start'],
        'test_end':          test_prices.index[-1].strftime('%Y-%m-%d') if has_test else 'N/A',
        'ftrl_total_return': 0.0,
        'agg_total_return':  0.0,
        'excess_return':     0.0,
        'ftrl_sharpe':       0.0,
        'ftrl_max_drawdown': 0.0,
        # FIX: predict_reverse.py looks for live_excess_return — alias here
        # since the test period IS the live 2025+ period for reverse windows
        'live_excess_return': None,
        'live_sharpe':        None,
        'live_n_days':        0,
        'best_train_epoch':  train_log['best_epoch'],
        'best_train_return': train_log['best_return'],
        'n_test_days':       len(test_prices),
    }

    daily_results = []

    if has_test and len(test_feat) > 0:
        # FIX: pass cfg.H explicitly
        test_mat = build_price_matrices(test_feat, cfg.H)
        test_ret = PortfolioEnv.compute_daily_returns(test_prices)

        # FIX: align returns to matrices using H-1 offset
        n_test   = len(test_mat)
        test_ret = test_ret[cfg.H - 1: cfg.H - 1 + n_test]
        n_test   = min(len(test_mat), len(test_ret))
        test_mat = test_mat[:n_test]
        test_ret = test_ret[:n_test]

        print(f"[test]  matrices={test_mat.shape}  returns={test_ret.shape}")

        test_env   = PortfolioEnv(test_mat, test_ret)
        trainer.load_best(local_model_dir)
        trainer.actor.eval()

        test_state = test_env.reset()
        done       = False
        test_dates = test_prices.index[cfg.H:]
        step       = 0

        while not done:
            mat = torch.FloatTensor(test_state['matrix']).unsqueeze(0)
            wts = torch.FloatTensor(test_state['weights']).unsqueeze(0)

            with torch.no_grad():
                action = trainer.actor(mat, wts).squeeze(0).numpy()

            next_state, reward, done, info = test_env.step(action)

            row = {
                'date':          test_dates[step].strftime('%Y-%m-%d') if step < len(test_dates) else 'unknown',
                'window_id':     wid,
                'train_start':   window['train_start'],
                'portfolio_val': info['portfolio_val'],
                'port_return':   info['port_return'],
                'net_return':    info['net_return'],
                'reward':        reward,
            }
            for i, asset in enumerate(cfg.ASSETS):
                row[f'w_{asset}'] = float(action[i])

            daily_results.append(row)
            test_state = next_state
            step += 1

        # Compute metrics
        port_vals       = np.array(test_env.portfolio_history)
        port_daily_rets = np.diff(port_vals) / port_vals[:-1]

        final_port_return = float(test_env.portfolio_history[-1] / cfg.INITIAL_CAPITAL - 1)

        if len(test_bench) > 1:
            bench_aligned      = test_bench[test_bench.index >= window['test_start']]
            final_bench_return = float(bench_aligned.iloc[-1] / bench_aligned.iloc[0] - 1)
        else:
            final_bench_return = 0.0

        excess    = float(final_port_return) - final_bench_return
        sharpe_v  = sharpe(port_daily_rets)
        max_dd    = max_drawdown(port_vals)

        summary.update({
            'ftrl_total_return':  json_safe(final_port_return),
            'agg_total_return':   json_safe(final_bench_return),
            'excess_return':      json_safe(excess),
            'ftrl_sharpe':        json_safe(sharpe_v),
            'ftrl_max_drawdown':  json_safe(max_dd),
            # FIX: populate live_* fields so predict_reverse.py can select best window
            'live_excess_return': json_safe(excess),
            'live_sharpe':        json_safe(sharpe_v),
            'live_n_days':        int(len(daily_results)),
        })

        print(f"\n── Window R{wid:02d} Results ──")
        print(f"  FTRL Return:  {final_port_return:.2%}")
        print(f"  AGG  Return:  {final_bench_return:.2%}")
        print(f"  Excess:       {excess:.2%}")
        print(f"  Sharpe:       {sharpe_v:.3f}")
        print(f"  Max Drawdown: {max_dd:.2%}")

    # ── 6. Save and push to HF ────────────────────────────────────────────────
    os.makedirs("/tmp/ftrl_reverse_results", exist_ok=True)

    summary_path = f"/tmp/ftrl_reverse_results/reverse_window_{wid:02d}{cfg.OUTPUT_SUFFIX}_summary.json"
    log_path     = f"/tmp/ftrl_reverse_results/reverse_window_{wid:02d}{cfg.OUTPUT_SUFFIX}_training_log.json"

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)

    push_to_hf(summary_path, f"results/reverse_window_{wid:02d}{cfg.OUTPUT_SUFFIX}_summary.json")
    push_to_hf(log_path,     f"results/reverse_window_{wid:02d}{cfg.OUTPUT_SUFFIX}_training_log.json")

    if daily_results:
        daily_df   = pd.DataFrame(daily_results)
        daily_path = f"/tmp/ftrl_reverse_results/reverse_window_{wid:02d}{cfg.OUTPUT_SUFFIX}_daily.csv"
        daily_df.to_csv(daily_path, index=False)
        push_to_hf(daily_path, f"results/reverse_window_{wid:02d}{cfg.OUTPUT_SUFFIX}_daily.csv")

    # FIX: push the suffixed checkpoint copy
    if os.path.exists(suffixed_ckpt):
        push_to_hf(suffixed_ckpt,
                   f"models/reverse_window_{wid:02d}{cfg.OUTPUT_SUFFIX}_best.pt")

    print(f"\n[Done] Reverse Window R{wid:02d} ({cfg.ASSET_GROUP}) complete. All outputs pushed to HF.")


if __name__ == "__main__":
    main()
