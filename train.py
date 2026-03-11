# train.py — GitHub Actions entry point for one walk-forward window
# Usage: python src/train.py --window <1-14>
#
# Each window:
#   1. Trains on historical window (e.g. 2008–2015)
#   2. Backtests on historical test year (e.g. 2016) → excess_return
#   3. Evaluates same trained model on live 2025+ data → live_excess_return
#
# predict.py picks best window by live_excess_return, making expanding windows
# directly comparable to reverse windows (both optimised on live 2025+ data).

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg
from dataset import load_etf_prices, load_benchmark_prices, get_window_data, align_dates
from features import compute_features, build_price_matrices, normalise_features
from environment import PortfolioEnv
from ddpg import DDPGTrainer

LIVE_START = '2025-01-01'   # live evaluation period — same as reverse windows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, required=True,
                        help='Walk-forward window id (1-14)')
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


def sharpe(rets, rf=0.0):
    excess = rets - rf / 252
    return float((excess.mean() / (excess.std() + 1e-8)) * np.sqrt(252))


def max_drawdown(vals):
    peak = np.maximum.accumulate(vals)
    dd   = (vals - peak) / (peak + 1e-8)
    return float(dd.min())


def run_backtest(trainer, prices, bench, feat_mean, feat_std,
                 start_date, end_date, label, wid):
    """
    Evaluate a trained model on any date range.
    Uses the supplied normalisation stats (always from training set).
    Returns dict of performance metrics + daily rows list.
    """
    eval_prices = prices[
        (prices.index >= start_date) &
        (prices.index <= end_date)
    ].copy()

    eval_bench = bench[
        (bench.index >= start_date) &
        (bench.index <= end_date)
    ]

    if len(eval_prices) < cfg.H + 5:
        print(f"[{label}] Not enough data ({len(eval_prices)} days) — skipping")
        return None, []

    eval_feat = compute_features(eval_prices)
    eval_feat = ((eval_feat - feat_mean) / feat_std).astype(np.float32)

    eval_mat = build_price_matrices(eval_feat)
    eval_ret = PortfolioEnv.compute_daily_returns(eval_prices)

    n = min(len(eval_mat), len(eval_ret))
    eval_mat = eval_mat[:n]
    eval_ret = eval_ret[:n]

    eval_env  = PortfolioEnv(eval_mat, eval_ret)
    state     = eval_env.reset()
    done      = False
    daily_rows = []

    eval_dates = eval_prices.index[cfg.H:]
    step = 0

    while not done:
        mat = torch.FloatTensor(state['matrix']).unsqueeze(0)
        wts = torch.FloatTensor(state['weights']).unsqueeze(0)

        with torch.no_grad():
            action = trainer.actor(mat, wts).squeeze(0).numpy()

        next_state, reward, done, info = eval_env.step(action)

        row = {
            'date':          eval_dates[step].strftime('%Y-%m-%d') if step < len(eval_dates) else 'unknown',
            'window_id':     wid,
            'portfolio_val': info['portfolio_val'],
            'port_return':   info['port_return'],
            'net_return':    info['net_return'],
            'reward':        reward,
        }
        for i, asset in enumerate(cfg.ASSETS):
            row[f'w_{asset}'] = float(action[i])

        daily_rows.append(row)
        state = next_state
        step += 1

    # Metrics
    port_vals       = np.array(eval_env.portfolio_history)
    port_daily_rets = np.diff(port_vals) / port_vals[:-1]
    ftrl_return     = float(port_vals[-1] / cfg.INITIAL_CAPITAL - 1)

    if len(eval_bench) > 1:
        bench_return = float(eval_bench.iloc[-1] / eval_bench.iloc[0] - 1)
    else:
        bench_return = 0.0

    metrics = {
        'ftrl_total_return': ftrl_return,
        'agg_total_return':  bench_return,
        'excess_return':     ftrl_return - bench_return,
        'ftrl_sharpe':       sharpe(port_daily_rets),
        'ftrl_max_drawdown': max_drawdown(port_vals),
        'n_days':            len(daily_rows),
    }

    print(f"[{label}] FTRL={ftrl_return:.2%} AGG={bench_return:.2%} "
          f"Excess={metrics['excess_return']:.2%} "
          f"Sharpe={metrics['ftrl_sharpe']:.3f}")

    return metrics, daily_rows


def main():
    args  = parse_args()
    wid   = args.window
    assert 1 <= wid <= 14, "Window must be 1-14"

    window = next(w for w in cfg.WINDOWS if w['id'] == wid)
    print(f"\n{'='*60}")
    print(f"FTRL Training — Window {wid:02d}")
    print(f"Train: {window['train_start']} → {window['train_end']}")
    print(f"Test:  {window['test_year']}  +  Live {LIVE_START}→today")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    prices = load_etf_prices()
    bench  = load_benchmark_prices()
    prices, bench = align_dates(prices, bench)

    train_prices, test_prices = get_window_data(window, prices)

    # ── 2. Features + normalisation ───────────────────────────────────────────
    train_feat = compute_features(train_prices)
    test_feat  = compute_features(test_prices)

    # Keep raw training stats for reuse in live evaluation
    feat_mean = train_feat.mean(axis=(0, 2), keepdims=True)
    feat_std  = train_feat.std(axis=(0, 2),  keepdims=True)
    feat_std  = np.where(feat_std < 1e-8, 1.0, feat_std)

    train_feat = ((train_feat - feat_mean) / feat_std).astype(np.float32)
    test_feat  = ((test_feat  - feat_mean) / feat_std).astype(np.float32)

    train_mat = build_price_matrices(train_feat)
    test_mat  = build_price_matrices(test_feat)

    train_ret = PortfolioEnv.compute_daily_returns(train_prices)
    test_ret  = PortfolioEnv.compute_daily_returns(test_prices)

    n_train = min(len(train_mat), len(train_ret))
    n_test  = min(len(test_mat),  len(test_ret))
    train_mat = train_mat[:n_train];  train_ret = train_ret[:n_train]
    test_mat  = test_mat[:n_test];    test_ret  = test_ret[:n_test]

    print(f"[train] matrices={train_mat.shape} returns={train_ret.shape}")
    print(f"[test]  matrices={test_mat.shape}  returns={test_ret.shape}")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    train_env       = PortfolioEnv(train_mat, train_ret)
    local_model_dir = "/tmp/ftrl_models"
    trainer         = DDPGTrainer(window_id=wid)
    train_log       = trainer.train(train_env, local_model_dir)
    trainer.load_best(local_model_dir)
    trainer.actor.eval()

    # ── 4. Historical backtest (original test year) ───────────────────────────
    print(f"\n[Backtest] Historical test year {window['test_year']}...")
    hist_metrics, hist_daily = run_backtest(
        trainer, prices, bench, feat_mean, feat_std,
        start_date = f"{window['test_year']}-01-01",
        end_date   = f"{window['test_year']}-12-31",
        label      = f"Hist {window['test_year']}",
        wid        = wid,
    )

    # ── 5. Live backtest (2025 → today) ───────────────────────────────────────
    print(f"\n[Backtest] Live period {LIVE_START} → today...")
    live_metrics, live_daily = run_backtest(
        trainer, prices, bench, feat_mean, feat_std,
        start_date = LIVE_START,
        end_date   = prices.index[-1].strftime('%Y-%m-%d'),
        label      = f"Live {LIVE_START}+",
        wid        = wid,
    )

    # ── 6. Build summary ──────────────────────────────────────────────────────
    summary = {
        'window_id':         wid,
        'test_year':         window['test_year'],
        'train_start':       window['train_start'],
        'train_end':         window['train_end'],

        # Historical test year metrics (used by Overview tab charts)
        'ftrl_total_return': hist_metrics['ftrl_total_return'] if hist_metrics else 0.0,
        'agg_total_return':  hist_metrics['agg_total_return']  if hist_metrics else 0.0,
        'excess_return':     hist_metrics['excess_return']      if hist_metrics else 0.0,
        'ftrl_sharpe':       hist_metrics['ftrl_sharpe']        if hist_metrics else 0.0,
        'ftrl_max_drawdown': hist_metrics['ftrl_max_drawdown']  if hist_metrics else 0.0,

        # Live 2025+ metrics (used by predict.py to pick best window)
        'live_ftrl_return':  live_metrics['ftrl_total_return'] if live_metrics else None,
        'live_agg_return':   live_metrics['agg_total_return']  if live_metrics else None,
        'live_excess_return':live_metrics['excess_return']      if live_metrics else None,
        'live_sharpe':       live_metrics['ftrl_sharpe']        if live_metrics else None,
        'live_max_drawdown': live_metrics['ftrl_max_drawdown']  if live_metrics else None,
        'live_n_days':       live_metrics['n_days']             if live_metrics else 0,
        'live_start':        LIVE_START,

        'best_train_epoch':  train_log['best_epoch'],
        'best_train_return': train_log['best_return'],
    }

    print(f"\n── Window {wid:02d} Summary ──")
    print(f"  Historical ({window['test_year']}):")
    print(f"    FTRL={summary['ftrl_total_return']:.2%}  "
          f"AGG={summary['agg_total_return']:.2%}  "
          f"Excess={summary['excess_return']:.2%}  "
          f"Sharpe={summary['ftrl_sharpe']:.3f}")
    if live_metrics:
        print(f"  Live (2025+, {live_metrics['n_days']} days):")
        print(f"    FTRL={summary['live_ftrl_return']:.2%}  "
              f"AGG={summary['live_agg_return']:.2%}  "
              f"Excess={summary['live_excess_return']:.2%}  "
              f"Sharpe={summary['live_sharpe']:.3f}")
    else:
        print(f"  Live (2025+): insufficient data")

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    os.makedirs("/tmp/ftrl_results", exist_ok=True)

    # Historical daily CSV (used by dashboard equity curves)
    if hist_daily:
        hist_df = pd.DataFrame(hist_daily)
        hist_df['test_year'] = window['test_year']
        daily_path = f"/tmp/ftrl_results/window_{wid:02d}_daily.csv"
        hist_df.to_csv(daily_path, index=False)
        push_to_hf(daily_path, f"results/window_{wid:02d}_daily.csv")

    # Live daily CSV (separate file for potential future dashboard use)
    if live_daily:
        live_df   = pd.DataFrame(live_daily)
        live_path = f"/tmp/ftrl_results/window_{wid:02d}_live_daily.csv"
        live_df.to_csv(live_path, index=False)
        push_to_hf(live_path, f"results/window_{wid:02d}_live_daily.csv")

    # Summary JSON
    summary_path = f"/tmp/ftrl_results/window_{wid:02d}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Training log
    log_path = f"/tmp/ftrl_results/window_{wid:02d}_training_log.json"
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)

    # Model checkpoint
    best_model_path = os.path.join(local_model_dir, f"window_{wid:02d}_best.pt")

    push_to_hf(summary_path,    f"results/window_{wid:02d}_summary.json")
    push_to_hf(log_path,        f"results/window_{wid:02d}_training_log.json")
    push_to_hf(best_model_path, f"models/window_{wid:02d}_best.pt")

    print(f"\n[Done] Window {wid:02d} complete. All outputs pushed to HF.")


if __name__ == "__main__":
    main()
