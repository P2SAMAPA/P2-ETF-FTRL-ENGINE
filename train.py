# train.py — GitHub Actions entry point for one walk-forward window
# Usage: python train.py --window <1-14>
# The ASSET_GROUP environment variable must be set to "FI" or "EQUITY".

import argparse
import json
import os
import shutil
import sys
import traceback
import numpy as np
import pandas as pd
import torch
from datetime import date
from huggingface_hub import HfApi

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg
from dataset import load_etf_prices, load_benchmark_prices, get_window_data, align_dates
from features import compute_features, build_price_matrices, normalise_features
from environment import PortfolioEnv
from ddpg import DDPGTrainer

LIVE_START = '2025-01-01'


def json_safe(v):
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, required=True)
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


def write_outcome(wid: int, status: str, error: str = ""):
    """Write a per-window outcome file so the report job can collect results."""
    os.makedirs("/tmp/ftrl_results", exist_ok=True)
    outcome = {
        "window":       wid,
        "status":       status,       # "success" or "failed"
        "error":        error,
        "trained_date": date.today().isoformat(),
        "asset_group":  cfg.ASSET_GROUP,
    }
    path = f"/tmp/ftrl_results/window_{wid:02d}{cfg.OUTPUT_SUFFIX}_outcome.json"
    with open(path, 'w') as f:
        json.dump(outcome, f, indent=2)
    print(f"[Outcome] Written {status} → {path}")


def sharpe(rets, rf=0.0):
    excess = rets - rf / 252
    return float((excess.mean() / (excess.std() + 1e-8)) * np.sqrt(252))


def max_drawdown(vals):
    peak = np.maximum.accumulate(vals)
    dd   = (vals - peak) / (peak + 1e-8)
    return float(dd.min())


def run_backtest(trainer, prices, bench, feat_mean, feat_std,
                 start_date, end_date, label, wid):
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

    eval_mat = build_price_matrices(eval_feat, cfg.H)
    eval_ret = PortfolioEnv.compute_daily_returns(eval_prices)

    N         = len(eval_mat)
    ret_start = cfg.H - 1
    eval_ret  = eval_ret[ret_start: ret_start + N]

    n = min(len(eval_mat), len(eval_ret))
    eval_mat = eval_mat[:n]
    eval_ret = eval_ret[:n]

    eval_env   = PortfolioEnv(eval_mat, eval_ret)
    state      = eval_env.reset()
    done       = False
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


def train_window(wid: int):
    """Full training pipeline for one window. Raises on failure."""
    window = next(w for w in cfg.WINDOWS if w['id'] == wid)
    print(f"\n{'='*60}")
    print(f"FTRL Training — {cfg.ASSET_GROUP} group — Window {wid:02d}")
    print(f"Train: {window['train_start']} → {window['train_end']}")
    print(f"Test:  {window['test_year']}  +  Live {LIVE_START}→today")
    print(f"{'='*60}\n")

    prices = load_etf_prices()
    bench  = load_benchmark_prices()
    prices, bench = align_dates(prices, bench)

    train_prices, test_prices = get_window_data(window, prices)

    train_feat = compute_features(train_prices)
    test_feat  = compute_features(test_prices)

    feat_mean = train_feat.mean(axis=(0, 2), keepdims=True)
    feat_std  = train_feat.std(axis=(0, 2),  keepdims=True)
    feat_std  = np.where(feat_std < 1e-8, 1.0, feat_std)

    train_feat = ((train_feat - feat_mean) / feat_std).astype(np.float32)
    test_feat  = ((test_feat  - feat_mean) / feat_std).astype(np.float32)

    train_mat = build_price_matrices(train_feat, cfg.H)
    test_mat  = build_price_matrices(test_feat,  cfg.H)

    train_ret = PortfolioEnv.compute_daily_returns(train_prices)
    test_ret  = PortfolioEnv.compute_daily_returns(test_prices)

    n_train   = len(train_mat)
    n_test    = len(test_mat)
    train_ret = train_ret[cfg.H - 1: cfg.H - 1 + n_train]
    test_ret  = test_ret[cfg.H - 1:  cfg.H - 1 + n_test]

    n_train = min(len(train_mat), len(train_ret))
    n_test  = min(len(test_mat),  len(test_ret))
    train_mat = train_mat[:n_train];  train_ret = train_ret[:n_train]
    test_mat  = test_mat[:n_test];    test_ret  = test_ret[:n_test]

    print(f"[train] matrices={train_mat.shape} returns={train_ret.shape}")
    print(f"[test]  matrices={test_mat.shape}  returns={test_ret.shape}")

    train_env       = PortfolioEnv(train_mat, train_ret)
    local_model_dir = "/tmp/ftrl_models"
    os.makedirs(local_model_dir, exist_ok=True)

    trainer   = DDPGTrainer(window_id=wid)
    train_log = trainer.train(train_env, local_model_dir)
    trainer.load_best(local_model_dir)
    trainer.actor.eval()

    raw_ckpt      = os.path.join(local_model_dir, f"window_{wid:02d}_best.pt")
    suffixed_ckpt = os.path.join(local_model_dir, f"window_{wid:02d}{cfg.OUTPUT_SUFFIX}_best.pt")
    if raw_ckpt != suffixed_ckpt:
        shutil.copy2(raw_ckpt, suffixed_ckpt)

    print(f"\n[Backtest] Historical test year {window['test_year']}...")
    hist_metrics, hist_daily = run_backtest(
        trainer, prices, bench, feat_mean, feat_std,
        start_date = f"{window['test_year']}-01-01",
        end_date   = f"{window['test_year']}-12-31",
        label      = f"Hist {window['test_year']}",
        wid        = wid,
    )

    print(f"\n[Backtest] Live period {LIVE_START} → today...")
    live_metrics, live_daily = run_backtest(
        trainer, prices, bench, feat_mean, feat_std,
        start_date = LIVE_START,
        end_date   = prices.index[-1].strftime('%Y-%m-%d'),
        label      = f"Live {LIVE_START}+",
        wid        = wid,
    )

    def hm(key, default=0.0):
        return json_safe(hist_metrics[key]) if hist_metrics else default

    def lm(key, default=None):
        return json_safe(live_metrics[key]) if live_metrics else default

    summary = {
        'window_id':         wid,
        'test_year':         window['test_year'],
        'train_start':       window['train_start'],
        'train_end':         window['train_end'],
        'trained_date':      date.today().isoformat(),   # ← NEW: for skip_completed check

        'ftrl_total_return': hm('ftrl_total_return'),
        'agg_total_return':  hm('agg_total_return'),
        'excess_return':     hm('excess_return'),
        'ftrl_sharpe':       hm('ftrl_sharpe'),
        'ftrl_max_drawdown': hm('ftrl_max_drawdown'),

        'live_ftrl_return':   lm('ftrl_total_return'),
        'live_agg_return':    lm('agg_total_return'),
        'live_excess_return': lm('excess_return'),
        'live_sharpe':        lm('ftrl_sharpe'),
        'live_max_drawdown':  lm('ftrl_max_drawdown'),
        'live_n_days':        int(live_metrics['n_days']) if live_metrics else 0,
        'live_start':         LIVE_START,

        'best_train_epoch':  int(train_log['best_epoch']),
        'best_train_return': float(train_log['best_return']),
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

    os.makedirs("/tmp/ftrl_results", exist_ok=True)

    if hist_daily:
        hist_df = pd.DataFrame(hist_daily)
        hist_df['test_year'] = window['test_year']
        daily_path = f"/tmp/ftrl_results/window_{wid:02d}{cfg.OUTPUT_SUFFIX}_daily.csv"
        hist_df.to_csv(daily_path, index=False)
        push_to_hf(daily_path, f"results/window_{wid:02d}{cfg.OUTPUT_SUFFIX}_daily.csv")

    if live_daily:
        live_df   = pd.DataFrame(live_daily)
        live_path = f"/tmp/ftrl_results/window_{wid:02d}{cfg.OUTPUT_SUFFIX}_live_daily.csv"
        live_df.to_csv(live_path, index=False)
        push_to_hf(live_path, f"results/window_{wid:02d}{cfg.OUTPUT_SUFFIX}_live_daily.csv")

    summary_path = f"/tmp/ftrl_results/window_{wid:02d}{cfg.OUTPUT_SUFFIX}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    push_to_hf(summary_path, f"results/window_{wid:02d}{cfg.OUTPUT_SUFFIX}_summary.json")

    log_path = f"/tmp/ftrl_results/window_{wid:02d}{cfg.OUTPUT_SUFFIX}_training_log.json"
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)
    push_to_hf(log_path, f"results/window_{wid:02d}{cfg.OUTPUT_SUFFIX}_training_log.json")

    push_to_hf(suffixed_ckpt, f"models/window_{wid:02d}{cfg.OUTPUT_SUFFIX}_best.pt")

    print(f"\n[Done] Window {wid:02d} ({cfg.ASSET_GROUP}) complete.")


def main():
    args = parse_args()
    wid  = args.window
    assert 1 <= wid <= 14, "Window must be 1-14"

    try:
        train_window(wid)
        write_outcome(wid, 'success')
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"\n[ERROR] Window {wid} failed:\n{err_msg}", file=sys.stderr)
        write_outcome(wid, 'failed', error=str(e))
        sys.exit(1)   # non-zero exit so GitHub marks this matrix job as failed


if __name__ == "__main__":
    main()
