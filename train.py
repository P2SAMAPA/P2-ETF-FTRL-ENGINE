# train.py — GitHub Actions entry point for one walk-forward window
# Usage: python src/train.py --window <1-14>

import argparse
import json
import os
import sys
import numpy as np
import torch
from huggingface_hub import HfApi

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg
from dataset import load_etf_prices, load_benchmark_prices, get_window_data, align_dates
from features import compute_features, build_price_matrices, normalise_features
from environment import PortfolioEnv
from ddpg import DDPGTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, required=True,
                        help='Walk-forward window id (1-14)')
    return parser.parse_args()


def push_to_hf(local_path: str, repo_path: str):
    """Push a file to the HF dataset repo."""
    api = HfApi(token=cfg.HF_TOKEN)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
    )
    print(f"[HF] Pushed {repo_path}")


def main():
    args  = parse_args()
    wid   = args.window
    assert 1 <= wid <= 14, "Window must be 1-14"

    window = next(w for w in cfg.WINDOWS if w['id'] == wid)
    print(f"\n{'='*60}")
    print(f"FTRL Training — Window {wid:02d}")
    print(f"Train: {window['train_start']} → {window['train_end']}")
    print(f"Test:  {window['test_year']}")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    prices = load_etf_prices()
    bench  = load_benchmark_prices()
    prices, bench = align_dates(prices, bench)

    train_prices, test_prices = get_window_data(window, prices)

    # ── 2. Compute features ───────────────────────────────────────────────────
    train_feat = compute_features(train_prices)   # (T_train, C, W)
    test_feat  = compute_features(test_prices)    # (T_test,  C, W)

    # Normalise using train statistics (no lookahead)
    train_feat, test_feat = normalise_features(train_feat, test_feat)

    # Build sliding window matrices
    train_mat = build_price_matrices(train_feat)  # (N_train, C, H, W)
    test_mat  = build_price_matrices(test_feat)   # (N_test,  C, H, W)

    # Daily returns for environment reward
    train_ret = PortfolioEnv.compute_daily_returns(train_prices)
    test_ret  = PortfolioEnv.compute_daily_returns(test_prices)

    # Align lengths (matrix needs return for each step)
    n_train = min(len(train_mat), len(train_ret))
    n_test  = min(len(test_mat),  len(test_ret))
    train_mat = train_mat[:n_train]
    train_ret = train_ret[:n_train]
    test_mat  = test_mat[:n_test]
    test_ret  = test_ret[:n_test]

    print(f"[train] matrices={train_mat.shape} returns={train_ret.shape}")
    print(f"[test]  matrices={test_mat.shape}  returns={test_ret.shape}")

    # ── 3. Build environments ─────────────────────────────────────────────────
    train_env = PortfolioEnv(train_mat, train_ret)
    test_env  = PortfolioEnv(test_mat,  test_ret)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    local_model_dir = f"/tmp/ftrl_models"
    trainer = DDPGTrainer(window_id=wid)
    train_log = trainer.train(train_env, local_model_dir)

    # ── 5. Back-test on test set ──────────────────────────────────────────────
    print(f"\n[Backtest] Running on {window['test_year']}...")
    trainer.load_best(local_model_dir)
    trainer.actor.eval()

    test_state = test_env.reset()
    done       = False
    daily_results = []

    test_dates = test_prices.index[cfg.H:]  # offset by lookback

    step = 0
    while not done:
        mat = torch.FloatTensor(test_state['matrix']).unsqueeze(0)
        wts = torch.FloatTensor(test_state['weights']).unsqueeze(0)

        with torch.no_grad():
            action = trainer.actor(mat, wts).squeeze(0).numpy()

        next_state, reward, done, info = test_env.step(action)

        row = {
            'date':          test_dates[step].strftime('%Y-%m-%d') if step < len(test_dates) else 'unknown',
            'window_id':     wid,
            'test_year':     window['test_year'],
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

    # ── 6. Compute benchmark returns ──────────────────────────────────────────
    test_bench = bench[
        (bench.index >= f"{window['test_year']}-01-01") &
        (bench.index <= f"{window['test_year']}-12-31")
    ]
    bench_ret_series = test_bench.pct_change().dropna()

    final_port_return = (
        test_env.portfolio_history[-1] / cfg.INITIAL_CAPITAL - 1
    )
    final_bench_return = float(
        (test_bench.iloc[-1] / test_bench.iloc[0]) - 1
    ) if len(test_bench) > 1 else 0.0

    port_vals = np.array(test_env.portfolio_history)
    port_daily_rets = np.diff(port_vals) / port_vals[:-1]

    def sharpe(rets, rf=0.0):
        excess = rets - rf / 252
        return (excess.mean() / (excess.std() + 1e-8)) * np.sqrt(252)

    def max_drawdown(vals):
        peak = np.maximum.accumulate(vals)
        dd   = (vals - peak) / (peak + 1e-8)
        return float(dd.min())

    summary = {
        'window_id':          wid,
        'test_year':          window['test_year'],
        'train_start':        window['train_start'],
        'train_end':          window['train_end'],
        'ftrl_total_return':  float(final_port_return),
        'agg_total_return':   final_bench_return,
        'excess_return':      float(final_port_return) - final_bench_return,
        'ftrl_sharpe':        float(sharpe(port_daily_rets)),
        'ftrl_max_drawdown':  max_drawdown(port_vals),
        'best_train_epoch':   train_log['best_epoch'],
        'best_train_return':  train_log['best_return'],
    }

    print(f"\n── Window {wid:02d} Results ──")
    print(f"  FTRL Return:  {final_port_return:.2%}")
    print(f"  AGG  Return:  {final_bench_return:.2%}")
    print(f"  Excess:       {summary['excess_return']:.2%}")
    print(f"  Sharpe:       {summary['ftrl_sharpe']:.3f}")
    print(f"  Max Drawdown: {summary['ftrl_max_drawdown']:.2%}")

    # ── 7. Save outputs locally then push to HF ───────────────────────────────
    import pandas as pd
    os.makedirs("/tmp/ftrl_results", exist_ok=True)

    # Daily results CSV
    daily_df   = pd.DataFrame(daily_results)
    daily_path = f"/tmp/ftrl_results/window_{wid:02d}_daily.csv"
    daily_df.to_csv(daily_path, index=False)

    # Summary JSON
    summary_path = f"/tmp/ftrl_results/window_{wid:02d}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Training log JSON
    log_path = f"/tmp/ftrl_results/window_{wid:02d}_training_log.json"
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)

    # Push to HF
    best_model_path = os.path.join(local_model_dir,
                                   f"window_{wid:02d}_best.pt")

    push_to_hf(daily_path,    f"results/window_{wid:02d}_daily.csv")
    push_to_hf(summary_path,  f"results/window_{wid:02d}_summary.json")
    push_to_hf(log_path,      f"results/window_{wid:02d}_training_log.json")
    push_to_hf(best_model_path, f"models/window_{wid:02d}_best.pt")

    print(f"\n[Done] Window {wid:02d} complete. All outputs pushed to HF.")


if __name__ == "__main__":
    main()
