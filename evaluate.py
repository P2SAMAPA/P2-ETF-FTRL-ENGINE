# evaluate.py — aggregates all window results and produces final summary
# Run after all training windows complete

import json
import os
import sys
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg


def load_all_results() -> tuple:
    """
    Load all completed window results from HF dataset repo.
    Returns (daily_df, summaries_list)
    """
    daily_frames = []
    summaries    = []

    for w in cfg.WINDOWS:
        wid = w['id']
        try:
            # Daily results
            path = hf_hub_download(
                repo_id=cfg.HF_DATASET_REPO,
                filename=f"results/window_{wid:02d}_daily.csv",
                repo_type="dataset",
                token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
            )
            df = pd.read_csv(path, parse_dates=['date'])
            daily_frames.append(df)

            # Summary
            path = hf_hub_download(
                repo_id=cfg.HF_DATASET_REPO,
                filename=f"results/window_{wid:02d}_summary.json",
                repo_type="dataset",
                token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
            )
            with open(path) as f:
                summaries.append(json.load(f))

            print(f"[eval] Window {wid:02d} loaded ✓")

        except Exception as e:
            print(f"[eval] Window {wid:02d} not yet available: {e}")
            continue

    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    return daily_df, summaries


def compute_aggregate_metrics(summaries: list) -> dict:
    """Compute aggregate performance across all completed windows."""
    if not summaries:
        return {}

    ftrl_rets  = [s['ftrl_total_return']  for s in summaries]
    agg_rets   = [s['agg_total_return']   for s in summaries]
    excess     = [s['excess_return']      for s in summaries]
    sharpes    = [s['ftrl_sharpe']        for s in summaries]
    drawdowns  = [s['ftrl_max_drawdown']  for s in summaries]

    n = len(summaries)
    win_rate = sum(1 for e in excess if e > 0) / n

    return {
        'windows_completed':       n,
        'avg_ftrl_return':         float(np.mean(ftrl_rets)),
        'avg_agg_return':          float(np.mean(agg_rets)),
        'avg_excess_return':       float(np.mean(excess)),
        'avg_sharpe':              float(np.mean(sharpes)),
        'avg_max_drawdown':        float(np.mean(drawdowns)),
        'win_rate_vs_agg':         float(win_rate),
        'best_year':               summaries[int(np.argmax(ftrl_rets))]['test_year'],
        'worst_year':              summaries[int(np.argmin(ftrl_rets))]['test_year'],
        'best_ftrl_return':        float(max(ftrl_rets)),
        'worst_ftrl_return':       float(min(ftrl_rets)),
        'years_tested':            [s['test_year'] for s in summaries],
    }


def build_equity_curve(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build continuous equity curve across all windows.
    Chain portfolio values: each window starts where previous ended.
    """
    if daily_df.empty:
        return pd.DataFrame()

    frames = []
    cumulative_base = cfg.INITIAL_CAPITAL

    for wid in sorted(daily_df['window_id'].unique()):
        w_df = daily_df[daily_df['window_id'] == wid].copy()
        w_df = w_df.sort_values('date')

        # Rescale portfolio values to chain from previous window
        first_val = w_df['portfolio_val'].iloc[0]
        scale     = cumulative_base / cfg.INITIAL_CAPITAL
        w_df['equity_curve'] = w_df['portfolio_val'] * scale

        cumulative_base = w_df['equity_curve'].iloc[-1]
        frames.append(w_df)

    return pd.concat(frames, ignore_index=True)


def push_to_hf(local_path: str, repo_path: str):
    api = HfApi(token=cfg.HF_TOKEN)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
    )
    print(f"[HF] Pushed {repo_path}")


def main():
    print("\n[Evaluate] Loading all window results from HF...")
    daily_df, summaries = load_all_results()

    if not summaries:
        print("[Evaluate] No completed windows found. Run training first.")
        return

    # Aggregate metrics
    agg_metrics = compute_aggregate_metrics(summaries)

    print(f"\n── Aggregate Performance ({agg_metrics['windows_completed']} windows) ──")
    print(f"  Avg FTRL Return:   {agg_metrics['avg_ftrl_return']:.2%}")
    print(f"  Avg AGG Return:    {agg_metrics['avg_agg_return']:.2%}")
    print(f"  Avg Excess:        {agg_metrics['avg_excess_return']:.2%}")
    print(f"  Avg Sharpe:        {agg_metrics['avg_sharpe']:.3f}")
    print(f"  Avg Max Drawdown:  {agg_metrics['avg_max_drawdown']:.2%}")
    print(f"  Win Rate vs AGG:   {agg_metrics['win_rate_vs_agg']:.1%}")
    print(f"  Best Year:         {agg_metrics['best_year']}")
    print(f"  Worst Year:        {agg_metrics['worst_year']}")

    # Per-window table
    print(f"\n── Per-Window Results ──")
    print(f"{'Win':>3} {'Year':>6} {'FTRL':>8} {'AGG':>8} {'Excess':>8} {'Sharpe':>7} {'MDD':>8}")
    print("-" * 55)
    for s in summaries:
        print(f"{s['window_id']:3d} {s['test_year']:>6} "
              f"{s['ftrl_total_return']:8.2%} "
              f"{s['agg_total_return']:8.2%} "
              f"{s['excess_return']:8.2%} "
              f"{s['ftrl_sharpe']:7.3f} "
              f"{s['ftrl_max_drawdown']:8.2%}")

    # Equity curve
    equity_df = build_equity_curve(daily_df)

    # Save and push
    os.makedirs("/tmp/ftrl_eval", exist_ok=True)

    summary_path = "/tmp/ftrl_eval/performance_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'aggregate':  agg_metrics,
            'per_window': summaries,
        }, f, indent=2)

    backtest_path = "/tmp/ftrl_eval/backtest_all_windows.csv"
    daily_df.to_csv(backtest_path, index=False)

    equity_path = "/tmp/ftrl_eval/equity_curve.csv"
    if not equity_df.empty:
        equity_df.to_csv(equity_path, index=False)
        push_to_hf(equity_path,   "results/equity_curve.csv")

    push_to_hf(summary_path,  "results/performance_summary.json")
    push_to_hf(backtest_path, "results/backtest_all_windows.csv")

    print("\n[Evaluate] Complete. All results pushed to HF.")


if __name__ == "__main__":
    main()
