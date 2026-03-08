# src/aggregate.py — Pull all window results from HF and build master summary

import os
import sys
import json
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files, HfApi
sys.path.insert(0, os.path.dirname(__file__))
import config


def pull_all_summaries() -> list:
    """Download all per-window summary JSONs from HF output repo."""
    results = []
    try:
        files = list(list_repo_files(config.HF_OUTPUT_REPO, repo_type="dataset"))
    except Exception as e:
        print(f"[aggregate] Cannot list HF repo files: {e}")
        return results

    summary_files = [f for f in files if f.startswith("results/summary_window_")]
    print(f"[aggregate] Found {len(summary_files)} window summaries")

    for fname in sorted(summary_files):
        try:
            path = hf_hub_download(
                repo_id=config.HF_OUTPUT_REPO,
                filename=fname,
                repo_type="dataset"
            )
            with open(path) as f:
                data = json.load(f)
            results.append(data['comparison'])
            print(f"  [aggregate] Loaded: {fname}")
        except Exception as e:
            print(f"  [aggregate] Failed {fname}: {e}")

    return results


def build_master_summary(results: list) -> dict:
    """Build aggregate statistics across all completed windows."""
    if not results:
        return {}

    df = pd.DataFrame(results)

    summary = {
        'n_windows_complete':  len(df),
        'n_windows_total':     14,
        'win_rate':            float(df['beats_benchmark'].mean()),
        'wins':                int(df['beats_benchmark'].sum()),
        'avg_ftrl_return':     float(df['ftrl_return'].mean()),
        'avg_agg_return':      float(df['agg_return'].mean()),
        'avg_excess_return':   float(df['excess_return'].mean()),
        'avg_ftrl_sharpe':     float(df['ftrl_sharpe'].mean()),
        'avg_agg_sharpe':      float(df['agg_sharpe'].mean()),
        'avg_ftrl_max_dd':     float(df['ftrl_max_dd'].mean()),
        'avg_agg_max_dd':      float(df['agg_max_dd'].mean()),
        'best_window':         int(df.loc[df['ftrl_return'].idxmax(), 'window_id']),
        'worst_window':        int(df.loc[df['ftrl_return'].idxmin(), 'window_id']),
        'best_year':           str(df.loc[df['ftrl_return'].idxmax(), 'test_year']),
        'worst_year':          str(df.loc[df['ftrl_return'].idxmin(), 'test_year']),
        'windows':             results,
    }

    # Print summary table
    print(f"\n{'='*55}")
    print(f"  FTRL ENGINE — AGGREGATE RESULTS")
    print(f"{'='*55}")
    print(f"  Windows complete:  {summary['n_windows_complete']}/14")
    print(f"  Win rate vs AGG:   {summary['win_rate']*100:.1f}% "
          f"({summary['wins']}/{summary['n_windows_complete']})")
    print(f"  Avg FTRL return:   {summary['avg_ftrl_return']*100:+.2f}%")
    print(f"  Avg AGG return:    {summary['avg_agg_return']*100:+.2f}%")
    print(f"  Avg excess:        {summary['avg_excess_return']*100:+.2f}%")
    print(f"  Avg FTRL Sharpe:   {summary['avg_ftrl_sharpe']:+.3f}")
    print(f"  Best year:         {summary['best_year']}")
    print(f"  Worst year:        {summary['worst_year']}")
    print(f"{'='*55}\n")

    return summary


def push_master_summary(summary: dict, results: list, hf_token: str):
    """Push master summary + all_windows CSV to HF output repo."""
    api      = HfApi()
    repo_id  = os.environ.get('HF_DATASET_REPO', config.HF_OUTPUT_REPO)
    out_dir  = Path('output/results')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Master summary JSON
    summary_path = out_dir / 'master_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # All windows CSV
    if results:
        df       = pd.DataFrame(results)
        csv_path = out_dir / 'all_windows_summary.json'
        with open(csv_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Push both
    for fpath in [summary_path, out_dir / 'all_windows_summary.json']:
        if fpath.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=f"results/{fpath.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token,
                )
                print(f"[aggregate] Pushed: {fpath.name}")
            except Exception as e:
                print(f"[aggregate] Push failed {fpath.name}: {e}")


if __name__ == "__main__":
    results = pull_all_summaries()
    if results:
        summary  = build_master_summary(results)
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            push_master_summary(summary, results, hf_token)
        else:
            print("[aggregate] No HF_TOKEN — results saved locally only")
    else:
        print("[aggregate] No results found yet.")
