#!/usr/bin/env python3
"""
report_status.py — run in the report job after all matrix jobs finish.
Reads the per-window outcome files written by train.py/train_reverse.py,
builds a training_status JSON, and pushes it to HF.

Each train job writes /tmp/ftrl_results/window_<N>_outcome.json with:
  {"window": N, "status": "success"|"failed", "error": "...", "trained_date": "YYYY-MM-DD"}

Usage:
    python report_status.py --suffix _fi
    python report_status.py --suffix _equity --reverse
"""
import argparse
import json
import os
import glob
from datetime import date

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix',  required=True)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()

    suffix    = args.suffix
    today     = date.today().isoformat()
    prefix    = 'reverse_' if args.reverse else ''
    results_dir = '/tmp/ftrl_results' if not args.reverse else '/tmp/ftrl_reverse_results'

    succeeded = []
    failed    = []
    errors    = {}

    for w_id in range(1, 15):
        outcome_path = os.path.join(results_dir,
                                    f"{prefix}window_{w_id:02d}{suffix}_outcome.json")
        if os.path.exists(outcome_path):
            with open(outcome_path) as f:
                data = json.load(f)
            if data.get('status') == 'success':
                succeeded.append(w_id)
            else:
                failed.append(w_id)
                errors[str(w_id)] = data.get('error', 'Unknown error')
        # If no outcome file exists the job either didn't run or was skipped
        # (already completed) — don't count as failed

    status = {
        'date':        today,
        'suffix':      suffix,
        'is_reverse':  args.reverse,
        'succeeded':   succeeded,
        'failed':      failed,
        'errors':      errors,
        'total_ran':   len(succeeded) + len(failed),
        'all_passed':  len(failed) == 0 and len(succeeded) > 0,
    }

    os.makedirs(results_dir, exist_ok=True)
    status_fname = f"{prefix}training_status{suffix}.json"
    status_path  = os.path.join(results_dir, status_fname)

    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)

    print(f"[Report] Succeeded: {succeeded}")
    print(f"[Report] Failed:    {failed}")
    if errors:
        for w, e in errors.items():
            print(f"[Report] Window {w} error: {e}")

    # Push to HF
    HF_DATASET_REPO = os.environ.get('HF_DATASET_REPO', '')
    HF_TOKEN        = os.environ.get('HF_TOKEN', '')

    if HF_DATASET_REPO:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN if HF_TOKEN else None)
            api.upload_file(
                path_or_fileobj=status_path,
                path_in_repo=f"results/{status_fname}",
                repo_id=HF_DATASET_REPO,
                repo_type='dataset',
            )
            print(f"[Report] Pushed results/{status_fname} to HF")
        except Exception as e:
            print(f"[Report] WARNING: Could not push to HF: {e}")
    else:
        print("[Report] No HF_DATASET_REPO set — skipping push")

if __name__ == '__main__':
    main()
