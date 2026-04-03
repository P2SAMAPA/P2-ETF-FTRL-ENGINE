#!/usr/bin/env python3
"""
check_completed.py — run in the setup job to determine which windows
already have a summary pushed to HF today for this asset group.
Outputs a GitHub Actions matrix JSON to GITHUB_OUTPUT.

Usage:
    python check_completed.py --suffix _fi --skip_completed true
    python check_completed.py --suffix _equity --skip_completed false
"""
import argparse
import json
import os
import sys
from datetime import date

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix',          required=True)
    parser.add_argument('--skip_completed',  default='false')
    parser.add_argument('--reverse',         action='store_true',
                        help='Check reverse window summaries instead')
    args = parser.parse_args()

    suffix         = args.suffix
    skip_completed = args.skip_completed.lower() == 'true'
    is_reverse     = args.reverse
    today          = date.today().isoformat()

    HF_DATASET_REPO = os.environ.get('HF_DATASET_REPO', '')
    HF_TOKEN        = os.environ.get('HF_TOKEN', '')

    completed = []

    if skip_completed and HF_DATASET_REPO:
        try:
            from huggingface_hub import hf_hub_download
            for w_id in range(1, 15):
                if is_reverse:
                    fname = f"results/reverse_window_{w_id:02d}{suffix}_summary.json"
                else:
                    fname = f"results/window_{w_id:02d}{suffix}_summary.json"
                try:
                    path = hf_hub_download(
                        repo_id=HF_DATASET_REPO,
                        filename=fname,
                        repo_type='dataset',
                        token=HF_TOKEN if HF_TOKEN else None,
                        force_download=True,
                    )
                    with open(path) as f:
                        data = json.load(f)
                    # Check if trained today
                    trained_date = data.get('trained_date', '')
                    if trained_date == today:
                        completed.append(w_id)
                        print(f"[Setup] Window {w_id} already trained today — skipping",
                              file=sys.stderr)
                except Exception:
                    pass  # not found or error = not completed
        except ImportError:
            print("[Setup] huggingface_hub not available — training all windows",
                  file=sys.stderr)

    all_windows = list(range(1, 15))
    pending     = [w for w in all_windows if w not in completed]

    if not pending:
        # All done — use a dummy so matrix doesn't fail with empty list
        pending = []
        print(f"[Setup] All 14 windows already trained today!", file=sys.stderr)

    matrix_json = json.dumps(pending)
    completed_json = json.dumps(completed)

    github_output = os.environ.get('GITHUB_OUTPUT', '')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f"matrix={matrix_json}\n")
            f.write(f"completed={completed_json}\n")
            f.write(f"has_pending={'true' if pending else 'false'}\n")
    else:
        print(f"matrix={matrix_json}")
        print(f"completed={completed_json}")
        print(f"has_pending={'true' if pending else 'false'}")

    print(f"[Setup] Pending windows: {pending}", file=sys.stderr)
    print(f"[Setup] Already done:    {completed}", file=sys.stderr)

if __name__ == '__main__':
    main()
