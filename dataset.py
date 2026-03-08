# dataset.py — loads data from HF source repo, returns clean DataFrames

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg

def load_etf_prices() -> pd.DataFrame:
    """
    Load ETF prices from HF source repo.
    Returns DataFrame indexed by Date, columns = ASSETS (6 ETFs).
    Total return adjusted prices confirmed from data inspection.
    """
    path = hf_hub_download(
        repo_id=cfg.HF_SOURCE_REPO,
        filename=cfg.ETF_FILE,
        repo_type="dataset",
        token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
    )
    df = pd.read_parquet(path)

    # Normalise date index
    if 'Date' in df.columns:
        df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'

    # Keep only our 6 assets — drop VCIT and anything else
    df = df[cfg.ASSETS].copy()

    # Filter from START_DATE
    df = df[df.index >= cfg.START_DATE]

    # Sort chronologically
    df = df.sort_index()

    # Forward-fill any isolated missing days (holidays misaligned)
    df = df.ffill()

    # Drop any remaining NaN rows (start of series)
    df = df.dropna()

    print(f"[dataset] ETF prices: {df.shape} | "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_benchmark_prices() -> pd.Series:
    """
    Load AGG benchmark prices from HF source repo.
    Returns Series indexed by Date.
    """
    path = hf_hub_download(
        repo_id=cfg.HF_SOURCE_REPO,
        filename=cfg.BENCH_FILE,
        repo_type="dataset",
        token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
    )
    df = pd.read_parquet(path)

    if 'Date' in df.columns:
        df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'

    agg = df[cfg.BENCHMARK].copy()
    agg = agg[agg.index >= cfg.START_DATE]
    agg = agg.sort_index().ffill().dropna()

    print(f"[dataset] AGG benchmark: {len(agg)} rows | "
          f"{agg.index[0].date()} → {agg.index[-1].date()}")
    return agg


def get_window_data(window: dict, prices: pd.DataFrame):
    """
    Split prices into train / test for a given walk-forward window.

    Args:
        window : dict with keys train_start, train_end, test_year
        prices : full price DataFrame from load_etf_prices()

    Returns:
        train_prices : DataFrame
        test_prices  : DataFrame
    """
    train = prices[
        (prices.index >= window['train_start']) &
        (prices.index <= window['train_end'])
    ].copy()

    test_start = f"{window['test_year']}-01-01"
    test_end   = f"{window['test_year']}-12-31"
    test = prices[
        (prices.index >= test_start) &
        (prices.index <= test_end)
    ].copy()

    print(f"[dataset] Window {window['id']:02d} | "
          f"Train: {len(train)} days | Test: {len(test)} days")
    return train, test


def align_dates(etf_prices: pd.DataFrame,
                bench_prices: pd.Series) -> tuple:
    """
    Align ETF and benchmark to same trading dates.
    """
    common = etf_prices.index.intersection(bench_prices.index)
    return etf_prices.loc[common], bench_prices.loc[common]


if __name__ == "__main__":
    # Quick validation run
    prices = load_etf_prices()
    bench  = load_benchmark_prices()
    prices, bench = align_dates(prices, bench)

    print("\n── Price sanity check ──")
    print(prices.head(3))
    print(f"\nTLT first price (2008-01-02): {prices['TLT'].iloc[0]:.4f}")
    print(f"Expected range: 50-60 (total return adjusted)")
    print(f"\nAGG first price (2008-01-02): {bench.iloc[0]:.4f}")
    print(f"Expected range: 55-65 (total return adjusted)")

    print("\n── Null check ──")
    print(prices.isnull().sum())

    print("\n── Window split test ──")
    w = cfg.WINDOWS[9]  # Window 10: train to 2019, test 2020 (COVID)
    train, test = get_window_data(w, prices)
    print(f"Window 10 train tail: {train.index[-1].date()}")
    print(f"Window 10 test head:  {test.index[0].date()}")
