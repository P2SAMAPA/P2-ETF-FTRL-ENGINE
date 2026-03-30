# features.py — computes 4 features from price-only data
# Input:  price DataFrame (Date x Assets)
# Output: feature tensor (T x C x W) ready for price matrix construction

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config as cfg


def compute_features(prices: pd.DataFrame) -> np.ndarray:
    """
    Compute 4 features for each asset from price series.

    F1: Daily return       = (p_t / p_{t-1}) - 1
    F2: Rolling 5d vol     = std of last 5 daily returns (short-term noise)
    F3: Vol ratio          = 5d_vol / 20d_vol  (regime: >1 = rising vol)
    F4: Momentum           = (p_t / p_{t-20}) - 1  (trend signal)

    Args:
        prices : DataFrame shape (T, W) — adjusted close prices

    Returns:
        features : ndarray shape (T, C, W)
                   T = trading days
                   C = 4 features
                   W = number of assets (now 18)
    """
    T, W = prices.shape
    assert W == cfg.W, f"Expected {cfg.W} assets, got {W}"

    # F1: Daily returns
    ret = prices.pct_change().fillna(0.0)                    # (T, W)

    # F2: Rolling short vol (5d)
    vol_short = ret.rolling(cfg.VOL_SHORT).std().fillna(0.0) # (T, W)

    # F3: Vol ratio = short vol / long vol — clipped to avoid div/0
    vol_long  = ret.rolling(cfg.VOL_LONG).std().fillna(0.0)  # (T, W)
    vol_ratio = (vol_short / vol_long.replace(0, np.nan)
                 ).fillna(1.0).clip(0.0, 5.0)                # (T, W)

    # F4: Momentum (20d)
    mom = prices.pct_change(cfg.MOM_WINDOW).fillna(0.0)      # (T, W)

    # Stack into (T, C, W)
    features = np.stack([
        ret.values,        # F1
        vol_short.values,  # F2
        vol_ratio.values,  # F3
        mom.values,        # F4
    ], axis=1)             # → (T, 4, W)

    # Clip extreme values — financial data has fat tails
    features = np.clip(features, -5.0, 5.0)

    return features.astype(np.float32)


def build_price_matrices(features: np.ndarray,
                         lookback: int = cfg.H) -> np.ndarray:
    """
    Build sliding window price matrices for training.

    Args:
        features : ndarray (T, C, W)
        lookback : H — window size in days

    Returns:
        matrices : ndarray (N, C, H, W)
                   N = number of valid windows = T - H
    """
    T, C, W = features.shape
    N = T - lookback
    assert N > 0, f"Not enough data: T={T}, lookback={lookback}"

    matrices = np.zeros((N, C, lookback, W), dtype=np.float32)
    for i in range(N):
        matrices[i] = features[i: i + lookback].transpose(1, 0, 2)
        # transpose: (H, C, W) → (C, H, W)

    return matrices


def get_return_series(prices: pd.DataFrame) -> pd.Series:
    """
    Compute daily log returns of equal-weight portfolio.
    Used as a sanity check during training.
    """
    ret = np.log(prices / prices.shift(1)).dropna()
    return ret.mean(axis=1)  # equal weight


def normalise_features(train_feat: np.ndarray,
                       test_feat: np.ndarray) -> tuple:
    """
    Z-score normalise features using TRAIN statistics only.
    Prevents lookahead bias.

    Args:
        train_feat : (T_train, C, W)
        test_feat  : (T_test,  C, W)

    Returns:
        Normalised train and test arrays, same shape
    """
    # Compute mean/std per feature channel across time and assets
    mean = train_feat.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
    std  = train_feat.std(axis=(0, 2), keepdims=True)    # (1, C, 1)
    std  = np.where(std < 1e-8, 1.0, std)               # avoid div/0

    train_norm = (train_feat - mean) / std
    test_norm  = (test_feat  - mean) / std               # use TRAIN stats

    return train_norm.astype(np.float32), test_norm.astype(np.float32)


if __name__ == "__main__":
    # Quick validation
    import sys
    sys.path.insert(0, '..')
    from dataset import load_etf_prices

    prices = load_etf_prices()
    print(f"Prices shape: {prices.shape}")

    feat = compute_features(prices)
    print(f"Features shape: {feat.shape}  — expected (T, 4, {cfg.W})")

    mats = build_price_matrices(feat)
    print(f"Price matrices shape: {mats.shape}  — expected (T-{cfg.H}, 4, {cfg.H}, {cfg.W})")

    print("\n── Feature stats (should be small, centred near 0) ──")
    for i, name in enumerate(['Return', 'Vol5d', 'VolRatio', 'Mom20d']):
        f = feat[:, i, :]
        print(f"  {name:12s}: mean={f.mean():.5f}  std={f.std():.5f}  "
              f"min={f.min():.4f}  max={f.max():.4f}")
