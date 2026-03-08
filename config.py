# config.py — single source of truth for all parameters

import os

# ── Data ──────────────────────────────────────────────────────────────────────
HF_SOURCE_REPO  = os.environ.get("HF_SOURCE_REPO", "P2SAMAPA/p2-etf-deepwave-dl")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-ftrl-engine")
HF_TOKEN        = os.environ.get("HF_TOKEN", "")

ETF_FILE        = "data/etf_price.parquet"
BENCH_FILE      = "data/bench_price.parquet"

ASSETS          = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV']
BENCHMARK       = 'AGG'
START_DATE      = '2008-01-02'

# ── Feature engineering ───────────────────────────────────────────────────────
W               = 6     # number of assets
H               = 40    # lookback window in trading days
C               = 4     # features per asset per day
VOL_SHORT       = 5
VOL_LONG        = 20
MOM_WINDOW      = 20

# ── Model architecture ────────────────────────────────────────────────────────
ATTENTION_BLOCKS = 3
N_HEADS          = 4
D_MODEL          = 64
D_FF             = 128
DROPOUT          = 0.1

# ── DDPG training ─────────────────────────────────────────────────────────────
LR_ACTOR         = 1e-5
LR_CRITIC        = 1e-4
GAMMA            = 0.99
TAU              = 0.005
BUFFER_SIZE      = 2000
BATCH_SIZE       = 16
MAX_EPOCHS       = 50
EARLY_STOP_PAT   = 10
NOISE_SIGMA      = 0.05
NOISE_DECAY      = 0.995

# ── Portfolio environment ─────────────────────────────────────────────────────
TRANSACTION_COST = 0.001
INITIAL_CAPITAL  = 100.0
RISK_AVERSION    = 0.0

# ── Walk-forward windows ──────────────────────────────────────────────────────
WINDOWS = [
    {'id':  1, 'train_start': '2008-01-02', 'train_end': '2010-12-31', 'test_year': '2011'},
    {'id':  2, 'train_start': '2008-01-02', 'train_end': '2011-12-31', 'test_year': '2012'},
    {'id':  3, 'train_start': '2008-01-02', 'train_end': '2012-12-31', 'test_year': '2013'},
    {'id':  4, 'train_start': '2008-01-02', 'train_end': '2013-12-31', 'test_year': '2014'},
    {'id':  5, 'train_start': '2008-01-02', 'train_end': '2014-12-31', 'test_year': '2015'},
    {'id':  6, 'train_start': '2008-01-02', 'train_end': '2015-12-31', 'test_year': '2016'},
    {'id':  7, 'train_start': '2008-01-02', 'train_end': '2016-12-31', 'test_year': '2017'},
    {'id':  8, 'train_start': '2008-01-02', 'train_end': '2017-12-31', 'test_year': '2018'},
    {'id':  9, 'train_start': '2008-01-02', 'train_end': '2018-12-31', 'test_year': '2019'},
    {'id': 10, 'train_start': '2008-01-02', 'train_end': '2019-12-31', 'test_year': '2020'},
    {'id': 11, 'train_start': '2008-01-02', 'train_end': '2020-12-31', 'test_year': '2021'},
    {'id': 12, 'train_start': '2008-01-02', 'train_end': '2021-12-31', 'test_year': '2022'},
    {'id': 13, 'train_start': '2008-01-02', 'train_end': '2022-12-31', 'test_year': '2023'},
    {'id': 14, 'train_start': '2008-01-02', 'train_end': '2023-12-31', 'test_year': '2024'},
]

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_DIR      = "results"
MODELS_DIR       = "models"
BACKTEST_FILE    = "results/backtest_all_windows.csv"
SUMMARY_FILE     = "results/performance_summary.json"
TRAINING_LOG     = "results/training_log.json"
