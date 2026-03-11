# P2 ETF FTRL Engine

**Financial Transformer Reinforcement Learning (FTRL) for 6-ETF portfolio management.**

A walk-forward validated, DDPG-based portfolio engine that learns to allocate across 6 fixed-income and real-asset ETFs. Produces a daily actionable signal: one ETF to hold for the next trading day, with a 15-day audit trail of predicted vs actual returns.

---

## Research Foundation

Based on: *"Time Series is Not Enough: Financial Transformer Reinforcement Learning for Portfolio Management"*
Xiaotian Ren, Ruoyu Sun, Zhengyong Jiang, Angelos Stefanidis, Hongbin Liu, Jionglong Su
School of AI and Advanced Computing, XJTLU Entrepreneur College (Taicang), Xi'an Jiaotong-Liverpool University, Suzhou, China. Submitted to Neurocomputing, July 2025.

The paper introduces a Financial Transformer actor that combines two novel attention blocks inside a standard DDPG framework. The key insight is that price time series alone is insufficient — the model must also learn latent inter-asset correlations (the LLB) alongside temporal dynamics (the TRB).

---

## Architecture

```
Price Matrix (B, C, H, W)
        │
        ├──► TRB (Temporal Relationship Block)
        │     ├── Daily patch stream   (patch_size=1, H tokens)
        │     ├── Weekly patch stream  (patch_size=5, H/5 tokens)
        │     ├── Class-token transformer attention
        │     └── Hadamard calibration → (B, W)
        │
        ├──► LLB (Latent Linkage Block)
        │     ├── Segment projection (H → n_segs × seg_len)
        │     ├── Inner attention (intra-segment, short-term dynamics)
        │     ├── Outer attention (inter-asset, ECA channel attention)
        │     └── Class token → (B, D_MODEL)
        │
        └──► Merge → Linear → LayerNorm → Softmax → Portfolio Weights (B, W)
```

| Component | Description |
|-----------|-------------|
| **TRB** | Temporal Relationship Block — dual-stream daily + weekly patch attention with calibration |
| **LLB** | Latent Linkage Block — hierarchical inner/outer token attention with ECA for inter-asset correlation |
| **Actor** | Financial Transformer → softmax portfolio weights summing to 1 |
| **Critic** | CNN state encoder + FC layers → Q-value scalar |
| **Algorithm** | DDPG with Ornstein-Uhlenbeck exploration noise, experience replay, soft target updates |
| **Reward** | Log return minus transaction cost (pure return maximisation, no Sharpe penalty) |

### Hyperparameters (CPU-optimised for GitHub Actions)

```python
ASSETS           = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV']
H                = 40       # lookback days
C                = 4        # features per asset
ATTENTION_BLOCKS = 3        # reduced from paper's 6 for CPU
N_HEADS          = 4
D_MODEL          = 64
BATCH_SIZE       = 16
LR               = 1e-5
GAMMA            = 0.99
TAU              = 0.005
BUFFER_SIZE      = 2000
TRANSACTION_COST = 0.001
MAX_EPOCHS       = 50       # capped at 30 for daily predict runs
EARLY_STOP_PAT   = 10
```

### Features (price-derived, no OHLC required)

```
F1: Daily return   = (p_t / p_{t-1}) - 1
F2: Rolling 5d vol = std of last 5 daily returns
F3: Vol ratio      = 5d_vol / 20d_vol
F4: Momentum       = (p_t / p_{t-20}) - 1
```

---

## Asset Universe

| ETF | Exposure | Role |
|-----|----------|------|
| TLT | Long-duration US Treasuries (20y+) | Rates / duration |
| LQD | Investment grade corporate bonds | Credit quality |
| HYG | High yield corporate bonds | Credit risk |
| VNQ | US Real Estate Investment Trusts | Real assets |
| GLD | Gold | Safe haven |
| SLV | Silver | Real assets / volatility |

**Benchmark:** AGG (US Aggregate Bond Market, total return adjusted)
**Data source:** `P2SAMAPA/p2-etf-deepwave-dl` — total return adjusted prices from 2008-01-02 to present, auto-updated daily.

---

## Infrastructure

```
READ  data from:  P2SAMAPA/p2-etf-deepwave-dl   (HF Dataset, existing live project)
WRITE results to: P2SAMAPA/p2-etf-ftrl-engine    (HF Dataset, this project)
                  └── models/window_XX_best.pt
                  └── results/window_XX_daily.csv
                  └── results/window_XX_live_daily.csv       ← NEW
                  └── results/window_XX_summary.json         ← now includes live_* fields
                  └── results/window_XX_training_log.json
                  └── results/reverse_window_XX_daily.csv
                  └── results/reverse_window_XX_summary.json
                  └── results/latest_signal.json             ← includes trained_on, train_start, train_end
                  └── results/signal_history.json            ← scored daily, persists across sessions
                  └── results/latest_reverse_signal.json
                  └── results/reverse_signal_history.json
DISPLAY at:       Streamlit.io (connected to this GitHub repo)
TRAINING on:      GitHub Actions free tier (2-core CPU, 7GB RAM, 6hr limit)
```

---

## Phase 1 — Walk-Forward Back-test (14 Expanding Windows)

Validates the strategy across 14 years of out-of-sample data using an expanding training window. Each window trains on all history up to that point and tests on the next unseen year.

```
Window 1:  Train 2008–2010 → Test 2011
Window 2:  Train 2008–2011 → Test 2012
Window 3:  Train 2008–2012 → Test 2013  (taper tantrum)
Window 4:  Train 2008–2013 → Test 2014
Window 5:  Train 2008–2014 → Test 2015  (oil/HYG stress)
Window 6:  Train 2008–2015 → Test 2016
Window 7:  Train 2008–2016 → Test 2017
Window 8:  Train 2008–2017 → Test 2018  (rate rise)
Window 9:  Train 2008–2018 → Test 2019
Window 10: Train 2008–2019 → Test 2020  (COVID crash)
Window 11: Train 2008–2020 → Test 2021  (reflation)
Window 12: Train 2008–2021 → Test 2022  (rate shock)
Window 13: Train 2008–2022 → Test 2023
Window 14: Train 2008–2023 → Test 2024
```

All 14 windows run in parallel via GitHub Actions matrix strategy (~90 min total).
Re-runs on demand — recommended 2–3 times per week to keep live metrics current.

### What `train.py` now produces per window

Each training run evaluates the same trained model on **two** periods:

| Evaluation | Period | Metric stored | Purpose |
|------------|--------|---------------|---------|
| Historical backtest | Window's test year (e.g. 2016) | `excess_return`, `ftrl_sharpe` | Powers Overview tab year-by-year charts |
| **Live backtest** | 2025-01-01 → today | `live_excess_return`, `live_sharpe` | **Selects best window for daily signal** |

The live backtest uses the same trained model and the same training-set normalisation stats — no lookahead bias. This makes the expanding window "best" selection directly comparable to reverse windows: both are now optimised on the same live 2025+ period.

---

## Phase 1 — Daily Live Signal

A daily prediction runs every trading day (Mon–Fri, 21:00 UTC — after US market close):

```
1. Score yesterday's signal → actual ETF return vs AGG → append to signal_history.json
2. Load all 14 window_XX_summary.json files from HF
3. Find best window by live_excess_return (2025+ period)
   └── Falls back to historical excess_return if live metrics not yet populated
4. Train actor on best window's exact date range (e.g. 2008–2015)
5. Run inference on latest 40-day window
6. Winner-takes-all: highest softmax weight = tomorrow's ETF signal
7. Save latest_signal.json (includes trained_on, train_start, train_end, basis)
8. Save signal_history.json to HF dataset repo
9. Streamlit dashboard updates automatically
```

### Best window selection — expanding vs reverse

Both signals now answer genuinely different questions, evaluated on the same live 2025+ period:

| Signal | Question answered | Varies |
|--------|------------------|--------|
| **Expanding** | Which training *end date* produces the best live model? | End date (2010→2023), fixed start 2008 |
| **Reverse** | Which training *start date* produces the best live model? | Start date (2008→2021), fixed end 2024 |

If W06 wins expanding: training beyond 2015 hurts (recent data adds noise).
If R08 wins reverse: starting before 2015 hurts (GFC-era data is stale).
When they agree on a similar period, that's high-conviction evidence of the optimal training window.

---

## Phase 2 — Reverse Expanding Windows

Determines the optimal training start date by dropping the oldest year from each window, all tested on the live 2025+ period.

```
Window R1:  Train 2008–2024 → Test 2025+2026YTD
Window R2:  Train 2009–2024 → Test 2025+2026YTD
Window R3:  Train 2010–2024 → Test 2025+2026YTD
...
Window R14: Train 2021–2024 → Test 2025+2026YTD
```

**Purpose:** Identifies how far back training data should go. Best window (e.g. R08: 2015–2024) tells you the GFC era is not helpful for current market regimes.

---

## Dashboard — Signal Panel

Three hero cards shown at the top of the dashboard every trading day:

| Card | Source | Trained on |
|------|--------|-----------|
| **Expanding Signal** | `predict.py` | Best window by live 2025+ excess return |
| **Reverse Signal** | `predict_reverse.py` | Best reverse window by live 2025+ excess return |
| **Consensus** | Weighted score engine | Activates after 5 scored days |

The info banner below each signal card shows exactly which window was used and why (live 2025+ or historical fallback).

### Weighted Consensus Engine

Once 5+ days of scored history exist, the consensus card activates:

```
Score = 0.50 × avg_daily_excess + 0.30 × live_sharpe − 0.20 × abs(live_max_dd)
```

| History available | Consensus level |
|-------------------|-----------------|
| < 5 days | Building history — no consensus |
| 5–14 days | ⚠️ Provisional |
| 15–29 days | 🔶 Moderate |
| 30+ days | ✅ High Conviction (rolling 60-day window) |

### Signal Audit Trail

Scored daily and persisted to `signal_history.json` / `reverse_signal_history.json`. Each record carries the full signal fields including `trained_on`, making the history fully auditable — you can see which window each historical signal used even after the best window changes.

---

## Repository Structure

```
p2-etf-ftrl-engine/
├── config.py              # All hyperparameters — single source of truth
├── dataset.py             # Loads prices from p2-etf-deepwave-dl HF repo
├── features.py            # Computes 4 price-derived features
├── environment.py         # Portfolio simulation + ReplayBuffer
├── trb.py                 # Temporal Relationship Block
├── llb.py                 # Latent Linkage Block
├── ft.py                  # Financial Transformer (Actor)
├── critic.py              # Critic network
├── ddpg.py                # DDPG training loop
├── train.py               # Walk-forward training (historical + live evaluation)
├── train_reverse.py       # Reverse window training
├── evaluate.py            # Aggregates all window results
├── aggregate.py           # Aggregates all window results
├── predict.py             # Daily expanding signal (best window by live 2025+)
├── predict_reverse.py     # Daily reverse signal (best window by live 2025+)
├── streamlit_app.py       # Dashboard (Streamlit.io)
├── requirements.txt       # Dependencies (no torch — Streamlit only)
└── .github/workflows/
    ├── train.yml          # Walk-forward training (parallel, on demand)
    ├── train_reverse.yml  # Reverse window training (parallel, on demand)
    ├── evaluate.yml       # Aggregation workflow
    ├── predict.yml        # Daily expanding signal (Mon–Fri 21:00 UTC)
    └── predict_reverse.yml # Daily reverse signal (Mon–Fri 21:00 UTC)
```

---

## Running

### Train all 14 expanding windows in parallel
Go to **Actions → Train FTRL Window → Run workflow → select "all" → Run**
Takes ~90 min. Recommended 2–3× per week to keep `live_excess_return` metrics current.

### Train all 14 reverse windows in parallel
Go to **Actions → Train FTRL Reverse Windows → Run workflow → select "all" → Run**
Takes ~90 min. Run on the same day as expanding windows so both signals compare windows evaluated on the same price data snapshot.

### Train a single window
Go to **Actions → Train FTRL Window → Run workflow → select window 1–14 → Run**

### Run daily prediction manually
Go to **Actions → Daily FTRL Signal → Run workflow → Run**
Go to **Actions → Daily FTRL Reverse Signal → Run workflow → Run**

### View results
Open the Streamlit dashboard linked in the repo description.

---

## Recommended Weekly Cadence

| Day | Action |
|-----|--------|
| Monday | Run both training workflows (expanding + reverse) |
| Thursday | Run both training workflows again |
| Mon–Fri 21:00 UTC | Daily predict workflows run automatically |

Running both on the same day ensures the two hero boxes are always selecting windows evaluated on the same underlying price data.

---

## GitHub Secrets Required

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | HuggingFace write token (Settings → Access Tokens) |
| `HF_DATASET_REPO` | `P2SAMAPA/p2-etf-ftrl-engine` |
| `HF_SOURCE_REPO` | `P2SAMAPA/p2-etf-deepwave-dl` |

---

## Important Notes

- **Prices are total return adjusted** — dividends and distributions are included. This matters significantly for fixed-income ETFs like TLT and LQD where yield is a large component of return.
- **No lookahead bias** — features are normalised using training set statistics only. The live 2025+ evaluation in `train.py` reuses the training-set mean/std, so there is no data leakage.
- **Transaction cost** — 0.1% of rebalanced value is charged at each step during training and back-test.
- **Long-only** — softmax output guarantees all weights ≥ 0 and sum to 1. No shorting.
- **CPU training** — AMP (mixed precision) is not used. Architecture is sized for GitHub Actions 2-core CPU within the 6-hour free tier limit.
- **Best window fallback** — if `live_excess_return` is not yet in the summary JSONs (first run before `train.py` upgrade), both predict scripts fall back to historical `excess_return` automatically. Nothing breaks mid-flight.
- **Signal history is persistent** — `signal_history.json` is appended to daily and pushed to HF with `force_download=True` on read, so the dashboard always reflects the latest scored records. History starts accumulating from the first daily run after deployment.
- **The daily signal is not financial advice.** It is an experimental output of a research implementation. Past back-test performance does not guarantee future results.
