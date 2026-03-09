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
ASSETS          = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV']
H               = 40       # lookback days
C               = 4        # features per asset
ATTENTION_BLOCKS = 3       # reduced from paper's 6 for CPU
N_HEADS         = 4
D_MODEL         = 64
BATCH_SIZE      = 16
LR              = 1e-5
GAMMA           = 0.99
TAU             = 0.005
BUFFER_SIZE     = 2000
TRANSACTION_COST = 0.001
MAX_EPOCHS      = 50
EARLY_STOP_PAT  = 10
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
                  └── results/window_XX_summary.json
                  └── results/window_XX_training_log.json
                  └── results/latest_signal.json
                  └── results/signal_history.json
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
Re-runs automatically on the first Monday of each month to incorporate new price data.

---

## Phase 1 — Daily Live Signal

After back-test validation, a daily prediction runs every trading day (Mon–Fri, 21:00 UTC — after US market close):

```
1. Score yesterday's signal → actual ETF return vs AGG → append to signal_history.json
2. Train actor on full dataset (2008 → today)
3. Run inference on latest 40-day window
4. Winner-takes-all: highest softmax weight = tomorrow's ETF signal
5. Save latest_signal.json to HF dataset repo
6. Streamlit dashboard updates automatically
```

Output is a single actionable ETF per day — not a price forecast, but an allocation signal: which ETF the model believes will outperform AGG the next trading day.

The dashboard shows:
- **Today's Signal** — ETF name, confidence (raw weight %), signal date
- **15-Day Audit Trail** — predicted ETF, actual return, vs AGG, ✓/✗

---

## Phase 2 — Reverse Expanding Windows (Planned)

A second validation experiment to determine how much historical data is optimal for the live signal. Instead of always starting from 2008, each window drops the oldest year:

```
Window R1:  Train 2008–2024 → Test 2025+2026YTD
Window R2:  Train 2009–2024 → Test 2025+2026YTD
Window R3:  Train 2010–2024 → Test 2025+2026YTD
...
Window R14: Train 2021–2024 → Test 2025+2026YTD
```

**Purpose:** If dropping pre-2012 data improves out-of-sample performance, the live model should only train on 2012→today. If GFC data (2008–2009) helps, keep the full history.

Phase 2 deliverables:
- `train_reverse.yml` — parallel run of all 14 reverse windows
- `predict_reverse.py` — live signal from optimal reverse window
- Updated dashboard — side-by-side signal comparison:

| Date | Expanding Signal | Reverse Signal | Actual | Winner |
|------|-----------------|----------------|--------|--------|
| 2026-03-10 | TLT | GLD | ... | ... |

After ~1 month of live signals, a **consensus signal** will be added: if both models agree → strong conviction trade, if they disagree → no trade / hold cash.

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
├── train.py               # GitHub Actions entry point (walk-forward)
├── evaluate.py            # Back-test evaluation
├── aggregate.py           # Aggregates all window results
├── predict.py             # Daily live signal generation
├── streamlit_app.py       # Dashboard (Streamlit.io)
├── requirements.txt       # Dependencies (no torch — Streamlit only)
└── .github/workflows/
    ├── train.yml          # Walk-forward training (parallel + monthly schedule)
    ├── evaluate.yml       # Aggregation workflow
    └── predict.yml        # Daily signal (Mon–Fri 21:00 UTC)
```

---

## Running

### Train all 14 windows in parallel (one-click)
Go to **Actions → Train FTRL Window → Run workflow → select "all" → Run**
Takes ~90 min. All 14 jobs run simultaneously on GitHub Actions free tier.

### Train a single window
Go to **Actions → Train FTRL Window → Run workflow → select window 1–14 → Run**

### Run daily prediction manually
Go to **Actions → Daily FTRL Signal → Run workflow → Run**

### View results
Open the Streamlit dashboard linked in the repo description.

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
- **No lookahead bias** — features are normalised using training set statistics only. Test set uses training mean/std.
- **Transaction cost** — 0.1% of rebalanced value is charged at each step during training and back-test.
- **Long-only** — softmax output guarantees all weights ≥ 0 and sum to 1. No shorting.
- **CPU training** — AMP (mixed precision) is not used. Architecture is sized for GitHub Actions 2-core CPU within the 6-hour free tier limit.
- **The daily signal is not financial advice.** It is an experimental output of a research implementation. Past back-test performance does not guarantee future results.
