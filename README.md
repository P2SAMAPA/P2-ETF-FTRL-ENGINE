# P2 ETF FTRL Engine

Financial Transformer Reinforcement Learning (FTRL) for 6-ETF portfolio management.

## Architecture

Based on: *"Time Series is Not Enough: Financial Transformer Reinforcement Learning for Portfolio Management"* by Xiaotian Rena,1, Ruoyu Suna,2, Zhengyong Jianga,3, Angelos Stefanidisa,4,
Hongbin Liua,5,∗∗, Jionglong Sua,6,
School of AI and Advanced Computing, XJTLU Entrepreneur College (Taicang), Xi’an
Jiaotong-Liverpool University, Suzhou, 215123, China

| Component | Description |
|-----------|-------------|
| **TRB** | Temporal Relationship Block — daily + weekly patch attention |
| **LLB** | Latent Linkage Block — inter-asset correlation discovery |
| **Actor** | Financial Transformer → softmax portfolio weights |
| **Critic** | Conv + FC network → Q-value estimate |
| **Algorithm** | DDPG with experience replay + soft target updates |

## Universe

| ETF | Exposure |
|-----|----------|
| TLT | Long-duration US Treasuries |
| LQD | Investment grade corporate bonds |
| HYG | High yield corporate bonds |
| VNQ | US Real Estate (REITs) |
| GLD | Gold |
| SLV | Silver |

**Benchmark:** AGG (US Aggregate Bond Market)

## Walk-Forward Back-test

14 expanding windows from 2008–2024:
- Train: 2008 → window end (expanding)
- Test: one held-out year per window
- Covers: GFC, taper tantrum, COVID crash, 2022 rate shock

## Infrastructure

```
Data source:   P2SAMAPA/p2-etf-deepwave-dl  (HF Dataset)
Model outputs: P2SAMAPA/p2-etf-ftrl-engine  (HF Dataset)
Training:      GitHub Actions (CPU, one window per run)
Dashboard:     Streamlit.io
```

## Running

### Train a single window
Go to **Actions → Train FTRL Window → Run workflow**
Select window ID (1–14) and click Run.

Each window takes ~3–4 hours on GitHub Actions free tier CPU.

### View results
Open the Streamlit dashboard linked in the repo description.

## Configuration

All hyperparameters in `src/config.py`:

```python
ASSETS     = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV']
LOOKBACK   = 40      # days
N_BLOCKS   = 3       # attention blocks
D_MODEL    = 64
BATCH_SIZE = 16
MAX_EPOCHS = 50
```
