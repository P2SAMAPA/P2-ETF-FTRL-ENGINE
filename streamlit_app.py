# streamlit_app.py — FTRL ETF Portfolio Dashboard
# Reads results from HuggingFace dataset repo
# Deploy at streamlit.io — connect to GitHub repo, entry point: streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FTRL ETF Portfolio Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

HF_REPO   = "P2SAMAPA/p2-etf-ftrl-engine"
ASSETS    = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV']
COLORS    = {
    'TLT': '#2196F3', 'LQD': '#4CAF50', 'HYG': '#FF9800',
    'VNQ': '#9C27B0', 'GLD': '#FFD700', 'SLV': '#90A4AE',
    'FTRL': '#E91E63', 'AGG': '#607D8B'
}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_summary() -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="results/all_windows_summary.json",
            repo_type="dataset"
        )
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_window_daily(window_id: int) -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"results/backtest_window_{window_id:02d}.csv",
            repo_type="dataset"
        )
        df = pd.read_csv(path, parse_dates=['date'], index_col='date')
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_all_daily() -> pd.DataFrame:
    """Load and concatenate all available window daily results."""
    frames = []
    for w_id in range(1, 15):
        df = load_window_daily(w_id)
        if not df.empty:
            df['window_id'] = w_id
            frames.append(df)
    if frames:
        return pd.concat(frames).sort_index()
    return pd.DataFrame()


# ── Helper functions ──────────────────────────────────────────────────────────
def format_pct(val):
    if pd.isna(val):
        return "—"
    color = "green" if val > 0 else "red"
    return f":{color}[{val*100:+.2f}%]"


def build_equity_curve(daily_df: pd.DataFrame) -> pd.Series:
    """Build normalised equity curve from net_return column."""
    if 'net_return' not in daily_df.columns:
        return pd.Series()
    curve = (1 + daily_df['net_return']).cumprod()
    return curve


def compute_rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    return (returns.rolling(window).mean() /
            returns.rolling(window).std()) * np.sqrt(252)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
             width=40)
    st.title("FTRL Engine")
    st.caption("Financial Transformer Reinforcement Learning\n6-ETF Portfolio")

    st.divider()
    st.markdown("**Universe**")
    for asset in ASSETS:
        st.markdown(f"- `{asset}`")

    st.divider()
    st.markdown("**Benchmark:** `AGG`")
    st.markdown("**Strategy:** Max Return DDPG")
    st.markdown("**Data:** 2008 → Present")
    st.markdown("**Walk-forward:** 14 windows")

    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────
st.title("📈 FTRL ETF Portfolio Engine")
st.caption("Financial Transformer Reinforcement Learning | Walk-Forward Back-test vs AGG")

# Load data
summary_df = load_summary()
all_daily  = load_all_daily()

if summary_df.empty:
    st.info("⏳ No results yet. Training is in progress. "
            "Check back after the first GitHub Actions run completes.")
    st.markdown("""
    **Pipeline status:**
    - Data source: `P2SAMAPA/p2-etf-deepwave-dl`
    - Output repo: `P2SAMAPA/p2-etf-ftrl-engine`
    - Training: GitHub Actions (one window per run)
    - Results appear here as each window completes
    """)
    st.stop()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "📈 Equity Curves", "⚖️ Weights", "🔍 Window Detail"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    n_complete = len(summary_df)
    st.subheader(f"Walk-Forward Results — {n_complete}/14 Windows Complete")

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    avg_ftrl_ret  = summary_df['ftrl_return'].mean()
    avg_agg_ret   = summary_df['agg_return'].mean()
    avg_excess    = summary_df['excess_return'].mean()
    win_rate      = summary_df['beats_benchmark'].mean()
    avg_sharpe    = summary_df['ftrl_sharpe'].mean()

    col1.metric("Avg Annual Return", f"{avg_ftrl_ret*100:.2f}%",
                f"{avg_excess*100:+.2f}% vs AGG")
    col2.metric("Avg AGG Return",    f"{avg_agg_ret*100:.2f}%")
    col3.metric("Win Rate vs AGG",   f"{win_rate*100:.0f}%",
                f"{int(summary_df['beats_benchmark'].sum())}/{n_complete} years")
    col4.metric("Avg Sharpe",        f"{avg_sharpe:.3f}")
    col5.metric("Windows Done",      f"{n_complete}/14")

    st.divider()

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Year-by-Year Performance")

    display_df = summary_df[[
        'test_year', 'ftrl_return', 'agg_return', 'excess_return',
        'ftrl_sharpe', 'agg_sharpe', 'ftrl_max_dd', 'agg_max_dd',
        'beats_benchmark'
    ]].copy()

    display_df.columns = [
        'Year', 'FTRL Return', 'AGG Return', 'Excess',
        'FTRL Sharpe', 'AGG Sharpe', 'FTRL MaxDD', 'AGG MaxDD', 'Beats'
    ]

    def style_row(row):
        color = 'background-color: #1a3a1a' if row['Beats'] else 'background-color: #3a1a1a'
        return [color] * len(row)

    pct_cols = ['FTRL Return', 'AGG Return', 'Excess', 'FTRL MaxDD', 'AGG MaxDD']
    for col in pct_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:+.2f}%")

    for col in ['FTRL Sharpe', 'AGG Sharpe']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")

    display_df['Beats'] = display_df['Beats'].apply(lambda x: "✓" if x else "✗")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Bar chart: FTRL vs AGG returns ────────────────────────────────────────
    st.subheader("Annual Returns: FTRL vs AGG")
    fig = go.Figure()
    fig.add_bar(
        x=summary_df['test_year'].astype(str),
        y=summary_df['ftrl_return'] * 100,
        name='FTRL', marker_color=COLORS['FTRL']
    )
    fig.add_bar(
        x=summary_df['test_year'].astype(str),
        y=summary_df['agg_return'] * 100,
        name='AGG', marker_color=COLORS['AGG']
    )
    fig.update_layout(
        barmode='group', template='plotly_dark',
        yaxis_title='Return (%)', xaxis_title='Test Year',
        height=400, legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: EQUITY CURVES
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Cumulative Portfolio Value (All Windows)")

    if all_daily.empty:
        st.info("Daily data not yet available.")
    else:
        fig = go.Figure()

        # Plot each window's equity curve
        for w_id in sorted(all_daily['window_id'].unique()):
            w_df  = all_daily[all_daily['window_id'] == w_id]
            curve = build_equity_curve(w_df)
            year  = summary_df[summary_df['window_id'] == w_id]['test_year'].values
            label = str(year[0]) if len(year) > 0 else f"W{w_id}"
            beats = summary_df[summary_df['window_id'] == w_id]['beats_benchmark'].values
            color = '#00C853' if (len(beats) > 0 and beats[0]) else '#FF1744'

            fig.add_scatter(
                x=w_df.index, y=curve,
                mode='lines', name=label,
                line=dict(color=color, width=1.5),
                opacity=0.7
            )

        fig.update_layout(
            template='plotly_dark', height=500,
            yaxis_title='Portfolio Value (normalised to 1.0)',
            xaxis_title='Date',
            legend=dict(x=1.01, y=1)
        )
        fig.add_hline(y=1.0, line_dash='dash',
                      line_color='white', opacity=0.3,
                      annotation_text='Start')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green = beats AGG benchmark | Red = trails AGG")

        # ── Rolling Sharpe ────────────────────────────────────────────────────
        st.subheader("Rolling 60-Day Sharpe (All Windows)")
        fig2 = go.Figure()
        for w_id in sorted(all_daily['window_id'].unique()):
            w_df    = all_daily[all_daily['window_id'] == w_id]
            rolling = compute_rolling_sharpe(w_df['net_return'])
            year    = summary_df[summary_df['window_id'] == w_id]['test_year'].values
            label   = str(year[0]) if len(year) > 0 else f"W{w_id}"
            fig2.add_scatter(
                x=w_df.index, y=rolling,
                mode='lines', name=label, line=dict(width=1.5)
            )
        fig2.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3)
        fig2.update_layout(
            template='plotly_dark', height=400,
            yaxis_title='Rolling Sharpe', xaxis_title='Date'
        )
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: PORTFOLIO WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Portfolio Weight Allocation")

    if all_daily.empty:
        st.info("Daily data not yet available.")
    else:
        weight_cols = [f'w_{a}' for a in ASSETS if f'w_{a}' in all_daily.columns]

        if weight_cols:
            # Average weights per window
            avg_weights = []
            for w_id in sorted(all_daily['window_id'].unique()):
                w_df = all_daily[all_daily['window_id'] == w_id]
                row  = {'window': w_id}
                year = summary_df[summary_df['window_id'] == w_id]['test_year'].values
                row['year'] = str(year[0]) if len(year) > 0 else f"W{w_id}"
                for col in weight_cols:
                    asset = col.replace('w_', '')
                    row[asset] = w_df[col].mean()
                avg_weights.append(row)

            wt_df = pd.DataFrame(avg_weights)

            # Stacked bar: average weights per year
            fig = go.Figure()
            for asset in ASSETS:
                if asset in wt_df.columns:
                    fig.add_bar(
                        x=wt_df['year'], y=wt_df[asset],
                        name=asset, marker_color=COLORS.get(asset, '#888')
                    )
            fig.update_layout(
                barmode='stack', template='plotly_dark',
                yaxis_title='Average Weight', xaxis_title='Test Year',
                height=400, yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)

            # Weight over time for selected window
            st.subheader("Daily Weights — Select Window")
            available_years = []
            for w_id in sorted(all_daily['window_id'].unique()):
                year = summary_df[summary_df['window_id'] == w_id]['test_year'].values
                if len(year) > 0:
                    available_years.append((w_id, str(year[0])))

            if available_years:
                sel_label = st.selectbox(
                    "Test Year",
                    [y for _, y in available_years]
                )
                sel_wid = next(w for w, y in available_years if y == sel_label)
                sel_df  = all_daily[all_daily['window_id'] == sel_wid]

                fig3 = go.Figure()
                for col in weight_cols:
                    asset = col.replace('w_', '')
                    fig3.add_scatter(
                        x=sel_df.index, y=sel_df[col],
                        mode='lines', name=asset,
                        stackgroup='one',
                        line=dict(color=COLORS.get(asset, '#888'))
                    )
                fig3.update_layout(
                    template='plotly_dark', height=400,
                    yaxis_title='Weight', xaxis_title='Date',
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: WINDOW DETAIL
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Individual Window Analysis")

    available_windows = summary_df['window_id'].tolist()
    if not available_windows:
        st.info("No windows complete yet.")
    else:
        sel_wid = st.selectbox(
            "Select Window",
            available_windows,
            format_func=lambda x: (
                f"Window {x:02d} — "
                f"Test {summary_df[summary_df['window_id']==x]['test_year'].values[0]}"
            )
        )

        row    = summary_df[summary_df['window_id'] == sel_wid].iloc[0]
        w_daily = load_window_daily(sel_wid)

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FTRL Return",  f"{row['ftrl_return']*100:+.2f}%",
                  f"{row['excess_return']*100:+.2f}% vs AGG")
        c2.metric("AGG Return",   f"{row['agg_return']*100:+.2f}%")
        c3.metric("FTRL Sharpe",  f"{row['ftrl_sharpe']:.3f}",
                  f"{row['ftrl_sharpe']-row['agg_sharpe']:+.3f} vs AGG")
        c4.metric("Max Drawdown", f"{row['ftrl_max_dd']*100:+.2f}%",
                  f"{(row['ftrl_max_dd']-row['agg_max_dd'])*100:+.2f}% vs AGG")

        if not w_daily.empty:
            # Equity curve for this window
            curve = build_equity_curve(w_daily)
            fig   = go.Figure()
            fig.add_scatter(
                x=w_daily.index, y=curve,
                mode='lines', name='FTRL',
                line=dict(color=COLORS['FTRL'], width=2)
            )
            fig.add_hline(y=1.0, line_dash='dash',
                          line_color='white', opacity=0.4)
            fig.update_layout(
                template='plotly_dark', height=350,
                title=f"Equity Curve — Test Year {row['test_year']}",
                yaxis_title='Portfolio Value', xaxis_title='Date'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown chart
            peak = curve.cummax()
            dd   = (curve - peak) / peak
            fig2 = go.Figure()
            fig2.add_scatter(
                x=w_daily.index, y=dd * 100,
                mode='lines', fill='tozeroy',
                line=dict(color='#FF1744', width=1),
                name='Drawdown'
            )
            fig2.update_layout(
                template='plotly_dark', height=250,
                title='Drawdown (%)', yaxis_title='%', xaxis_title='Date'
            )
            st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "FTRL Engine | Data: P2SAMAPA/p2-etf-deepwave-dl | "
    "Model: P2SAMAPA/p2-etf-ftrl-engine | "
    "Architecture: Financial Transformer + DDPG"
)
