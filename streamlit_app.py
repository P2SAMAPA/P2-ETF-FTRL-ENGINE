# streamlit_app.py — FTRL ETF Portfolio Dashboard
# Reads results from HuggingFace dataset repo
# Deploy at streamlit.io — connect to GitHub repo, entry point: streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FTRL ETF Portfolio Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

HF_REPO = "P2SAMAPA/p2-etf-ftrl-engine"
ASSETS  = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV']
COLORS  = {
    'TLT': '#2196F3', 'LQD': '#4CAF50', 'HYG': '#FF9800',
    'VNQ': '#9C27B0', 'GLD': '#FFD700', 'SLV': '#90A4AE',
    'FTRL': '#E91E63', 'AGG': '#607D8B'
}
ETF_DESC = {
    'TLT': 'Long-duration US Treasuries',
    'LQD': 'Investment Grade Corporate Bonds',
    'HYG': 'High Yield Corporate Bonds',
    'VNQ': 'US Real Estate (REITs)',
    'GLD': 'Gold',
    'SLV': 'Silver',
}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_summary() -> pd.DataFrame:
    frames = []
    for w_id in range(1, 15):
        try:
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"results/window_{w_id:02d}_summary.json",
                repo_type="dataset"
            )
            with open(path) as f:
                data = json.load(f)
            frames.append(data)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df = df.rename(columns={
        'ftrl_total_return': 'ftrl_return',
        'agg_total_return':  'agg_return',
        'ftrl_max_drawdown': 'ftrl_max_dd',
    })
    df['beats_benchmark'] = df['ftrl_return'] > df['agg_return']
    df['agg_sharpe'] = 0.0
    df['agg_max_dd'] = 0.0
    return df


@st.cache_data(ttl=3600)
def load_window_daily(window_id: int) -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"results/window_{window_id:02d}_daily.csv",
            repo_type="dataset"
        )
        df = pd.read_csv(path, parse_dates=['date'], index_col='date')
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_all_daily() -> pd.DataFrame:
    frames = []
    for w_id in range(1, 15):
        df = load_window_daily(w_id)
        if not df.empty:
            df['window_id'] = w_id
            frames.append(df)
    if frames:
        return pd.concat(frames).sort_index()
    return pd.DataFrame()


@st.cache_data(ttl=300)   # 5 min cache — refresh more frequently for signal
def load_latest_signal() -> dict:
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="results/latest_signal.json",
            repo_type="dataset",
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_signal_history() -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="results/signal_history.json",
            repo_type="dataset",
            force_download=True,
        )
        with open(path) as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


# ── Helper functions ──────────────────────────────────────────────────────────
def build_equity_curve(daily_df: pd.DataFrame) -> pd.Series:
    if 'net_return' not in daily_df.columns:
        return pd.Series()
    return (1 + daily_df['net_return']).cumprod()


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

# Load all data
summary_df     = load_summary()
all_daily      = load_all_daily()
latest_signal  = load_latest_signal()
signal_history = load_signal_history()

# ═══════════════════════════════════════════════════════════════════════════════
# TODAY'S SIGNAL — always shown at top if available
# ═══════════════════════════════════════════════════════════════════════════════
if latest_signal:
    signal_etf  = latest_signal.get('signal', '—')
    confidence  = latest_signal.get('confidence', 0)
    signal_date = latest_signal.get('date', '—')
    raw_weights = latest_signal.get('raw_weights', {})

    st.markdown("---")
    sig_col1, sig_col2, sig_col3 = st.columns([2, 2, 3])

    with sig_col1:
        etf_color = COLORS.get(signal_etf, '#E91E63')
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {etf_color}22, {etf_color}44);
            border: 2px solid {etf_color};
            border-radius: 12px;
            padding: 20px 24px;
            text-align: center;
        ">
            <div style="font-size:13px; color:#aaa; margin-bottom:4px;">
                TODAY'S SIGNAL · {signal_date}
            </div>
            <div style="font-size:48px; font-weight:900; color:{etf_color}; 
                        letter-spacing:2px;">
                {signal_etf}
            </div>
            <div style="font-size:14px; color:#ccc; margin-top:4px;">
                {ETF_DESC.get(signal_etf, '')}
            </div>
            <div style="font-size:20px; font-weight:700; color:{etf_color}; 
                        margin-top:8px;">
                {confidence:.1%} confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sig_col2:
        st.markdown("**Raw Model Weights**")
        if raw_weights:
            sorted_weights = sorted(raw_weights.items(),
                                    key=lambda x: x[1], reverse=True)
            for etf, w in sorted_weights:
                bar_color = COLORS.get(etf, '#888')
                is_winner = etf == signal_etf
                prefix = "🏆 " if is_winner else "   "
                st.markdown(
                    f"{prefix}**`{etf}`** "
                    f"{'&nbsp;' * 2}"
                    f"<span style='color:{bar_color}'>{w:.1%}</span>",
                    unsafe_allow_html=True
                )
                st.progress(float(w))

    with sig_col3:
        # Weight donut chart
        if raw_weights:
            labels = list(raw_weights.keys())
            values = list(raw_weights.values())
            colors = [COLORS.get(l, '#888') for l in labels]

            fig_donut = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker_colors=colors,
                textinfo='label+percent',
                hovertemplate='%{label}: %{value:.1%}<extra></extra>',
            ))
            fig_donut.update_layout(
                template='plotly_dark',
                height=220,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False,
                annotations=[dict(
                    text=f'<b>{signal_etf}</b>',
                    x=0.5, y=0.5,
                    font_size=20,
                    font_color=COLORS.get(signal_etf, 'white'),
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# 15-DAY AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════════════
if not signal_history.empty:
    st.subheader("📋 Signal Audit Trail — Last 15 Days")

    display_hist = signal_history.head(15).copy()

    # Format columns
    display_hist['Date']           = display_hist['date'].dt.strftime('%Y-%m-%d')
    display_hist['Signal']         = display_hist['signal']
    display_hist['Confidence']     = display_hist['confidence'].apply(
                                        lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    display_hist['Actual Return']  = display_hist['actual_return'].apply(
                                        lambda x: f"{x:+.2%}" if pd.notna(x) else "Pending")
    display_hist['vs AGG']         = display_hist['excess_return'].apply(
                                        lambda x: f"{x:+.2%}" if pd.notna(x) else "Pending")
    display_hist['Result']         = display_hist['beats_agg'].apply(
                                        lambda x: "✓" if x is True else
                                                  ("✗" if x is False else "⏳"))

    audit_display = display_hist[[
        'Date', 'Signal', 'Confidence', 'Actual Return', 'vs AGG', 'Result'
    ]]

    # Compute hit rate from scored records
    scored = signal_history[signal_history['beats_agg'].notna()]
    if not scored.empty:
        hit_rate = scored['beats_agg'].mean()
        avg_excess = scored['excess_return'].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Signal Hit Rate", f"{hit_rate:.0%}",
                  f"{int(scored['beats_agg'].sum())}/{len(scored)} scored")
        c2.metric("Avg Daily Excess vs AGG",
                  f"{avg_excess:+.2%}")
        c3.metric("Signals Generated", str(len(signal_history)))

    st.dataframe(audit_display, use_container_width=True, hide_index=True)
    st.markdown("---")

# ── No backtest data yet ──────────────────────────────────────────────────────
if summary_df.empty:
    st.info("⏳ Back-test results not yet available. "
            "Training is in progress via GitHub Actions.")
    st.markdown("""
    **Pipeline status:**
    - Data source: `P2SAMAPA/p2-etf-deepwave-dl`
    - Output repo: `P2SAMAPA/p2-etf-ftrl-engine`
    - Training: GitHub Actions (14 parallel windows)
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

    col1, col2, col3, col4, col5 = st.columns(5)

    avg_ftrl_ret = summary_df['ftrl_return'].mean()
    avg_agg_ret  = summary_df['agg_return'].mean()
    avg_excess   = summary_df['excess_return'].mean()
    win_rate     = summary_df['beats_benchmark'].mean()
    avg_sharpe   = summary_df['ftrl_sharpe'].mean()

    col1.metric("Avg Annual Return", f"{avg_ftrl_ret*100:.2f}%",
                f"{avg_excess*100:+.2f}% vs AGG")
    col2.metric("Avg AGG Return",    f"{avg_agg_ret*100:.2f}%")
    col3.metric("Win Rate vs AGG",   f"{win_rate*100:.0f}%",
                f"{int(summary_df['beats_benchmark'].sum())}/{n_complete} years")
    col4.metric("Avg Sharpe",        f"{avg_sharpe:.3f}")
    col5.metric("Windows Done",      f"{n_complete}/14")

    st.divider()
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

    pct_cols = ['FTRL Return', 'AGG Return', 'Excess', 'FTRL MaxDD', 'AGG MaxDD']
    for col in pct_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:+.2f}%")
    for col in ['FTRL Sharpe', 'AGG Sharpe']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    display_df['Beats'] = display_df['Beats'].apply(lambda x: "✓" if x else "✗")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Annual Returns: FTRL vs AGG")
    fig = go.Figure()
    fig.add_bar(x=summary_df['test_year'].astype(str),
                y=summary_df['ftrl_return'] * 100,
                name='FTRL', marker_color=COLORS['FTRL'])
    fig.add_bar(x=summary_df['test_year'].astype(str),
                y=summary_df['agg_return'] * 100,
                name='AGG', marker_color=COLORS['AGG'])
    fig.update_layout(barmode='group', template='plotly_dark',
                      yaxis_title='Return (%)', xaxis_title='Test Year',
                      height=400, legend=dict(x=0.01, y=0.99))
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
        for w_id in sorted(all_daily['window_id'].unique()):
            w_df  = all_daily[all_daily['window_id'] == w_id]
            curve = build_equity_curve(w_df)
            year  = summary_df[summary_df['window_id'] == w_id]['test_year'].values
            label = str(year[0]) if len(year) > 0 else f"W{w_id}"
            beats = summary_df[summary_df['window_id'] == w_id]['beats_benchmark'].values
            color = '#00C853' if (len(beats) > 0 and beats[0]) else '#FF1744'
            fig.add_scatter(x=w_df.index, y=curve, mode='lines', name=label,
                            line=dict(color=color, width=1.5), opacity=0.7)

        fig.update_layout(template='plotly_dark', height=500,
                          yaxis_title='Portfolio Value (normalised to 1.0)',
                          xaxis_title='Date', legend=dict(x=1.01, y=1))
        fig.add_hline(y=1.0, line_dash='dash', line_color='white',
                      opacity=0.3, annotation_text='Start')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green = beats AGG benchmark | Red = trails AGG")

        st.subheader("Rolling 60-Day Sharpe (All Windows)")
        fig2 = go.Figure()
        for w_id in sorted(all_daily['window_id'].unique()):
            w_df    = all_daily[all_daily['window_id'] == w_id]
            rolling = compute_rolling_sharpe(w_df['net_return'])
            year    = summary_df[summary_df['window_id'] == w_id]['test_year'].values
            label   = str(year[0]) if len(year) > 0 else f"W{w_id}"
            fig2.add_scatter(x=w_df.index, y=rolling, mode='lines',
                             name=label, line=dict(width=1.5))
        fig2.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3)
        fig2.update_layout(template='plotly_dark', height=400,
                           yaxis_title='Rolling Sharpe', xaxis_title='Date')
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
            fig = go.Figure()
            for asset in ASSETS:
                if asset in wt_df.columns:
                    fig.add_bar(x=wt_df['year'], y=wt_df[asset],
                                name=asset, marker_color=COLORS.get(asset, '#888'))
            fig.update_layout(barmode='stack', template='plotly_dark',
                              yaxis_title='Average Weight', xaxis_title='Test Year',
                              height=400, yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Daily Weights — Select Window")
            available_years = []
            for w_id in sorted(all_daily['window_id'].unique()):
                year = summary_df[summary_df['window_id'] == w_id]['test_year'].values
                if len(year) > 0:
                    available_years.append((w_id, str(year[0])))

            if available_years:
                sel_label = st.selectbox("Test Year",
                                         [y for _, y in available_years])
                sel_wid = next(w for w, y in available_years if y == sel_label)
                sel_df  = all_daily[all_daily['window_id'] == sel_wid]

                fig3 = go.Figure()
                for col in weight_cols:
                    asset = col.replace('w_', '')
                    fig3.add_scatter(x=sel_df.index, y=sel_df[col],
                                     mode='lines', name=asset,
                                     stackgroup='one',
                                     line=dict(color=COLORS.get(asset, '#888')))
                fig3.update_layout(template='plotly_dark', height=400,
                                   yaxis_title='Weight', xaxis_title='Date',
                                   yaxis=dict(range=[0, 1]))
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

        row     = summary_df[summary_df['window_id'] == sel_wid].iloc[0]
        w_daily = load_window_daily(sel_wid)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FTRL Return",  f"{row['ftrl_return']*100:+.2f}%",
                  f"{row['excess_return']*100:+.2f}% vs AGG")
        c2.metric("AGG Return",   f"{row['agg_return']*100:+.2f}%")
        c3.metric("FTRL Sharpe",  f"{row['ftrl_sharpe']:.3f}",
                  f"{row['ftrl_sharpe']-row['agg_sharpe']:+.3f} vs AGG")
        c4.metric("Max Drawdown", f"{row['ftrl_max_dd']*100:+.2f}%",
                  f"{(row['ftrl_max_dd']-row['agg_max_dd'])*100:+.2f}% vs AGG")

        if not w_daily.empty:
            curve = build_equity_curve(w_daily)
            fig   = go.Figure()
            fig.add_scatter(x=w_daily.index, y=curve, mode='lines', name='FTRL',
                            line=dict(color=COLORS['FTRL'], width=2))
            fig.add_hline(y=1.0, line_dash='dash', line_color='white', opacity=0.4)
            fig.update_layout(template='plotly_dark', height=350,
                              title=f"Equity Curve — Test Year {row['test_year']}",
                              yaxis_title='Portfolio Value', xaxis_title='Date')
            st.plotly_chart(fig, use_container_width=True)

            peak = curve.cummax()
            dd   = (curve - peak) / peak
            fig2 = go.Figure()
            fig2.add_scatter(x=w_daily.index, y=dd * 100,
                             mode='lines', fill='tozeroy',
                             line=dict(color='#FF1744', width=1), name='Drawdown')
            fig2.update_layout(template='plotly_dark', height=250,
                               title='Drawdown (%)',
                               yaxis_title='%', xaxis_title='Date')
            st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "FTRL Engine | Data: P2SAMAPA/p2-etf-deepwave-dl | "
    "Model: P2SAMAPA/p2-etf-ftrl-engine | "
    "Architecture: Financial Transformer + DDPG | "
    "⚠️ Not financial advice"
)
