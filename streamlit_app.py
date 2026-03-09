# streamlit_app.py — FTRL ETF Portfolio Dashboard (Phase 1 + Phase 2)
# Phase 1: Walk-forward back-test + expanding window daily signal
# Phase 2: Reverse expanding windows + side-by-side signal + consensus

import streamlit as st
import pandas as pd
import numpy as np
import json
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go

st.set_page_config(page_title="FTRL ETF Portfolio Engine", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

HF_REPO = "P2SAMAPA/p2-etf-ftrl-engine"
ASSETS  = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV']
COLORS  = {'TLT':'#2196F3','LQD':'#4CAF50','HYG':'#FF9800',
           'VNQ':'#9C27B0','GLD':'#FFD700','SLV':'#90A4AE',
           'FTRL':'#E91E63','AGG':'#607D8B'}
ETF_DESC = {
    'TLT': 'Long-duration US Treasuries',
    'LQD': 'Investment Grade Corporate Bonds',
    'HYG': 'High Yield Corporate Bonds',
    'VNQ': 'US Real Estate (REITs)',
    'GLD': 'Gold',
    'SLV': 'Silver',
}


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_summary() -> pd.DataFrame:
    frames = []
    for w_id in range(1, 15):
        try:
            path = hf_hub_download(repo_id=HF_REPO,
                filename=f"results/window_{w_id:02d}_summary.json",
                repo_type="dataset")
            with open(path) as f:
                frames.append(json.load(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df = df.rename(columns={'ftrl_total_return':'ftrl_return',
                            'agg_total_return':'agg_return',
                            'ftrl_max_drawdown':'ftrl_max_dd'})
    df['beats_benchmark'] = df['ftrl_return'] > df['agg_return']
    df['agg_sharpe'] = 0.0
    df['agg_max_dd'] = 0.0
    return df


@st.cache_data(ttl=3600)
def load_reverse_summary() -> pd.DataFrame:
    frames = []
    for w_id in range(1, 15):
        try:
            path = hf_hub_download(repo_id=HF_REPO,
                filename=f"results/reverse_window_{w_id:02d}_summary.json",
                repo_type="dataset")
            with open(path) as f:
                frames.append(json.load(f))
        except Exception:
            continue
    return pd.DataFrame(frames) if frames else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_window_daily(window_id: int) -> pd.DataFrame:
    try:
        path = hf_hub_download(repo_id=HF_REPO,
            filename=f"results/window_{window_id:02d}_daily.csv",
            repo_type="dataset")
        return pd.read_csv(path, parse_dates=['date'], index_col='date')
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
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


@st.cache_data(ttl=300)
def load_latest_signal() -> dict:
    try:
        path = hf_hub_download(repo_id=HF_REPO,
            filename="results/latest_signal.json",
            repo_type="dataset", force_download=True)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_latest_reverse_signal() -> dict:
    try:
        path = hf_hub_download(repo_id=HF_REPO,
            filename="results/latest_reverse_signal.json",
            repo_type="dataset", force_download=True)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_signal_history() -> pd.DataFrame:
    try:
        path = hf_hub_download(repo_id=HF_REPO,
            filename="results/signal_history.json",
            repo_type="dataset", force_download=True)
        with open(path) as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date', ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_reverse_signal_history() -> pd.DataFrame:
    try:
        path = hf_hub_download(repo_id=HF_REPO,
            filename="results/reverse_signal_history.json",
            repo_type="dataset", force_download=True)
        with open(path) as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date', ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_equity_curve(daily_df):
    if 'net_return' not in daily_df.columns:
        return pd.Series()
    return (1 + daily_df['net_return']).cumprod()


def compute_rolling_sharpe(returns, window=60):
    return (returns.rolling(window).mean() /
            returns.rolling(window).std()) * np.sqrt(252)


def signal_card(sig: dict, label: str, color: str = None) -> str:
    etf     = sig.get('signal', '—')
    conf    = sig.get('confidence', 0)
    dt      = sig.get('date', '—')
    trained = sig.get('trained_on', '—')
    c       = color or COLORS.get(etf, '#E91E63')
    return f"""
    <div style="background:linear-gradient(135deg,{c}22,{c}44);
                border:2px solid {c};border-radius:12px;
                padding:18px 22px;text-align:center;">
        <div style="font-size:12px;color:#aaa;margin-bottom:4px;">{label} · {dt}</div>
        <div style="font-size:44px;font-weight:900;color:{c};letter-spacing:2px;">{etf}</div>
        <div style="font-size:13px;color:#ccc;margin-top:4px;">{ETF_DESC.get(etf,'')}</div>
        <div style="font-size:18px;font-weight:700;color:{c};margin-top:8px;">{conf:.1%} confidence</div>
        <div style="font-size:11px;color:#888;margin-top:6px;">{trained}</div>
    </div>"""


def render_audit_table(history_df: pd.DataFrame):
    if history_df.empty:
        st.info("No scored signals yet.")
        return
    display = history_df.head(15).copy()
    display['Date']          = display['date'].dt.strftime('%Y-%m-%d')
    display['Signal']        = display['signal']
    display['Confidence']    = display['confidence'].apply(
                                lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    display['Actual Return'] = display['actual_return'].apply(
                                lambda x: f"{x:+.2%}" if pd.notna(x) else "⏳")
    display['vs AGG']        = display['excess_return'].apply(
                                lambda x: f"{x:+.2%}" if pd.notna(x) else "⏳")
    display['Result']        = display['beats_agg'].apply(
                                lambda x: "✓" if x is True else
                                          ("✗" if x is False else "⏳"))
    scored = history_df[history_df['beats_agg'].notna()]
    if not scored.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Hit Rate", f"{scored['beats_agg'].mean():.0%}",
                  f"{int(scored['beats_agg'].sum())}/{len(scored)} scored")
        c2.metric("Avg Daily Excess", f"{scored['excess_return'].mean():+.2%}")
        c3.metric("Signals", str(len(history_df)))
    st.dataframe(display[['Date','Signal','Confidence',
                           'Actual Return','vs AGG','Result']],
                 use_container_width=True, hide_index=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
             width=40)
    st.title("FTRL Engine")
    st.caption("Financial Transformer Reinforcement Learning\n6-ETF Portfolio")
    st.divider()
    st.markdown("**Universe**")
    for a in ASSETS:
        st.markdown(f"- `{a}`")
    st.divider()
    st.markdown("**Benchmark:** `AGG`")
    st.markdown("**Strategy:** Max Return DDPG")
    st.markdown("**Data:** 2008 → Present")
    st.markdown("**Walk-forward:** 14 windows")
    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 FTRL ETF Portfolio Engine")
st.caption("Financial Transformer Reinforcement Learning | Walk-Forward Back-test vs AGG")

# Load everything
summary_df         = load_summary()
reverse_summary_df = load_reverse_summary()
all_daily          = load_all_daily()
signal_exp         = load_latest_signal()
signal_rev         = load_latest_reverse_signal()
history_exp        = load_signal_history()
history_rev        = load_reverse_signal_history()

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL PANEL
# ═══════════════════════════════════════════════════════════════════════════════
has_exp = bool(signal_exp)
has_rev = bool(signal_rev)

if has_exp or has_rev:
    st.markdown("---")

    if has_exp and has_rev:
        exp_etf = signal_exp.get('signal')
        rev_etf = signal_rev.get('signal')
        agree   = exp_etf == rev_etf

        col_e, col_r, col_c = st.columns(3)
        with col_e:
            st.markdown(signal_card(signal_exp, "EXPANDING SIGNAL"),
                        unsafe_allow_html=True)
        with col_r:
            st.markdown(signal_card(signal_rev, "REVERSE SIGNAL", '#FF9800'),
                        unsafe_allow_html=True)
        with col_c:
            if agree:
                c, etf = '#00E676', exp_etf
                lbl    = "✅ CONSENSUS — HIGH CONVICTION"
                conf   = max(signal_exp.get('confidence', 0),
                             signal_rev.get('confidence', 0))
                body   = f"{conf:.1%} confidence"
                desc   = ETF_DESC.get(etf, '')
            else:
                c, etf = '#FF5252', '—'
                lbl    = "⚠️ NO CONSENSUS — HOLD CASH"
                body   = "Models disagree"
                desc   = f"Expanding→{exp_etf} | Reverse→{rev_etf}"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{c}22,{c}44);
                        border:2px solid {c};border-radius:12px;
                        padding:18px 22px;text-align:center;">
                <div style="font-size:12px;color:#aaa;margin-bottom:4px;">{lbl}</div>
                <div style="font-size:44px;font-weight:900;color:{c};
                            letter-spacing:2px;">{etf}</div>
                <div style="font-size:13px;color:#ccc;margin-top:4px;">{desc}</div>
                <div style="font-size:18px;font-weight:700;color:{c};
                            margin-top:8px;">{body}</div>
            </div>""", unsafe_allow_html=True)

    elif has_exp:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(signal_card(signal_exp, "TODAY'S SIGNAL (EXPANDING)"),
                        unsafe_allow_html=True)
        with col2:
            rw = signal_exp.get('raw_weights', {})
            if rw:
                st.markdown("**Raw Model Weights**")
                for etf, w in sorted(rw.items(), key=lambda x: x[1], reverse=True):
                    prefix = "🏆 " if etf == signal_exp.get('signal') else "   "
                    etf_color = COLORS.get(etf, '#888')
                    st.markdown(
                        f"{prefix}**`{etf}`** "
                        f"<span style='color:{etf_color}'>{w:.1%}</span>",
                        unsafe_allow_html=True)
                    st.progress(float(w))

    elif has_rev:
        col1, _ = st.columns([1, 2])
        with col1:
            st.markdown(
                signal_card(signal_rev, "TODAY'S SIGNAL (REVERSE)", '#FF9800'),
                unsafe_allow_html=True)

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════════════
if not history_exp.empty or not history_rev.empty:
    st.subheader("📋 Signal Audit Trail — Last 15 Days")
    at1, at2, at3 = st.tabs(["📊 Expanding", "🔄 Reverse", "⚖️ Comparison"])

    with at1:
        render_audit_table(history_exp)

    with at2:
        if not history_rev.empty:
            render_audit_table(history_rev)
        else:
            st.info("Run reverse window training first.")

    with at3:
        if not history_exp.empty and not history_rev.empty:
            merged = pd.merge(
                history_exp[['date','signal','actual_return',
                             'excess_return','beats_agg']].rename(
                    columns={'signal':'Expanding','actual_return':'Exp Return',
                             'excess_return':'Exp Excess','beats_agg':'Exp Beats'}),
                history_rev[['date','signal','beats_agg']].rename(
                    columns={'signal':'Reverse','beats_agg':'Rev Beats'}),
                on='date', how='inner').head(15)

            if not merged.empty:
                merged['Date']      = merged['date'].dt.strftime('%Y-%m-%d')
                merged['Agree']     = merged['Expanding'] == merged['Reverse']
                merged['Consensus'] = merged.apply(
                    lambda r: r['Expanding'] if r['Agree'] else '—', axis=1)
                merged['Exp ✓']    = merged['Exp Beats'].apply(
                    lambda x: "✓" if x is True else
                              ("✗" if x is False else "⏳"))
                merged['Rev ✓']    = merged['Rev Beats'].apply(
                    lambda x: "✓" if x is True else
                              ("✗" if x is False else "⏳"))
                merged['🤝']       = merged['Agree'].apply(
                    lambda x: "✅" if x else "❌")

                c1, c2 = st.columns(2)
                c1.metric("Agreement Rate",
                          f"{merged['Agree'].mean():.0%}",
                          f"{int(merged['Agree'].sum())}/{len(merged)} days")

                scored_con = merged[merged['Agree'] &
                                    merged['Exp Beats'].notna()]
                if not scored_con.empty:
                    c2.metric("Consensus Hit Rate",
                              f"{scored_con['Exp Beats'].mean():.0%}",
                              f"{int(scored_con['Exp Beats'].sum())}"
                              f"/{len(scored_con)} days")

                st.dataframe(
                    merged[['Date','Expanding','Reverse',
                            'Consensus','Exp ✓','Rev ✓','🤝']],
                    use_container_width=True, hide_index=True)
        else:
            st.info("Both expanding and reverse signals needed for comparison.")

    st.markdown("---")

if summary_df.empty:
    st.info("⏳ Back-test results not yet available.")
    st.stop()

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📈 Equity Curves", "⚖️ Weights",
    "🔍 Window Detail", "🔄 Reverse Windows"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    n = len(summary_df)
    st.subheader(f"Walk-Forward Results — {n}/14 Windows Complete")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Annual Return",
              f"{summary_df['ftrl_return'].mean()*100:.2f}%",
              f"{summary_df['excess_return'].mean()*100:+.2f}% vs AGG")
    c2.metric("Avg AGG Return",
              f"{summary_df['agg_return'].mean()*100:.2f}%")
    c3.metric("Win Rate vs AGG",
              f"{summary_df['beats_benchmark'].mean()*100:.0f}%",
              f"{int(summary_df['beats_benchmark'].sum())}/{n} years")
    c4.metric("Avg Sharpe", f"{summary_df['ftrl_sharpe'].mean():.3f}")
    c5.metric("Windows Done", f"{n}/14")

    st.divider()
    st.subheader("Year-by-Year Performance")
    disp = summary_df[['test_year','ftrl_return','agg_return','excess_return',
                        'ftrl_sharpe','agg_sharpe','ftrl_max_dd','agg_max_dd',
                        'beats_benchmark']].copy()
    disp.columns = ['Year','FTRL Return','AGG Return','Excess',
                    'FTRL Sharpe','AGG Sharpe','FTRL MaxDD','AGG MaxDD','Beats']
    for col in ['FTRL Return','AGG Return','Excess','FTRL MaxDD','AGG MaxDD']:
        disp[col] = disp[col].apply(lambda x: f"{x*100:+.2f}%")
    for col in ['FTRL Sharpe','AGG Sharpe']:
        disp[col] = disp[col].apply(lambda x: f"{x:.3f}")
    disp['Beats'] = disp['Beats'].apply(lambda x: "✓" if x else "✗")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.subheader("Annual Returns: FTRL vs AGG")
    fig = go.Figure()
    fig.add_bar(x=summary_df['test_year'].astype(str),
                y=summary_df['ftrl_return']*100,
                name='FTRL', marker_color=COLORS['FTRL'])
    fig.add_bar(x=summary_df['test_year'].astype(str),
                y=summary_df['agg_return']*100,
                name='AGG', marker_color=COLORS['AGG'])
    fig.update_layout(barmode='group', template='plotly_dark',
                      yaxis_title='Return (%)', height=400)
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
                          legend=dict(x=1.01, y=1))
        fig.add_hline(y=1.0, line_dash='dash', line_color='white',
                      opacity=0.3, annotation_text='Start')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green = beats AGG | Red = trails AGG")

        st.subheader("Rolling 60-Day Sharpe")
        fig2 = go.Figure()
        for w_id in sorted(all_daily['window_id'].unique()):
            w_df  = all_daily[all_daily['window_id'] == w_id]
            year  = summary_df[summary_df['window_id'] == w_id]['test_year'].values
            label = str(year[0]) if len(year) > 0 else f"W{w_id}"
            fig2.add_scatter(x=w_df.index,
                             y=compute_rolling_sharpe(w_df['net_return']),
                             mode='lines', name=label, line=dict(width=1.5))
        fig2.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3)
        fig2.update_layout(template='plotly_dark', height=400,
                           yaxis_title='Rolling Sharpe')
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Portfolio Weight Allocation")
    if all_daily.empty:
        st.info("Daily data not yet available.")
    else:
        weight_cols = [f'w_{a}' for a in ASSETS if f'w_{a}' in all_daily.columns]
        if weight_cols:
            rows = []
            for w_id in sorted(all_daily['window_id'].unique()):
                w_df = all_daily[all_daily['window_id'] == w_id]
                year = summary_df[summary_df['window_id'] == w_id]['test_year'].values
                row  = {'year': str(year[0]) if len(year) > 0 else f"W{w_id}"}
                for col in weight_cols:
                    row[col.replace('w_', '')] = w_df[col].mean()
                rows.append(row)
            wt_df = pd.DataFrame(rows)
            fig = go.Figure()
            for asset in ASSETS:
                if asset in wt_df.columns:
                    fig.add_bar(x=wt_df['year'], y=wt_df[asset],
                                name=asset,
                                marker_color=COLORS.get(asset,'#888'))
            fig.update_layout(barmode='stack', template='plotly_dark',
                              height=400, yaxis=dict(range=[0,1]))
            st.plotly_chart(fig, use_container_width=True)

            available = [
                (w_id, str(summary_df[summary_df['window_id']==w_id]
                           ['test_year'].values[0]))
                for w_id in sorted(all_daily['window_id'].unique())
                if len(summary_df[summary_df['window_id']==w_id]) > 0
            ]
            if available:
                sel = st.selectbox("Test Year", [y for _, y in available])
                wid = next(w for w, y in available if y == sel)
                sdf = all_daily[all_daily['window_id'] == wid]
                fig3 = go.Figure()
                for col in weight_cols:
                    fig3.add_scatter(
                        x=sdf.index, y=sdf[col], mode='lines',
                        name=col.replace('w_',''), stackgroup='one',
                        line=dict(color=COLORS.get(col.replace('w_',''),'#888')))
                fig3.update_layout(template='plotly_dark', height=400,
                                   yaxis=dict(range=[0,1]))
                st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: WINDOW DETAIL
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Individual Window Analysis")
    avail = summary_df['window_id'].tolist()
    if not avail:
        st.info("No windows complete yet.")
    else:
        sel = st.selectbox("Select Window", avail,
            format_func=lambda x: (
                f"Window {x:02d} — "
                f"Test {summary_df[summary_df['window_id']==x]['test_year'].values[0]}"))
        row = summary_df[summary_df['window_id'] == sel].iloc[0]
        wd  = load_window_daily(sel)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FTRL Return",  f"{row['ftrl_return']*100:+.2f}%",
                  f"{row['excess_return']*100:+.2f}% vs AGG")
        c2.metric("AGG Return",   f"{row['agg_return']*100:+.2f}%")
        c3.metric("FTRL Sharpe",  f"{row['ftrl_sharpe']:.3f}")
        c4.metric("Max Drawdown", f"{row['ftrl_max_dd']*100:+.2f}%")

        if not wd.empty:
            curve = build_equity_curve(wd)
            fig = go.Figure()
            fig.add_scatter(x=wd.index, y=curve, mode='lines', name='FTRL',
                            line=dict(color=COLORS['FTRL'], width=2))
            fig.add_hline(y=1.0, line_dash='dash',
                          line_color='white', opacity=0.4)
            fig.update_layout(template='plotly_dark', height=350,
                              title=f"Equity Curve — Test {row['test_year']}")
            st.plotly_chart(fig, use_container_width=True)

            peak = curve.cummax()
            dd   = (curve - peak) / peak
            fig2 = go.Figure()
            fig2.add_scatter(x=wd.index, y=dd*100, mode='lines',
                             fill='tozeroy',
                             line=dict(color='#FF1744', width=1))
            fig2.update_layout(template='plotly_dark', height=250,
                               title='Drawdown (%)')
            st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: REVERSE WINDOWS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Phase 2 — Reverse Expanding Windows")
    st.caption("Drops oldest year each window — tests on 2025+2026YTD. "
               "Identifies optimal training history length.")

    if reverse_summary_df.empty:
        st.info("Reverse window training not yet run. "
                "Go to Actions → Train FTRL Reverse Windows → "
                "Run workflow → all")
    else:
        n_rev = len(reverse_summary_df)
        st.markdown(f"**{n_rev}/14 reverse windows complete**")

        if 'excess_return' in reverse_summary_df.columns:
            best = reverse_summary_df.loc[
                reverse_summary_df['excess_return'].idxmax()]
            st.success(
                f"🏆 Best reverse window: **R{int(best['window_id']):02d}** "
                f"({best.get('label','—')}) — "
                f"Excess: {best['excess_return']:.2%} | "
                f"Sharpe: {best['ftrl_sharpe']:.3f}"
            )

        show_cols = ['window_id','label','ftrl_total_return',
                     'agg_total_return','excess_return',
                     'ftrl_sharpe','ftrl_max_drawdown']
        if 'n_test_days' in reverse_summary_df.columns:
            show_cols.append('n_test_days')
        disp = reverse_summary_df[
            [c for c in show_cols if c in reverse_summary_df.columns]
        ].copy()
        rename = {'window_id':'Window','label':'Label',
                  'ftrl_total_return':'FTRL Return',
                  'agg_total_return':'AGG Return',
                  'excess_return':'Excess','ftrl_sharpe':'Sharpe',
                  'ftrl_max_drawdown':'MaxDD','n_test_days':'Test Days'}
        disp = disp.rename(columns=rename)
        for col in ['FTRL Return','AGG Return','Excess','MaxDD']:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: f"{x*100:+.2f}%")
        if 'Sharpe' in disp.columns:
            disp['Sharpe'] = disp['Sharpe'].apply(lambda x: f"{x:.3f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        if 'excess_return' in reverse_summary_df.columns:
            bar_c = ['#00C853' if v > 0 else '#FF1744'
                     for v in reverse_summary_df['excess_return']]
            labels = (reverse_summary_df['label'].tolist()
                      if 'label' in reverse_summary_df.columns
                      else reverse_summary_df['window_id'].apply(
                          lambda x: f"R{x}").tolist())
            fig = go.Figure()
            fig.add_bar(
                x=reverse_summary_df['window_id'].apply(lambda x: f"R{x:02d}"),
                y=reverse_summary_df['excess_return'] * 100,
                marker_color=bar_c, text=labels,
                hovertemplate='%{text}<br>Excess: %{y:.2f}%<extra></extra>',
            )
            fig.add_hline(y=0, line_dash='dash',
                          line_color='white', opacity=0.4)
            fig.update_layout(
                template='plotly_dark', height=400,
                title='Excess Return vs AGG by Training Start Year',
                xaxis_title='Reverse Window (R01=2008, R14=2021)',
                yaxis_title='Excess Return (%)',
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "If R08 (starting 2015) outperforms R01 (starting 2008), "
                "pre-2015 data adds noise rather than signal for recent regimes."
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "FTRL Engine | Data: P2SAMAPA/p2-etf-deepwave-dl | "
    "Model: P2SAMAPA/p2-etf-ftrl-engine | "
    "Architecture: Financial Transformer + DDPG | "
    "⚠️ Not financial advice"
)
