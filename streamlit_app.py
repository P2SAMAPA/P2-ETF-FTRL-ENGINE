# streamlit_app.py — FTRL ETF Portfolio Dashboard (Phase 1 + Phase 2)
# Phase 1: Walk-forward back-test + expanding window daily signal
# Phase 2: Reverse expanding windows + side-by-side signal + consensus

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go

st.set_page_config(page_title="FTRL ETF Portfolio Engine", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

HF_REPO  = "P2SAMAPA/p2-etf-ftrl-engine"
GH_PAT   = os.environ.get("GH_PAT", "")
GH_REPO  = os.environ.get("GITHUB_REPO", "P2SAMAPA/P2-ETF-FTRL-ENGINE")
ASSETS   = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV']
COLORS   = {'TLT':'#2196F3','LQD':'#4CAF50','HYG':'#FF9800',
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


# ── GitHub Actions trigger helper ─────────────────────────────────────────────

def trigger_workflow(workflow_file: str) -> tuple[bool, str]:
    """Trigger a GitHub Actions workflow_dispatch. Returns (success, message)."""
    if not GH_PAT:
        return False, "GH_PAT secret not set in Streamlit"
    url = (f"https://api.github.com/repos/{GH_REPO}"
           f"/actions/workflows/{workflow_file}/dispatches")
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {GH_PAT}",
                 "Accept": "application/vnd.github+json"},
        json={"ref": "main"},
        timeout=15,
    )
    if resp.status_code == 204:
        return True, "✅ Triggered successfully"
    return False, f"❌ GitHub API returned {resp.status_code}: {resp.text[:200]}"


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


@st.cache_data(ttl=3600)
def load_reverse_window_daily(window_id: int) -> pd.DataFrame:
    try:
        path = hf_hub_download(repo_id=HF_REPO,
            filename=f"results/reverse_window_{window_id:02d}_daily.csv",
            repo_type="dataset")
        return pd.read_csv(path, parse_dates=['date'], index_col='date')
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_all_reverse_daily() -> pd.DataFrame:
    frames = []
    for w_id in range(1, 15):
        df = load_reverse_window_daily(w_id)
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
        if 'date' not in df.columns:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        for col in ['beats_agg', 'excess_return', 'actual_return', 'agg_return']:
            if col not in df.columns:
                df[col] = np.nan
        return df.sort_values('date', ascending=False).reset_index(drop=True)
    except Exception as e:
        print(f"[load_signal_history] Error: {e}")
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
        if 'date' not in df.columns:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        for col in ['beats_agg', 'excess_return', 'actual_return', 'agg_return']:
            if col not in df.columns:
                df[col] = np.nan
        return df.sort_values('date', ascending=False).reset_index(drop=True)
    except Exception as e:
        print(f"[load_reverse_signal_history] Error: {e}")
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


# ── Consensus engine ──────────────────────────────────────────────────────────

MIN_DAYS_PROVISIONAL = 5
MIN_DAYS_MODERATE    = 15
MIN_DAYS_FULL        = 30
SCORE_WINDOW         = 60
W_RETURN = 0.50
W_SHARPE = 0.30
W_DD     = 0.20


def compute_live_score(history_df: pd.DataFrame, n_total: int) -> tuple:
    if history_df.empty:
        return None, 0.0, 0.0, 0.0, 0, 0
    scored = history_df[history_df['beats_agg'].notna() &
                        history_df['excess_return'].notna()].copy()
    scored = scored.sort_values('date', ascending=True)
    n = len(scored)
    if n == 0:
        return None, 0.0, 0.0, 0.0, 0, 0
    if n_total >= MIN_DAYS_FULL:
        scored      = scored.tail(SCORE_WINDOW)
        window_used = min(SCORE_WINDOW, n)
    else:
        window_used = n
    daily_exc   = scored['excess_return'].values
    avg_excess  = float(daily_exc.mean())
    live_sharpe = float((daily_exc.mean() / (daily_exc.std() + 1e-8)) * np.sqrt(252))
    cum    = np.cumprod(1 + daily_exc)
    peak   = np.maximum.accumulate(cum)
    dd     = (cum - peak) / (peak + 1e-8)
    max_dd = float(dd.min())
    score  = W_RETURN * avg_excess + W_SHARPE * live_sharpe - W_DD * abs(max_dd)
    return score, avg_excess, live_sharpe, max_dd, n, window_used


def consensus_label(n_scored: int) -> str:
    if n_scored < MIN_DAYS_PROVISIONAL:
        return None
    if n_scored < MIN_DAYS_MODERATE:
        return "provisional"
    if n_scored < MIN_DAYS_FULL:
        return "moderate"
    return "full"


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

    # ── Run Signal button ─────────────────────────────────────────────────────
    st.subheader("🎯 Generate Signal")
    st.caption(
        "Run **after** the deepwave-dl dataset has been updated "
        "so yesterday's signal is scored correctly."
    )

    run_expanding = st.button(
        "▶ Run Expanding Signal",
        use_container_width=True,
        type="primary",
        help="Triggers predict.py on GitHub Actions",
    )
    run_reverse = st.button(
        "▶ Run Reverse Signal",
        use_container_width=True,
        type="primary",
        help="Triggers predict_reverse.py on GitHub Actions",
    )
    run_both = st.button(
        "▶ Run Both Signals",
        use_container_width=True,
        type="primary",
        help="Triggers both workflows simultaneously",
    )

    if run_expanding or run_both:
        ok, msg = trigger_workflow("predict.yml")
        if ok:
            st.success("Expanding signal triggered ✅\n~10-20 min to complete.")
        else:
            st.error(f"Expanding: {msg}")

    if run_reverse or run_both:
        ok, msg = trigger_workflow("predict_reverse.yml")
        if ok:
            st.success("Reverse signal triggered ✅\n~10-20 min to complete.")
        else:
            st.error(f"Reverse: {msg}")

    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────

st.title("📈 FTRL ETF Portfolio Engine")
st.caption("Financial Transformer Reinforcement Learning | Walk-Forward Back-test vs AGG")

# Load everything
summary_df         = load_summary()
reverse_summary_df = load_reverse_summary()
all_daily          = load_all_daily()
all_reverse_daily  = load_all_reverse_daily()
signal_exp         = load_latest_signal()
signal_rev         = load_latest_reverse_signal()
history_exp        = load_signal_history()
history_rev        = load_reverse_signal_history()

# Consensus scoring
exp_n_scored = int(history_exp['beats_agg'].notna().sum()) if not history_exp.empty else 0
rev_n_scored = int(history_rev['beats_agg'].notna().sum()) if not history_rev.empty else 0
exp_n_total  = len(history_exp) if not history_exp.empty else 0
rev_n_total  = len(history_rev) if not history_rev.empty else 0

exp_score, exp_avg_exc, exp_sharpe, exp_max_dd, exp_n, exp_win = compute_live_score(history_exp, exp_n_total)
rev_score, rev_avg_exc, rev_sharpe, rev_max_dd, rev_n, rev_win = compute_live_score(history_rev, rev_n_total)

min_n     = min(exp_n_scored, rev_n_scored)
con_level = consensus_label(min_n)

# ── SIGNAL PANEL ──────────────────────────────────────────────────────────────

has_exp = bool(signal_exp)
has_rev = bool(signal_rev)

if has_exp or has_rev:
    st.markdown("---")

    if has_exp and has_rev:
        exp_etf = signal_exp.get('signal')
        rev_etf = signal_rev.get('signal')

        col_e, col_r, col_c = st.columns(3)
        with col_e:
            st.markdown(signal_card(signal_exp, "EXPANDING SIGNAL"),
                        unsafe_allow_html=True)
        with col_r:
            st.markdown(signal_card(signal_rev, "REVERSE SIGNAL", '#FF9800'),
                        unsafe_allow_html=True)

        with col_c:
            if con_level is None:
                c    = '#78909C'
                etf  = '—'
                lbl  = f"⏳ BUILDING HISTORY ({min_n}/{MIN_DAYS_PROVISIONAL} days)"
                body = f"Need {MIN_DAYS_PROVISIONAL}+ scored days for consensus"
                desc = f"Expanding→{exp_etf} | Reverse→{rev_etf}"
            elif exp_score is not None and rev_score is not None:
                if exp_score >= rev_score:
                    winner_etf   = exp_etf
                    winner_label = "EXPANDING"
                    score_gap    = exp_score - rev_score
                else:
                    winner_etf   = rev_etf
                    winner_label = "REVERSE"
                    score_gap    = rev_score - exp_score
                conf_tag = {"provisional": "⚠️ PROVISIONAL",
                            "moderate":    "🔶 MODERATE",
                            "full":        "✅ HIGH CONVICTION"}[con_level]
                c    = COLORS.get(winner_etf, '#E91E63')
                etf  = winner_etf
                lbl  = f"{conf_tag} CONSENSUS ({min_n} days)"
                desc = ETF_DESC.get(winner_etf, '')
                body = f"{winner_label} wins | Score gap: {score_gap:.4f}"
            else:
                if exp_etf == rev_etf:
                    c, etf = '#00E676', exp_etf
                    lbl    = "✅ CONSENSUS (no history yet)"
                    body   = "Both models agree"
                    desc   = ETF_DESC.get(etf, '')
                else:
                    c, etf = '#FF5252', exp_etf
                    lbl    = "⚠️ NO HISTORY — USING EXPANDING"
                    body   = f"Expanding→{exp_etf} | Reverse→{rev_etf}"
                    desc   = ETF_DESC.get(exp_etf, '')

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{c}22,{c}44);
                        border:2px solid {c};border-radius:12px;
                        padding:18px 22px;text-align:center;">
                <div style="font-size:12px;color:#aaa;margin-bottom:4px;">{lbl}</div>
                <div style="font-size:44px;font-weight:900;color:{c};
                            letter-spacing:2px;">{etf}</div>
                <div style="font-size:13px;color:#ccc;margin-top:4px;">{desc}</div>
                <div style="font-size:14px;font-weight:600;color:{c};
                            margin-top:8px;">{body}</div>
            </div>""", unsafe_allow_html=True)

            if con_level is not None and exp_score is not None:
                with st.expander("📊 Score breakdown"):
                    if min_n >= MIN_DAYS_FULL:
                        st.caption(f"🔄 Rolling {SCORE_WINDOW}-day window active")
                    else:
                        st.caption(f"📈 Using all {min_n} available scored days")
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.markdown("**Expanding**")
                        st.markdown(f"Score: `{exp_score:.4f}`")
                        st.markdown(f"Avg daily excess: `{exp_avg_exc:+.3%}`")
                        st.markdown(f"Live Sharpe: `{exp_sharpe:.3f}`")
                        st.markdown(f"Live Max DD: `{exp_max_dd:.2%}`")
                        st.markdown(f"Scored days: `{exp_n}` (window: {exp_win})")
                    with sc2:
                        st.markdown("**Reverse**")
                        st.markdown(f"Score: `{rev_score:.4f}`")
                        st.markdown(f"Avg daily excess: `{rev_avg_exc:+.3%}`")
                        st.markdown(f"Live Sharpe: `{rev_sharpe:.3f}`")
                        st.markdown(f"Live Max DD: `{rev_max_dd:.2%}`")
                        st.markdown(f"Scored days: `{rev_n}` (window: {rev_win})")

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
                    prefix    = "🏆 " if etf == signal_exp.get('signal') else "   "
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

    # ── Consensus progress tracker ────────────────────────────────────────────
    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)

    # Build day-by-day log from signal history
    # Each entry in history_exp has a date — total entries = signals saved
    # scored entries = actual_return is not null
    all_days = []
    if not history_exp.empty and 'date' in history_exp.columns:
        exp_sorted = history_exp.sort_values('date', ascending=True).reset_index(drop=True)
        for i, row in exp_sorted.iterrows():
            day_num  = i + 1
            scored   = pd.notna(row.get('beats_agg'))
            date_str = row['date'].strftime('%d %b') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            etf      = row.get('signal', '?')
            if scored:
                beats    = row.get('beats_agg')
                exc      = row.get('excess_return', 0)
                result   = "✓" if beats else "✗"
                exc_str  = f"{exc*100:+.2f}%" if pd.notna(exc) else ""
                status   = "scored"
            else:
                result  = "⏳"
                exc_str = ""
                status  = "saved"
            all_days.append({
                "day":     day_num,
                "date":    date_str,
                "etf":     etf,
                "result":  result,
                "exc":     exc_str,
                "status":  status,
            })

    total_days  = len(all_days)
    scored_days = sum(1 for d in all_days if d["status"] == "scored")
    target      = MIN_DAYS_PROVISIONAL

    if total_days > 0:
        # Progress bar toward consensus activation
        pct      = min(scored_days / target, 1.0)
        bar_fill = int(pct * 20)
        bar_str  = "█" * bar_fill + "░" * (20 - bar_fill)

        if scored_days >= MIN_DAYS_FULL:
            bar_color  = "#00C853"
            status_lbl = f"✅ HIGH CONVICTION — rolling {SCORE_WINDOW}-day window active"
        elif scored_days >= MIN_DAYS_MODERATE:
            bar_color  = "#FF9800"
            status_lbl = f"🔶 MODERATE — {MIN_DAYS_FULL - scored_days} days to high conviction"
        elif scored_days >= target:
            bar_color  = "#FF9800"
            status_lbl = f"⚠️ PROVISIONAL — {MIN_DAYS_MODERATE - scored_days} days to moderate"
        else:
            bar_color  = "#78909C"
            status_lbl = f"⏳ Building history — {target - scored_days} more scored day(s) needed"

        st.markdown(f"""
        <div style="background:rgba(120,144,156,0.08);border:1px solid rgba(120,144,156,0.25);
                    border-radius:10px;padding:14px 18px;margin-bottom:8px;">
            <div style="font-size:12px;letter-spacing:2px;color:#aaa;margin-bottom:8px;">
                CONSENSUS PROGRESS — {scored_days}/{target} SCORED DAYS
            </div>
            <div style="font-family:monospace;font-size:15px;color:{bar_color};
                        letter-spacing:1px;margin-bottom:6px;">[{bar_str}] {pct*100:.0f}%</div>
            <div style="font-size:13px;color:#ccc;">{status_lbl}</div>
        </div>
        """, unsafe_allow_html=True)

        # Day-by-day log — show last 10 days max, most recent first
        recent = list(reversed(all_days[-10:]))
        cols   = st.columns(min(len(recent), 10))
        for col, d in zip(cols, recent):
            color = ("#00C853" if d["result"] == "✓"
                     else "#FF1744" if d["result"] == "✗"
                     else "#78909C")
            etf_color = COLORS.get(d["etf"], "#aaa")
            col.markdown(f"""
            <div style="text-align:center;background:rgba(0,0,0,0.15);
                        border-radius:8px;padding:8px 4px;border:1px solid {color}33;">
                <div style="font-size:10px;color:#888;">Day {d['day']}</div>
                <div style="font-size:11px;color:#aaa;">{d['date']}</div>
                <div style="font-size:16px;font-weight:700;color:{etf_color};">{d['etf']}</div>
                <div style="font-size:18px;">{d['result']}</div>
                <div style="font-size:10px;color:{color};">{d['exc']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

# ── AUDIT TRAIL ───────────────────────────────────────────────────────────────

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
                    lambda x: "✓" if x is True else ("✗" if x is False else "⏳"))
                merged['Rev ✓']    = merged['Rev Beats'].apply(
                    lambda x: "✓" if x is True else ("✗" if x is False else "⏳"))
                merged['🤝']       = merged['Agree'].apply(
                    lambda x: "✅" if x else "❌")

                c1, c2 = st.columns(2)
                c1.metric("Agreement Rate",
                          f"{merged['Agree'].mean():.0%}",
                          f"{int(merged['Agree'].sum())}/{len(merged)} days")
                scored_con = merged[merged['Agree'] & merged['Exp Beats'].notna()]
                if not scored_con.empty:
                    c2.metric("Consensus Hit Rate",
                              f"{scored_con['Exp Beats'].mean():.0%}",
                              f"{int(scored_con['Exp Beats'].sum())}/{len(scored_con)} days")

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

    if 'excess_return' in summary_df.columns:
        has_live_exp = (
            'live_excess_return' in summary_df.columns and
            summary_df['live_excess_return'].notna().any()
        )
        if has_live_exp:
            live_pool = summary_df[summary_df['live_excess_return'].notna()]
            best_exp  = live_pool.loc[live_pool['live_excess_return'].idxmax()]
            n_days    = int(best_exp.get('live_n_days') or 0)
            st.success(
                f"🏆 Best window (live 2025+, {n_days} days): "
                f"**W{int(best_exp['window_id']):02d}** "
                f"(trained 2008–{str(best_exp.get('train_end',''))[:4]}) — "
                f"Live Excess: {best_exp['live_excess_return']:.2%} | "
                f"Live Sharpe: {float(best_exp.get('live_sharpe') or 0):.3f}"
            )
        else:
            best_exp = summary_df.loc[summary_df['excess_return'].idxmax()]
            st.success(
                f"🏆 Best window (historical): "
                f"**W{int(best_exp['window_id']):02d}** "
                f"(Test {int(best_exp['test_year'])}) — "
                f"Excess: {best_exp['excess_return']:.2%} | "
                f"Sharpe: {best_exp['ftrl_sharpe']:.3f}"
            )

    if signal_exp:
        train_start = signal_exp.get('train_start', '')
        train_end   = signal_exp.get('train_end', '')
        basis       = signal_exp.get('basis', '')
        trained_on  = signal_exp.get('trained_on', '—')
        if train_start and train_end:
            detail = f"from {train_start} → {train_end}"
            detail += f" · selected by {basis}" if basis else ""
            st.info(f"ℹ️ Today's expanding signal was trained on: **{trained_on}** ({detail})")
        else:
            st.info(f"ℹ️ Today's expanding signal was trained on: **{trained_on}**")

    ov_t1, ov_t2, ov_t3, ov_t4 = st.tabs([
        "📊 Summary", "📈 Equity Curves", "⚖️ Weights", "🔍 Window Detail"
    ])

    with ov_t1:
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

        if 'excess_return' in summary_df.columns:
            bar_c = ['#00C853' if v > 0 else '#FF1744'
                     for v in summary_df['excess_return']]
            fig = go.Figure()
            fig.add_bar(x=summary_df['test_year'].astype(str),
                        y=summary_df['excess_return'] * 100,
                        marker_color=bar_c,
                        hovertemplate='%{x}<br>Excess: %{y:.2f}%<extra></extra>')
            fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
            fig.update_layout(template='plotly_dark', height=400,
                              title='Excess Return vs AGG by Test Year',
                              xaxis_title='Test Year',
                              yaxis_title='Excess Return (%)')
            st.plotly_chart(fig, use_container_width=True, key='ov_t1_excess_bar')

    with ov_t2:
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
            fig.add_hline(y=1.0, line_dash='dash', line_color='white', opacity=0.3)
            st.plotly_chart(fig, use_container_width=True, key='ov_t2_equity')
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
            st.plotly_chart(fig2, use_container_width=True, key='ov_t2_sharpe')

    with ov_t3:
        st.subheader("Average Portfolio Weight Allocation")
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
                                    name=asset, marker_color=COLORS.get(asset,'#888'))
                fig.update_layout(barmode='stack', template='plotly_dark',
                                  height=400, yaxis=dict(range=[0,1]))
                st.plotly_chart(fig, use_container_width=True, key='ov_t3_weights_bar')

                available = [
                    (w_id, str(summary_df[summary_df['window_id']==w_id]
                               ['test_year'].values[0]))
                    for w_id in sorted(all_daily['window_id'].unique())
                    if len(summary_df[summary_df['window_id']==w_id]) > 0
                ]
                if available:
                    sel = st.selectbox("Test Year", [y for _, y in available],
                                       key='exp_weight_sel')
                    wid = next(w for w, y in available if y == sel)
                    sdf = all_daily[all_daily['window_id'] == wid]
                    fig3 = go.Figure()
                    for col in weight_cols:
                        fig3.add_scatter(x=sdf.index, y=sdf[col], mode='lines',
                                         name=col.replace('w_',''), stackgroup='one',
                                         line=dict(color=COLORS.get(col.replace('w_',''),'#888')))
                    fig3.update_layout(template='plotly_dark', height=400,
                                       yaxis=dict(range=[0,1]))
                    st.plotly_chart(fig3, use_container_width=True, key='ov_t3_weights_ts')

    with ov_t4:
        st.subheader("Individual Window Analysis")
        avail = summary_df['window_id'].tolist()
        if not avail:
            st.info("No windows complete yet.")
        else:
            sel = st.selectbox("Select Window", avail,
                format_func=lambda x: (
                    f"W{x:02d} — "
                    f"Test {summary_df[summary_df['window_id']==x]['test_year'].values[0]}"),
                key='exp_detail_sel')
            row = summary_df[summary_df['window_id'] == sel].iloc[0]
            wd  = load_window_daily(sel)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("FTRL Return", f"{row['ftrl_return']*100:+.2f}%",
                      f"{row['excess_return']*100:+.2f}% vs AGG")
            c2.metric("AGG Return",   f"{row['agg_return']*100:+.2f}%")
            c3.metric("FTRL Sharpe",  f"{row['ftrl_sharpe']:.3f}")
            c4.metric("Max Drawdown", f"{row['ftrl_max_dd']*100:+.2f}%")
            if not wd.empty:
                curve = build_equity_curve(wd)
                fig = go.Figure()
                fig.add_scatter(x=wd.index, y=curve, mode='lines', name='FTRL',
                                line=dict(color=COLORS['FTRL'], width=2))
                fig.add_hline(y=1.0, line_dash='dash', line_color='white', opacity=0.4)
                fig.update_layout(template='plotly_dark', height=350,
                                  title=f"Equity Curve — Test {row['test_year']}")
                st.plotly_chart(fig, use_container_width=True, key='ov_t4_equity')
                peak = curve.cummax()
                dd   = (curve - peak) / peak
                fig2 = go.Figure()
                fig2.add_scatter(x=wd.index, y=dd*100, mode='lines', fill='tozeroy',
                                 line=dict(color='#FF1744', width=1))
                fig2.update_layout(template='plotly_dark', height=250, title='Drawdown (%)')
                st.plotly_chart(fig2, use_container_width=True, key='ov_t4_dd')

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
        fig.add_hline(y=1.0, line_dash='dash', line_color='white', opacity=0.3)
        st.plotly_chart(fig, use_container_width=True, key='tab2_equity')
        st.caption("Green = beats AGG | Red = trails AGG")

        fig2 = go.Figure()
        for w_id in sorted(all_daily['window_id'].unique()):
            w_df  = all_daily[all_daily['window_id'] == w_id]
            year  = summary_df[summary_df['window_id'] == w_id]['test_year'].values
            label = str(year[0]) if len(year) > 0 else f"W{w_id}"
            fig2.add_scatter(x=w_df.index,
                             y=compute_rolling_sharpe(w_df['net_return']),
                             mode='lines', name=label, line=dict(width=1.5))
        fig2.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3)
        fig2.update_layout(template='plotly_dark', height=400, yaxis_title='Rolling Sharpe')
        st.plotly_chart(fig2, use_container_width=True, key='tab2_sharpe')

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
                    fig.add_bar(x=wt_df['year'], y=wt_df[asset], name=asset,
                                marker_color=COLORS.get(asset,'#888'))
            fig.update_layout(barmode='stack', template='plotly_dark',
                              height=400, yaxis=dict(range=[0,1]))
            st.plotly_chart(fig, use_container_width=True, key='tab3_weights_bar')

            available = [
                (w_id, str(summary_df[summary_df['window_id']==w_id]
                           ['test_year'].values[0]))
                for w_id in sorted(all_daily['window_id'].unique())
                if len(summary_df[summary_df['window_id']==w_id]) > 0
            ]
            if available:
                sel = st.selectbox("Test Year", [y for _, y in available],
                                   key='tab3_weight_sel')
                wid = next(w for w, y in available if y == sel)
                sdf = all_daily[all_daily['window_id'] == wid]
                fig3 = go.Figure()
                for col in weight_cols:
                    fig3.add_scatter(x=sdf.index, y=sdf[col], mode='lines',
                                     name=col.replace('w_',''), stackgroup='one',
                                     line=dict(color=COLORS.get(col.replace('w_',''),'#888')))
                fig3.update_layout(template='plotly_dark', height=400,
                                   yaxis=dict(range=[0,1]))
                st.plotly_chart(fig3, use_container_width=True, key='tab3_weights_ts')

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
                f"Test {summary_df[summary_df['window_id']==x]['test_year'].values[0]}"),
            key='tab4_detail_sel')
        row = summary_df[summary_df['window_id'] == sel].iloc[0]
        wd  = load_window_daily(sel)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FTRL Return", f"{row['ftrl_return']*100:+.2f}%",
                  f"{row['excess_return']*100:+.2f}% vs AGG")
        c2.metric("AGG Return",   f"{row['agg_return']*100:+.2f}%")
        c3.metric("FTRL Sharpe",  f"{row['ftrl_sharpe']:.3f}")
        c4.metric("Max Drawdown", f"{row['ftrl_max_dd']*100:+.2f}%")
        if not wd.empty:
            curve = build_equity_curve(wd)
            fig = go.Figure()
            fig.add_scatter(x=wd.index, y=curve, mode='lines', name='FTRL',
                            line=dict(color=COLORS['FTRL'], width=2))
            fig.add_hline(y=1.0, line_dash='dash', line_color='white', opacity=0.4)
            fig.update_layout(template='plotly_dark', height=350,
                              title=f"Equity Curve — Test {row['test_year']}")
            st.plotly_chart(fig, use_container_width=True, key='tab4_equity')
            peak = curve.cummax()
            dd   = (curve - peak) / peak
            fig2 = go.Figure()
            fig2.add_scatter(x=wd.index, y=dd*100, mode='lines', fill='tozeroy',
                             line=dict(color='#FF1744', width=1))
            fig2.update_layout(template='plotly_dark', height=250, title='Drawdown (%)')
            st.plotly_chart(fig2, use_container_width=True, key='tab4_dd')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: REVERSE WINDOWS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Phase 2 — Reverse Expanding Windows")
    st.caption("Drops oldest year each window — tests on 2025+2026YTD. "
               "Identifies optimal training history length.")

    if reverse_summary_df.empty:
        st.info("Reverse window training not yet run. "
                "Go to Actions → Train FTRL Reverse Windows → Run workflow → all")
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

        if signal_rev:
            st.info(
                f"ℹ️ Today's reverse signal was trained on: "
                f"**{signal_rev.get('trained_on', '—')}** "
                f"(from {signal_rev.get('train_start','?')} → "
                f"{signal_rev.get('train_end','?')})"
            )

        rev_t1, rev_t2, rev_t3, rev_t4 = st.tabs([
            "📊 Summary", "📈 Equity Curves", "⚖️ Weights", "🔍 Window Detail"
        ])

        with rev_t1:
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
                bar_c  = ['#00C853' if v > 0 else '#FF1744'
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
                    hovertemplate='%{text}<br>Excess: %{y:.2f}%<extra></extra>')
                fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
                fig.update_layout(template='plotly_dark', height=400,
                                  title='Excess Return vs AGG by Training Start Year',
                                  xaxis_title='Reverse Window',
                                  yaxis_title='Excess Return (%)')
                st.plotly_chart(fig, use_container_width=True, key='rev_t1_excess_bar')

        with rev_t2:
            if all_reverse_daily.empty:
                st.info("Daily data not yet available.")
            else:
                fig = go.Figure()
                for w_id in sorted(all_reverse_daily['window_id'].unique()):
                    w_df  = all_reverse_daily[all_reverse_daily['window_id'] == w_id]
                    curve = build_equity_curve(w_df)
                    row   = reverse_summary_df[reverse_summary_df['window_id'] == w_id]
                    label = row['label'].values[0] if len(row) > 0 else f"R{w_id}"
                    exc   = row['excess_return'].values[0] if len(row) > 0 else 0
                    color = '#00C853' if exc > 0 else '#FF1744'
                    fig.add_scatter(x=w_df.index, y=curve, mode='lines', name=label,
                                    line=dict(color=color, width=1.5), opacity=0.75)
                fig.add_hline(y=1.0, line_dash='dash', line_color='white', opacity=0.3)
                fig.update_layout(template='plotly_dark', height=500,
                                  yaxis_title='Portfolio Value (normalised to 1.0)',
                                  legend=dict(x=1.01, y=1))
                st.plotly_chart(fig, use_container_width=True, key='rev_t2_equity')

                fig2 = go.Figure()
                for w_id in sorted(all_reverse_daily['window_id'].unique()):
                    w_df  = all_reverse_daily[all_reverse_daily['window_id'] == w_id]
                    row   = reverse_summary_df[reverse_summary_df['window_id'] == w_id]
                    label = row['label'].values[0] if len(row) > 0 else f"R{w_id}"
                    fig2.add_scatter(x=w_df.index,
                                     y=compute_rolling_sharpe(w_df['net_return']),
                                     mode='lines', name=label, line=dict(width=1.5))
                fig2.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3)
                fig2.update_layout(template='plotly_dark', height=400,
                                   yaxis_title='Rolling Sharpe')
                st.plotly_chart(fig2, use_container_width=True, key='rev_t2_sharpe')

        with rev_t3:
            if all_reverse_daily.empty:
                st.info("Daily data not yet available.")
            else:
                weight_cols = [f'w_{a}' for a in ASSETS
                               if f'w_{a}' in all_reverse_daily.columns]
                if weight_cols:
                    rows = []
                    for w_id in sorted(all_reverse_daily['window_id'].unique()):
                        w_df  = all_reverse_daily[all_reverse_daily['window_id'] == w_id]
                        row_s = reverse_summary_df[reverse_summary_df['window_id'] == w_id]
                        label = row_s['label'].values[0] if len(row_s) > 0 else f"R{w_id}"
                        row   = {'label': label}
                        for col in weight_cols:
                            row[col.replace('w_', '')] = w_df[col].mean()
                        rows.append(row)
                    wt_df = pd.DataFrame(rows)
                    fig = go.Figure()
                    for asset in ASSETS:
                        if asset in wt_df.columns:
                            fig.add_bar(x=wt_df['label'], y=wt_df[asset], name=asset,
                                        marker_color=COLORS.get(asset, '#888'))
                    fig.update_layout(barmode='stack', template='plotly_dark',
                                      height=400, yaxis=dict(range=[0,1]))
                    st.plotly_chart(fig, use_container_width=True, key='rev_t3_weights_bar')

                    available_rev = [
                        (w_id, reverse_summary_df[
                            reverse_summary_df['window_id'] == w_id]['label'].values[0])
                        for w_id in sorted(all_reverse_daily['window_id'].unique())
                        if len(reverse_summary_df[
                            reverse_summary_df['window_id'] == w_id]) > 0
                    ]
                    if available_rev:
                        sel_label = st.selectbox("Select Reverse Window",
                                                 [lbl for _, lbl in available_rev],
                                                 key='rev_weight_sel')
                        sel_wid = next(w for w, lbl in available_rev if lbl == sel_label)
                        sel_df  = all_reverse_daily[all_reverse_daily['window_id'] == sel_wid]
                        fig3 = go.Figure()
                        for col in weight_cols:
                            fig3.add_scatter(x=sel_df.index, y=sel_df[col], mode='lines',
                                             name=col.replace('w_', ''), stackgroup='one',
                                             line=dict(color=COLORS.get(col.replace('w_',''),'#888')))
                        fig3.update_layout(template='plotly_dark', height=400,
                                           yaxis=dict(range=[0,1]))
                        st.plotly_chart(fig3, use_container_width=True, key='rev_t3_weights_ts')

        with rev_t4:
            if all_reverse_daily.empty:
                st.info("Daily data not yet available.")
            else:
                available_rev2 = [
                    (w_id, reverse_summary_df[
                        reverse_summary_df['window_id'] == w_id]['label'].values[0])
                    for w_id in sorted(all_reverse_daily['window_id'].unique())
                    if len(reverse_summary_df[
                        reverse_summary_df['window_id'] == w_id]) > 0
                ]
                if available_rev2:
                    sel_label2 = st.selectbox("Select Reverse Window",
                                              [lbl for _, lbl in available_rev2],
                                              key='rev_detail_sel')
                    sel_wid2 = next(w for w, lbl in available_rev2 if lbl == sel_label2)
                    row2     = reverse_summary_df[
                        reverse_summary_df['window_id'] == sel_wid2].iloc[0]
                    wd2      = load_reverse_window_daily(sel_wid2)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("FTRL Return",
                              f"{row2['ftrl_total_return']*100:+.2f}%",
                              f"{row2['excess_return']*100:+.2f}% vs AGG")
                    c2.metric("AGG Return",  f"{row2['agg_total_return']*100:+.2f}%")
                    c3.metric("FTRL Sharpe", f"{row2['ftrl_sharpe']:.3f}")
                    c4.metric("Max Drawdown",f"{row2['ftrl_max_drawdown']*100:+.2f}%")
                    if not wd2.empty:
                        curve2 = build_equity_curve(wd2)
                        fig4   = go.Figure()
                        fig4.add_scatter(x=wd2.index, y=curve2, mode='lines', name='FTRL',
                                         line=dict(color=COLORS['FTRL'], width=2))
                        fig4.add_hline(y=1.0, line_dash='dash', line_color='white', opacity=0.4)
                        fig4.update_layout(template='plotly_dark', height=350,
                                           title=f"Equity Curve — {sel_label2}")
                        st.plotly_chart(fig4, use_container_width=True, key='rev_t4_equity')
                        peak2 = curve2.cummax()
                        dd2   = (curve2 - peak2) / peak2
                        fig5  = go.Figure()
                        fig5.add_scatter(x=wd2.index, y=dd2*100, mode='lines', fill='tozeroy',
                                         line=dict(color='#FF1744', width=1))
                        fig5.update_layout(template='plotly_dark', height=250, title='Drawdown (%)')
                        st.plotly_chart(fig5, use_container_width=True, key='rev_t4_dd')

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "FTRL Engine | Data: P2SAMAPA/p2-etf-deepwave-dl | "
    "Model: P2SAMAPA/p2-etf-ftrl-engine | "
    "Architecture: Financial Transformer + DDPG | "
    "⚠️ Not financial advice"
)
