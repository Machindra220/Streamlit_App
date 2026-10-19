"""
EMA Squeeze Breakout Screener
Detects stocks in uptrend where EMA 10/20/50 recently compressed,
then price broke out above the cluster — the pattern visible in the
annotated screenshots.
"""

import io
import json
import os
import sqlite3
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EMA Squeeze Screener",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS  (same dark theme as main app)
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] { background:#0E1117; color:#C9D1D9; }
[data-testid="stSidebar"] { background:#131722; border-right:1px solid #21262D; }
h1 { color:#A78BFA !important; font-weight:700; letter-spacing:-.5px; }
h2, h3 { color:#818CF8 !important; }
.stProgress > div > div > div { background:#7C3AED !important; }
.stButton > button {
    background:#7C3AED !important; color:#fff !important;
    border:none !important; border-radius:6px !important; font-weight:600 !important;
}
[data-testid="metric-container"] {
    background:#131722; border:1px solid #30363D;
    border-radius:10px; padding:12px 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE  (shared DB file with main screener)
# ─────────────────────────────────────────────
try:
    _BASE = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(_BASE, "..", "screener_history.db")
except NameError:
    DB_PATH = "screener_history.db"

DB_PATH = os.path.normpath(DB_PATH)


def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _init_pattern_table():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS pattern_scan_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                universe        TEXT    NOT NULL,
                total_processed INTEGER NOT NULL DEFAULT 0,
                passed_count    INTEGER NOT NULL DEFAULT 0,
                params_summary  TEXT,
                results_json    TEXT
            )
        """)


def _save_pattern_scan(universe, total, passed, params_summary, df):
    rjson = df.to_json(orient="records") if (df is not None and not df.empty) else "[]"
    with _conn() as c:
        c.execute(
            "INSERT INTO pattern_scan_history "
            "(timestamp,universe,total_processed,passed_count,params_summary,results_json) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), universe, total, passed, params_summary, rjson),
        )
        c.execute("""
            DELETE FROM pattern_scan_history WHERE id NOT IN (
                SELECT id FROM pattern_scan_history ORDER BY id DESC LIMIT 5
            )
        """)


def _load_pattern_history():
    with _conn() as c:
        return c.execute(
            "SELECT id,timestamp,universe,total_processed,passed_count,params_summary,results_json "
            "FROM pattern_scan_history ORDER BY id DESC LIMIT 5"
        ).fetchall()


_init_pattern_table()

# ─────────────────────────────────────────────
# TICKER UNIVERSE  (same CSV files as main app)
# ─────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(DB_PATH), "data")


def _load_nifty500():
    try:
        df = pd.read_csv(os.path.join(_DATA_DIR, "nifty_500.csv"), encoding="utf-8-sig")
        col = df.columns[0]
        return [s.strip() + ".NS" for s in df[col].dropna() if str(s).strip()]
    except Exception:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]


def _load_sp500():
    try:
        df = pd.read_csv(os.path.join(_DATA_DIR, "sp500.csv"))
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        return [str(s).strip().replace(".", "-") for s in df[col].dropna() if str(s).strip()]
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]


@st.cache_data(ttl=86400, show_spinner=False)
def _get_sp500_wiki():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return _load_sp500()


# ─────────────────────────────────────────────
# EMA HELPERS
# ─────────────────────────────────────────────
def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────
# PATTERN DETECTION
# ─────────────────────────────────────────────
def detect_squeeze_breakout(hist: pd.DataFrame, params: dict) -> tuple:
    """
    Returns (True, metrics_dict) when the stock shows the EMA-squeeze-breakout
    pattern:
      1. Uptrend  : price > EMA200, EMA50 > EMA200
      2. Squeeze  : within the last `lookback` bars, the spread between
                    EMA10 / EMA20 / EMA50 dropped below `comp_pct`% of EMA50
      3. Breakout : price is now above all three compressed EMAs AND
                    bars since the tightest-squeeze bar ≤ `max_bars_since`
      4. Extension: price not more than `max_ext_pct`% above EMA50 (still buyable)
    """
    c = hist["Close"]
    v = hist["Volume"]
    if len(c) < 210:
        return False, {}

    e10  = _ema(c, 10)
    e20  = _ema(c, 20)
    e50  = _ema(c, 50)
    e200 = _ema(c, 200)

    price = float(c.iloc[-1])
    cur_e10  = float(e10.iloc[-1])
    cur_e20  = float(e20.iloc[-1])
    cur_e50  = float(e50.iloc[-1])
    cur_e200 = float(e200.iloc[-1])

    # 1. Uptrend
    if price < cur_e200 or cur_e50 < cur_e200:
        return False, {}

    # 2. Spread series over the lookback window
    lookback = params.get("lookback", 40)
    ema_mat = pd.concat([e10, e20, e50], axis=1)
    ema_mat.columns = ["e10", "e20", "e50"]
    ema_mat["spread_pct"] = (
        (ema_mat.max(axis=1) - ema_mat.min(axis=1)) / ema_mat["e50"] * 100
    )
    window = ema_mat["spread_pct"].iloc[-lookback:]
    min_spread = float(window.min())
    comp_threshold = params.get("comp_pct", 2.5)

    if min_spread > comp_threshold:
        return False, {}

    # 3. Bars since the squeeze low
    squeeze_pos  = int(window.values.argmin())           # position inside window
    bars_since   = len(window) - 1 - squeeze_pos         # 0 = happening right now
    max_bars     = params.get("max_bars_since", 25)

    if bars_since > max_bars:
        return False, {}

    # Price must be above all three EMAs (breakout confirmed)
    if price < cur_e10 or price < cur_e20 or price < cur_e50:
        return False, {}

    # 4. Extension check
    ext_pct = (price - cur_e50) / cur_e50 * 100
    max_ext = params.get("max_ext_pct", 20.0)
    if ext_pct > max_ext:
        return False, {}

    # Volume expansion vs 50-bar average
    avg_vol    = float(v.iloc[-50:-1].mean()) if len(v) > 50 else float(v.mean())
    recent_vol = float(v.iloc[-5:].mean())
    vol_ratio  = round(recent_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    # Pattern score: lower min_spread + more recent squeeze + higher vol = better
    score = round(
        (comp_threshold - min_spread) * 20          # tighter squeeze → higher
        + max(0, max_bars - bars_since) * 1.5        # more recent → higher
        + min(vol_ratio, 3.0) * 5,                   # volume expansion bonus
        1,
    )

    metrics = {
        "min_spread_pct":    round(min_spread, 2),
        "cur_spread_pct":    round(float(ema_mat["spread_pct"].iloc[-1]), 2),
        "bars_since_squeeze": int(bars_since),
        "ext_from_50ema%":   round(ext_pct, 2),
        "vol_ratio":         vol_ratio,
        "score":             score,
    }
    return True, metrics


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Pattern Screener")
st.sidebar.markdown("---")

universe_choice = st.sidebar.selectbox(
    "📊 Universe",
    ["US Top 500 (S&P 500)", "NSE Nifty 500", "Custom Tickers"],
)

custom_input = ""
if universe_choice == "Custom Tickers":
    custom_input = st.sidebar.text_area(
        "Tickers (comma-separated)",
        placeholder="AAPL, MSFT, RELIANCE.NS",
        height=100,
    )

max_tickers = st.sidebar.slider(
    "Max Tickers to Scan", 10, 500, 200, step=10,
    help="Start with 100–200 for speed.",
)

st.sidebar.markdown("### 🎛️ Pattern Parameters")

comp_pct = st.sidebar.slider(
    "Max EMA Spread % (squeeze tightness)",
    min_value=0.5, max_value=6.0, value=2.5, step=0.25,
    help="Looser = more results; tighter = higher quality squeeze. 2–3% is ideal.",
)
lookback = st.sidebar.slider(
    "Squeeze Lookback Bars", 15, 60, 40, step=5,
    help="How many recent daily bars to search for a squeeze low.",
)
max_bars_since = st.sidebar.slider(
    "Max Bars Since Squeeze Low", 5, 30, 20, step=5,
    help="How recently the squeeze must have occurred.",
)
max_ext_pct = st.sidebar.slider(
    "Max Extension from 50 EMA %", 5, 40, 20, step=5,
    help="Filters out stocks that have already run too far.",
)

params = dict(
    comp_pct=comp_pct,
    lookback=lookback,
    max_bars_since=max_bars_since,
    max_ext_pct=max_ext_pct,
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Scan for Pattern", use_container_width=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_t, col_ts = st.columns([3, 1])
with col_t:
    st.markdown("# 🔍 EMA Squeeze Breakout Screener")
    st.caption(
        "Finds stocks where EMA 10/20/50 compressed tightly, then price broke out above the cluster. "
        "Pattern visible in: VCP base breakouts, pocket pivots, EMA reclaim setups."
    )
with col_ts:
    st.markdown(
        f"<p style='text-align:right;color:#60A5FA;padding-top:14px'>"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
        unsafe_allow_html=True,
    )

# Pattern explanation card
with st.expander("📖 Pattern Explained", expanded=False):
    st.markdown("""
**EMA Squeeze → Breakout** (the highlighted pattern in the screenshots):

| Phase | What happens |
|-------|-------------|
| **1. Uptrend base** | Price well above EMA 200 (red), EMA 50 > EMA 200 |
| **2. Compression** | EMA 10 (orange), EMA 20 (yellow), EMA 50 (blue) bunch together tightly |
| **3. Breakout** | Price crosses above the compressed EMA cluster with expanding spread |
| **4. Extension** | Price not yet over-extended (within 20% of EMA 50) — still buyable |

**Screener Parameters:**
- **EMA Spread %** — how tightly the three EMAs must bunch (lower = tighter = rarer)
- **Lookback bars** — search window to find the squeeze low
- **Bars since squeeze** — how recently the breakout must have started
- **Extension %** — reject stocks already up > X% from EMA 50
    """)

st.markdown("---")

# ─────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────
if run_btn:
    is_us = "US" in universe_choice

    if universe_choice == "US Top 500 (S&P 500)":
        with st.spinner("Fetching S&P 500 list…"):
            all_tickers = _get_sp500_wiki()
    elif universe_choice == "NSE Nifty 500":
        all_tickers = _load_nifty500()
    else:
        all_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
        if not all_tickers:
            st.error("Please enter at least one ticker.")
            st.stop()

    tickers = all_tickers[:max_tickers]
    currency = "$" if is_us else "₹"

    st.markdown(f"### 🔍 Scanning **{len(tickers)}** tickers — `{universe_choice}`")

    progress  = st.progress(0, text="Starting…")
    status_ph = st.empty()
    n = len(tickers)
    rows = []

    for i, sym in enumerate(tickers):
        progress.progress((i + 1) / n, text=f"Scanning {sym} … ({i+1}/{n})")
        status_ph.markdown(
            f"<span style='color:#A78BFA;font-size:13px'>▶ {sym}</span>",
            unsafe_allow_html=True,
        )
        try:
            hist = yf.Ticker(sym).history(period="2y")
            if hist.empty or len(hist) < 210:
                continue

            matched, metrics = detect_squeeze_breakout(hist, params)
            if not matched:
                continue

            price      = round(float(hist["Close"].iloc[-1]), 2)
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
            day_chg    = round((price / prev_close - 1) * 100, 2) if prev_close else 0.0

            hi52 = float(hist["Close"].tail(252).max())
            dist_hi = round((hi52 - price) / hi52 * 100, 2) if hi52 else np.nan

            try:
                name = yf.Ticker(sym).info.get("longName", sym)[:35]
            except Exception:
                name = sym

            rows.append({
                "Ticker":          sym,
                "Name":            name,
                "Price":           price,
                "Day Chg%":        day_chg,
                "Dist 52wH%":      dist_hi,
                "Score":           metrics["score"],
                "Squeeze %":       metrics["min_spread_pct"],
                "Bars Since Sqz":  metrics["bars_since_squeeze"],
                "Ext 50EMA%":      metrics["ext_from_50ema%"],
                "Vol Ratio":       metrics["vol_ratio"],
            })
        except Exception:
            continue

    progress.progress(1.0, text="✅ Scan complete!")
    status_ph.empty()

    if not rows:
        st.warning("No stocks matched the pattern. Try relaxing the parameters (increase EMA Spread %, extend lookback).")
        st.stop()

    result_df = (
        pd.DataFrame(rows)
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scanned",        len(tickers))
    c2.metric("Matched",        len(result_df), delta=f"{len(result_df)/len(tickers)*100:.1f}%")
    c3.metric("Best Score",     result_df["Score"].max())
    c4.metric("Avg Vol Ratio",  round(result_df["Vol Ratio"].mean(), 2))

    st.markdown("### ✅ EMA Squeeze Breakout Matches (sorted by Score ↓)")
    _display = result_df.copy()
    _display.insert(0, "📊", _display["Ticker"].apply(lambda s: f"/Chart?symbol={s}"))
    st.dataframe(
        _display,
        use_container_width=True,
        height=480,
        column_config={
            "📊":           st.column_config.LinkColumn("📊", display_text="📊 Chart", help="Open chart in new tab"),
            "Price":        st.column_config.NumberColumn(format="%.2f"),
            "Day Chg%":     st.column_config.NumberColumn(format="%.2f %%"),
            "Dist 52wH%":   st.column_config.NumberColumn(format="%.1f %%"),
            "Score":        st.column_config.NumberColumn(format="%.1f"),
            "Squeeze %":    st.column_config.NumberColumn(format="%.2f %%", help="Tightest EMA spread found"),
            "Bars Since Sqz": st.column_config.NumberColumn(format="%d bars"),
            "Ext 50EMA%":   st.column_config.NumberColumn(format="%.1f %%"),
            "Vol Ratio":    st.column_config.NumberColumn(format="%.2f×"),
        },
    )

    # Score column tooltip guide
    st.caption(
        "**Score** = composite of squeeze tightness + recency + volume expansion. "
        "**Squeeze %** = how tight EMA10/20/50 bunched (lower = tighter). "
        "**Vol Ratio** = recent 5-bar avg volume ÷ 50-bar avg."
    )

    # Export
    st.markdown("### 💾 Export")
    ec1, ec2 = st.columns(2)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    with ec1:
        buf = io.StringIO()
        result_df.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download CSV",
            data=buf.getvalue(),
            file_name=f"squeeze_breakout_{ts_str}.csv",
            mime="text/csv",
        )
    with ec2:
        st.download_button(
            "⬇️ Download JSON",
            data=result_df.to_json(orient="records", indent=2),
            file_name=f"squeeze_breakout_{ts_str}.json",
            mime="application/json",
        )

    # Save to DB
    params_str = (
        f"Spread≤{comp_pct}% | Lookback={lookback}bars | "
        f"MaxSinceSqz={max_bars_since}bars | Ext≤{max_ext_pct}%"
    )
    _save_pattern_scan(universe_choice, len(tickers), len(result_df), params_str, result_df)
    st.success("✅ Saved to scan history.")


# ─────────────────────────────────────────────
# LAST 5 SCANS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🕐 Last 5 Scans")

history = _load_pattern_history()
if not history:
    st.info("No scans saved yet. Run a scan above to populate history.")
else:
    for row in history:
        _id, ts, universe, total, passed, params_summary, rjson = row
        with st.expander(
            f"🗓 {ts}  ·  **{universe}**  ·  {passed} matches / {total} scanned",
            expanded=False,
        ):
            st.caption(f"Parameters: {params_summary}")
            try:
                hist_df = pd.read_json(io.StringIO(rjson))
                if not hist_df.empty:
                    # Add chart links
                    _h_display = hist_df.copy()
                    if "Ticker" in _h_display.columns:
                        _h_display.insert(0, "📊", _h_display["Ticker"].apply(lambda s: f"/Chart?symbol={s}"))
                    st.dataframe(
                        _h_display,
                        use_container_width=True,
                        column_config={
                            "📊":         st.column_config.LinkColumn("📊", display_text="📊 Chart"),
                            "Price":      st.column_config.NumberColumn(format="%.2f"),
                            "Day Chg%":   st.column_config.NumberColumn(format="%.2f %%"),
                            "Score":      st.column_config.NumberColumn(format="%.1f"),
                            "Squeeze %":  st.column_config.NumberColumn(format="%.2f %%"),
                        },
                    )
                else:
                    st.info("No matches were found in this scan.")
            except Exception:
                st.warning("Could not load saved results.")
