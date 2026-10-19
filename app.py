"""
Techno-Fundamental Stock Screener
US & Indian Equities | yFinance | SQLite History
"""

import io
import os
import sqlite3
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Techno-Fundamental Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DARK DASHBOARD CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #0E1117;
    color: #C9D1D9;
}
[data-testid="stSidebar"] {
    background-color: #131722;
    border-right: 1px solid #21262D;
}
h1 { color: #A78BFA !important; font-weight: 700; letter-spacing: -.5px; }
h2, h3 { color: #818CF8 !important; }
.stProgress > div > div > div { background: #7C3AED !important; }
.stButton > button {
    background: #7C3AED !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.stButton > button:hover { background: #6D28D9 !important; }
.stDownloadButton > button {
    background: #065F46 !important;
    color: #6EE7B7 !important;
    border: 1px solid #10B981 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
[data-testid="metric-container"] {
    background: #131722;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SQLITE — Persistent Scan History
# ─────────────────────────────────────────────
try:
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screener_history.db")
except NameError:
    DB_PATH = "screener_history.db"


def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                universe        TEXT    NOT NULL,
                total_processed INTEGER NOT NULL DEFAULT 0,
                passed_count    INTEGER NOT NULL DEFAULT 0,
                filters_summary TEXT,
                results_json    TEXT
            )
        """)


def save_scan(universe: str, total: int, passed: int, filters_summary: str, df: pd.DataFrame):
    rjson = df.to_json(orient="records") if (df is not None and not df.empty) else "[]"
    with _conn() as c:
        c.execute(
            "INSERT INTO scan_history "
            "(timestamp, universe, total_processed, passed_count, filters_summary, results_json) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), universe, total, passed, filters_summary, rjson),
        )
        # Keep only the last 20 scans
        c.execute("""
            DELETE FROM scan_history WHERE id NOT IN (
                SELECT id FROM scan_history ORDER BY id DESC LIMIT 20
            )
        """)


def load_history(limit: int = 5):
    with _conn() as c:
        return c.execute(
            "SELECT id, timestamp, universe, total_processed, passed_count, "
            "filters_summary, results_json "
            "FROM scan_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()


init_db()


# ─────────────────────────────────────────────
# STOCK UNIVERSE LISTS
# ─────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_nifty500_from_csv() -> list:
    csv_path = os.path.join(_DATA_DIR, "nifty_500.csv")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        col = df.columns[0]
        return [s.strip() + ".NS" for s in df[col].dropna() if str(s).strip()]
    except Exception:
        return []


def _load_sp500_from_csv() -> list:
    csv_path = os.path.join(_DATA_DIR, "sp500.csv")
    try:
        df = pd.read_csv(csv_path)
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        return [str(s).strip().replace(".", "-") for s in df[col].dropna() if str(s).strip()]
    except Exception:
        return []


_NIFTY_TICKERS = _load_nifty500_from_csv()
_US_FALLBACK = _load_sp500_from_csv()


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> list:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return _US_FALLBACK


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark(is_us: bool) -> pd.DataFrame:
    sym = "^GSPC" if is_us else "^NSEI"
    try:
        h = yf.Ticker(sym).history(period="2y")
        return h if not h.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# CALCULATION HELPERS
# ─────────────────────────────────────────────

def safe_pct(a, b) -> float:
    try:
        if pd.isna(b) or b == 0:
            return np.nan
        return round((float(a) / float(b) - 1) * 100, 2)
    except Exception:
        return np.nan


def rs_excess_return(hist: pd.DataFrame, bench: pd.DataFrame, days: int = 252) -> float:
    """1-year excess return vs benchmark; percentile-ranked after batch completes."""
    try:
        s = hist["Close"].dropna()
        b = bench["Close"].dropna()
        if len(s) < days or len(b) < days:
            return np.nan
        return round((s.iloc[-1] / s.iloc[-days] - 1) * 100 - (b.iloc[-1] / b.iloc[-days] - 1) * 100, 2)
    except Exception:
        return np.nan


def compute_smas(hist: pd.DataFrame) -> dict:
    c = hist["Close"]
    return {p: c.rolling(p).mean().iloc[-1] for p in (20, 50, 150, 200)}


def minervini_check(price: float, lo52: float, hi52: float, smas: dict) -> bool:
    s50, s150, s200 = smas[50], smas[150], smas[200]
    if any(pd.isna(x) for x in (s50, s150, s200)):
        return False
    return (
        price > s150 and price > s200 and
        s150 > s200 and s50 > s150 and price > s50 and
        price >= lo52 * 1.25 and price >= hi52 * 0.75
    )


def quarterly_growth(t: yf.Ticker):
    """Returns (eps_yoy%, rev_yoy%, eps_accel_bool)."""
    eps_g, rev_g, accel = np.nan, np.nan, False
    try:
        q = None
        for attr in ("quarterly_income_stmt", "quarterly_financials"):
            try:
                q = getattr(t, attr, None)
                if q is not None and not q.empty:
                    break
            except Exception:
                continue
        if q is None or q.empty:
            return eps_g, rev_g, accel

        q = q.sort_index(axis=1)
        cols = q.columns.tolist()
        if len(cols) < 5:
            return eps_g, rev_g, accel

        rev_row = next((q.loc[l] for l in ("Total Revenue", "Revenue") if l in q.index), None)
        eps_row = next((q.loc[l] for l in ("Basic EPS", "Diluted EPS", "Net Income") if l in q.index), None)

        curr, yr_ago = cols[-1], cols[-5]

        if rev_row is not None:
            rev_g = safe_pct(rev_row.get(curr, np.nan), rev_row.get(yr_ago, np.nan))

        if eps_row is not None:
            eps_g = safe_pct(eps_row.get(curr, np.nan), eps_row.get(yr_ago, np.nan))
            if not np.isnan(eps_g) and len(cols) >= 6:
                prev_g = safe_pct(eps_row.get(cols[-2], np.nan), eps_row.get(cols[-6], np.nan))
                if not np.isnan(prev_g):
                    accel = eps_g > prev_g
    except Exception:
        pass
    return eps_g, rev_g, accel


def apply_pass_fail(row: dict, f: dict) -> str:
    eps_g = row.get("EPS Growth%", np.nan)
    rev_g = row.get("Rev Growth%", np.nan)
    dist  = row.get("Dist 52wH%",  np.nan)
    if not np.isnan(eps_g) and eps_g < f["min_eps"]:
        return "FAIL"
    if not np.isnan(rev_g) and rev_g < f["min_rev"]:
        return "FAIL"
    if not np.isnan(dist)  and dist  > f["max_dist"]:
        return "FAIL"
    if f["minervini"] and not row.get("_minervini", False):
        return "FAIL"
    return "PASS"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Screener Controls")
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
    "Max Tickers to Scan",
    min_value=10, max_value=500, value=500, step=10,
    help="Reduce for speed. 500 = full universe (may take several minutes).",
)

st.sidebar.markdown("### 📐 Fundamental Filters")
min_eps_f = st.sidebar.slider("Min EPS Growth %",      -100, 500,  15, step=5)
min_rev_f = st.sidebar.slider("Min Revenue Growth %",  -100, 300,  10, step=5)
max_dist_f = st.sidebar.slider("Max % Below 52-Wk High",  5, 100,  30, step=5)

st.sidebar.markdown("### 🔬 Technical Filters")
req_minervini = st.sidebar.toggle(
    "Require Minervini Trend Template",
    value=False,
    help="Price > 150/200 SMA, proper SMA alignment, price ≥25% above 52w low",
)

st.sidebar.caption("ℹ️ Tickers with missing fundamental data bypass EPS/Revenue filters.")

filters = dict(min_eps=min_eps_f, min_rev=min_rev_f, max_dist=max_dist_f, minervini=req_minervini)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Screener", use_container_width=True)


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
col_t, col_ts = st.columns([3, 1])
with col_t:
    st.markdown("# 📈 Techno-Fundamental Stock Screener")
    st.caption("US & Indian Equities · yFinance Data · SQLite History · Not financial advice")
with col_ts:
    st.markdown(
        f"<p style='text-align:right;color:#60A5FA;padding-top:14px'>"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
        unsafe_allow_html=True,
    )
st.markdown("---")


# ─────────────────────────────────────────────
# SCREENER EXECUTION
# ─────────────────────────────────────────────
def run_screener(tickers: list, bench: pd.DataFrame, f: dict) -> pd.DataFrame:
    progress  = st.progress(0, text="Starting scan…")
    status_ph = st.empty()
    rows = []
    n = len(tickers)

    for i, sym in enumerate(tickers):
        progress.progress((i + 1) / n, text=f"Scanning {sym} … ({i+1}/{n})")
        status_ph.markdown(
            f"<span style='color:#A78BFA;font-size:13px'>▶ {sym}</span>",
            unsafe_allow_html=True,
        )
        try:
            t    = yf.Ticker(sym)
            hist = t.history(period="2y")
            if hist.empty or len(hist) < 60:
                continue

            price      = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2] if len(hist) >= 2 else price
            day_chg    = safe_pct(price, prev_close)
            smas       = compute_smas(hist)
            hi52       = hist["Close"].tail(252).max()
            lo52       = hist["Close"].tail(252).min()
            dist_hi    = round((hi52 - price) / hi52 * 100, 2) if hi52 else np.nan
            above_200  = price > smas[200] if not pd.isna(smas[200]) else False
            gold_align = (not pd.isna(smas[50]) and not pd.isna(smas[200]) and smas[50] > smas[200])
            minv       = minervini_check(price, lo52, hi52, smas)
            rs_raw     = rs_excess_return(hist, bench)

            try:
                name = t.info.get("longName", sym)[:35]
            except Exception:
                name = sym

            eps_g, rev_g, accel = quarterly_growth(t)

            rows.append({
                "Ticker":       sym,
                "Name":         name,
                "Price":        round(price, 2),
                "Day Chg%":     day_chg,
                "_rs_raw":      rs_raw,
                "Dist 52wH%":   dist_hi,
                "Above 200SMA": "✅" if above_200 else "❌",
                "EPS Growth%":  eps_g,
                "Rev Growth%":  rev_g,
                "EPS Accel":    "✅" if accel else "❌",
                "Golden Align": "✅" if gold_align else "❌",
                "Minervini":    "✅" if minv else "❌",
                "_minervini":   minv,
            })
        except Exception:
            continue

    progress.progress(1.0, text="✅ Scan complete!")
    status_ph.empty()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Percentile RS Rating (1–99)
    df["RS Rating"] = (
        df["_rs_raw"].rank(pct=True, na_option="keep") * 98 + 1
    ).round(0).astype("Int64")

    df["Status"] = df.apply(lambda r: apply_pass_fail(r, f), axis=1)
    df = df.drop(columns=["_rs_raw", "_minervini"])

    col_order = [
        "Ticker", "Name", "Status", "Price", "Day Chg%",
        "RS Rating", "Dist 52wH%", "Above 200SMA",
        "EPS Growth%", "Rev Growth%", "EPS Accel",
        "Golden Align", "Minervini",
    ]
    return df[[c for c in col_order if c in df.columns]]


# ─────────────────────────────────────────────
# RUN SCAN
# ─────────────────────────────────────────────
if run_btn:
    is_us = "US" in universe_choice

    if universe_choice == "US Top 500 (S&P 500)":
        with st.spinner("Fetching S&P 500 ticker list…"):
            all_tickers = get_sp500_tickers()
    elif universe_choice == "NSE Nifty 500":
        all_tickers = _NIFTY_TICKERS
    else:
        all_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
        if not all_tickers:
            st.error("Please enter at least one ticker.")
            st.stop()

    tickers = all_tickers[:max_tickers]
    bench   = fetch_benchmark(is_us)

    st.markdown(f"### 🔍 Scanning **{len(tickers)}** tickers — `{universe_choice}`")

    df = run_screener(tickers, bench, filters)

    if df.empty:
        st.warning("No data retrieved. Check tickers or network connection.")
        st.stop()

    passed_df = df[df["Status"] == "PASS"].sort_values("RS Rating", ascending=False, na_position="last")
    failed_df = df[df["Status"] == "FAIL"]

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers Scanned",       len(df))
    c2.metric("✅ Passed",              len(passed_df), delta=f"{len(passed_df)/len(df)*100:.1f}%")
    c3.metric("❌ Failed / No Data",    len(failed_df))
    avg_eps = passed_df["EPS Growth%"].dropna().mean()
    c4.metric("Avg EPS Growth (Pass)", f"{avg_eps:.1f}%" if not np.isnan(avg_eps) else "–")

    # PASS table
    st.markdown("### ✅ Screened Results — PASS (sorted by RS Rating ↓)")
    _pass_display = passed_df.reset_index(drop=True).copy()
    _pass_display.insert(0, "📊", _pass_display["Ticker"].apply(lambda s: f"/Chart?symbol={s}"))
    st.dataframe(
        _pass_display,
        use_container_width=True,
        height=440,
        column_config={
            "📊":          st.column_config.LinkColumn("📊", display_text="📊 Chart", help="Open chart in new tab"),
            "Price":       st.column_config.NumberColumn(format="%.2f"),
            "Day Chg%":    st.column_config.NumberColumn(format="%.2f %%"),
            "RS Rating":   st.column_config.NumberColumn(format="%d"),
            "Dist 52wH%":  st.column_config.NumberColumn(format="%.1f %%"),
            "EPS Growth%": st.column_config.NumberColumn(format="%.1f %%"),
            "Rev Growth%": st.column_config.NumberColumn(format="%.1f %%"),
        },
    )

    # All results (collapsible)
    with st.expander(f"📋 All Results ({len(df)} tickers, PASS + FAIL)"):
        sort_options = [c for c in df.columns if c not in ("Ticker", "Name", "Status")]
        sort_col = st.selectbox(
            "Sort by",
            sort_options,
            index=sort_options.index("RS Rating") if "RS Rating" in sort_options else 0,
        )
        sort_asc = st.checkbox("Ascending", value=False)
        _all_display = df.sort_values(sort_col, ascending=sort_asc, na_position="last").reset_index(drop=True).copy()
        _all_display.insert(0, "📊", _all_display["Ticker"].apply(lambda s: f"/Chart?symbol={s}"))
        st.dataframe(
            _all_display,
            use_container_width=True,
            column_config={
                "📊": st.column_config.LinkColumn("📊", display_text="📊 Chart", help="Open chart in new tab"),
            },
        )

    # Export
    st.markdown("### 💾 Export")
    ec1, ec2 = st.columns(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with ec1:
        buf = io.StringIO()
        passed_df.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download CSV (PASS)",
            data=buf.getvalue(),
            file_name=f"screener_pass_{ts}.csv",
            mime="text/csv",
        )
    with ec2:
        st.download_button(
            "⬇️ Download JSON (PASS)",
            data=passed_df.to_json(orient="records", indent=2),
            file_name=f"screener_pass_{ts}.json",
            mime="application/json",
        )

    # Persist to SQLite
    fs = (
        f"EPS≥{min_eps_f}% | Rev≥{min_rev_f}% | Dist≤{max_dist_f}%"
        + (" | Minervini" if req_minervini else "")
    )
    save_scan(universe_choice, len(df), len(passed_df), fs, passed_df)
    st.success("✅ Results saved to scan history (SQLite).")


# ─────────────────────────────────────────────
# SCAN HISTORY
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🕐 Scan History (Last 5 Runs)")

hist_rows = load_history(5)

if hist_rows:
    hist_display = pd.DataFrame(
        [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in hist_rows],
        columns=["ID", "Timestamp", "Universe", "Processed", "Passed", "Filters"],
    )
    st.dataframe(hist_display, use_container_width=True, hide_index=True)

    st.markdown("#### 🔁 Reload Previous Scan Results")
    options = {
        f"#{r[0]} · {r[1]} · {r[2]} ({r[4]} passed)": r
        for r in hist_rows
    }
    sel = st.selectbox("Choose scan to reload", list(options.keys()))
    if st.button("Load Selected Scan"):
        chosen = options[sel]
        rjson = chosen[6]
        if not rjson or rjson == "[]":
            st.info("No results stored for this scan.")
        else:
            try:
                prev = pd.read_json(io.StringIO(rjson))
                st.markdown(
                    f"**Scan `{chosen[1]}`** — {chosen[2]}  |  Filters: *{chosen[5]}*"
                )
                st.dataframe(prev, use_container_width=True)
            except Exception as e:
                st.error(f"Could not reload results: {e}")
else:
    st.info("No scan history yet. Run the screener above to start building history.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4B5563;font-size:12px'>"
    "Techno-Fundamental Screener · Data: yFinance · History: SQLite · "
    "<span style='color:#7C3AED'>Not financial advice.</span>"
    "</div>",
    unsafe_allow_html=True,
)
