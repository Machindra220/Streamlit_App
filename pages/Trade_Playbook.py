"""
Swing Trading Playbook
Full workflow: position sizing → entry → management → exit → journaling
"""

import io
import os
import sqlite3
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Trade Playbook",
    page_icon="📒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{background:#0E1117;color:#C9D1D9;}
[data-testid="stSidebar"]{background:#131722;border-right:1px solid #21262D;}
h1{color:#A78BFA!important;font-weight:700;}
h2,h3{color:#818CF8!important;}
.stTabs [data-baseweb="tab-list"]{background:#131722;border-radius:10px;padding:4px;}
.stTabs [data-baseweb="tab"]{color:#C9D1D9;border-radius:8px;padding:8px 18px;}
.stTabs [aria-selected="true"]{background:#7C3AED!important;color:#fff!important;}
.stButton>button{background:#7C3AED!important;color:#fff!important;border:none!important;border-radius:6px!important;font-weight:600!important;}
[data-testid="metric-container"]{background:#131722;border:1px solid #30363D;border-radius:10px;padding:12px 16px;}
.rule-card{background:#131722;border:1px solid #30363D;border-radius:10px;padding:14px 18px;margin:6px 0;}
.green{color:#26a69a!important;font-weight:600;}
.red{color:#ef5350!important;font-weight:600;}
.yellow{color:#F1C40F!important;font-weight:600;}
.stNumberInput>div>div>input,.stTextInput>div>div>input,.stSelectbox>div>div{background:#161B22!important;color:#C9D1D9!important;border:1px solid #30363D!important;}
table{width:100%;}
th{background:#1E222D;color:#818CF8;padding:8px 12px;text-align:left;}
td{padding:6px 12px;border-bottom:1px solid #21262D;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
try:
    _BASE   = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.normpath(os.path.join(_BASE, "..", "screener_history.db"))
except NameError:
    DB_PATH = "screener_history.db"


def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _init_trades_table():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                date_open    TEXT NOT NULL,
                symbol       TEXT NOT NULL,
                market       TEXT NOT NULL DEFAULT 'US',
                entry_price  REAL NOT NULL,
                stop_price   REAL NOT NULL,
                target1      REAL,
                target2      REAL,
                qty          REAL NOT NULL,
                risk_amount  REAL,
                setup_notes  TEXT,
                status       TEXT NOT NULL DEFAULT 'OPEN',
                date_closed  TEXT,
                exit_price   REAL,
                pnl          REAL,
                exit_notes   TEXT
            )
        """)


_init_trades_table()


def _get_trades(status_filter="ALL"):
    with _conn() as c:
        if status_filter == "ALL":
            return pd.read_sql("SELECT * FROM trades ORDER BY id DESC", c)
        return pd.read_sql(
            "SELECT * FROM trades WHERE status=? ORDER BY id DESC",
            c, params=(status_filter,)
        )


def _add_trade(row: dict):
    with _conn() as c:
        c.execute("""
            INSERT INTO trades
            (date_open,symbol,market,entry_price,stop_price,target1,target2,qty,risk_amount,setup_notes,status)
            VALUES (:date_open,:symbol,:market,:entry_price,:stop_price,:target1,:target2,:qty,:risk_amount,:setup_notes,'OPEN')
        """, row)


def _close_trade(trade_id: int, exit_price: float, status: str, notes: str):
    with _conn() as c:
        row = c.execute("SELECT entry_price,qty FROM trades WHERE id=?", (trade_id,)).fetchone()
        if row:
            pnl = round((exit_price - row[0]) * row[1], 2)
            c.execute("""
                UPDATE trades SET status=?,date_closed=?,exit_price=?,pnl=?,exit_notes=?
                WHERE id=?
            """, (status, date.today().isoformat(), exit_price, pnl, notes, trade_id))


def _delete_trade(trade_id: int):
    with _conn() as c:
        c.execute("DELETE FROM trades WHERE id=?", (trade_id,))


# ─────────────────────────────────────────────
# LIVE PRICE FETCH
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _live_price(sym: str) -> float:
    try:
        h = yf.Ticker(sym).history(period="2d")
        if not h.empty:
            return round(float(h["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 📒 Swing Trading Playbook")
st.caption("Your complete workflow: position sizing → entry → management → exit. Not financial advice.")
st.markdown("---")

tabs = st.tabs([
    "📐 Position Sizing",
    "🎯 Entry Rules",
    "📈 Trade Management",
    "🚪 Exit Rules",
    "📋 Trade Tracker",
    "📚 Quick Reference",
])


# ══════════════════════════════════════════════
# TAB 1: POSITION SIZING
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 📐 Position Size Calculator")
    st.markdown(
        "_Never risk more than **1–2% of your total portfolio on a single trade**. "
        "This single rule keeps you in the game through losing streaks._"
    )

    col_calc, col_rules = st.columns([1, 1], gap="large")

    with col_calc:
        st.markdown("### Calculator")
        portfolio = st.number_input("Portfolio Value (₹ or $)", value=500000, step=10000, min_value=1000)
        risk_pct  = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.0, 0.25,
                              help="1% = conservative, 2% = moderate, >2% = aggressive")
        entry     = st.number_input("Entry Price", value=1000.0, step=0.5, min_value=0.01)
        stop      = st.number_input("Stop Loss Price", value=950.0, step=0.5, min_value=0.01)
        target1   = st.number_input("Target 1 (Partial Exit)", value=1100.0, step=0.5, min_value=0.01)
        target2   = st.number_input("Target 2 (Full Exit)", value=1200.0, step=0.5, min_value=0.01)

        if entry > stop and entry > 0:
            risk_per_share  = entry - stop
            reward1         = target1 - entry
            reward2         = target2 - entry
            max_risk_amount = portfolio * risk_pct / 100
            shares          = int(max_risk_amount / risk_per_share)
            position_value  = round(shares * entry, 2)
            position_pct    = round(position_value / portfolio * 100, 1)
            rr1             = round(reward1 / risk_per_share, 2)
            rr2             = round(reward2 / risk_per_share, 2)
            pnl_stop        = round(-risk_per_share * shares, 2)
            pnl_t1          = round(reward1 * shares, 2)
            pnl_t2          = round(reward2 * shares, 2)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Shares to Buy",   f"{shares:,}")
            c2.metric("Position Value",  f"{position_value:,.0f}")
            c3.metric("Portfolio %",     f"{position_pct}%")

            c4, c5, c6 = st.columns(3)
            c4.metric("Max Loss (₹/$)",  f"{abs(pnl_stop):,.0f}", delta=f"-{risk_pct}%", delta_color="inverse")
            c5.metric("P&L at Target 1", f"{pnl_t1:,.0f}",        delta=f"R:R {rr1}:1")
            c6.metric("P&L at Target 2", f"{pnl_t2:,.0f}",        delta=f"R:R {rr2}:1")

            # Color warnings
            if rr1 < 1.5:
                st.warning("⚠️ R:R at Target 1 is below 1.5:1 — consider moving your target or tightening the stop.")
            if rr2 >= 2.0:
                st.success(f"✅ R:R at Target 2 = {rr2}:1 — good setup.")
            if position_pct > 25:
                st.error("🚨 Position > 25% of portfolio — too concentrated, reduce size.")
        else:
            st.error("Entry price must be above stop loss price.")

    with col_rules:
        st.markdown("### Money Management Rules")
        rules = [
            ("💰", "Max Risk Per Trade",     "1–2% of total portfolio. At 1%, you can lose 50 trades in a row and still have half your capital."),
            ("📦", "Max Position Size",      "No single stock > 20–25% of portfolio. Diversify across 5–10 positions."),
            ("🔥", "Portfolio Heat Limit",   "Total open risk across ALL positions ≤ 6–8% of portfolio at any time."),
            ("🚫", "Never Average Down",     "Do not add to a losing position. Only add to winners showing strength."),
            ("💵", "Keep Cash Reserve",      "Always keep 20–30% in cash for new opportunities or a market correction."),
            ("📊", "Pyramid Scaling",        "Add only 50% of initial position on first add, 25% on second. Never add more than 3 times."),
            ("⛔", "Hard Stop — No Excuses", "If price hits your stop, EXIT. No hoping, no averaging. The market is always right."),
            ("📅", "Weekly Review",          "Review all open positions every weekend. Raise stops on winners; cut laggards."),
        ]
        for icon, title, detail in rules:
            st.markdown(
                f"<div class='rule-card'><b>{icon} {title}</b><br>"
                f"<small style='color:#8B949E'>{detail}</small></div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════
# TAB 2: ENTRY RULES
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 🎯 Entry Rules — When to Pull the Trigger")

    col_yes, col_no = st.columns(2, gap="large")

    with col_yes:
        st.markdown("### ✅ Enter When ALL of These Are True")
        entry_rules = [
            ("📈 Market Direction",
             "Major index (Nifty / S&P 500) is in a confirmed uptrend. "
             "Do NOT buy stocks in a bear market — swim with the tide, not against it."),
            ("🏗️ Sound Base / Setup",
             "Stock built a proper base (3–8 weeks minimum). "
             "Look for EMA squeeze (10/20/50 bunching) or tight price consolidation. "
             "Avoid stocks that are extended — buy bases, not spikes."),
            ("📐 Price Above All Key EMAs",
             "Price > EMA 10, 20, 50, 200. All EMAs aligned bullishly (10 > 20 > 50 > 200). "
             "EMA 50 > EMA 200 is the minimum uptrend gate."),
            ("🚀 Breakout / Pivot Point",
             "Price breaks above the resistance level of the base with volume ≥ 150% of 50-day avg. "
             "For EMA squeeze: price crosses above compressed EMAs with expanding spread."),
            ("📊 Volume Confirmation",
             "Breakout bar volume must be above average. "
             "Low-volume breakouts have a higher failure rate — wait for conviction."),
            ("💪 Relative Strength",
             "RS Rating ≥ 70 (top 30% performers vs the market). "
             "Ideally RS line making new highs before or on the breakout — a leading indicator."),
            ("💰 Risk/Reward Check",
             "Before entering, define your stop and target. "
             "Only take the trade if R:R ≥ 2:1 at Target 1. ≥3:1 is ideal."),
        ]
        for title, detail in entry_rules:
            st.markdown(
                f"<div class='rule-card'><b>{title}</b><br>"
                f"<small style='color:#8B949E'>{detail}</small></div>",
                unsafe_allow_html=True,
            )

    with col_no:
        st.markdown("### ❌ Do NOT Enter When")
        avoid_rules = [
            ("📉 Major Index Below 200 EMA",
             "If Nifty or S&P 500 is below the 200 EMA, go to cash. "
             "Even great stocks fall in a market downtrend."),
            ("💨 Stock Is Extended",
             "Price more than 15–20% above the 50 EMA = too late, chasing. "
             "Wait for the next base to form."),
            ("📢 News-Driven Gap",
             "Never chase a gap-up from earnings or news. "
             "Let it consolidate and form a new base first."),
            ("🔇 Low Volume Breakout",
             "Volume below 50-day average on breakout day = weak conviction. "
             "Institutional money is not participating — skip it."),
            ("📆 Earnings Within 2 Weeks",
             "Avoid entering within 2 weeks of an earnings report. "
             "Earnings is a binary event — unpredictable."),
            ("🔻 Downtrending EMAs",
             "If EMA 50 is below EMA 200 (death cross), the stock is not in a valid uptrend setup. "
             "Skip entirely."),
            ("🌊 High Market Volatility",
             "During high-uncertainty events (Fed meetings, elections, geopolitical events), "
             "reduce position sizes by 50% or sit in cash."),
        ]
        for title, detail in avoid_rules:
            st.markdown(
                f"<div class='rule-card'><b>{title}</b><br>"
                f"<small style='color:#8B949E'>{detail}</small></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### 🎯 Entry Types")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**Breakout Entry** (Aggressive)
- Buy at the exact pivot/resistance break
- Stop: 5–7% below entry OR below the base low
- Requires high volume confirmation
- Best for strong momentum setups
        """)
    with c2:
        st.markdown("""
**Pullback to EMA Entry** (Conservative)
- Wait for stock to pull back to EMA 10 or 20 after a breakout
- Buy when price bounces off the EMA with a reversal candle
- Stop: just below the EMA + small buffer
- Lower risk, but may miss fast movers
        """)
    with c3:
        st.markdown("""
**EMA Reclaim Entry** (Reversal)
- Stock broke below EMAs, now reclaiming them with volume
- Buy on close above EMA 50 (or EMA 20 for stronger setups)
- Stop: below the recent swing low
- Higher risk — confirm with market uptrend
        """)


# ══════════════════════════════════════════════
# TAB 3: TRADE MANAGEMENT
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 📈 Active Trade Management")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("### 🛡️ Stop Loss Management")
        stop_rules = [
            ("Initial Stop",
             "Place stop 1–2% below the base low OR below a key EMA. "
             "Never set it at round numbers ($100, ₹500) — market makers target those."),
            ("Time Stop",
             "If a stock hasn't moved meaningfully in 3–4 weeks after entry, "
             "it's dead money — exit and redeploy capital elsewhere."),
            ("Break-Even Stop (1R)",
             "Once price reaches your first R target (1× your initial risk above entry), "
             "move stop to break-even. You now have a 'free trade'."),
            ("Trail Stop at EMA 10",
             "After the stock is up 2R, trail stop to just below EMA 10 on daily chart. "
             "Exit on a daily close below EMA 10."),
            ("Trail Stop at EMA 20",
             "For longer swings (4–12 weeks), trail stop below EMA 20. "
             "Let winners run until EMA 20 is violated."),
            ("Hard Stop Rule",
             "If price closes 8% below your purchase price, EXIT without question. "
             "This is Minervini's 8% hard cut rule — it caps your maximum loss."),
        ]
        for title, detail in stop_rules:
            st.markdown(
                f"<div class='rule-card'><b>📍 {title}</b><br>"
                f"<small style='color:#8B949E'>{detail}</small></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### ➕ Adding to a Winning Position")
        add_rules = [
            ("First Add (Pyramid)",
             "Add 50% of initial position size ONLY after the stock is up 1.5–2R AND "
             "shows a secondary base or tight consolidation. Do NOT add on a gap-up."),
            ("Second Add",
             "Add 25% of initial position size after the stock clears the second base. "
             "Stop for the new add = below the second base low."),
            ("Stop on Full Position",
             "After adding, recalculate your combined cost basis. "
             "If price hits -8% from your average cost, exit the ENTIRE position."),
        ]
        for title, detail in add_rules:
            st.markdown(
                f"<div class='rule-card'><b>➕ {title}</b><br>"
                f"<small style='color:#8B949E'>{detail}</small></div>",
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown("### 📊 Partial Exit (Scaling Out)")
        partial_rules = [
            ("Take Partial at Target 1 (1R–2R)",
             "Sell 25–33% of your position at the first target. "
             "This locks in real profit and reduces psychological pressure. "
             "Raise stop to break-even on remaining shares."),
            ("Take More at Target 2 (3R–4R)",
             "Sell another 33–50% at the second target. "
             "Let the remaining shares ('free runner') run with a trailing stop."),
            ("Sell Into Strength",
             "When a stock gaps up or has a parabolic day with very high volume "
             "(climax run), consider selling 50–100%. "
             "These reversals can be fast and painful."),
            ("Earnings Partial Exit",
             "If you hold into earnings: sell 50% before the announcement. "
             "You keep upside if it gaps up; limit damage if it gaps down."),
            ("Market Deterioration",
             "If the major index breaks below its 50 EMA with high volume, "
             "sell weakest positions first and raise cash to 50%+."),
        ]
        for title, detail in partial_rules:
            st.markdown(
                f"<div class='rule-card'><b>💰 {title}</b><br>"
                f"<small style='color:#8B949E'>{detail}</small></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### 📅 Weekly Checklist for Running Trades")
        checklist = [
            "Is the stock still above EMA 10 and EMA 20?",
            "Is the major index still in an uptrend (above its 50/200 EMA)?",
            "Has the RS line weakened vs the market?",
            "Is the stock forming a new base (good) or erratic selling (bad)?",
            "Did any insider selling or negative news come out?",
            "Is your stop still correctly placed? Raise it if price has moved up.",
            "Is the stock near a resistance level / prior high? Plan partial exit.",
            "Total portfolio heat: is it above 8%? Scale back if so.",
        ]
        for item in checklist:
            st.markdown(f"☐ {item}")


# ══════════════════════════════════════════════
# TAB 4: EXIT RULES
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🚪 Exit Rules — When to Close a Trade")

    col_exit1, col_exit2 = st.columns(2, gap="large")

    with col_exit1:
        st.markdown("### 🔴 Exit Immediately (No Debate)")
        hard_exits = [
            ("Stop Loss Hit", "Price touches your pre-defined stop. EXIT in full. No averaging, no hoping."),
            ("8% Hard Rule",  "Price closes 8% below your average cost at any point. Exit the full position."),
            ("Daily Close Below EMA 50", "After being in profit, a daily close below EMA 50 on high volume = distribution. Exit."),
            ("Climax Top Signals",
             "Parabolic move: largest up-day in the entire trend, gap-up exhaustion, "
             "volume 3× the average = likely climax. Sell into strength."),
            ("Failed Breakout",
             "Price breaks out but closes back below the pivot on the same day. "
             "Exit the next morning. Do not hold a failed breakout."),
        ]
        for title, detail in hard_exits:
            st.markdown(
                f"<div class='rule-card' style='border-color:#ef535066'>"
                f"<b class='red'>🚨 {title}</b><br>"
                f"<small style='color:#8B949E'>{detail}</small></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### 🟡 Exit on Next Open (Review First)")
        soft_exits = [
            ("EMA 10 Violation on Daily Close", "If still early in the trend (<2R profit): exit or tighten stop aggressively."),
            ("EMA 20 Violation on Daily Close", "If in significant profit: sell 50%, trail remaining to EMA 50."),
            ("Volume Drying Up After Breakout",
             "Three consecutive days of declining volume after a breakout = weak follow-through. "
             "Reduce position by 50%."),
            ("3-Weeks Tight Pattern Failure",
             "Stock built a tight pattern but is now widening and making lower lows — "
             "the base failed. Exit and wait for reset."),
            ("Relative Strength Deteriorating",
             "RS Rating drops from above 80 to below 60 while market is going up = "
             "sector rotation or stock-specific weakness. Exit."),
        ]
        for title, detail in soft_exits:
            st.markdown(
                f"<div class='rule-card' style='border-color:#F1C40F66'>"
                f"<b class='yellow'>⚠️ {title}</b><br>"
                f"<small style='color:#8B949E'>{detail}</small></div>",
                unsafe_allow_html=True,
            )

    with col_exit2:
        st.markdown("### 🟢 Take Profit Targets")
        profit_exits = [
            ("Target 1: Sell 25–33% at +1R to +2R",
             "Lock in real profit. Raise stop to break-even. Now you play with house money."),
            ("Target 2: Sell 33–50% at +3R to +4R",
             "Reduce position further. Let the remaining shares run with a trailing stop."),
            ("Target 3: Trail remaining to EMA 20",
             "For the 'free runner' portion, trail stop under EMA 20. "
             "Ride it until the EMA is violated."),
            ("20–25% Gain in ≤3 Weeks",
             "Sell at least 50% of the position. Stocks that gain 20–25% in 3 weeks "
             "often need to rest. Lock it in."),
        ]
        for title, detail in profit_exits:
            st.markdown(
                f"<div class='rule-card' style='border-color:#26a69a66'>"
                f"<b class='green'>✅ {title}</b><br>"
                f"<small style='color:#8B949E'>{detail}</small></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### 🔁 The Full Trade Lifecycle")
        st.markdown("""
```
SCAN → Find EMA Squeeze + Uptrend setup
         ↓
SIZE → Calculate shares using 1% risk rule
         ↓
ENTER → Breakout day with high volume
         ↓
MANAGE → Trail stop as stock moves up
         ↓
PARTIAL → Sell 30% at 2R, raise stop to BE
         ↓
PARTIAL → Sell 30% at 4R, trail EMA 20
         ↓
FINAL EXIT → EMA 20 violated / 8% hard rule
         ↓
JOURNAL → Record: what worked, what didn't
```
        """)

        st.markdown("### ⚖️ R-Multiple Reference")
        r_df = pd.DataFrame({
            "R-Multiple": ["0.5R", "1R", "1.5R", "2R", "3R", "4R", "5R"],
            "Meaning":    [
                "Half your risk gained — consider partial",
                "Break-even stop → move stop here",
                "Sell 20–25%, tight stop",
                "Sell 30%, excellent trade",
                "Outstanding — sell 30%, trail rest",
                "Exceptional — protect gains aggressively",
                "Home run — trail EMA 10 very tightly",
            ],
        })
        st.dataframe(r_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 5: TRADE TRACKER
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 📋 Running Trade Tracker")

    # ── Add new trade ──
    with st.expander("➕ Add New Trade", expanded=False):
        with st.form("add_trade_form", clear_on_submit=True):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                t_sym    = st.text_input("Symbol", placeholder="RELIANCE.NS or AAPL").strip().upper()
                t_market = st.selectbox("Market", ["India (.NS)", "US"])
                t_date   = st.date_input("Date Opened", value=date.today())
            with fc2:
                t_entry  = st.number_input("Entry Price", min_value=0.01, value=1000.0, step=0.5)
                t_stop   = st.number_input("Stop Loss",   min_value=0.01, value=950.0,  step=0.5)
                t_qty    = st.number_input("Qty / Shares", min_value=1,   value=10,     step=1)
            with fc3:
                t_t1     = st.number_input("Target 1",  min_value=0.01, value=1100.0, step=0.5)
                t_t2     = st.number_input("Target 2",  min_value=0.01, value=1200.0, step=0.5)
                t_notes  = st.text_input("Setup Notes", placeholder="EMA squeeze breakout…")

            submitted = st.form_submit_button("Add Trade", use_container_width=True)
            if submitted:
                if not t_sym:
                    st.error("Symbol is required.")
                elif t_entry <= t_stop:
                    st.error("Entry must be above stop loss.")
                else:
                    risk_amt = round((t_entry - t_stop) * t_qty, 2)
                    mkt = "India" if "India" in t_market else "US"
                    if mkt == "India" and not t_sym.endswith(".NS"):
                        t_sym += ".NS"
                    _add_trade({
                        "date_open":   t_date.isoformat(),
                        "symbol":      t_sym,
                        "market":      mkt,
                        "entry_price": t_entry,
                        "stop_price":  t_stop,
                        "target1":     t_t1,
                        "target2":     t_t2,
                        "qty":         t_qty,
                        "risk_amount": risk_amt,
                        "setup_notes": t_notes,
                    })
                    st.success(f"✅ Trade added: {t_sym} — risk {risk_amt:.0f}")
                    st.rerun()

    # ── Open trades ──
    st.markdown("### 🟢 Open Positions")
    open_df = _get_trades("OPEN")

    if open_df.empty:
        st.info("No open trades. Add one above.")
    else:
        refresh = st.button("🔄 Refresh Live Prices")
        display_rows = []
        total_risk = 0.0
        total_unrealised = 0.0

        for _, row in open_df.iterrows():
            sym       = row["symbol"]
            live      = _live_price(sym) if refresh or True else row["entry_price"]
            entry     = float(row["entry_price"])
            stop_p    = float(row["stop_price"])
            qty       = float(row["qty"])
            t1        = float(row["target1"])  if row["target1"]  else None
            t2        = float(row["target2"])  if row["target2"]  else None
            risk_amt  = float(row["risk_amount"]) if row["risk_amount"] else (entry - stop_p) * qty
            unrealised = round((live - entry) * qty, 2)
            pct_chg    = round((live / entry - 1) * 100, 2) if entry else 0
            r_multiple = round((live - entry) / (entry - stop_p), 2) if (entry - stop_p) != 0 else 0

            total_risk       += risk_amt
            total_unrealised += unrealised

            # Status flag
            if live <= stop_p:
                flag = "🚨 AT STOP"
            elif t1 and live >= t1 and (t2 is None or live < t2):
                flag = "🎯 T1 HIT"
            elif t2 and live >= t2:
                flag = "🏆 T2 HIT"
            elif r_multiple >= 1.0:
                flag = "✅ In Profit"
            elif pct_chg <= -5:
                flag = "⚠️ Weak"
            else:
                flag = "📊 Holding"

            display_rows.append({
                "Symbol":    sym,
                "Entry":     entry,
                "Stop":      stop_p,
                "T1":        t1 or "–",
                "T2":        t2 or "–",
                "Qty":       int(qty),
                "Live":      live,
                "Chg%":      pct_chg,
                "Unrealised": unrealised,
                "R-Mult":    r_multiple,
                "Status":    flag,
                "Notes":     row["setup_notes"] or "",
                "_id":       int(row["id"]),
            })

        disp_df = pd.DataFrame(display_rows)

        # Add chart link column
        disp_df.insert(0, "📊", disp_df["Symbol"].apply(lambda s: f"/Chart?symbol={s}"))

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Open Positions",   len(open_df))
        m2.metric("Total Risk",       f"{total_risk:,.0f}")
        m3.metric("Total Unrealised", f"{total_unrealised:,.0f}",
                  delta_color="normal" if total_unrealised >= 0 else "inverse")
        m4.metric("Avg R-Multiple",   round(disp_df["R-Mult"].mean(), 2))

        show_cols = ["📊", "Symbol", "Entry", "Stop", "T1", "T2", "Qty",
                     "Live", "Chg%", "Unrealised", "R-Mult", "Status", "Notes"]
        st.dataframe(
            disp_df[show_cols],
            use_container_width=True,
            column_config={
                "📊":         st.column_config.LinkColumn("📊", display_text="📊"),
                "Entry":      st.column_config.NumberColumn(format="%.2f"),
                "Stop":       st.column_config.NumberColumn(format="%.2f"),
                "Live":       st.column_config.NumberColumn(format="%.2f"),
                "Chg%":       st.column_config.NumberColumn(format="%.2f %%"),
                "Unrealised": st.column_config.NumberColumn(format="%.0f"),
                "R-Mult":     st.column_config.NumberColumn(format="%.2f×"),
            },
        )

        # Close trade form
        st.markdown("#### Close / Partial Exit a Trade")
        cl1, cl2, cl3, cl4, cl5 = st.columns([2, 1, 1, 1, 1])
        id_options = {f"{r['Symbol']} (#{r['_id']})": r["_id"] for r in display_rows}
        with cl1:
            sel_label  = st.selectbox("Select Trade", list(id_options.keys()), key="close_sel")
        with cl2:
            exit_price = st.number_input("Exit Price", min_value=0.01, value=1000.0, key="exit_px")
        with cl3:
            exit_type  = st.selectbox("Type", ["CLOSED", "PARTIAL", "STOPPED"], key="exit_type")
        with cl4:
            exit_notes = st.text_input("Notes", placeholder="Reason…", key="exit_notes")
        with cl5:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Close Trade", key="close_btn"):
                _close_trade(id_options[sel_label], exit_price, exit_type, exit_notes)
                st.success("✅ Trade closed.")
                st.rerun()

        # Delete trade
        with st.expander("🗑️ Delete a Trade", expanded=False):
            del_label = st.selectbox("Select Trade to Delete", list(id_options.keys()), key="del_sel")
            if st.button("Delete Trade", key="del_btn"):
                _delete_trade(id_options[del_label])
                st.warning("Trade deleted.")
                st.rerun()

    # ── Closed trades ──
    st.markdown("---")
    st.markdown("### 📁 Closed / Partial Trades")
    closed_df = _get_trades("ALL")
    closed_df = closed_df[closed_df["status"] != "OPEN"]
    if closed_df.empty:
        st.info("No closed trades yet.")
    else:
        wins = closed_df[closed_df["pnl"] > 0]
        losses = closed_df[closed_df["pnl"] <= 0]
        win_rate = round(len(wins) / len(closed_df) * 100, 1) if len(closed_df) else 0
        total_pnl = round(closed_df["pnl"].sum(), 2)

        cs1, cs2, cs3, cs4 = st.columns(4)
        cs1.metric("Total Closed", len(closed_df))
        cs2.metric("Win Rate",     f"{win_rate}%")
        cs3.metric("Total P&L",    f"{total_pnl:,.0f}",
                   delta_color="normal" if total_pnl >= 0 else "inverse")
        cs4.metric("Avg Win / Loss",
                   f"{wins['pnl'].mean():.0f} / {losses['pnl'].mean():.0f}"
                   if (len(wins) and len(losses)) else "–")

        show = ["date_open", "symbol", "entry_price", "stop_price",
                "exit_price", "qty", "pnl", "status", "exit_notes"]
        st.dataframe(
            closed_df[[c for c in show if c in closed_df.columns]].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "pnl":         st.column_config.NumberColumn("P&L", format="%.0f"),
                "entry_price": st.column_config.NumberColumn("Entry",  format="%.2f"),
                "stop_price":  st.column_config.NumberColumn("Stop",   format="%.2f"),
                "exit_price":  st.column_config.NumberColumn("Exit",   format="%.2f"),
            },
        )


# ══════════════════════════════════════════════
# TAB 6: QUICK REFERENCE
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 📚 Quick Reference — Rules at a Glance")

    col_q1, col_q2, col_q3 = st.columns(3, gap="large")

    with col_q1:
        st.markdown("### ✅ The 10 Entry Commandments")
        for i, rule in enumerate([
            "Market index must be in an uptrend",
            "Stock price > EMA 50 > EMA 200",
            "EMA squeeze or tight base (3+ weeks)",
            "Breakout with volume ≥ 150% of avg",
            "RS Rating ≥ 70",
            "R:R ≥ 2:1 before entering",
            "Stop defined before you enter",
            "Position size via 1% risk rule",
            "No earnings within 2 weeks",
            "Not extended (< 15% from 50 EMA)",
        ], 1):
            st.markdown(f"**{i}.** {rule}")

    with col_q2:
        st.markdown("### 🚪 Exit Triggers (Priority Order)")
        for i, rule in enumerate([
            "Stop loss hit → EXIT FULL, no debate",
            "8% loss from cost → hard cut",
            "Failed breakout (close below pivot) → next morning exit",
            "Climax run (parabolic + 3× volume) → sell 50–100%",
            "Earnings within 1 week → sell 50%",
            "Daily close below EMA 50 (from profit) → exit",
            "Daily close below EMA 20 → tighten stop",
            "20%+ gain in < 3 weeks → book partial",
            "Target 1 hit → sell 30%, raise stop to BE",
            "Target 2 hit → sell 30%, trail EMA 20",
        ], 1):
            st.markdown(f"**{i}.** {rule}")

    with col_q3:
        st.markdown("### 🧠 Mindset Rules")
        for rule in [
            "The market is always right. You are not.",
            "Cut losses short. Let winners run.",
            "A small loss is information. A big loss is a mistake.",
            "Never move a stop wider — only tighter.",
            "Journal every trade. Patterns of error become visible.",
            "After 3 consecutive losses, reduce size by 50%. Take a break.",
            "Never risk more than 6–8% of portfolio on all open trades combined.",
            "Miss 100 trades. Never miss a stop.",
            "Compound your gains — don't rush to deploy all cash.",
            "Preparation before market open beats reaction during it.",
        ]:
            st.markdown(f"▸ {rule}")

    st.markdown("---")
    st.markdown("### 📊 Trade Grade Card")
    grade_df = pd.DataFrame({
        "Grade": ["A+", "A", "B", "C", "D", "F"],
        "Description": [
            "Perfect setup: all 10 entry rules met, breakout with 2× volume",
            "9/10 rules, minor flaw (e.g., volume only 130% of avg)",
            "7–8/10 rules, small extension or weak RS",
            "6/10 rules, marginal setup — reduce position to 50%",
            "< 6 rules — should not have been taken",
            "Stop violated before setup could develop",
        ],
        "Position Size": ["100%", "100%", "75%", "50%", "25%", "Exit"],
        "Action": [
            "Full size, pyramid on strength",
            "Full size",
            "Normal size, no adding",
            "Half size, no adding",
            "Quarter size or skip",
            "Exit immediately",
        ],
    })
    st.dataframe(grade_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "This playbook is based on momentum/trend-following principles (Minervini SEPA, O'Neil CAN SLIM). "
        "All rules are guidelines — adapt to your risk tolerance. "
        "**Not financial advice.**"
    )
