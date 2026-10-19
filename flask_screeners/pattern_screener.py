"""
EMA Squeeze Breakout Pattern Screener Blueprint
Routes: /pattern-screener            (main page)
        /api/pattern/scan            POST  – start async scan
        /api/pattern/progress        GET   – poll progress
        /api/pattern/results         GET   – latest results JSON
        /api/pattern/history         GET   – last 5 scans
        /api/pattern/history/<id>    GET   – one scan's results
        /api/pattern/export          GET   – CSV download
"""
import io
import os
import json
import sqlite3
import threading
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Blueprint, render_template, jsonify, request, send_file

warnings.filterwarnings("ignore")

pattern_bp = Blueprint("pattern", __name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_UPLOAD       = os.path.join(_PROJECT_ROOT, 'uploads', 'pattern_screener')
_DATA_DIR     = os.path.join(_PROJECT_ROOT, 'data')
_DB           = os.path.join(_UPLOAD, 'pattern_history.db')
os.makedirs(_UPLOAD, exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
def _conn():
    return sqlite3.connect(_DB, check_same_thread=False)


def _init():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                universe        TEXT    NOT NULL,
                total_processed INTEGER NOT NULL DEFAULT 0,
                passed_count    INTEGER NOT NULL DEFAULT 0,
                params_summary  TEXT,
                results_json    TEXT
            )
        """)


_init()


def _save_scan(universe, total, passed, params_summary, results):
    rjson = json.dumps(results)
    with _conn() as c:
        c.execute(
            "INSERT INTO scan_history "
            "(timestamp,universe,total_processed,passed_count,params_summary,results_json) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             universe, total, passed, params_summary, rjson),
        )
        c.execute("""
            DELETE FROM scan_history WHERE id NOT IN (
                SELECT id FROM scan_history ORDER BY id DESC LIMIT 5
            )
        """)


def _load_history():
    with _conn() as c:
        rows = c.execute(
            "SELECT id,timestamp,universe,total_processed,passed_count,params_summary "
            "FROM scan_history ORDER BY id DESC LIMIT 5"
        ).fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "universe": r[2],
         "total": r[3], "passed": r[4], "params": r[5]}
        for r in rows
    ]


def _load_scan_results(scan_id):
    with _conn() as c:
        row = c.execute(
            "SELECT results_json FROM scan_history WHERE id=?", (scan_id,)
        ).fetchone()
    return json.loads(row[0]) if row else []


# ── Ticker loaders ────────────────────────────────────────────────────────────
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


# ── EMA & pattern ─────────────────────────────────────────────────────────────
def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def detect_squeeze_breakout(hist, params):
    c = hist["Close"]
    v = hist["Volume"]
    if len(c) < 210:
        return False, {}

    e10  = _ema(c, 10)
    e20  = _ema(c, 20)
    e50  = _ema(c, 50)
    e200 = _ema(c, 200)

    price    = float(c.iloc[-1])
    cur_e10  = float(e10.iloc[-1])
    cur_e20  = float(e20.iloc[-1])
    cur_e50  = float(e50.iloc[-1])
    cur_e200 = float(e200.iloc[-1])

    if price < cur_e200 or cur_e50 < cur_e200:
        return False, {}

    lookback = int(params.get("lookback", 40))
    ema_mat  = pd.concat([e10, e20, e50], axis=1)
    ema_mat.columns = ["e10", "e20", "e50"]
    ema_mat["spread_pct"] = (
        (ema_mat.max(axis=1) - ema_mat.min(axis=1)) / ema_mat["e50"] * 100
    )
    window     = ema_mat["spread_pct"].iloc[-lookback:]
    min_spread = float(window.min())
    comp_pct   = float(params.get("comp_pct", 2.5))

    if min_spread > comp_pct:
        return False, {}

    squeeze_pos = int(window.values.argmin())
    bars_since  = len(window) - 1 - squeeze_pos
    max_bars    = int(params.get("max_bars_since", 25))

    if bars_since > max_bars:
        return False, {}

    if price < cur_e10 or price < cur_e20 or price < cur_e50:
        return False, {}

    ext_pct = (price - cur_e50) / cur_e50 * 100
    max_ext = float(params.get("max_ext_pct", 20.0))
    if ext_pct > max_ext:
        return False, {}

    avg_vol    = float(v.iloc[-50:-1].mean()) if len(v) > 50 else float(v.mean())
    recent_vol = float(v.iloc[-5:].mean())
    vol_ratio  = round(recent_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    score = round(
        (comp_pct - min_spread) * 20
        + max(0, max_bars - bars_since) * 1.5
        + min(vol_ratio, 3.0) * 5,
        1,
    )

    return True, {
        "min_spread_pct":     round(min_spread, 2),
        "cur_spread_pct":     round(float(ema_mat["spread_pct"].iloc[-1]), 2),
        "bars_since_squeeze": int(bars_since),
        "ext_from_50ema":     round(ext_pct, 2),
        "vol_ratio":          vol_ratio,
        "score":              score,
    }


# ── Async scan state ──────────────────────────────────────────────────────────
_state = {
    "running": False,
    "pct":     0,
    "msg":     "Idle",
    "results": [],
    "total":   0,
    "passed":  0,
    "universe": "",
    "params":  "",
    "error":   None,
}
_lock = threading.Lock()


def _set(key, val):
    with _lock:
        _state[key] = val


def _run_scan(tickers, universe, params):
    _set("running", True)
    _set("pct", 0)
    _set("msg", "Starting…")
    _set("results", [])
    _set("passed", 0)
    _set("error", None)
    _set("universe", universe)

    rows = []
    n = len(tickers)
    for i, sym in enumerate(tickers):
        pct = int((i + 1) / n * 100)
        with _lock:
            _state["pct"] = pct
            _state["msg"] = f"Scanning {sym} ({i+1}/{n})"
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
            hi52       = float(hist["Close"].tail(252).max())
            dist_hi    = round((hi52 - price) / hi52 * 100, 2) if hi52 else None
            try:
                name = yf.Ticker(sym).info.get("longName", sym)[:35]
            except Exception:
                name = sym
            rows.append({
                "ticker":        sym,
                "name":          name,
                "price":         price,
                "day_chg":       day_chg,
                "dist_52wh":     dist_hi,
                "score":         metrics["score"],
                "squeeze_pct":   metrics["min_spread_pct"],
                "bars_since_sqz": metrics["bars_since_squeeze"],
                "ext_50ema":     metrics["ext_from_50ema"],
                "vol_ratio":     metrics["vol_ratio"],
            })
        except Exception:
            continue

    rows.sort(key=lambda r: r["score"], reverse=True)
    params_str = (
        f"Spread≤{params['comp_pct']}% | Lookback={params['lookback']}bars | "
        f"MaxSinceSqz={params['max_bars_since']}bars | Ext≤{params['max_ext_pct']}%"
    )
    with _lock:
        _state["results"] = rows
        _state["passed"]  = len(rows)
        _state["total"]   = n
        _state["pct"]     = 100
        _state["msg"]     = f"Done — {len(rows)} matches found"
        _state["running"] = False
        _state["params"]  = params_str

    _save_scan(universe, n, len(rows), params_str, rows)


# ── Routes ─────────────────────────────────────────────────────────────────────
@pattern_bp.route("/pattern-screener")
def pattern_screener_view():
    return render_template("pattern_screener.html")


@pattern_bp.route("/api/pattern/scan", methods=["POST"])
def start_scan():
    with _lock:
        if _state["running"]:
            return jsonify({"error": "Scan already running"}), 409

    d = request.get_json(force=True)
    universe = d.get("universe", "sp500")
    max_t    = int(d.get("max_tickers", 200))
    params   = {
        "comp_pct":       float(d.get("comp_pct",       2.5)),
        "lookback":       int(d.get("lookback",          40)),
        "max_bars_since": int(d.get("max_bars_since",    20)),
        "max_ext_pct":    float(d.get("max_ext_pct",     20.0)),
    }
    custom = d.get("custom_tickers", "")

    if universe == "nifty500":
        tickers = _load_nifty500()[:max_t]
        label   = "NSE Nifty 500"
    elif universe == "custom":
        tickers = [t.strip().upper() for t in custom.split(",") if t.strip()][:max_t]
        label   = "Custom"
    else:
        tickers = _load_sp500()[:max_t]
        label   = "US S&P 500"

    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    t = threading.Thread(target=_run_scan, args=(tickers, label, params), daemon=True)
    t.start()
    return jsonify({"started": True, "total": len(tickers)})


@pattern_bp.route("/api/pattern/progress")
def scan_progress():
    with _lock:
        snap = dict(_state)
    snap.pop("results", None)
    return jsonify(snap)


@pattern_bp.route("/api/pattern/results")
def scan_results():
    with _lock:
        rows = list(_state["results"])
    return jsonify(rows)


@pattern_bp.route("/api/pattern/history")
def scan_history():
    return jsonify(_load_history())


@pattern_bp.route("/api/pattern/history/<int:scan_id>")
def scan_history_detail(scan_id):
    return jsonify(_load_scan_results(scan_id))


@pattern_bp.route("/api/pattern/export")
def export_csv():
    with _lock:
        rows = list(_state["results"])
    if not rows:
        return jsonify({"error": "No results to export"}), 404
    df  = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"ema_squeeze_{ts}.csv",
    )
