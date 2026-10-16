import os, json
import pandas as pd
import yfinance as yf
from flask import Blueprint, render_template, request, jsonify

chart_bp = Blueprint("chart", __name__)

_PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UPLOAD_FOLDER  = os.path.join(_PROJECT_ROOT, 'uploads', 'chart')
WATCHLIST_FILE = os.path.join(UPLOAD_FOLDER, 'watchlist.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    from app.services.market_data_cache import ind_cache as _ind_cache, us_cache as _us_cache
    _CACHE_AVAILABLE = True
except ImportError:
    _ind_cache = _us_cache = None
    _CACHE_AVAILABLE = False

DEFAULT_WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA",
]

INTERVAL_CFG = {
    "1m":  ("7d",  "1m"),
    "5m":  ("60d", "5m"),
    "15m": ("60d", "15m"),
    "30m": ("60d", "30m"),
    "1h":  ("1y",  "60m"),
    "4h":  ("1y",  "90m"),
    "D":   ("2y",  "1d"),
    "W":   ("5y",  "1wk"),
    "M":   ("10y", "1mo"),
}

def _load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            return json.load(open(WATCHLIST_FILE))
        except Exception:
            pass
    return DEFAULT_WATCHLIST[:]

def _save_watchlist(wl):
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(wl, f)

def _normalise(df):
    if df is None or df.empty:
        return None
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        flat = {}
        for c in cols:
            field = c[0] if isinstance(c, tuple) else c
            if str(field).lower() in ('open','high','low','close','volume'):
                flat[c] = str(field).title()
        if flat:
            df = df[list(flat.keys())].copy()
            df.columns = list(flat.values())
    else:
        df = df.copy()
        df.columns = [str(c).title() if str(c).lower() in
                      ('open','high','low','close','volume') else c for c in df.columns]
    return df

def _fetch_ohlcv(symbol, period, yf_interval):
    is_india = symbol.endswith(".NS")
    if _CACHE_AVAILABLE and yf_interval == "1d":
        cache = _ind_cache if is_india else _us_cache
        try:
            data, _ = cache.get_price_history_bulk(
                [symbol], interval="1d", lookback_days=504,
                progress_callback=lambda i, t, s: None,
            )
            df = _normalise(data.get(symbol))
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"[Chart] cache error: {e}")
    return yf.Ticker(symbol).history(period=period, interval=yf_interval)

# ── Routes ────────────────────────────────────────────────────────────────────

@chart_bp.route("/chart")
def chart_view():
    sym = request.args.get("symbol", "RELIANCE.NS")
    return render_template("chart_screener.html",
                           default_symbol=sym,
                           watchlist=_load_watchlist())

@chart_bp.route("/api/chart/ohlcv")
def chart_ohlcv():
    symbol   = request.args.get("symbol", "RELIANCE.NS").upper().strip()
    interval = request.args.get("interval", "D")
    period, yf_iv = INTERVAL_CFG.get(interval, ("2y", "1d"))
    is_india  = symbol.endswith(".NS")
    bench     = "^NSEI" if is_india else "^GSPC"
    intraday  = interval in ("1m", "5m", "15m", "30m", "1h", "4h")

    try:
        df = _fetch_ohlcv(symbol, period, yf_iv)
        if df is None or df.empty:
            return jsonify({"error": f"No data for {symbol}"}), 404

        df = _normalise(df) or df
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, 'tz', None):
            df.index = df.index.tz_localize(None)
        df = df.sort_index().dropna(subset=['Close'])

        def gv(row, *keys):
            for k in keys:
                if k in row.index:
                    return float(row[k])
            return 0.0

        ohlcv = []
        for ts, row in df.iterrows():
            t = int(ts.timestamp()) if intraday else ts.strftime("%Y-%m-%d")
            ohlcv.append({
                "time":   t,
                "open":   round(gv(row, "Open",   "open"),   4),
                "high":   round(gv(row, "High",   "high"),   4),
                "low":    round(gv(row, "Low",    "low"),    4),
                "close":  round(gv(row, "Close",  "close"),  4),
                "volume": int(gv(row, "Volume", "volume") or 0),
            })

        # RS vs benchmark (daily only)
        rs_data = []
        if not intraday:
            try:
                bdf = yf.Ticker(bench).history(period=period, interval=yf_iv)
                if not bdf.empty:
                    bdf.index = pd.to_datetime(bdf.index)
                    if getattr(bdf.index, 'tz', None):
                        bdf.index = bdf.index.tz_localize(None)
                    bdf = bdf.sort_index()
                    sc = df["Close"].reindex(bdf.index, method="ffill").dropna()
                    bc = bdf["Close"].reindex(sc.index).dropna()
                    sc = sc.reindex(bc.index)
                    if len(sc) > 1 and sc.iloc[0] and bc.iloc[0]:
                        rs = (sc / sc.iloc[0]) / (bc / bc.iloc[0])
                        rs_data = [
                            {"time": ts.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
                            for ts, v in rs.items() if not pd.isna(v)
                        ]
            except Exception as e:
                print(f"[Chart] RS error: {e}")

        last = round(float(df["Close"].iloc[-1]), 2)
        prev = round(float(df["Close"].iloc[-2]), 2) if len(df) > 1 else last
        chg  = round(last - prev, 4)
        chgp = round(chg / prev * 100 if prev else 0, 2)

        return jsonify({
            "symbol": symbol, "interval": interval,
            "market": "ind" if is_india else "us",
            "benchmark": bench,
            "ohlcv": ohlcv, "rs": rs_data,
            "last": last, "chg": chg, "chg_pct": chgp,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@chart_bp.route("/api/chart/prices")
def chart_prices():
    wl = _load_watchlist()
    if not wl:
        return jsonify([])
    result = []
    try:
        raw = yf.download(" ".join(wl), period="2d", interval="1d",
                          group_by="ticker", progress=False, auto_adjust=True)
        single = len(wl) == 1
        for sym in wl:
            try:
                c = raw["Close"] if single else raw[sym]["Close"]
                c = c.dropna()
                last = round(float(c.iloc[-1]), 2)
                prev = round(float(c.iloc[-2]), 2) if len(c) > 1 else last
                chg  = round(last - prev, 2)
                pct  = round(chg / prev * 100 if prev else 0, 2)
                result.append({"symbol": sym, "last": last, "chg": chg, "chg_pct": pct})
            except Exception:
                result.append({"symbol": sym, "last": 0, "chg": 0, "chg_pct": 0})
    except Exception:
        result = [{"symbol": s, "last": 0, "chg": 0, "chg_pct": 0} for s in wl]
    return jsonify(result)


@chart_bp.route("/api/chart/watchlist/add", methods=["POST"])
def chart_wl_add():
    sym = (request.get_json(force=True) or {}).get("symbol", "").upper().strip()
    if not sym:
        return jsonify({"error": "No symbol"}), 400
    wl = _load_watchlist()
    if sym not in wl:
        wl.append(sym)
        _save_watchlist(wl)
    return jsonify({"watchlist": wl})


@chart_bp.route("/api/chart/watchlist/remove", methods=["POST"])
def chart_wl_remove():
    sym = (request.get_json(force=True) or {}).get("symbol", "").upper().strip()
    wl  = [s for s in _load_watchlist() if s != sym]
    _save_watchlist(wl)
    return jsonify({"watchlist": wl})
