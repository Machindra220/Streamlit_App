import os
import json
import glob
import threading
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Blueprint, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime, date

techfund_us_bp = Blueprint("techfund_us", __name__)

# ── Path anchoring ────────────────────────────────────────────────────────────
_PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UPLOAD_FOLDER   = os.path.join(_PROJECT_ROOT, 'uploads', 'techfund_us')
DATA_FOLDER     = os.path.join(_PROJECT_ROOT, 'data')
RESULTS_JSON    = os.path.join(UPLOAD_FOLDER, 'last_techfund_us_results.json')
LAST_CSV_CONFIG = os.path.join(UPLOAD_FOLDER, 'last_csv_path.json')
HISTORY_DIR     = os.path.join(UPLOAD_FOLDER, 'history_cache')
SECTOR_CACHE    = os.path.join(_PROJECT_ROOT, 'instance', 'sector_cache.json')
DEFAULT_CSV     = os.path.join(DATA_FOLDER, 'sp500.csv')
_SECTOR_TTL_DAYS = 15

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HISTORY_DIR,   exist_ok=True)
os.makedirs(DATA_FOLDER,   exist_ok=True)
os.makedirs(os.path.join(_PROJECT_ROOT, 'instance'), exist_ok=True)

# ── Shared market-data cache (US) ─────────────────────────────────────────────
try:
    from app.services.market_data_cache import us_cache as _us_cache
    _US_CACHE_AVAILABLE = True
except ImportError:
    _us_cache = None
    _US_CACHE_AVAILABLE = False

# ── Progress tracking ─────────────────────────────────────────────────────────
_progress = {"pct": 0, "msg": "Idle", "running": False,
             "cache_hits": 0, "yf_fetches": 0, "failed": 0, "phase": ""}
_lock = threading.Lock()

def _set_progress(pct, msg, phase=""):
    with _lock:
        _progress["pct"]   = pct
        _progress["msg"]   = msg
        _progress["phase"] = phase

# ── S&P 500 symbol scraper ─────────────────────────────────────────────────────
_SP500_CACHE = {"syms": None, "fetched": None}

def _fetch_sp500_wikipedia():
    try:
        url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0 (screener-bot)"})
        dfs  = pd.read_html(resp.text)
        df   = dfs[0]
        col  = next(c for c in df.columns if 'ticker' in str(c).lower() or 'symbol' in str(c).lower())
        syms = df[col].dropna().str.strip().tolist()
        syms = [s.replace('.', '-') for s in syms]
        return syms
    except Exception as e:
        print(f"[TechFund-US] Wikipedia scrape error: {e}")
        return None

def _get_default_symbols():
    if os.path.exists(DEFAULT_CSV):
        try:
            df = pd.read_csv(DEFAULT_CSV)
            df.columns = df.columns.str.strip().str.lower()
            col = next((c for c in df.columns if c in ('symbol', 'ticker')), None)
            if col:
                return df[col].dropna().unique().tolist()
        except Exception:
            pass

    if _SP500_CACHE["syms"] and _SP500_CACHE["fetched"]:
        age = (datetime.now() - _SP500_CACHE["fetched"]).total_seconds()
        if age < 86400:
            return _SP500_CACHE["syms"]

    syms = _fetch_sp500_wikipedia()
    if syms:
        _SP500_CACHE["syms"]    = syms
        _SP500_CACHE["fetched"] = datetime.now()
        try:
            pd.DataFrame({"symbol": syms}).to_csv(DEFAULT_CSV, index=False)
        except Exception:
            pass
        return syms

    return ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","BRK-B","JPM","V",
            "UNH","JNJ","XOM","PG","MA","HD","CVX","ABBV","MRK","PFE",
            "AVGO","LLY","TMO","COST","PEP","CSCO","ADBE","ACN","NEE","DHR",
            "VZ","T","INTC","AMD","CRM","QCOM","TXN","WMT","AMGN","GILD",
            "IBM","ORCL","AMAT","MU","LRCX","KLAC","SNPS","CDNS","NOW","INTU"]

def _get_active_source():
    if os.path.exists(LAST_CSV_CONFIG):
        try:
            cfg   = json.load(open(LAST_CSV_CONFIG))
            fname = cfg.get("last_used_csv", "")
            fpath = os.path.join(UPLOAD_FOLDER, fname)
            if fname and os.path.exists(fpath):
                return fpath, fname, False
        except Exception:
            pass
    return DEFAULT_CSV, "S&P 500 (Default)", True

# ── Sector cache ──────────────────────────────────────────────────────────────
def _load_sector_cache():
    if os.path.exists(SECTOR_CACHE):
        try:
            return json.load(open(SECTOR_CACHE))
        except Exception:
            pass
    return {}

def _save_sector_cache(cache):
    try:
        with open(SECTOR_CACHE, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"[TechFund-US] sector cache save error: {e}")

def _get_sector_cached(symbol, sector_cache):
    today = date.today().isoformat()
    entry = sector_cache.get(symbol)
    if entry:
        try:
            age = (date.today() - date.fromisoformat(entry.get("fetched", ""))).days
            if age < _SECTOR_TTL_DAYS:
                return entry.get("sector", "N/A")
        except Exception:
            pass
    try:
        s = yf.Ticker(symbol).info.get('sector', 'N/A') or 'N/A'
    except Exception:
        s = 'N/A'
    sector_cache[symbol] = {"sector": s, "fetched": today}
    return s

# ── DataFrame normaliser ──────────────────────────────────────────────────────
def _normalise_df(df, sym=None):
    if df is None or df.empty:
        return None
    cols = df.columns
    if isinstance(cols, pd.MultiIndex) or (len(cols) > 0 and isinstance(cols[0], tuple)):
        if sym is not None:
            for cand in [sym, sym.replace('-', '.'), sym.replace('.', '-')]:
                try:
                    sliced = df.xs(cand, axis=1, level=1)
                    sliced.columns = [c.title() for c in sliced.columns]
                    return sliced
                except KeyError:
                    pass
        flat_cols = {}
        for c in cols:
            field = c[0] if isinstance(c, tuple) else c
            if field.lower() in ('open', 'high', 'low', 'close', 'volume'):
                flat_cols[c] = field.title()
        if flat_cols:
            df = df[list(flat_cols.keys())].copy()
            df.columns = list(flat_cols.values())
            return df
        return None
    df = df.copy()
    df.columns = [c.title() if isinstance(c, str) and c.lower() in
                  ('open', 'high', 'low', 'close', 'volume') else c for c in df.columns]
    return df

# ── Quarterly fundamental data ─────────────────────────────────────────────────
def _quarterly_growth(symbol):
    """Returns (eps_yoy%, rev_yoy%, eps_accel_bool). None when data unavailable."""
    try:
        t   = yf.Ticker(symbol)
        fin = None
        for attr in ('quarterly_income_stmt', 'quarterly_financials'):
            try:
                f = getattr(t, attr, None)
                if f is not None and not f.empty:
                    fin = f
                    break
            except Exception:
                pass

        if fin is None or fin.empty:
            return None, None, False

        eps_row = rev_row = None
        for idx in fin.index:
            idx_l = str(idx).lower()
            if eps_row is None and any(k in idx_l for k in ('basic eps', 'diluted eps', ' eps')):
                eps_row = fin.loc[idx]
            if rev_row is None and 'total revenue' in idx_l:
                rev_row = fin.loc[idx]
            if eps_row is None and 'net income' in idx_l:
                eps_row = fin.loc[idx]

        cols = list(fin.columns)
        eps_yoy = rev_yoy = None
        eps_accel = False

        if rev_row is not None and len(cols) >= 5:
            try:
                r0 = float(rev_row.iloc[0])
                r4 = float(rev_row.iloc[4])
                if r4 and r4 != 0 and not pd.isna(r0) and not pd.isna(r4):
                    rev_yoy = round((r0 - r4) / abs(r4) * 100, 1)
            except Exception:
                pass

        if eps_row is not None and len(cols) >= 5:
            try:
                e0 = float(eps_row.iloc[0])
                e4 = float(eps_row.iloc[4])
                if e4 and e4 != 0 and not pd.isna(e0) and not pd.isna(e4):
                    eps_yoy = round((e0 - e4) / abs(e4) * 100, 1)
                    if len(cols) >= 6:
                        e1 = float(eps_row.iloc[1])
                        e5 = float(eps_row.iloc[5])
                        if e5 and e5 != 0 and not pd.isna(e1) and not pd.isna(e5):
                            prior_yoy = (e1 - e5) / abs(e5) * 100
                            eps_accel = eps_yoy > prior_yoy
            except Exception:
                pass

        return eps_yoy, rev_yoy, eps_accel
    except Exception:
        return None, None, False

# ── Technical analysis ────────────────────────────────────────────────────────
def _analyse_us(symbol, df, sector, bench_close):
    try:
        if df is None or df.empty or len(df) < 200:
            return None

        df = _normalise_df(df, symbol)
        if df is None or df.empty:
            return None

        if not {'Close', 'High', 'Low'}.issubset(df.columns):
            return None

        close  = df['Close'].dropna()
        high_s = df['High'].dropna()
        low_s  = df['Low'].dropna()

        if len(close) < 200:
            return None

        curr = float(close.iloc[-1])

        sma20_s  = close.rolling(20).mean()
        sma50_s  = close.rolling(50).mean()
        sma150_s = close.rolling(150).mean()
        sma200_s = close.rolling(200).mean()

        sma20  = float(sma20_s.iloc[-1])
        sma50  = float(sma50_s.iloc[-1])
        sma150 = float(sma150_s.iloc[-1])
        sma200 = float(sma200_s.iloc[-1])

        if pd.isna(sma200) or sma200 == 0:
            return None

        high_52w = float(high_s.tail(252).max()) if len(high_s) >= 252 else float(high_s.max())
        low_52w  = float(low_s.tail(252).min())  if len(low_s)  >= 252 else float(low_s.min())
        dist_52w_high = round((high_52w - curr) / high_52w * 100, 2) if high_52w > 0 else 0.0

        # Minervini Trend Template (6 conditions)
        sma200_20ago = float(sma200_s.iloc[-21]) if len(close) >= 221 and not pd.isna(sma200_s.iloc[-21]) else sma200
        m1 = bool(curr > sma150 and curr > sma200)
        m2 = bool(sma150 > sma200)
        m3 = bool(sma200 > sma200_20ago)
        m4 = bool(sma50 > sma150 and sma50 > sma200)
        m5 = bool(curr > sma50)
        m6 = bool(low_52w > 0 and curr >= low_52w * 1.25)

        minervini       = all([m1, m2, m3, m4, m5, m6])
        minervini_score = sum([m1, m2, m3, m4, m5, m6])

        # RS score vs S&P 500
        rs_score = 0.0
        if bench_close is not None and len(bench_close) > 0:
            try:
                bc = bench_close.copy()
                cl = close.copy()
                if getattr(bc.index, 'tz', None) is not None:
                    bc.index = bc.index.tz_localize(None)
                if getattr(cl.index, 'tz', None) is not None:
                    cl.index = cl.index.tz_localize(None)
                aligned = bc.reindex(cl.index, method='ffill').dropna()
                common  = cl.reindex(aligned.index).dropna()
                aligned = aligned.reindex(common.index)
                window  = min(252, len(common))
                if window >= 63:
                    s_ret = float(common.iloc[-1]) / float(common.iloc[-window]) - 1
                    b_ret = float(aligned.iloc[-1]) / float(aligned.iloc[-window]) - 1
                    rs_score = ((1 + s_ret) / (1 + b_ret) - 1) if (1 + b_ret) != 0 else s_ret
            except Exception:
                pass
        elif len(close) >= 63:
            rs_score = float(close.iloc[-1]) / float(close.iloc[-63]) - 1

        return {
            "symbol":          symbol,
            "sector":          sector,
            "price":           round(curr, 2),
            "sma20":           round(sma20, 2),
            "sma50":           round(sma50, 2),
            "sma150":          round(sma150, 2),
            "sma200":          round(sma200, 2),
            "high_52w":        round(high_52w, 2),
            "low_52w":         round(low_52w, 2),
            "dist_52w_high":   dist_52w_high,
            "minervini":       minervini,
            "minervini_score": minervini_score,
            "m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6,
            "rs_score":  rs_score,
            "rs_rating": 0,
            "eps_yoy":   None,
            "rev_yoy":   None,
            "eps_accel": False,
            "result":    "N/A",
        }
    except Exception as e:
        print(f"[TechFund-US] _analyse error {symbol}: {e}")
        return None

# ── Scan worker ───────────────────────────────────────────────────────────────
def _run_scan(symbols, source_name, filters):
    global _progress
    total = len(symbols)

    sector_cache = _load_sector_cache()

    # Step 0: Benchmark
    _set_progress(1, "Fetching ^GSPC benchmark…", "benchmark")
    bench_close = None
    try:
        if _US_CACHE_AVAILABLE:
            bench_data, _ = _us_cache.get_price_history_bulk(
                ["^GSPC"], interval="1d", lookback_days=504,
                progress_callback=lambda i, t, s: None,
            )
            bdf = bench_data.get("^GSPC")
            if bdf is not None and not bdf.empty:
                bdf = _normalise_df(bdf, "^GSPC")
                if bdf is not None and "Close" in bdf.columns:
                    bench_close = bdf["Close"]
        if bench_close is None:
            bdf = yf.Ticker("^GSPC").history(period="2y")
            if not bdf.empty:
                bench_close = bdf["Close"]
    except Exception as e:
        print(f"[TechFund-US] benchmark error: {e}")

    # Step 1: Bulk price fetch
    price_data   = {}
    fetch_report = {"from_cache": 0, "fetched": 0, "failed": []}

    if _US_CACHE_AVAILABLE:
        _set_progress(2, f"Loading {total} symbols from cache…", "price_fetch")

        def _cb(idx, ttl, sym):
            pct = 2 + int((idx / max(ttl, 1)) * 36)
            _set_progress(pct, f"[Cache] {sym} ({idx}/{ttl})", "price_fetch")

        try:
            price_data, fetch_report = _us_cache.get_price_history_bulk(
                symbols, interval="1d", lookback_days=504, progress_callback=_cb,
            )
        except Exception as e:
            print(f"[TechFund-US] Cache bulk fetch error: {e}")
            fetch_report = {"from_cache": 0, "fetched": 0, "failed": list(symbols)}

    if not _US_CACHE_AVAILABLE:
        _set_progress(2, f"Fetching {total} symbols via yfinance…", "price_fetch")
        for i, sym in enumerate(symbols):
            pct = 2 + int((i / max(total, 1)) * 36)
            _set_progress(pct, f"[yfinance] {sym} ({i+1}/{total})", "price_fetch")
            try:
                df = yf.Ticker(sym).history(period="2y")
                price_data[sym] = df if not df.empty else None
                fetch_report["fetched"] += 1
            except Exception:
                price_data[sym] = None
                fetch_report["failed"].append(sym)

    # Step 2: Technical analysis
    _set_progress(39, "Computing technical indicators…", "technical")
    tech_results = []
    for sym in symbols:
        res = _analyse_us(sym, price_data.get(sym), "N/A", bench_close)
        if res:
            tech_results.append((sym, res))

    tech_total = len(tech_results)

    # Step 3: Sector lookup
    _set_progress(42, f"Sector lookup for {tech_total} stocks…", "sector")
    for i, (sym, res) in enumerate(tech_results):
        pct = 42 + int((i / max(tech_total, 1)) * 8)
        _set_progress(pct, f"Sector: {sym} ({i+1}/{tech_total})", "sector")
        res["sector"] = _get_sector_cached(sym, sector_cache)
    _save_sector_cache(sector_cache)

    # Step 4: Fundamental data
    _set_progress(51, f"Fetching fundamentals for {tech_total} stocks…", "fundamentals")
    for i, (sym, res) in enumerate(tech_results):
        pct = 51 + int((i / max(tech_total, 1)) * 35)
        _set_progress(pct, f"Fundamentals: {sym} ({i+1}/{tech_total})", "fundamentals")
        eps_yoy, rev_yoy, eps_accel = _quarterly_growth(sym)
        res["eps_yoy"]   = eps_yoy
        res["rev_yoy"]   = rev_yoy
        res["eps_accel"] = eps_accel

    # Step 5: RS percentile ranking
    _set_progress(87, "Computing RS percentile ranking…", "ranking")
    stocks = [r for _, r in tech_results]
    if stocks:
        scores = np.array([s["rs_score"] for s in stocks])
        n = len(scores)
        for s in stocks:
            rank = int(np.sum(scores <= s["rs_score"]))
            s["rs_rating"] = max(1, min(99, round(rank / n * 99)))

    # Step 6: PASS/FAIL
    _set_progress(91, "Applying PASS/FAIL filters…", "filter")
    min_eps  = float(filters.get("min_eps_growth",    20))
    min_rev  = float(filters.get("min_rev_growth",    10))
    max_dist = float(filters.get("max_52w_high_dist", 25))
    req_min  = bool(filters.get("require_minervini",  True))

    for s in stocks:
        tech_ok = True
        if req_min and not s["minervini"]:
            tech_ok = False
        if s["dist_52w_high"] > max_dist:
            tech_ok = False
        eps_ok = s["eps_yoy"] is None or s["eps_yoy"] >= min_eps
        rev_ok = s["rev_yoy"] is None or s["rev_yoy"] >= min_rev
        s["result"] = "PASS" if (tech_ok and eps_ok and rev_ok) else "FAIL"

    stocks.sort(key=lambda x: x["rs_rating"], reverse=True)

    pass_count  = sum(1 for s in stocks if s["result"] == "PASS")
    rs80_count  = sum(1 for s in stocks if s["rs_rating"] >= 80)
    accel_count = sum(1 for s in stocks if s["eps_accel"])
    last_run    = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    ts_stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[TechFund-US] {total} scanned | {tech_total} valid | {pass_count} PASS | "
          f"cache={fetch_report.get('from_cache',0)} yf={fetch_report.get('fetched',0)}")

    payload = {
        "last_run": last_run, "source": source_name,
        "count": len(stocks), "pass_count": pass_count,
        "rs80_count": rs80_count, "accel_count": accel_count,
        "cache_hits": fetch_report.get("from_cache", 0),
        "yf_fetches": fetch_report.get("fetched", 0),
        "failed": len(fetch_report.get("failed", [])),
        "filters": filters, "stocks": stocks,
    }
    with open(RESULTS_JSON, 'w') as f:
        json.dump(payload, f)

    hist_file = os.path.join(HISTORY_DIR, f"techfund_us_{ts_stamp}.json")
    with open(hist_file, 'w') as f:
        json.dump({"timestamp": last_run, "source": source_name, "count": len(stocks),
                   "pass_count": pass_count, "rs80_count": rs80_count,
                   "filters": filters, "stocks": stocks}, f)
    for old_f in sorted(glob.glob(os.path.join(HISTORY_DIR, 'techfund_us_*.json')))[:-5]:
        try: os.remove(old_f)
        except Exception: pass

    with _lock:
        _progress.update({
            "pct": 100, "msg": f"Done — {pass_count} PASS / {len(stocks)} scanned",
            "running": False, "phase": "done",
            "cache_hits": fetch_report.get("from_cache", 0),
            "yf_fetches": fetch_report.get("fetched", 0),
            "failed": len(fetch_report.get("failed", [])),
        })

# ── Shared constants ──────────────────────────────────────────────────────────
DEFAULT_FILTERS = {
    "min_eps_growth":    20.0,
    "min_rev_growth":    10.0,
    "max_52w_high_dist": 25.0,
    "require_minervini": True,
    "max_tickers":       500,
}

PALETTE = [
    {"bg":"rgba(16,185,129,.15)","text":"#34d399","badge":"#10b981","border":"rgba(16,185,129,.3)"},
    {"bg":"rgba(59,130,246,.15)","text":"#60a5fa","badge":"#3b82f6","border":"rgba(59,130,246,.3)"},
    {"bg":"rgba(168,85,247,.15)","text":"#c084fc","badge":"#a855f7","border":"rgba(168,85,247,.3)"},
    {"bg":"rgba(245,158,11,.15)","text":"#fbbf24","badge":"#f59e0b","border":"rgba(245,158,11,.3)"},
    {"bg":"rgba(244,63,94,.15)","text":"#f472b6","badge":"#f43f5e","border":"rgba(244,63,94,.3)"},
]

# ── Routes ────────────────────────────────────────────────────────────────────
@techfund_us_bp.route("/techfund-us/progress")
def techfund_us_progress():
    with _lock:
        return jsonify(dict(_progress))

@techfund_us_bp.route("/techfund-us", methods=["GET", "POST"])
def techfund_us_view():
    global _progress
    summary_message = None
    filepath, source_name, is_default = _get_active_source()

    if request.method == "POST":
        file = request.files.get('file')
        if file and file.filename:
            fname = secure_filename(file.filename)
            ext   = os.path.splitext(fname)[1].lower()
            saved = f"uploaded_us_techfund{ext}"
            file.save(os.path.join(UPLOAD_FOLDER, saved))
            with open(LAST_CSV_CONFIG, 'w') as cf:
                json.dump({"last_used_csv": saved}, cf)
            filepath, source_name, is_default = _get_active_source()

        if request.form.get('use_default') == '1':
            if os.path.exists(LAST_CSV_CONFIG): os.remove(LAST_CSV_CONFIG)
            filepath, source_name, is_default = _get_active_source()

        if request.form.get('clear_source') == '1':
            if os.path.exists(LAST_CSV_CONFIG): os.remove(LAST_CSV_CONFIG)
            filepath, source_name, is_default = _get_active_source()
            summary_message = "✅ Source cleared. Using S&P 500 default."

        try:
            filters = {
                "min_eps_growth":    float(request.form.get('min_eps_growth',    DEFAULT_FILTERS["min_eps_growth"])),
                "min_rev_growth":    float(request.form.get('min_rev_growth',    DEFAULT_FILTERS["min_rev_growth"])),
                "max_52w_high_dist": float(request.form.get('max_52w_high_dist', DEFAULT_FILTERS["max_52w_high_dist"])),
                "require_minervini": request.form.get('require_minervini') == '1',
                "max_tickers":       int(request.form.get('max_tickers',         DEFAULT_FILTERS["max_tickers"])),
            }
        except (ValueError, TypeError):
            filters = DEFAULT_FILTERS.copy()

        symbols = _get_default_symbols()
        if not is_default and os.path.exists(filepath):
            try:
                ext   = os.path.splitext(filepath)[1].lower()
                df_in = pd.read_excel(filepath) if ext in ('.xlsx', '.xls') else pd.read_csv(filepath)
                df_in.columns = df_in.columns.str.strip().str.lower()
                col = next((c for c in df_in.columns if c in ('symbol', 'ticker')), None)
                if col:
                    symbols = df_in[col].dropna().unique().tolist()
            except Exception as e:
                summary_message = f"❌ File read error: {e}"

        symbols = symbols[:filters["max_tickers"]]

        if not _progress.get("running") and 'clear_source' not in request.form:
            with _lock:
                _progress.update({"pct": 0, "msg": "Starting…", "running": True,
                                   "cache_hits": 0, "yf_fetches": 0, "failed": 0, "phase": ""})
            t = threading.Thread(target=_run_scan, args=(symbols, source_name, filters), daemon=True)
            t.start()
            summary_message = summary_message or f"🔄 Scan started — {len(symbols)} tickers queued."

    stocks, last_run_time, cached_source = [], "Never", source_name
    pass_count = rs80_count = cache_hits = yf_fetches = failed_count = accel_count = 0
    last_filters = DEFAULT_FILTERS.copy()

    if os.path.exists(RESULTS_JSON):
        try:
            cached = json.load(open(RESULTS_JSON))
            if isinstance(cached, dict):
                stocks        = cached.get("stocks", [])
                last_run_time = cached.get("last_run", "Unknown")
                cached_source = cached.get("source", source_name)
                pass_count    = cached.get("pass_count", 0)
                rs80_count    = cached.get("rs80_count", 0)
                accel_count   = cached.get("accel_count", 0)
                cache_hits    = cached.get("cache_hits", 0)
                yf_fetches    = cached.get("yf_fetches", 0)
                failed_count  = cached.get("failed", 0)
                last_filters  = cached.get("filters", DEFAULT_FILTERS)
        except Exception:
            pass

    for s in stocks:
        s.setdefault("minervini", False)
        s.setdefault("minervini_score", 0)
        s.setdefault("m1", False); s.setdefault("m2", False); s.setdefault("m3", False)
        s.setdefault("m4", False); s.setdefault("m5", False); s.setdefault("m6", False)
        s.setdefault("eps_yoy", None)
        s.setdefault("rev_yoy", None)
        s.setdefault("eps_accel", False)
        s.setdefault("result", "N/A")
        s.setdefault("rs_rating", 0)
        s.setdefault("dist_52w_high", 0.0)
        s.setdefault("sector", "N/A")

    sector_counts = {}
    for s in stocks:
        sec = s.get('sector', 'N/A')
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    top_5 = [k for k, _ in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5] if k != 'N/A']
    top_sectors_meta = []
    sector_color_map = {}
    for idx, sec in enumerate(top_5):
        theme = PALETTE[idx]
        sector_color_map[sec] = theme
        top_sectors_meta.append({"name": sec, "count": sector_counts[sec], "theme": theme})

    history = []
    for hf in sorted(glob.glob(os.path.join(HISTORY_DIR, 'techfund_us_*.json')), reverse=True)[:5]:
        try:
            h = json.load(open(hf))
            h['_file'] = os.path.basename(hf)
            history.append(h)
        except Exception:
            pass

    with _lock:
        scan_running = _progress.get("running", False)

    return render_template(
        "techfund_screener_us.html",
        stocks=stocks,
        summary_message=summary_message,
        last_file=source_name,
        is_default=is_default,
        last_run_time=last_run_time,
        top_sectors=top_sectors_meta,
        sector_color_map=sector_color_map,
        history=history,
        pass_count=pass_count,
        rs80_count=rs80_count,
        accel_count=accel_count,
        scan_count=len(stocks),
        scan_running=scan_running,
        cache_hits=cache_hits,
        yf_fetches=yf_fetches,
        failed_count=failed_count,
        last_filters=last_filters,
        default_filters=DEFAULT_FILTERS,
    )

@techfund_us_bp.route("/techfund-us/restore/<filename>")
def techfund_us_restore(filename):
    fpath = os.path.join(HISTORY_DIR, filename)
    if not os.path.exists(fpath):
        return "Not found", 404
    data = json.load(open(fpath))
    with open(RESULTS_JSON, 'w') as f:
        json.dump(data, f)
    return "", 204

@techfund_us_bp.route("/export-techfund-us")
def export_techfund_us():
    if os.path.exists(RESULTS_JSON):
        cached = json.load(open(RESULTS_JSON))
        data = cached.get("stocks", []) if isinstance(cached, dict) else cached
        df = pd.DataFrame(data)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"TechFund_US_{ts}.csv"
        out = os.path.join(UPLOAD_FOLDER, fname)
        df.to_csv(out, index=False)
        return send_file(out, as_attachment=True, download_name=fname)
    return "No data to export", 404
