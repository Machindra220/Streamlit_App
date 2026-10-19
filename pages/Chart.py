"""
Interactive Stock Chart Page
Candlestick + EMA overlays + Volume + RS pane
Powered by Lightweight Charts v4
"""

import json
import warnings

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stock Chart",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# MINIMAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] { background:#0E1117; color:#C9D1D9; }
[data-testid="stSidebar"] { background:#131722; }
h1 { color:#A78BFA !important; font-weight:700; }
.stButton > button {
    background:#7C3AED !important; color:#fff !important;
    border:none !important; border-radius:6px !important;
}
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background:#161B22 !important; color:#C9D1D9 !important;
    border:1px solid #30363D !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INTERVAL CONFIG  (must match chart_screener.py)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# HEADER + CONTROLS
# ─────────────────────────────────────────────
params = st.query_params
default_sym = params.get("symbol", "RELIANCE.NS")

st.markdown("# 📊 Stock Chart")

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    symbol = st.text_input("Symbol", value=default_sym, placeholder="e.g. RELIANCE.NS or AAPL").strip().upper()
with col2:
    interval = st.selectbox("Interval", ["D", "W", "M", "1h", "4h", "15m", "5m", "1m"], index=0)
with col3:
    show_ema10  = st.checkbox("EMA 10",  value=True)
    show_ema20  = st.checkbox("EMA 20",  value=True)
with col4:
    show_ema50  = st.checkbox("EMA 50",  value=True)
    show_ema200 = st.checkbox("EMA 200", value=True)

load_btn = st.button("Load Chart", use_container_width=False)

if not symbol:
    st.info("Enter a symbol above and click **Load Chart**.")
    st.stop()

# Auto-load if symbol provided via query param, or on button click
if not load_btn and "symbol" not in params:
    st.info("Click **Load Chart** to render.")
    st.stop()


# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────
def _normalise(df):
    if df is None or df.empty:
        return None
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        flat = {}
        for c in cols:
            field = c[0] if isinstance(c, tuple) else c
            if str(field).lower() in ("open", "high", "low", "close", "volume"):
                flat[c] = str(field).title()
        if flat:
            df = df[list(flat.keys())].copy()
            df.columns = list(flat.values())
    else:
        df = df.copy()
        df.columns = [
            str(c).title() if str(c).lower() in ("open", "high", "low", "close", "volume") else c
            for c in df.columns
        ]
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_chart_data(sym: str, tf: str):
    period, yf_iv = INTERVAL_CFG.get(tf, ("2y", "1d"))
    intraday = tf in ("1m", "5m", "15m", "30m", "1h", "4h")
    is_india = sym.endswith(".NS")
    bench_sym = "^NSEI" if is_india else "^GSPC"

    df = yf.Ticker(sym).history(period=period, interval=yf_iv)
    n = _normalise(df)
    if n is not None:
        df = n
    if df is None or df.empty:
        return None, None, None, None

    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None):
        df.index = df.index.tz_localize(None)
    df = df.sort_index().dropna(subset=["Close"])

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
            "open":   round(gv(row, "Open"),   4),
            "high":   round(gv(row, "High"),   4),
            "low":    round(gv(row, "Low"),    4),
            "close":  round(gv(row, "Close"),  4),
            "volume": int(gv(row, "Volume") or 0),
        })

    # RS line (daily only)
    rs_data = []
    if not intraday:
        try:
            bdf = yf.Ticker(bench_sym).history(period=period, interval=yf_iv)
            if not bdf.empty:
                bdf.index = pd.to_datetime(bdf.index)
                if getattr(bdf.index, "tz", None):
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
        except Exception:
            pass

    last  = round(float(df["Close"].iloc[-1]), 2)
    prev  = round(float(df["Close"].iloc[-2]), 2) if len(df) > 1 else last
    chg   = round(last - prev, 4)
    chgp  = round(chg / prev * 100 if prev else 0, 2)
    currency = "₹" if is_india else "$"

    info = {
        "symbol":   sym,
        "last":     last,
        "chg":      chg,
        "chgp":     chgp,
        "currency": currency,
        "interval": tf,
    }

    return ohlcv, rs_data, info, intraday


with st.spinner(f"Fetching {symbol} [{interval}]…"):
    ohlcv, rs_data, info, intraday = fetch_chart_data(symbol, interval)

if ohlcv is None:
    st.error(f"No data for **{symbol}**. Check the symbol and try again.")
    st.stop()

# ─────────────────────────────────────────────
# INLINE CHART (Lightweight Charts v4)
# ─────────────────────────────────────────────
ema_flags = {
    "ema10":  show_ema10,
    "ema20":  show_ema20,
    "ema50":  show_ema50,
    "ema200": show_ema200,
}

chart_html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#131722; color:#C9D1D9; font-family:'Segoe UI',sans-serif; overflow:hidden; }}
  #header {{ display:flex; gap:16px; align-items:center; padding:8px 12px; background:#1E222D;
             border-bottom:1px solid #2A2E39; flex-wrap:wrap; }}
  #sym  {{ font-size:18px; font-weight:700; color:#fff; }}
  #price{{ font-size:18px; font-weight:700; }}
  #chg  {{ font-size:14px; }}
  #wrap {{ display:flex; flex-direction:column; height:calc(100vh - 44px); }}
  #main {{ flex:1; min-height:0; }}
  #rs   {{ height:120px; border-top:1px solid #2A2E39; }}
</style>
</head>
<body>
<div id="header">
  <span id="sym">{info['symbol']}</span>
  <span id="price"></span>
  <span id="chg"></span>
  <span style="margin-left:auto;color:#60A5FA;font-size:12px">{interval} · Lightweight Charts</span>
</div>
<div id="wrap">
  <div id="main"></div>
  <div id="rs"></div>
</div>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
const OHLCV    = {json.dumps(ohlcv)};
const RS_DATA  = {json.dumps(rs_data)};
const INFO     = {json.dumps(info)};
const INTRADAY = {'true' if intraday else 'false'};
const EMA_FLAGS = {json.dumps(ema_flags)};

const DARK = {{
  layout:     {{ background:{{ type:'solid', color:'#131722' }}, textColor:'#C9D1D9' }},
  grid:       {{ vertLines:{{ color:'#1E222D' }}, horzLines:{{ color:'#1E222D' }} }},
  crosshair:  {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor:'#2A2E39' }},
  timeScale:  {{ borderColor:'#2A2E39', timeVisible:INTRADAY, secondsVisible:false }},
}};

// ── Main chart ──
const mainChart = LightweightCharts.createChart(document.getElementById('main'), {{
  ...DARK,
  width:  document.getElementById('main').clientWidth,
  height: document.getElementById('main').clientHeight,
}});

const candleSeries = mainChart.addCandlestickSeries({{
  upColor:'#26a69a', downColor:'#ef5350',
  borderUpColor:'#26a69a', borderDownColor:'#ef5350',
  wickUpColor:'#26a69a', wickDownColor:'#ef5350',
}});
candleSeries.setData(OHLCV);

// Volume histogram (overlay on price scale)
const volSeries = mainChart.addHistogramSeries({{
  color:'#26a69a', priceFormat:{{ type:'volume' }},
  priceScaleId:'vol',
}});
mainChart.priceScale('vol').applyOptions({{ scaleMargins:{{ top:0.8, bottom:0 }} }});
volSeries.setData(OHLCV.map(d => ({{
  time:d.time,
  value:d.volume,
  color: d.close >= d.open ? '#26a69a55' : '#ef535055',
}})));

// ── EMA helper ──
function calcEma(data, period) {{
  const k = 2/(period+1);
  const closes = data.map(d=>d.close);
  let ema = closes.slice(0,period).reduce((a,b)=>a+b,0)/period;
  const out = [];
  for (let i=0;i<data.length;i++) {{
    if (i < period-1) {{ out.push(null); continue; }}
    if (i === period-1) {{ out.push({{ time:data[i].time, value:+ema.toFixed(4) }}); continue; }}
    ema = closes[i]*k + ema*(1-k);
    out.push({{ time:data[i].time, value:+ema.toFixed(4) }});
  }}
  return out.filter(x=>x);
}}

const EMA_COLORS = {{ ema10:'#EE9322', ema20:'#F1C40F', ema50:'#3498DB', ema100:'#9B59B6', ema200:'#E74C3C' }};
const EMA_PERIODS = {{ ema10:10, ema20:20, ema50:50, ema100:100, ema200:200 }};

Object.entries(EMA_FLAGS).forEach(([key, on]) => {{
  if (!on) return;
  const period = EMA_PERIODS[key];
  const color  = EMA_COLORS[key];
  const s = mainChart.addLineSeries({{ color, lineWidth:1.5, priceLineVisible:false, lastValueVisible:false }});
  s.setData(calcEma(OHLCV, period));
}});

// ── Price display ──
const pEl  = document.getElementById('price');
const cEl  = document.getElementById('chg');
const cur  = INFO.currency;
pEl.textContent = cur + INFO.last.toFixed(2);
pEl.style.color = INFO.chg >= 0 ? '#26a69a' : '#ef5350';
cEl.textContent = (INFO.chg >= 0 ? '+' : '') + INFO.chg.toFixed(2) +
                  ' (' + (INFO.chg >= 0 ? '+' : '') + INFO.chgp.toFixed(2) + '%)';
cEl.style.color = INFO.chg >= 0 ? '#26a69a' : '#ef5350';

mainChart.subscribeCrosshairMove(p => {{
  if (!p.seriesData.size) return;
  const d = p.seriesData.get(candleSeries);
  if (!d) return;
  pEl.textContent = cur + d.close.toFixed(2);
  const chg = d.close - d.open;
  const chgp = (chg/d.open*100);
  pEl.style.color = chg >= 0 ? '#26a69a' : '#ef5350';
  cEl.textContent = 'O:'+d.open.toFixed(2)+' H:'+d.high.toFixed(2)+' L:'+d.low.toFixed(2)+' C:'+d.close.toFixed(2);
}});

// ── RS pane ──
const rsDiv = document.getElementById('rs');
if (RS_DATA.length > 0) {{
  const rsChart = LightweightCharts.createChart(rsDiv, {{
    ...DARK,
    width:  rsDiv.clientWidth,
    height: rsDiv.clientHeight,
    rightPriceScale: {{ visible:true, borderColor:'#2A2E39' }},
    timeScale: {{ visible:false }},
  }});
  const rsSeries = rsChart.addLineSeries({{ color:'#A78BFA', lineWidth:2, priceLineVisible:false, lastValueVisible:false }});
  rsSeries.setData(RS_DATA);

  // Add baseline at 1.0
  rsSeries.createPriceLine({{ price:1.0, color:'#60A5FA55', lineWidth:1, lineStyle:2, axisLabelVisible:false }});

  // Sync time range
  mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
    if (range) rsChart.timeScale().setVisibleLogicalRange(range);
  }});
  rsChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
    if (range) mainChart.timeScale().setVisibleLogicalRange(range);
  }});

  // RS label
  const label = document.createElement('div');
  label.style.cssText = 'position:absolute;bottom:6px;left:8px;font-size:11px;color:#A78BFA;pointer-events:none;z-index:10;';
  label.textContent = 'RS vs ' + (INFO.symbol.endsWith('.NS') ? 'Nifty' : 'S&P 500');
  rsDiv.style.position = 'relative';
  rsDiv.appendChild(label);
}} else {{
  rsDiv.style.display = 'none';
  document.getElementById('main').style.flex = '1';
}}

// ── Resize ──
window.addEventListener('resize', () => {{
  mainChart.resize(document.getElementById('main').clientWidth, document.getElementById('main').clientHeight);
  if (RS_DATA.length > 0) {{
    const rsChart = LightweightCharts.createChart; // already created, just need refs
  }}
}});
</script>
</body>
</html>"""

components.html(chart_html, height=600, scrolling=False)

# ─────────────────────────────────────────────
# INFO ROW BELOW CHART
# ─────────────────────────────────────────────
st.markdown(
    f"<small style='color:#60A5FA'>Data via yFinance · "
    f"Chart: Lightweight Charts v4 · "
    f"{'Intraday' if intraday else 'Daily/Weekly/Monthly'}</small>",
    unsafe_allow_html=True,
)
