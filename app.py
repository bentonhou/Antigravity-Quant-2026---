import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np
import sys
import os
import json
from datetime import date as date_cls

# --- 設定字體與 UI 規範 ---
ui_path = os.path.abspath(os.path.join("Standards", "fonts and UI"))
if ui_path not in sys.path:
    sys.path.append(ui_path)
import ui_config# --- Configuration ---
STOCKS_CONFIG = {
    # Targets updated: 2026-05-13 (based on Wall Street analyst consensus)
    "TSM":  {"start": 319.61, "target": 460.0},   # 舊:435 → 分析師共識 400-480
    "NVDA": {"start": 187.20, "target": 275.0},   # 舊:270 → 分析師共識 270-280
    "AMD":  {"start": 214.30, "target": 460.0},   # 舊:290 → 現價~461！分析師 390-420
    "MSFT": {"start": 472.94, "target": 580.0},   # 舊:590 → 分析師共識 560-590
    "GOOG": {"start": 315.32, "target": 410.0},   # 舊:360 → 分析師共識 360-425
    "QCOM": {"start": 173.00, "target": 178.0},   # 舊:155 → 分析師共識 173-180
    "AMZN": {"start": 237.21, "target": 315.0},   # 舊:285 → 分析師共識 312-318
    "AVGO": {"start": 347.62, "target": 470.0},   # 舊:475 → 分析師共識 436-477
    "MRVL": {"start": 89.39,  "target": 128.0},   # 舊:123 → 分析師共識 121-130
    "ANET": {"start": 133.60, "target": 188.0},   # 舊:235 → 分析師共識 181-186
    "ETN":  {"start": 326.29, "target": 455.0},   # 舊:485 → 分析師共識 420-470
    "NOK":  {"start": 6.51,   "target": 10.0},    # 舊:8.50 → 分析師共識 9.70-10.30
    "UMC":  {"start": 7.77,   "target": 8.50},    # 舊:13.00 → 分析師共識 7.40-8.60 (Bearish)
    "HPE":  {"start": 19.24,  "target": 27.0},    # 維持，分析師共識 26-27
    "TTD":  {"start": 24.73,  "target": 36.0},    # 舊:46 → 分析師共識 33-38
    "INTC": {"start": 36.16,  "target": 120.0},   # 舊:55 → 現價~129！分析師 75-124
    "NBIS": {"start": 100.45, "target": 170.0},   # 舊:210 → 分析師共識 159-174
    "CRWV": {"start": 71.61,  "target": 135.0},   # 舊:160 → 分析師共識 131-133
    "NOW":  {"start": 147.45, "target": 145.0},   # 舊:230 → 分割後共識 140-145 (5-for-1 split Dec 2025)
    "DELL": {"start": 114.44, "target": 190.0},   # 舊:185 → 分析師共識 180-193
}

# --- Target Price Meta File ---
META_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_price_meta.json")

def load_target_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_updated": "2026-05-13", "next_update_allowed": "2026-09-01", "targets": {}}

def save_target_meta(meta):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def get_next_update_allowed(from_date):
    """Returns next quarterly date (Mar1/Jun1/Sep1/Dec1) at least 60 days away."""
    min_date = from_date + timedelta(days=60)
    for year_offset in [0, 1]:
        year = from_date.year + year_offset
        for month in [3, 6, 9, 12]:
            candidate = date_cls(year, month, 1)
            if candidate >= min_date:
                return candidate
    return date_cls(from_date.year + 1, 3, 1)

# Apply saved targets to STOCKS_CONFIG
_meta_init = load_target_meta()
for _sym, _tgt in _meta_init.get("targets", {}).items():
    if _sym in STOCKS_CONFIG:
        STOCKS_CONFIG[_sym]["target"] = _tgt

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days + 1

st.set_page_config(page_title="Antigravity Quant 2026", layout="wide")

# --- Session State Init ---
if "show_refresh_panel" not in st.session_state:
    st.session_state.show_refresh_panel = False

# 載入外部的 CSS 樣式
def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>\n{f.read()}\n</style>", unsafe_allow_html=True)

load_css(os.path.join(ui_path, "ui_styles.css"))

st.markdown("<h1>Antigravity Quant 2026 - 波段導航儀</h1>", unsafe_allow_html=True)

# --- Sidebar Global Settings (Placed early for data dependency) ---
st.sidebar.markdown("<h1>參數調整(目標價校正)</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h2>Market Sentiment</h2>", unsafe_allow_html=True)
sentiment_label = st.sidebar.select_slider(
    "Market Sentiment",
    options=["Optimistic (1.05)", "Base Target (1.00)", "Conservative (0.90)"],
    value="Base Target (1.00)",
    label_visibility="collapsed"
)
sentiment_mapping = {
    "Optimistic (1.05)": 1.05,
    "Base Target (1.00)": 1.0,
    "Conservative (0.90)": 0.90
}
sentiment_factor = sentiment_mapping[sentiment_label]

st.sidebar.markdown("<h2>起點錨定校正</h2>", unsafe_allow_html=True)
_blend_label = st.sidebar.select_slider(
    "起點錨定校正",
    options=["Q4均價 (0.0)", "混合 (0.5)", "手動起點 (1.0)"],
    value="混合 (0.5)",
    label_visibility="collapsed"
)
_blend_mapping = {"Q4均價 (0.0)": 0.0, "混合 (0.5)": 0.5, "手動起點 (1.0)": 1.0}
start_blend_alpha = _blend_mapping[_blend_label]


# --- Helper: Data Fetching with Cache ---
@st.cache_data(ttl=60) # Cache for 60 seconds
def get_stock_data(ticker_or_tickers):
    if not ticker_or_tickers.strip():
        return pd.DataFrame()
    try:
        # Fetch data from 2026-01-01 to NOW
        df = yf.download(ticker_or_tickers, start="2026-01-01", interval="1d", progress=False)
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_yahoo_targets(tickers_tuple):
    """Fetch analyst consensus target prices from Yahoo Finance."""
    results = {}
    for sym in tickers_tuple:
        try:
            info = yf.Ticker(sym).info
            results[sym] = {
                "mean":   info.get("targetMeanPrice"),
                "high":   info.get("targetHighPrice"),
                "low":    info.get("targetLowPrice"),
            }
        except Exception:
            results[sym] = {"mean": None, "high": None, "low": None}
    return results

@st.cache_data(ttl=86400)  # 歷史數據快取 24 小時（前年Q4不會再變動）
def get_prev_q4_avg(tickers_tuple, base_year=2026):
    """取前一年 Q4（10-12月）各股平均收盤價，作為年初基準起點的市場錨點。"""
    prev_year = base_year - 1
    q4_start  = f"{prev_year}-10-01"
    q4_end    = f"{prev_year}-12-31"
    results   = {}
    try:
        df = yf.download(" ".join(tickers_tuple), start=q4_start, end=q4_end,
                         interval="1d", progress=False)
        if df.empty:
            return {t: None for t in tickers_tuple}
        for ticker in tickers_tuple:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    series = df['Close'][ticker].dropna()
                else:
                    series = df['Close'].dropna()
                results[ticker] = float(series.mean()) if not series.empty else None
            except Exception:
                results[ticker] = None
    except Exception:
        results = {t: None for t in tickers_tuple}
    return results

def calculate_status(ticker, price, date_obj, sentiment=1.0, p_start_override=None):
    # Re-implement baseline logic here for single point check
    config = STOCKS_CONFIG[ticker]
    p_start = p_start_override if p_start_override is not None else config["start"]
    p_target = config["target"] * sentiment # Apply Sentiment Adjustment
    slope = (p_target - p_start) / (TOTAL_DAYS - 1)
    
    # 確保日期對象為 naive (移除時區)，以便進行計算
    if hasattr(date_obj, 'tzinfo') and date_obj.tzinfo is not None:
        date_obj = date_obj.replace(tzinfo=None)
    
    status_icon = ":gray[●]" # Default
    
    # Baseline for date
    day_diff = (date_obj - START_DATE).days
    
    if 0 <= day_diff < TOTAL_DAYS:
        curr_baseline = p_start + slope * day_diff
        upper_bound_1 = curr_baseline * 1.25
        upper_bound_2 = curr_baseline * 1.375
        lower_bound = curr_baseline * 0.90
        
        if price <= lower_bound:
            status_icon = ":green[●]"
        elif price >= upper_bound_2:
            status_icon = ":red[●]"
        elif price >= upper_bound_1:
            status_icon = ":orange[●]"
            
    return status_icon

def calculate_trend(series, window=5):
    """
    Calculate trend based on the slope of the last 'window' days.
    Returns: "↗" (positive slope) or "↘" (negative/flat slope).
    """
    if len(series) < 2:
        return "ERROR" # Not enough data
    
    # Take last N days
    y = series.tail(window).values
    x = np.arange(len(y))
    
    # Linear Regression: Slope = Cov(x, y) / Var(x)
    # Or simple: if len is small, just last - first?
    # Let's use simple numpy polyfit for robustness
    try:
        slope, _ = np.polyfit(x, y, 1)
        return "↗" if slope > 0 else "↘"
    except:
        # Fallback to simple diff
        return "↗" if y[-1] >= y[0] else "↘"

# --- Sidebar Preparation ---

all_tickers_list = list(STOCKS_CONFIG.keys())
sidebar_options = {}

prev_q4_avg_data = {}  # 前年Q4均價初始化（起點錨點備援空字典）
with st.spinner("Updating Market Signals..."):
    # 1. 抓取日線歷史數據
    df_all = get_stock_data(" ".join(all_tickers_list))
    # 2. 抓取前年 Q4（10-12月）均價，作為起點偏差校正的市場錨點（快取 24 小時）
    try:
        prev_q4_avg_data = get_prev_q4_avg(tuple(all_tickers_list))
    except Exception:
        prev_q4_avg_data = {}
    # 3. 抓取所有 Ticker 的最新分時數據 (含盤前盤後) - 使用 5d 確保能抓到最近的資料
    try:
        df_ext_all = yf.download(" ".join(all_tickers_list), period='5d', interval='15m', prepost=True, progress=False)
    except:
        df_ext_all = pd.DataFrame()

for ticker in all_tickers_list:
    icon = ":gray[●]"
    trend = ""
    try:
        final_price = None
        final_date = None
        
        # 1. 抓取日線數據 (原始邏輯)
        if isinstance(df_all.columns, pd.MultiIndex):
            if ticker in df_all['Close'].columns:
                series = df_all['Close'][ticker].dropna()
                if not series.empty:
                    final_price = float(series.iloc[-1])
                    final_date = pd.to_datetime(series.index[-1]).replace(tzinfo=None)
                    trend = calculate_trend(series)
        
        # 2. 抓取延伸/盤前數據 (橋接邏輯)
        if not df_ext_all.empty:
            try:
                # 判斷 MultiIndex 結構並尋找對應 Ticker 的 Close
                ext_s = None
                if isinstance(df_ext_all.columns, pd.MultiIndex):
                    if 'Close' in df_ext_all.columns.get_level_values(0) and ticker in df_ext_all['Close'].columns:
                        ext_s = df_ext_all['Close'][ticker].dropna()
                
                if ext_s is not None and not ext_s.empty:
                    e_price = float(ext_s.iloc[-1])
                    _raw_ts = pd.to_datetime(ext_s.index[-1])
                    if _raw_ts.tzinfo is None:
                        _raw_ts = _raw_ts.tz_localize('UTC')
                    e_date = _raw_ts.tz_convert('America/New_York').replace(tzinfo=None)
                    
                    # 若盤前數據較新或屬於同一天，則採用之
                    if final_date is None or e_date.date() >= final_date.date():
                        final_price = e_price
                        final_date = e_date
            except:
                pass

        # 計算此 Ticker 的校正後起點（與主圖基準線邏輯一致）
        _q4_ticker = prev_q4_avg_data.get(ticker)
        _p_start_corr = None
        if _q4_ticker is not None and start_blend_alpha < 1.0:
            _p_start_corr = (1.0 - start_blend_alpha) * _q4_ticker + start_blend_alpha * STOCKS_CONFIG[ticker]["start"]

        if final_price is not None and final_date is not None:
             icon = calculate_status(ticker, final_price, final_date, sentiment_factor, p_start_override=_p_start_corr)
             
    except Exception:
        pass 
    
    display_name = f"{ticker}_" if ticker in ("NOK", "TTD") else ticker
    label = f"*{display_name}* {icon}"
    if trend and trend != "ERROR":
        label += f" {trend}"
    
    sidebar_options[label] = ticker

# --- Sidebar ---
st.sidebar.markdown("<h2>Asset Selection</h2>", unsafe_allow_html=True)
# Create reverse mapping or just use keys
display_keys = list(sidebar_options.keys())
selected_display = st.sidebar.radio("Ticker", display_keys, label_visibility="collapsed")
selected_ticker = sidebar_options[selected_display]

auto_refresh = st.sidebar.checkbox("Auto-Refresh (60s)", value=True)
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# --- Sidebar: Target Price Update ---
st.sidebar.markdown("---")
st.sidebar.markdown("<h2>目標價管理</h2>", unsafe_allow_html=True)
_meta_sb = load_target_meta()
_last_upd_str = _meta_sb.get("last_updated", "N/A")
_next_upd_str = _meta_sb.get("next_update_allowed", "2026-09-01")
try:
    _next_upd_dt = datetime.strptime(_next_upd_str, "%Y-%m-%d").date()
    _today_dt = datetime.now().date()
    _can_update = _today_dt >= _next_upd_dt
    _days_left = (_next_upd_dt - _today_dt).days
except Exception:
    _can_update = False
    _days_left = 999
if _can_update:
    if st.sidebar.button("🔄 刷新目標價", key="btn_refresh_target", use_container_width=True):
        fetch_yahoo_targets.clear()
        st.session_state.show_refresh_panel = True
else:
    st.sidebar.button(
        f"🔒 {_next_upd_str} 開放",
        disabled=True, key="btn_refresh_locked", use_container_width=True
    )
    st.sidebar.caption(f"還有 {_days_left} 天｜上次更新：{_last_upd_str}")

# --- Logic: Baseline ---
config = STOCKS_CONFIG[selected_ticker]
p_start_manual = config["start"]

# --- 起點偏差校正：混合手動起點（主觀基準）與前年Q4市場均價（客觀錨點）---
# 公式：p_start = (1 - alpha) × Q4均價 + alpha × 手動起點
# alpha=1.0 → 完全使用手動起點（原始行為）
# alpha=0.0 → 完全使用前年Q4均價（消除偏差）
_q4_anchor = prev_q4_avg_data.get(selected_ticker)
if _q4_anchor is not None and start_blend_alpha < 1.0:
    p_start = (1.0 - start_blend_alpha) * _q4_anchor + start_blend_alpha * p_start_manual
else:
    p_start = p_start_manual  # 無Q4數據或 alpha=1 時，回退原始手動值

# 1. Base Logic (Sentiment = 1.0)
p_target_base = config["target"]
slope_base = (p_target_base - p_start) / (TOTAL_DAYS - 1)
dates_2026 = [START_DATE + timedelta(days=i) for i in range(TOTAL_DAYS)]
baseline_prices_base = [p_start + slope_base * i for i in range(TOTAL_DAYS)]

df_base = pd.DataFrame({"Date": dates_2026, "Baseline": baseline_prices_base})
df_base["Upper_37_5"] = df_base["Baseline"] * 1.375
df_base["Upper_25"] = df_base["Baseline"] * 1.25
df_base["Lower_10"] = df_base["Baseline"] * 0.90

# 2. Adjusted Logic (Current Sentiment)
p_target_adj = config["target"] * sentiment_factor
slope_adj = (p_target_adj - p_start) / (TOTAL_DAYS - 1)
baseline_prices_adj = [p_start + slope_adj * i for i in range(TOTAL_DAYS)]

df_adj = pd.DataFrame({"Date": dates_2026, "Baseline": baseline_prices_adj})
df_adj["Upper_37_5"] = df_adj["Baseline"] * 1.375
df_adj["Upper_25"] = df_adj["Baseline"] * 1.25
df_adj["Lower_10"] = df_adj["Baseline"] * 0.90




# --- Helper: Extended Hours Price Logic ---
@st.cache_data(ttl=30)
def get_latest_price(ticker):
    """
    獲取最新價格（包含盤前與盤後）。回傳價格、標籤、時間。
    注意：不快取整個 DataFrame，避免 st.cache_data 的序列化問題。
    """
    try:
        t = yf.Ticker(ticker)
        df_h = t.history(period='2d', interval='1m', prepost=True)
        
        if df_h.empty:
            info = t.info
            price = info.get('preMarketPrice') or info.get('postMarketPrice') or info.get('regularMarketPrice') or info.get('currentPrice')
            if price:
                return {"price": float(price), "label": "Live", "time": pd.Timestamp.now(tz='America/New_York')}
            return None

        # Guard: tz_convert requires tz-aware index; localize to UTC first if naive
        if df_h.index.tzinfo is None:
            df_h.index = df_h.index.tz_localize('UTC')
        last_ts = df_h.index[-1].tz_convert('America/New_York')
        last_price = float(df_h['Close'].iloc[-1])
        
        hour = last_ts.hour
        minute = last_ts.minute
        time_f = hour + minute/60.0
        
        label = ""
        if 4.0 <= time_f < 9.5:
            label = "Pre"
        elif 9.5 <= time_f < 16.0:
            label = "Live"
        elif 16.0 <= time_f <= 20.0:
            label = "Post"
        else:
            label = "Ext"
            
        return {"price": last_price, "label": label, "time": last_ts}
    except Exception:
        pass
    return None

def get_ext_df(ticker):
    """
    獲取最新的分時 DataFrame（不快取，每次直接抓取）。
    避免大型 DataFrame 放入 st.cache_data 造成序列化問題。
    """
    try:
        t = yf.Ticker(ticker)
        df_h = t.history(period='2d', interval='1m', prepost=True)
        return df_h
    except Exception:
        return pd.DataFrame()

# --- Main Logic ---
st.markdown(f"<h2>{selected_ticker} Wave Navigator</h2>", unsafe_allow_html=True)

# --- Refresh Panel (shown when user clicks 刷新目標價) ---
if st.session_state.get("show_refresh_panel", False):
    st.markdown("---")
    st.markdown("### 🔄 目標價刷新預覽")
    st.info("以下為 Yahoo Finance 分析師共識目標價與目前設定值的比對。確認後將套用並鎖定至下一季度。")
    with st.spinner("正在從 Yahoo Finance 抓取分析師目標價（約需 20 秒）..."):
        _yahoo_data = fetch_yahoo_targets(tuple(all_tickers_list))
    _rows = []
    for _sym in all_tickers_list:
        _cur_tgt = STOCKS_CONFIG[_sym]["target"]
        _yd = _yahoo_data.get(_sym, {})
        _mean = _yd.get("mean")
        _high = _yd.get("high")
        _low  = _yd.get("low")
        _delta_str = f"{(_mean - _cur_tgt) / _cur_tgt * 100:+.1f}%" if _mean else "N/A"
        _flag = "⚠️" if _mean and abs(_mean - _cur_tgt) / _cur_tgt > 0.15 else ""
        _rows.append({
            "代碼": _sym,
            "目前目標": f"${_cur_tgt:.1f}",
            "Yahoo 均值": f"${_mean:.1f}" if _mean else "N/A",
            "Yahoo 最高": f"${_high:.1f}" if _high else "N/A",
            "Yahoo 最低": f"${_low:.1f}"  if _low  else "N/A",
            "變動幅度": f"{_flag} {_delta_str}".strip(),
        })
    st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
    _col1, _col2, _col3 = st.columns([1, 1, 2])
    with _col1:
        if st.button("✅ 套用 Yahoo 均值", key="btn_apply_mean", use_container_width=True):
            _new_targets = {}
            for _sym in all_tickers_list:
                _mean_val = _yahoo_data.get(_sym, {}).get("mean")
                _new_targets[_sym] = float(_mean_val) if _mean_val else STOCKS_CONFIG[_sym]["target"]
            _today_dt2 = datetime.now().date()
            _new_meta = {
                "last_updated": _today_dt2.strftime("%Y-%m-%d"),
                "next_update_allowed": get_next_update_allowed(_today_dt2).strftime("%Y-%m-%d"),
                "targets": _new_targets,
            }
            save_target_meta(_new_meta)
            st.session_state.show_refresh_panel = False
            get_stock_data.clear()
            st.success("✅ 目標價已更新！下次更新：" + _new_meta["next_update_allowed"])
            st.rerun()
    with _col2:
        if st.button("❌ 取消", key="btn_cancel_refresh", use_container_width=True):
            st.session_state.show_refresh_panel = False
            st.rerun()
    st.stop()

# Re-fetch specific ticker to ensure we have full history for plotting
with st.spinner(f"Loading chart for {selected_ticker}..."):
    df_real = get_stock_data(selected_ticker)

# --- Signal Logic & Metrics ---
current_price = 0.0
signal_status = "Waiting for data..."
signal_type = "neutral" # neutral, buy, reduce_1, reduce_2

# Use ADJUSTED values for Logic/Metrics
curr_baseline_val = 0
lower_bound_val = 0
upper_bound_1_val = 0
upper_bound_2_val = 0
delta_pct = 0.0

# --- IMPORTANT: initialize ext vars at outer scope so chart code can always access them ---
ext_info = None
is_ext_active = False
last_row = None
last_date = None

if not df_real.empty:
    # Flatten MultiIndex if necessary for consistent extraction
    if isinstance(df_real.columns, pd.MultiIndex):
        try:
             y_data_series = df_real['Close'].iloc[:, 0]
        except:
             y_data_series = df_real.iloc[:, 0]
    else:
        y_data_series = df_real['Close']

    df_plot = df_real.copy()
    
    # Ensure columns are single-level strings (flatten MultiIndex if necessary)
    if isinstance(df_plot.columns, pd.MultiIndex):
        df_plot.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df_plot.columns]
    
    df_plot['Close_Flat'] = y_data_series
    df_plot = df_plot.reset_index()

    # Find the last row with a valid price to avoid TypeError during float conversion
    subset_col = 'Close_Flat' if 'Close_Flat' in df_plot.columns else df_plot.columns[-1]
    df_valid = df_plot.dropna(subset=[subset_col])
    if not df_valid.empty:
        last_row = df_valid.iloc[-1]
        last_date = pd.to_datetime(last_row['Date'])
        if isinstance(last_date, pd.Series):
            last_date = last_date.iloc[0]
        
        current_price = float(last_row['Close_Flat'])
        daily_close_price = current_price  # 保留每日收盤做顯示用
        
        # --- 集成最新擴展時段價格 ---
        ext_info = get_latest_price(selected_ticker)
        display_price = current_price
        
        if ext_info:
            ext_time = ext_info['time']
            if ext_time.date() >= last_date.date():
                display_price = ext_info['price']
                is_ext_active = True
                # 信號邏輯改用盤前價格
                current_price = display_price
    else:
        # Fallback if no valid price data exists
        last_row = df_plot.iloc[-1]
        last_date = pd.to_datetime(last_row['Date'])
        if isinstance(last_date, pd.Series):
            last_date = last_date.iloc[0]
        
        ext_info = get_latest_price(selected_ticker)
        if ext_info:
            current_price = ext_info['price']
            display_price = current_price
            is_ext_active = True
        else:
            current_price = 0.0
            display_price = 0.0

    # 處理可能的時區問題
    calc_date = last_date
    if hasattr(calc_date, 'tzinfo') and calc_date.tzinfo is not None:
        calc_date = calc_date.replace(tzinfo=None)
    
    day_diff = (calc_date - START_DATE).days
    
    if 0 <= day_diff < TOTAL_DAYS:
        curr_baseline_val = baseline_prices_adj[day_diff]
        upper_bound_1_val = curr_baseline_val * 1.25
        upper_bound_2_val = curr_baseline_val * 1.375
        lower_bound_val = curr_baseline_val * 0.90
        
        delta = current_price - curr_baseline_val
        delta_pct = (delta / curr_baseline_val) * 100 if curr_baseline_val != 0 else 0.0
        
        if current_price <= lower_bound_val:
            signal_status = "觸發『買入點 a』 (Buy!)"
            signal_type = "buy"
        elif current_price >= upper_bound_2_val:
            signal_status = "觸發『第二階段全賣』 (Exit B - Sell Remaining 50%)"
            signal_type = "reduce_2"
        elif current_price >= upper_bound_1_val:
            signal_status = "觸發『第一階段減碼』 (sell 50%)"
            signal_type = "reduce_1"
        else:
            signal_status = "觀望 / 持有 (Hold)"
            signal_type = "neutral"
            
        trend_arrow = calculate_trend(df_plot['Close_Flat'])
        if trend_arrow != "ERROR":
            signal_status += f" {trend_arrow}"
            
        import textwrap
        # Color mapping for Signal Card
        card_styles = {
            "buy":      "background-color: rgba(0, 255, 0, 0.2); border: 1px solid #00ff00;",
            "reduce_1": "background-color: rgba(255, 204, 0, 0.2); border: 1px solid #ffcc00;",
            "reduce_2": "background-color: rgba(255, 68, 68, 0.2); border: 1px solid #ff4444;",
            "neutral":  "background-color: #1e1e1e; border: 1px solid rgba(255, 255, 255, 0.1);"
        }
        current_card_style = card_styles.get(signal_type, card_styles["neutral"])
        
        color_map = {
            "buy": "#00ff00",
            "reduce_1": "#ffcc00",
            "reduce_2": "#ff4444",
            "neutral": "#e0e0e0"
        }
        main_color = color_map.get(signal_type, "#e0e0e0")

        # --- 小卡盤前顯示字串 ---
        # 分別計算 daily close 偏差 與 pre-market 偏差
        daily_close_price = daily_close_price if is_ext_active else current_price
        daily_delta_pct = (daily_close_price - curr_baseline_val) / curr_baseline_val * 100 if curr_baseline_val != 0 else 0.0
        ext_delta_pct = (current_price - curr_baseline_val) / curr_baseline_val * 100 if (is_ext_active and curr_baseline_val != 0) else None

        # Card 1: pre-market price (same font, green)
        pre_price_str = ""
        pre_dev_str = ""
        if is_ext_active and ext_info:
            ext_label = ext_info['label']
            pre_price_str = f'<span class="pre-market-text"> ({ext_label}: ${current_price:.2f})</span>'
            if ext_delta_pct is not None:
                pre_dev_str = f'<span class="pre-market-text"> ({ext_label}: {ext_delta_pct:+.2f}%)</span>'

        # Container for Metrics
        chart_space = st.empty()
        
        with chart_space.container():
            # Decide on the main label
            main_label = "Current Price"
            if is_ext_active and ext_info:
                main_label = f"{ext_info['label']} Price"

            st.markdown(textwrap.dedent(f"""
    <div class="metric-container">
    <div class="metric-card">
    <div class="metric-label">{main_label}</div>
    <div class="metric-value">${daily_close_price:.2f}{pre_price_str}</div>
    <div class="metric-sub">{last_date.strftime('%Y-%m-%d')}</div>
    </div>
    <div class="metric-card">
    <div class="metric-label">Adj Target</div>
    <div class="metric-value">${curr_baseline_val:.2f}</div>
    <div class="metric-sub">Base: ${baseline_prices_base[day_diff]:.2f} (x{sentiment_factor})</div>
    </div>
    <div class="metric-card">
    <div class="metric-label">Deviation</div>
    <div class="metric-value">{daily_delta_pct:+.2f}%{pre_dev_str}</div>
    <div class="metric-sub">from Adj Base</div>
    </div>
    <div class="metric-card" style="{current_card_style}">
    <div class="metric-label" style="color: #ffffff !important;">Signal</div>
    <div class="signal-value" style="display: flex; justify-content: center; align-items: center; min-height: 43px;">
        <span style="display: inline-block; width: 36px; height: 36px; border-radius: 50%; background-color: {main_color};"></span>
    </div>
    <div class="metric-sub" style="color: #ffffff !important;">{signal_status}</div>
    </div>
    </div>
    """), unsafe_allow_html=True)
            
    else:
        st.warning("Date out of 2026 range.")
else:
    st.warning("No data found for 2026. Market may be closed or future date not reached.")
    df_plot = pd.DataFrame()

# --- Debug Output (shown in sidebar when enabled) ---
if debug_mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔍 Debug Info**")
    st.sidebar.write(f"is_ext_active: `{is_ext_active}`")
    if ext_info:
        st.sidebar.write(f"ext label: `{ext_info['label']}`")
        st.sidebar.write(f"ext price: `{ext_info['price']:.2f}`")
        st.sidebar.write(f"ext time: `{ext_info['time']}`")
    else:
        st.sidebar.write("ext_info: `None`")
    st.sidebar.write(f"last_date: `{last_date}`")
    st.sidebar.write(f"current_price: `{current_price:.2f}`")
    st.sidebar.markdown("**📐 起點校正**")
    st.sidebar.write(f"手動起點 (p_start_manual): `{p_start_manual:.2f}`")
    _q4_dbg = prev_q4_avg_data.get(selected_ticker)
    st.sidebar.write(f"Q4均價錨點: `{_q4_dbg:.2f}`" if _q4_dbg else "Q4均價錨點: `N/A`")
    st.sidebar.write(f"校正後起點 (p_start): `{p_start:.2f}`")
    st.sidebar.write(f"混合比例 alpha: `{start_blend_alpha:.1f}`")

# --- Visualization ---
fig = go.Figure()

def add_trace_pair(fig, x, y_base, y_adj, name, color, base_dash=None):
    # Base
    fig.add_trace(go.Scatter(
        x=x, y=y_base, mode='lines', name=f"{name} (Base)", 
        line=dict(color=color, width=1 if base_dash else 2, dash=base_dash),
        showlegend=True
    ))
    # Adj Shadow
    if sentiment_factor != 1.0:
        fig.add_trace(go.Scatter(
            x=x, y=y_adj, mode='lines', name=f"{name} (Adj)", 
            line=dict(width=0),
            fill='tonexty',
            fillcolor=color.replace(')', ', 0.1)').replace('rgb', 'rgba'),
            showlegend=False, hoverinfo='skip'
        ))

# Add Pairs - Reordered for Logical Legend (Top to Bottom)
# +37.5% (Red) - Highest
add_trace_pair(fig, df_base["Date"], df_base["Upper_37_5"], df_adj["Upper_37_5"], "+37.5% (Exit)", "rgb(255, 0, 0)", "dash")

# +25% (Orange)
add_trace_pair(fig, df_base["Date"], df_base["Upper_25"], df_adj["Upper_25"], "+25% (Reduce)", "rgb(255, 165, 0)", "dash")

# --- Ghost Baseline (原始未校正基準線 fadeout，僅校正生效時顯示) ---
# 將全年切為 N 段，opacity 從左端 0.55 線性遞減至右端 0.03，形成淡出效果
if start_blend_alpha < 1.0 and _q4_anchor is not None and abs(p_start - p_start_manual) > 0.01:
    _ghost_slope  = (p_target_base - p_start_manual) / (TOTAL_DAYS - 1)
    _ghost_prices = [p_start_manual + _ghost_slope * i for i in range(TOTAL_DAYS)]
    _N   = 15
    _seg = TOTAL_DAYS // _N
    for _i in range(_N):
        _alpha = 0.55 - (0.55 - 0.03) * (_i / (_N - 1))   # 0.55 → 0.03
        _s = _i * _seg
        _e = (_s + _seg + 1) if _i < _N - 1 else TOTAL_DAYS
        fig.add_trace(go.Scatter(
            x=dates_2026[_s:_e],
            y=_ghost_prices[_s:_e],
            mode='lines',
            name='Baseline (校正前)' if _i == 0 else None,
            showlegend=(_i == 0),
            legendgroup='ghost_baseline',
            line=dict(color=f'rgba(210, 175, 100, {_alpha:.3f})', width=1.5, dash='dot'),
            hoverinfo='skip'
        ))

# Baseline (Gray)
add_trace_pair(fig, df_base["Date"], df_base["Baseline"], df_adj["Baseline"], "Baseline", "rgb(128, 128, 128)")


# -10% (Green) - Lowest
add_trace_pair(fig, df_base["Date"], df_base["Lower_10"], df_adj["Lower_10"], "-10% (Buy)", "rgb(0, 128, 0)", "dash")

if not df_plot.empty:
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["Close_Flat"], mode='lines', name='Actual Price', 
        line=dict(color='red', width=4)
    ))

# --- Continuous Extended Hours Line (Green Line) ---
if is_ext_active and ext_info:
    # Fetch the raw intraday DataFrame (not cached)
    df_h_raw = get_ext_df(selected_ticker)
    if not df_h_raw.empty:
        try:
            df_h = df_h_raw.copy()
            # Ensure index is timezone-aware and convert to America/New_York
            if df_h.index.tzinfo is None:
                df_h.index = df_h.index.tz_localize('UTC')
            df_h.index = df_h.index.tz_convert('America/New_York')
            
            # Extract Close column safely
            if isinstance(df_h.columns, pd.MultiIndex):
                close_col = df_h['Close'].iloc[:, 0].values
            else:
                close_col = df_h['Close'].values
            
            df_ext = pd.DataFrame({'_Close': close_col}, index=df_h.index)
            
            if not df_plot.empty and last_date is not None:
                last_daily_date_naive = pd.to_datetime(last_date).replace(tzinfo=None)
                last_daily_price = float(last_row['Close_Flat']) if last_row is not None else current_price
                
                # Strip tz safely for comparison
                idx_naive = df_ext.index.tz_convert('UTC').tz_localize(None)
                df_ext_plot = df_ext[idx_naive >= last_daily_date_naive].copy()
                
                if not df_ext_plot.empty:
                    # Prepend last daily close as connecting anchor
                    try:
                        conn_idx = pd.Timestamp(last_daily_date_naive).tz_localize('America/New_York')
                        df_conn = pd.DataFrame({'_Close': [last_daily_price]}, index=[conn_idx])
                        df_ext_plot = pd.concat([df_conn, df_ext_plot])
                    except Exception:
                        pass
                    
                    fig.add_trace(go.Scatter(
                        x=df_ext_plot.index,
                        y=df_ext_plot['_Close'],
                        mode='lines',
                        name=f'{ext_info["label"]}-market',
                        line=dict(color='#80CF59', width=4),
                        hovertemplate="%{y:.2f}<br>%{x}<extra>Pre/Post</extra>"
                    ))
        except Exception as _e:
            pass


# --- Calculate Fixed Axis Range ---
# Determine min/max across all relevant series to fix the view
y_max_candidates = [df_base["Upper_37_5"].max()]
y_min_candidates = [df_base["Lower_10"].min()]

if sentiment_factor != 1.0:
    y_max_candidates.append(df_adj["Upper_37_5"].max())
    y_min_candidates.append(df_adj["Lower_10"].min())

if not df_plot.empty:
    y_max_candidates.append(df_plot["Close_Flat"].max())
    y_min_candidates.append(df_plot["Close_Flat"].min())

y_max_val = max(y_max_candidates)
y_min_val = min(y_min_candidates)
y_margin = (y_max_val - y_min_val) * 0.05

# 準備合併圖表設定
merged_legend = dict(ui_config.PLOTLY_LAYOUT.get("legend", {}))
merged_legend.update({
    "orientation": "h",
    "yanchor": "bottom",
    "y": 1.02,
    "xanchor": "right",
    "x": 1
})

layout_kwargs = dict(ui_config.PLOTLY_LAYOUT)
layout_kwargs.update({
    "height": 600,
    "hovermode": "x unified",
    "legend": merged_legend,
    "yaxis": dict(
        range=[y_min_val - y_margin, y_max_val + y_margin],
        fixedrange=True, # Disable zoom on Y
        **ui_config.PLOTLY_AXES
    ),
    "xaxis": dict(
        fixedrange=True, # Disable zoom on X
        **ui_config.PLOTLY_AXES
    )
})

fig.update_layout(**layout_kwargs)

# RENDER CHART IN CONTAINER
# If chart_space was defined earlier (inside the if), use it. But fig creation is outside.
# Let's clean up structure. 
# We'll use a main_block container for everything below header.
with st.container(border=False):
    st.plotly_chart(
        fig, 
        theme=None,
        use_container_width=True,
        config={
            'displayModeBar': True,
            'modeBarButtons': [['toImage']], 
            'displaylogo': False,
            'scrollZoom': False
        }
    )

# st.markdown("---") # Removed per user request
st.markdown(f"**Update Status:** Fetched at {datetime.now().strftime('%H:%M:%S')}")

# --- Auto Refresh ---
if auto_refresh:
    time.sleep(60)
    st.rerun()
