import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np
import sys
import os

# --- 設定字體與 UI 規範 ---
ui_path = os.path.abspath(os.path.join("Standards", "fonts and UI"))
if ui_path not in sys.path:
    sys.path.append(ui_path)
import ui_config# --- Configuration ---
STOCKS_CONFIG = {
    "TSM":  {"start": 319.61, "target": 435.0},
    "NVDA": {"start": 187.20, "target": 270.0},
    "AMD":  {"start": 214.30, "target": 290.0},
    "MSFT": {"start": 472.94, "target": 590.0},
    "GOOG": {"start": 315.32, "target": 360.0},
    "QCOM": {"start": 173.00, "target": 155.0},
    "AMZN": {"start": 237.21, "target": 285.0},
    "AVGO": {"start": 347.62, "target": 475.0},
    "MRVL": {"start": 89.39,  "target": 123.0},
    "NOK":  {"start": 6.51,   "target": 8.50},
}

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days + 1

st.set_page_config(page_title="Antigravity Quant 2026", layout="wide")

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

def calculate_status(ticker, price, date_obj, sentiment=1.0):
    # Re-implement baseline logic here for single point check
    config = STOCKS_CONFIG[ticker]
    p_start = config["start"]
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

with st.spinner("Updating Market Signals..."):
    # 1. 抓取日線歷史數據
    df_all = get_stock_data(" ".join(all_tickers_list))
    # 2. 抓取所有 Ticker 的最新分時數據 (含盤前盤後) - 使用 5d 確保能抓到最近的資料
    try:
        df_ext_all = yf.download(" ".join(all_tickers_list), period='5d', interval='15m', prepost=True, progress=False)
    except:
        df_ext_all = pd.DataFrame()

for ticker in all_tickers_list:
    icon = ":gray[●]"
    trend = ""
    try:
        # 決定最新價格
        final_price = None
        final_date = None
        
        # Helper 提取最新值
        def get_series_last(df, attr, tk):
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    if attr in df.columns.levels[0] and tk in df[attr].columns:
                        s = df[attr][tk].dropna()
                        if not s.empty:
                            return float(s.iloc[-1]), pd.to_datetime(s.index[-1])
                elif attr in df.columns: # Single ticker structure
                    s = df[attr].dropna()
                    if not s.empty:
                        return float(s.iloc[-1]), pd.to_datetime(s.index[-1])
            except: pass
            return None, None

        # 先從日線提取
        p_day, d_day = get_series_last(df_all, 'Close', ticker)
        if p_day is not None:
            final_price = p_day
            final_date = d_day
            # 計算趨勢 (僅用日線)
            try:
                if isinstance(df_all.columns, pd.MultiIndex):
                    trend = calculate_trend(df_all['Close'][ticker].dropna())
                else:
                    trend = calculate_trend(df_all['Close'].dropna())
            except: pass

        # 再從擴展時段數據提取 (若有更晚的數據則覆寫)
        p_ext, d_ext = get_series_last(df_ext_all, 'Close', ticker)
        if p_ext is not None:
            # 轉換為美東時間進行比較
            if d_ext.tzinfo is not None:
                d_ext_cmp = d_ext.tz_convert('US/Eastern').replace(tzinfo=None)
            else:
                d_ext_cmp = d_ext
            
            # 如果擴展數據的日期 >= 日線數據日期，則更新
            if final_date is None:
                final_price = p_ext
                final_date = d_ext_cmp
            else:
                d_day_cmp = final_date.replace(tzinfo=None) if hasattr(final_date, 'replace') else final_date
                if d_ext_cmp.date() >= d_day_cmp.date():
                    final_price = p_ext
                    final_date = d_ext_cmp

        if final_price is not None:
             icon = calculate_status(ticker, final_price, final_date, sentiment_factor)
             
    except Exception as e:
        pass 
    
    label = f"*{ticker}* {icon}"
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

# --- Logic: Baseline ---
config = STOCKS_CONFIG[selected_ticker]
p_start = config["start"]

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
    獲取最新價格（包含盤前與盤後）。
    """
    try:
        t = yf.Ticker(ticker)
        # 抓取最近 2 天的高頻分時數據，包含盤前與盤後 (prepost=True)
        df_h = t.history(period='2d', interval='1m', prepost=True)
        
        if df_h.empty:
            # 備援：如果 history 為空，嘗試從 info 獲取 (雖然較不穩定)
            info = t.info
            price = info.get('preMarketPrice') or info.get('postMarketPrice') or info.get('regularMarketPrice') or info.get('currentPrice')
            if price:
                return {"price": float(price), "label": "Live", "time": pd.Timestamp.now(tz='US/Eastern')}
            return None

        # 取得最後一個有效的價格點
        last_ts = df_h.index[-1].tz_convert('US/Eastern')
        last_price = float(df_h['Close'].iloc[-1])
        
        # 根據美東時間小時判斷當前市場狀態標籤
        hour = last_ts.hour
        minute = last_ts.minute
        time_f = hour + minute/60.0
        
        label = ""
        if 4.0 <= time_f < 9.5:
            label = "Pre"
        elif 9.5 <= time_f < 16.0:
            label = "Live" # Regular session
        elif 16.0 <= time_f <= 20.0:
            label = "Post"
        else:
            label = "Ext" # Extended (凌晨或其他)
            
        return {"price": last_price, "label": label, "time": last_ts}
    except Exception:
        pass
    return None

# --- Main Logic ---
st.markdown(f"<h2>{selected_ticker} Wave Navigator</h2>", unsafe_allow_html=True)

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
        
        # --- 集成最新擴展時段價格 ---
        ext_info = get_latest_price(selected_ticker)
        is_ext_active = False
        display_price = current_price
        
        if ext_info:
            # 只有當擴展價格的時間晚於或等於日線最後日期時才考慮顯現
            ext_time = ext_info['time']
            if ext_time.date() >= last_date.date():
                display_price = ext_info['price']
                is_ext_active = True
                # 若擴展價格存在，決策邏輯改用擴展價格
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
            is_ext_active = False

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

        # Pre-market / Post-market strings for Display
        pm_price_str = ""
        pm_dev_str = ""
        
        if is_ext_active and ext_info:
            label = ext_info['label']
            price_val = ext_info['price']
            pm_price_str = f'<span class="pre-market-text">({label}: ${price_val:.2f})</span>'
            
            # Calculate Deviation for Pre-market compared to Adj Base
            if curr_baseline_val > 0:
                pm_delta = price_val - curr_baseline_val
                pm_pct = (pm_delta / curr_baseline_val) * 100
                pm_dev_str = f'<span class="pre-market-text">({label}: {pm_pct:+.2f}%)</span>'

        # Container for Metrics
        chart_space = st.empty()
        
        with chart_space.container():
            st.markdown(textwrap.dedent(f"""
    <div class="metric-container">
    <div class="metric-card">
    <div class="metric-label">Current Price</div>
    <div class="metric-value">${current_price:.2f}{pm_price_str}</div>
    <div class="metric-sub">{last_date.strftime('%Y-%m-%d')}</div>
    </div>
    <div class="metric-card">
    <div class="metric-label">Adj Target</div>
    <div class="metric-value">${curr_baseline_val:.2f}</div>
    <div class="metric-sub">Base: ${baseline_prices_base[day_diff]:.2f} (x{sentiment_factor})</div>
    </div>
    <div class="metric-card">
    <div class="metric-label">Deviation</div>
    <div class="metric-value">{delta_pct:+.2f}%{pm_dev_str}</div>
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

# Baseline (Gray)
add_trace_pair(fig, df_base["Date"], df_base["Baseline"], df_adj["Baseline"], "Baseline", "rgb(128, 128, 128)")

# -10% (Green) - Lowest
add_trace_pair(fig, df_base["Date"], df_base["Lower_10"], df_adj["Lower_10"], "-10% (Buy)", "rgb(0, 128, 0)", "dash")

if not df_plot.empty:
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], y=df_plot["Close_Flat"], mode='lines', name='Actual Price', 
        line=dict(color='red', width=4)
    ))

# Add Pre-market / Extended Point if active
if is_ext_active and ext_info:
    # 判斷繪圖日期點
    ext_ts = ext_info['time']
    # 放在該日期的 00:00:00 (與日線對齊)
    target_date_plot = pd.to_datetime(ext_ts.date())
    ext_price_plot = ext_info['price']
    label_plot = ext_info['label']

    # 1. Add Marker
    fig.add_trace(go.Scatter(
        x=[target_date_plot], 
        y=[ext_price_plot], 
        mode='markers+text', 
        name=f'{label_plot}-market',
        text=[f" {label_plot}"],
        textposition="top right",
        marker=dict(color='#00ff00', size=12, symbol='diamond', line=dict(color='white', width=1)),
        showlegend=False
    ))

    # 2. Add Connecting Line (from last valid close to extended point)
    if not df_plot.empty:
        df_valid_conn = df_plot.dropna(subset=['Close_Flat'])
        if not df_valid_conn.empty:
            last_valid_row = df_valid_conn.iloc[-1]
            last_date_v = pd.to_datetime(last_valid_row['Date'])
            last_price_v = float(last_valid_row['Close_Flat'])
            
            # 只有當時間點不同時才拉線，否則只是重疊
            if last_date_v != target_date_plot:
                fig.add_trace(go.Scatter(
                    x=[last_date_v, target_date_plot],
                    y=[last_price_v, ext_price_plot],
                    mode='lines',
                    name='Ext-market Link',
                    line=dict(color='#00ff00', width=2, dash='dot'),
                    showlegend=False
                ))

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
