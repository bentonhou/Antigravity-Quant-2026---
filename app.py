import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- Configuration ---
STOCKS_CONFIG = {
    "NVDA": {"start": 187.20, "target": 265.0},
    "AMD":  {"start": 214.30, "target": 250.0},
    "AMZN": {"start": 237.21, "target": 280.0},
    "QCOM": {"start": 173.00, "target": 210.0},
    "AVGO": {"start": 347.62, "target": 435.0},
    "MRVL": {"start": 89.39,  "target": 125.0},
}

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days + 1

st.set_page_config(page_title="Antigravity Quant 2026", layout="wide")
st.title("Antigravity Quant 2026 - 波段導航儀")


# --- Helper: Data Fetching with Cache ---
import numpy as np

# --- Helper: Data Fetching with Cache ---
@st.cache_data(ttl=60) # Cache for 60 seconds
def get_stock_data(ticker_or_tickers):
    try:
        # Fetch data from 2026-01-01 to NOW
        df = yf.download(ticker_or_tickers, start="2026-01-01", interval="1d", progress=False)
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_status(ticker, price, date_obj):
    # Re-implement baseline logic here for single point check
    config = STOCKS_CONFIG[ticker]
    p_start = config["start"]
    p_target = config["target"]
    slope = (p_target - p_start) / (TOTAL_DAYS - 1)
    
    # Baseline for date
    day_diff = (date_obj - START_DATE).days
    
    status_icon = "⚪" # Default
    
    if 0 <= day_diff < TOTAL_DAYS:
        curr_baseline = p_start + slope * day_diff
        upper_bound_1 = curr_baseline * 1.25
        upper_bound_2 = curr_baseline * 1.375
        lower_bound = curr_baseline * 0.90
        
        if price <= lower_bound:
            status_icon = "🟢"
        elif price >= upper_bound_2:
            status_icon = "🔴"
        elif price >= upper_bound_1:
            status_icon = "🟡"
            
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
sidebar_options = {}

# Batch fetch latest checking
all_tickers_list = list(STOCKS_CONFIG.keys())
with st.spinner("Updating Market Signals..."):
    df_all = get_stock_data(" ".join(all_tickers_list))

for ticker in all_tickers_list:
    icon = "⚪"
    trend = ""
    try:
        # Handle MultiIndex or Single ticker return structure
        if isinstance(df_all.columns, pd.MultiIndex):
            if ticker in df_all['Close'].columns:
                series = df_all['Close'][ticker].dropna()
                if not series.empty:
                    last_val = float(series.iloc[-1])
                    last_t = pd.to_datetime(series.index[-1])
                    if isinstance(last_t, pd.Series): last_t = last_t.iloc[0]
                    icon = calculate_status(ticker, last_val, last_t)
                    trend = calculate_trend(series)
        else:
            if len(all_tickers_list) == 1:
               series = df_all['Close'].dropna() if 'Close' in df_all else df_all.iloc[:,0].dropna()
               if not series.empty:
                   last_val = float(series.iloc[-1])
                   last_t = pd.to_datetime(series.index[-1])
                   icon = calculate_status(ticker, last_val, last_t)
                   trend = calculate_trend(series)
    except Exception as e:
        pass 
    
    label = f"{ticker} {icon}"
    if trend and trend != "ERROR":
        label += f" {trend}"
        
    sidebar_options[label] = ticker

# --- Sidebar ---
st.sidebar.header("Asset Selection")
# Create reverse mapping or just use keys
display_keys = list(sidebar_options.keys())
selected_display = st.sidebar.radio("Ticker", display_keys)
selected_ticker = sidebar_options[selected_display]

auto_refresh = st.sidebar.checkbox("Auto-Refresh (60s)", value=True)

# --- Logic: Baseline ---
config = STOCKS_CONFIG[selected_ticker]
p_start = config["start"]
p_target = config["target"]
slope = (p_target - p_start) / (TOTAL_DAYS - 1)

# Generate Baseline Series
dates_2026 = [START_DATE + timedelta(days=i) for i in range(TOTAL_DAYS)]
baseline_prices = [p_start + slope * i for i in range(TOTAL_DAYS)]
df_baseline = pd.DataFrame({"Date": dates_2026, "Baseline": baseline_prices})
df_baseline["Upper_25"] = df_baseline["Baseline"] * 1.25
df_baseline["Upper_37_5"] = df_baseline["Baseline"] * 1.375  # Exit B: 1.10 * 1.25 = 1.375 (+37.5%)
df_baseline["Lower_10"] = df_baseline["Baseline"] * 0.90

# --- Logic: Baseline ---
config = STOCKS_CONFIG[selected_ticker]
p_start = config["start"]
p_target = config["target"]
slope = (p_target - p_start) / (TOTAL_DAYS - 1)

# Generate Baseline Series
dates_2026 = [START_DATE + timedelta(days=i) for i in range(TOTAL_DAYS)]
baseline_prices = [p_start + slope * i for i in range(TOTAL_DAYS)]
df_baseline = pd.DataFrame({"Date": dates_2026, "Baseline": baseline_prices})
df_baseline["Upper_25"] = df_baseline["Baseline"] * 1.25
df_baseline["Upper_37_5"] = df_baseline["Baseline"] * 1.375  # Exit B: 1.10 * 1.25 = 1.375 (+37.5%)
df_baseline["Lower_10"] = df_baseline["Baseline"] * 0.90


# --- Main Logic ---
# Re-fetch specific ticker to ensure we have full history for plotting (though we could slice from df_all)
# reusing get_stock_data will hit cache if we call it same way. 
# BUT above we called with " ".join(all), here we usually call single.
# yfinance might cache locally. Let's just use get_stock_data(selected_ticker) for simplicity of plotting code.
with st.spinner(f"Loading chart for {selected_ticker}..."):
    df_real = get_stock_data(selected_ticker)

# --- Signal Logic & Metrics ---
current_price = 0.0
signal_status = "Waiting for data..."
signal_type = "neutral" # neutral, buy, reduce_1, reduce_2

if not df_real.empty:
    # Flatten MultiIndex if present (yfinance update)
    if isinstance(df_real.columns, pd.MultiIndex):
        try:
             y_data_series = df_real['Close'].iloc[:, 0]
        except:
             y_data_series = df_real.iloc[:, 0] # Fallback
    else:
        y_data_series = df_real['Close']

    df_plot = df_real.copy()
    df_plot['Close_Flat'] = y_data_series
    df_plot = df_plot.reset_index() # Ensure Date is a column

    # Get latest
    last_row = df_plot.iloc[-1]
    
    # Ensure pandas timestamp conversion
    last_date = pd.to_datetime(last_row['Date'])
    
    # Force scalar extraction logic
    if isinstance(last_date, pd.Series):
        last_date = last_date.iloc[0]

    current_price = float(last_row['Close_Flat'])
    
    # Find matching baseline price for this specific date
    if hasattr(last_date, 'date'):
        day_diff = (last_date - START_DATE).days
    else:
        # Fallback if somehow still not datetime (e.g. string)
        day_diff = (pd.to_datetime(last_date) - START_DATE).days
    
    if 0 <= day_diff < TOTAL_DAYS:
        curr_baseline = baseline_prices[day_diff]
        upper_bound_1 = curr_baseline * 1.25   # +25%
        upper_bound_2 = curr_baseline * 1.375  # +37.5%
        lower_bound = curr_baseline * 0.90     # -10%
        
        delta = current_price - curr_baseline
        delta_pct = (delta / curr_baseline) * 100
        
        # Signal Judgment
        if current_price <= lower_bound:
            signal_status = "🟢 觸發『買入點 a』 (Buy!)"
            signal_type = "buy"
        elif current_price >= upper_bound_2:
            signal_status = "🔴 觸發『第二階段全賣』 (Exit B - Sell Remaining 50%)"
            signal_type = "reduce_2"
        elif current_price >= upper_bound_1:
            signal_status = "🟡 觸發『第一階段減碼』 (Sell 50%)"
            signal_type = "reduce_1"
        else:
            signal_status = "⚪ 觀望 / 持有 (Hold)"
            signal_type = "neutral"
            
        # Calculate Trend for Main View
        trend_arrow = calculate_trend(df_plot['Close_Flat'])
        if trend_arrow != "ERROR":
            signal_status += f" {trend_arrow}"
            
        # Metrics Row (Responsive HTML)
        import textwrap
        
        # Color mapping
        color_map = {
            "buy": "#00ff00",       # Green
            "reduce_1": "#ffcc00",  # Yellow
            "reduce_2": "#ff4444",  # Red
            "neutral": "#e0e0e0"    # White/Grey
        }
        main_color = color_map.get(signal_type, "#e0e0e0")

        st.markdown(textwrap.dedent(f"""
<style>
.metric-container {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 20px;
    justify-content: space-between;
}}
            .metric-card {{
                flex: 1 1 140px;
                background-color: #1e1e1e; /* Solid dark background for contrast */
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 10px;
                text-align: center;
                transition: all 0.3s ease;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); /* Add shadow for depth on light bg */
            }}
            .metric-card:hover {{
                border-color: rgba(255, 255, 255, 0.3);
                transform: translateY(-2px);
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.4);
            }}
.metric-label {{
    color: #aaaaaa;
    font-size: clamp(0.7rem, 2vw, 0.9rem);
    margin-bottom: 4px;
}}
.metric-value {{
    color: #ffffff;
    font-size: clamp(1.1rem, 4vw, 1.8rem);
    font-weight: 600;
    line-height: 1.2;
}}
.metric-sub {{
    color: #666666;
    font-size: clamp(0.6rem, 1.5vw, 0.75rem);
    margin-top: 4px;
}}
.signal-value {{
    font-size: clamp(0.9rem, 3vw, 1.4rem);
    font-weight: bold;
    color: {main_color};
}}
</style>
<div class="metric-container">
<div class="metric-card">
<div class="metric-label">Current Price</div>
<div class="metric-value">${current_price:.2f}</div>
<div class="metric-sub">{last_date.strftime('%Y-%m-%d')}</div>
</div>
<div class="metric-card">
<div class="metric-label">Baseline Target</div>
<div class="metric-value" style="color: #cccccc;">${curr_baseline:.2f}</div>
<div class="metric-sub">Expected Value</div>
</div>
<div class="metric-card">
<div class="metric-label">Deviation</div>
<div class="metric-value" style="color: {main_color};">{delta_pct:+.2f}%</div>
<div class="metric-sub">from Baseline</div>
</div>
<div class="metric-card" style="border-top: 3px solid {main_color};">
<div class="metric-label">Signal</div>
<div class="signal-value">{signal_status.split(' ')[0]}</div>
<div class="metric-sub" style="color: {main_color};">{signal_status.split(' ', 1)[1] if ' ' in signal_status else ''}</div>
</div>
</div>
"""), unsafe_allow_html=True)
        
        # Banner
        if signal_type == "buy":
            st.success(f"### ACTION REQUIRED: {signal_status}")
            st.markdown(f"Price is below -10% (${lower_bound:.2f}).")
        elif signal_type == "reduce_2":
            st.error(f"### ACTION REQUIRED: {signal_status}")
            st.markdown(f"**CRITICAL**: Price has reached +37.5% (${upper_bound_2:.2f}). Sell remaining position!")
        elif signal_type == "reduce_1":
            st.warning(f"### ACTION REQUIRED: {signal_status}")
            st.markdown(f"Price has reached +25% (${upper_bound_1:.2f}). Lock in profits on half.")
    else:
        st.warning("Date out of 2026 range.")
else:
    st.warning("No data found for 2026. Market may be closed or future date not reached.")
    df_plot = pd.DataFrame()


# --- Visualization ---
fig = go.Figure()

# 1. Baseline
fig.add_trace(go.Scatter(x=df_baseline["Date"], y=df_baseline["Baseline"], mode='lines', name='Baseline', line=dict(color='gray', width=2)))
# 2. Bands
fig.add_trace(go.Scatter(x=df_baseline["Date"], y=df_baseline["Upper_37_5"], mode='lines', name='+37.5% (Full Exit)', line=dict(color='red', width=1, dash='dash')))
fig.add_trace(go.Scatter(x=df_baseline["Date"], y=df_baseline["Upper_25"], mode='lines', name='+25% (Reduce)', line=dict(color='orange', width=1, dash='dash')))
fig.add_trace(go.Scatter(x=df_baseline["Date"], y=df_baseline["Lower_10"], mode='lines', name='-10% (Buy)', line=dict(color='green', width=1, dash='dash')))

# 3. Real Data (RED Line)
if not df_plot.empty:
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], 
        y=df_plot["Close_Flat"], 
        mode='lines', 
        name='Actual Price', 
        line=dict(color='red', width=4)
    ))

# 4. Pre-market Data (Green Line)
# Fetch intraday data for pre-market
@st.cache_data(ttl=60)
def get_intraday_data(ticker):
    try:
        # Fetch 1 day of 5m data including pre/post market
        df = yf.download(ticker, period="1d", interval="5m", prepost=True, progress=False)
        return df
    except:
        return pd.DataFrame()

with st.spinner(f"Checking pre-market data for {selected_ticker}..."):
    df_intra = get_intraday_data(selected_ticker)

if not df_intra.empty:
    # Handle MultiIndex if present
    if isinstance(df_intra.columns, pd.MultiIndex):
        try:
             close_series = df_intra['Close'].iloc[:, 0]
        except:
             close_series = df_intra.iloc[:, 0]
    else:
        close_series = df_intra['Close']
    
    # Filter for Pre-market (Before 09:30 Local Market Time)
    # yfinance usually returns timezone-aware intervals (e.g. America/New_York)
    # We'll assume the last day in the data is the "current" day we care about
    if not close_series.empty:
        last_day = close_series.index[-1].date()
        day_data = close_series[close_series.index.date == last_day]
        
        # 09:30 AM define
        # We need to be careful with timezones. simplest is to check string time or hour/minute
        # Market open is 9:30. Pre-market is < 9:30.
        # Create a boolean mask
        mask = (day_data.index.hour < 9) | ((day_data.index.hour == 9) & (day_data.index.minute < 30))
        pre_market_data = day_data[mask]
        
        if not pre_market_data.empty:
            # Check if market has officially opened (Data exists >= 09:30 NY Time)
            last_timestamp = day_data.index[-1]
            
            # Ensure timezone is NY
            if last_timestamp.tzinfo is None:
                 last_timestamp = last_timestamp.tz_localize('UTC')
            last_timestamp_ny = last_timestamp.tz_convert('America/New_York')
            
            is_market_open = (last_timestamp_ny.hour > 9) or (last_timestamp_ny.hour == 9 and last_timestamp_ny.minute >= 30)

            # Show Green Line ONLY if market is NOT open
            if not is_market_open:
                # Connect the last point of Red Line to first point of Green Line
                if not df_plot.empty:
                    last_red_date = df_plot["Date"].iloc[-1]
                    last_red_price = df_plot["Close_Flat"].iloc[-1]
                    
                    # Create a connector point
                    connector = pd.Series([last_red_price], index=[last_red_date])
                    
                    # Prepend to pre-market data
                    pre_market_data = pd.concat([connector, pre_market_data])

                fig.add_trace(go.Scatter(
                    x=pre_market_data.index, 
                    y=pre_market_data.values, 
                    mode='lines', 
                    name='Pre-market', 
                    line=dict(color='#00ff00', width=4) # Matched width
                ))

fig.update_layout(title=f"{selected_ticker} Wave Navigator", height=600, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(f"**Update Status:** Fetched at {datetime.now().strftime('%H:%M:%S')}")

# --- Auto Refresh ---
if auto_refresh:
    time.sleep(60)
    st.rerun()
