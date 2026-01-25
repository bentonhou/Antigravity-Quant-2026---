import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np

# --- Configuration ---
STOCKS_CONFIG = {
    "TSM":  {"start": 319.61, "target": 400.0},
    "NVDA": {"start": 187.20, "target": 265.0},
    "AMD":  {"start": 214.30, "target": 250.0},
    "GOOG": {"start": 315.32, "target": 380.0},
    "QCOM": {"start": 173.00, "target": 210.0},
    "AMZN": {"start": 237.21, "target": 280.0},
    "AVGO": {"start": 347.62, "target": 435.0},
    "MRVL": {"start": 89.39,  "target": 125.0},
    "NOK":  {"start": 6.51,   "target": 8.00},
}

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days + 1

st.set_page_config(page_title="Antigravity Quant 2026", layout="wide")
st.markdown("<h1 style='font-size: 30px;'>Antigravity Quant 2026 - 波段導航儀</h1>", unsafe_allow_html=True)
st.markdown("""
<style>
    /* Hide the drag handle for the sidebar - Multiple selectors for robustness */
    div[data-testid="stSidebar"] > div:nth-child(2) {
        display: none !important;
    }
    .stSidebar > div:nth-child(2) {
        display: none !important;
    }
    div[class^="stSidebar"] > div[class^="resize-tr"] {
        display: none !important;
    }
    /* Hide Streamlit element toolbar (fullscreen button) */
    [data-testid="stElementToolbar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Global Settings (Placed early for data dependency) ---
st.sidebar.header("參數調整(目標價校正)")
sentiment_label = st.sidebar.select_slider(
    "Market Sentiment",
    options=["Optimistic (1.05)", "Base Target (1.00)", "Conservative (0.90)"],
    value="Base Target (1.00)"
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

all_tickers_list = list(STOCKS_CONFIG.keys())
sidebar_options = {}

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
                    # Pass sentiment_factor here
                    icon = calculate_status(ticker, last_val, last_t, sentiment_factor)
                    trend = calculate_trend(series)
        else:
            if len(all_tickers_list) == 1 and all_tickers_list[0] == ticker:
               series = df_all['Close'].dropna() if 'Close' in df_all else df_all.iloc[:,0].dropna()
               if not series.empty:
                   last_val = float(series.iloc[-1])
                   last_t = pd.to_datetime(series.index[-1])
                   icon = calculate_status(ticker, last_val, last_t, sentiment_factor)
                   trend = calculate_trend(series)
            elif ticker in df_all.columns: 
                 pass

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


# --- Main Logic ---
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
    # Flatten MultiIndex
    if isinstance(df_real.columns, pd.MultiIndex):
        try:
             y_data_series = df_real['Close'].iloc[:, 0]
        except:
             y_data_series = df_real.iloc[:, 0]
    else:
        y_data_series = df_real['Close']

    df_plot = df_real.copy()
    df_plot['Close_Flat'] = y_data_series
    df_plot = df_plot.reset_index()

    last_row = df_plot.iloc[-1]
    last_date = pd.to_datetime(last_row['Date'])
    if isinstance(last_date, pd.Series):
        last_date = last_date.iloc[0]

    current_price = float(last_row['Close_Flat'])
    
    if hasattr(last_date, 'date'):
        day_diff = (last_date - START_DATE).days
    else:
        day_diff = (pd.to_datetime(last_date) - START_DATE).days
    
    if 0 <= day_diff < TOTAL_DAYS:
        curr_baseline_val = baseline_prices_adj[day_diff]
        upper_bound_1_val = curr_baseline_val * 1.25
        upper_bound_2_val = curr_baseline_val * 1.375
        lower_bound_val = curr_baseline_val * 0.90
        
        delta = current_price - curr_baseline_val
        delta_pct = (delta / curr_baseline_val) * 100
        
        if current_price <= lower_bound_val:
            signal_status = "🟢 觸發『買入點 a』 (Buy!)"
            signal_type = "buy"
        elif current_price >= upper_bound_2_val:
            signal_status = "🔴 觸發『第二階段全賣』 (Exit B - Sell Remaining 50%)"
            signal_type = "reduce_2"
        elif current_price >= upper_bound_1_val:
            signal_status = "🟡 觸發『第一階段減碼』 (Sell 50%)"
            signal_type = "reduce_1"
        else:
            signal_status = "⚪ 觀望 / 持有 (Hold)"
            signal_type = "neutral"
            
        trend_arrow = calculate_trend(df_plot['Close_Flat'])
        if trend_arrow != "ERROR":
            signal_status += f" {trend_arrow}"
            
        import textwrap
        # Color mapping for Signal Card
        # Buy: Green, Reduce_1: Yellow, Reduce_2: Red, Neutral: Dark
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

        # Container for Metrics
        chart_space = st.empty()
        
        with chart_space.container():
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
                    background-color: #1e1e1e;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    padding: 10px;
                    text-align: center;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                }}
                .metric-card:hover {{
                    border-color: rgba(255, 255, 255, 0.3);
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
        color: #ffffff;
    }}
    </style>
    <div class="metric-container">
    <div class="metric-card">
    <div class="metric-label">Current Price</div>
    <div class="metric-value">${current_price:.2f}</div>
    <div class="metric-sub">{last_date.strftime('%Y-%m-%d')}</div>
    </div>
    <div class="metric-card">
    <div class="metric-label">Adj Target</div>
    <div class="metric-value" style="color: #cccccc;">${curr_baseline_val:.2f}</div>
    <div class="metric-sub">Base: ${baseline_prices_base[day_diff]:.2f} (x{sentiment_factor})</div>
    </div>
    <div class="metric-card">
    <div class="metric-label">Deviation</div>
    <div class="metric-value" style="color: {main_color};">{delta_pct:+.2f}%</div>
    <div class="metric-sub">from Adj Base</div>
    </div>
    <div class="metric-card" style="{current_card_style}">
    <div class="metric-label" style="color: #ffffff;">Signal</div>
    <div class="signal-value">{signal_status.split(' ')[0]}</div>
    <div class="metric-sub" style="color: #ffffff; opacity: 0.8;">{signal_status.split(' ', 1)[1] if ' ' in signal_status else ''}</div>
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

fig.update_layout(
    title=f"{selected_ticker} Wave Navigator", 
    height=600, 
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    yaxis=dict(
        range=[y_min_val - y_margin, y_max_val + y_margin],
        fixedrange=True # Disable zoom on Y
    ),
    xaxis=dict(
        fixedrange=True # Disable zoom on X
    )
)

# RENDER CHART IN CONTAINER
# If chart_space was defined earlier (inside the if), use it. But fig creation is outside.
# Let's clean up structure. 
# We'll use a main_block container for everything below header.
with st.container(border=True):
    st.plotly_chart(
        fig, 
        use_container_width=True,
        config={
            'displayModeBar': True,
            'modeBarButtons': [['toImage']], 
            'displaylogo': False,
            'scrollZoom': False
        }
    )

st.markdown("---")
st.markdown(f"**Update Status:** Fetched at {datetime.now().strftime('%H:%M:%S')}")

# --- Auto Refresh ---
if auto_refresh:
    time.sleep(60)
    st.rerun()
