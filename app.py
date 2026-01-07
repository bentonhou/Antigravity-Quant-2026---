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
    "MRVL": {"start": 89.39,  "target": 125.0},
}

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days + 1

st.set_page_config(page_title="Antigravity Quant 2026", layout="wide")
st.title("Antigravity Quant 2026 - 波段導航儀")

# --- Sidebar ---
st.sidebar.header("Asset Selection")
selected_ticker = st.sidebar.radio("Ticker", list(STOCKS_CONFIG.keys()))
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
df_baseline["Lower_10"] = df_baseline["Baseline"] * 0.90

# --- Helper: Data Fetching with Cache ---
@st.cache_data(ttl=60) # Cache for 60 seconds
def get_stock_data(ticker):
    try:
        # Fetch data from 2026-01-01 to NOW
        # We use a broad range to ensure we capture defaults
        df = yf.download(ticker, start="2026-01-01",  interval="1d", progress=False)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- Main Logic ---
with st.spinner(f"Fetching live data for {selected_ticker}..."):
    df_real = get_stock_data(selected_ticker)

# --- Signal Logic & Metrics ---
current_price = 0.0
signal_status = "Waiting for data..."
signal_type = "neutral" # neutral, buy, reduce

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
        upper_bound = curr_baseline * 1.25
        lower_bound = curr_baseline * 0.90
        
        delta = current_price - curr_baseline
        delta_pct = (delta / curr_baseline) * 100
        
        # Signal Judgment
        if current_price <= lower_bound:
            signal_status = "🟢 觸發『買入點 a』 (Buy!)"
            signal_type = "buy"
        elif current_price >= upper_bound:
            signal_status = "🔴 觸發『減碼點 a'』 (Reduce!)"
            signal_type = "reduce"
        else:
            signal_status = "⚪ 觀望 / 持有 (Hold)"
            signal_type = "neutral"
            
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"${current_price:.2f}", f"{last_date.date()}")
        m2.metric("Baseline (Target)", f"${curr_baseline:.2f}")
        m3.metric("Deviation", f"{delta_pct:.2f}%")
        m4.metric("Signal", signal_status)
        
        # Banner
        if signal_type == "buy":
            st.success(f"### ACTION REQUIRED: {signal_status}")
        elif signal_type == "reduce":
            st.error(f"### ACTION REQUIRED: {signal_status}")
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
fig.add_trace(go.Scatter(x=df_baseline["Date"], y=df_baseline["Upper_25"], mode='lines', name='+25% (Reduce)', line=dict(color='gray', width=1, dash='dash')))
fig.add_trace(go.Scatter(x=df_baseline["Date"], y=df_baseline["Lower_10"], mode='lines', name='-10% (Buy)', line=dict(color='gray', width=1, dash='dash')))

# 3. Real Data (RED Line)
if not df_plot.empty:
    fig.add_trace(go.Scatter(
        x=df_plot["Date"], 
        y=df_plot["Close_Flat"], 
        mode='lines', 
        name='Actual Price', 
        line=dict(color='red', width=4)
    ))

fig.update_layout(title=f"{selected_ticker} Wave Navigator", height=600, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(f"**Update Status:** Fetched at {datetime.now().strftime('%H:%M:%S')}")

# --- Auto Refresh ---
if auto_refresh:
    time.sleep(60)
    st.rerun()
