import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# --- BACKEND BIAS ENGINE (Hidden from UI) ---
def calculate_global_bias():
    # Japan (Nikkei) and US (S&P) provide the "Trend Lead"
    indices = ["^N225", "^GSPC"]
    score = 0
    for idx in indices:
        try:
            d = yf.download(idx, period="2d", interval="15m", progress=False)
            change = (d['Close'].iloc[-1] - d['Close'].iloc[0]) / d['Close'].iloc[0]
            score += change
        except: continue
    return score

# --- THE VIEW ENGINE ---
st.set_page_config(page_title="NIFTY AI", layout="wide")
st_autorefresh(interval=60 * 1000, key="sync")

# Fetch Nifty Data
nifty = yf.download("^NSEI", period="2d", interval="15m", progress=False)
if isinstance(nifty.columns, pd.MultiIndex): nifty.columns = [c[0] for c in nifty.columns]

# Calculate Channels for Charting
nifty['MA20'] = nifty['Close'].rolling(20).mean()
nifty['Upper'] = nifty['MA20'] + (nifty['Close'].rolling(20).std() * 1.5)
nifty['Lower'] = nifty['MA20'] - (nifty['Close'].rolling(20).std() * 1.5)

# 1. THE LIVE CHART (Main Focus)
fig = go.Figure()
fig.add_trace(go.Candlestick(x=nifty.index, open=nifty['Open'], high=nifty['High'], low=nifty['Low'], close=nifty['Close'], name="Market"))
fig.add_trace(go.Scatter(x=nifty.index, y=nifty['Upper'], line=dict(color='rgba(255,0,0,0.2)'), name="Resistance"))
fig.add_trace(go.Scatter(x=nifty.index, y=nifty['Lower'], line=dict(color='rgba(0,255,0,0.2)'), name="Support"))

fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# 2. THE SIGNAL BAR (The "Execution" Center)
bias = calculate_global_bias()
ltp = nifty['Close'].iloc[-1]

st.divider()
col1, col2 = st.columns([3, 1])

with col1:
    if ltp > nifty['Upper'].iloc[-1] and bias < 0:
        st.error(f"📉 **PUT ENTRY SIGNAL** | Entry: {ltp:.2f} | Target: {ltp-40:.2f} | SL: {ltp+15:.2f}")
    elif ltp < nifty['Lower'].iloc[-1] and bias > 0:
        st.success(f"🚀 **CALL ENTRY SIGNAL** | Entry: {ltp:.2f} | Target: {ltp+40:.2f} | SL: {ltp-15:.2f}")
    else:
        st.warning("⚖️ **NEUTRAL ZONE** | Global Bias: " + ("Bullish" if bias > 0 else "Bearish"))

with col2:
    st.metric("NIFTY 50", f"{ltp:.2f}", f"{ltp - nifty['Close'].iloc[0]:+.2f}")
