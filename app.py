import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz

# --- 1. SETTINGS & REFRESH ---
st.set_page_config(page_title="NIFTY AI MASTER", layout="wide", page_icon="🎯")

# Auto-refresh every 60 seconds for live market tracking
st_autorefresh(interval=60 * 1000, key="live_sync")

# Timezone Handling
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

# --- 2. BACKEND BIAS ENGINE (Hidden Analysis) ---
@st.cache_data(ttl=300)
def get_global_bias():
    # Weights: S&P 500 (50%), Nikkei (50%)
    indices = {"^GSPC": 0.5, "^N225": 0.5}
    bias_score = 0.0
    for ticker, weight in indices.items():
        try:
            data = yf.download(ticker, period="2d", interval="15m", progress=False)
            if not data.empty:
                # Calculate % change from start of previous session
                change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
                bias_score += float(change) * weight
        except:
            continue
    return bias_score

# --- 3. LIVE DATA & TECHNICALS ---
@st.cache_data(ttl=60)
def fetch_nifty_data():
    df = yf.download("^NSEI", period="2d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Calculate Linear Channels (Bollinger-style)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Std'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (1.5 * df['Std'])
    df['Lower'] = df['MA20'] - (1.5 * df['Std'])
    
    return df

# --- 4. DASHBOARD UI ---
st.markdown("<h2 style='text-align: center; color: #00ffcc;'>NIFTY 50 LIVE AI TERMINAL</h2>", unsafe_allow_html=True)

# Run Logic
nifty = fetch_nifty_data()
global_bias = float(get_global_bias())
ltp = float(nifty['Close'].iloc[-1])
upper_band = float(nifty['Upper'].iloc[-1])
lower_band = float(nifty['Lower'].iloc[-1])

# Header Metrics
m1, m2, m3 = st.columns(3)
m1.metric("LIVE NIFTY", f"{ltp:.2f}", f"{ltp - nifty['Close'].iloc[0]:+.2f}")
m2.metric("GLOBAL BIAS", "BULLISH" if global_bias > 0 else "BEARISH", f"{global_bias:+.4f}")
m3.metric("EXPIRY MODE", "TUESDAY HERO-ZERO" if now_ist.weekday() == 1 else "NORMAL")

# 5. THE CHART (Plotly Interactive)

fig = go.Figure()

# Candles
fig.add_trace(go.Candlestick(
    x=nifty.index, open=nifty['Open'], high=nifty['High'],
    low=nifty['Low'], close=nifty['Close'], name="Price"
))

# Channels
fig.add_trace(go.Scatter(x=nifty.index, y=nifty['Upper'], line=dict(color='rgba(255,0,0,0.3)', width=1), name="Resistance"))
fig.add_trace(go.Scatter(x=nifty.index, y=nifty['Lower'], line=dict(color='rgba(0,255,0,0.3)', width=1), name="Support"))

fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=550, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# 6. EXECUTION SIGNALS (The Float-Fixed Logic)
st.divider()
sig_col, hero_col = st.columns([2, 1])

with sig_col:
    st.subheader("🎯 Trade Signal")
    # THE FIX: Using explicit float comparisons
    if ltp < lower_band and global_bias > 0:
        st.success(f"🚀 **ACTION: BUY CALL (CE)**\n\nEntry: {ltp:.2f} | Target: {ltp+45:.2f} | SL: {ltp-20:.2f}")
    elif ltp > upper_band and global_bias < 0:
        st.error(f"📉 **ACTION: BUY PUT (PE)**\n\nEntry: {ltp:.2f} | Target: {ltp-45:.2f} | SL: {ltp+20:.2f}")
    else:
        st.warning("⚖️ **STATUS: NEUTRAL** (Wait for Channel Breakout)")

with hero_col:
    st.subheader("⚡ Hero-Zero")
    if now_ist.weekday() == 1 and now_ist.hour >= 13:
        strike = int(round(ltp / 50) * 50)
        st.warning(f"🔥 ACTIVE: {strike} {'CE' if global_bias > 0 else 'PE'}")
        st.caption("Suggested entry: ₹5 - ₹12")
    else:
        st.write("Inactive (Only on Tuesdays after 1:30 PM)")

st.caption(f"Last Refreshed: {now_ist.strftime('%H:%M:%S')} IST")
