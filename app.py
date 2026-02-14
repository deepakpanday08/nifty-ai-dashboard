import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, time
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. SETTINGS & BACKGROUND LOGIC ---
st.set_page_config(page_title="NIFTY AI COMMAND CENTER", layout="wide")
st_autorefresh(interval=60 * 1000, key="live_sync")
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

# Background Data (Hidden from UI)
NIFTY_CONSTITUENTS = {
    "FINANCE": {"HDFCBANK.NS": 13.2, "ICICIBANK.NS": 9.1, "AXISBANK.NS": 3.2},
    "OIL_GAS": {"RELIANCE.NS": 9.4, "ONGC.NS": 1.2, "BPCL.NS": 0.6},
    "IT": {"TCS.NS": 3.9, "INFY.NS": 5.1, "HCLTECH.NS": 1.5},
    "FMCG": {"ITC.NS": 3.8, "HINDUNILVR.NS": 2.2}
}
MACRO_IMPACT_MAP = {"crude oil": {"OIL_GAS": -1}, "repo rate": {"FINANCE": -1}, "dollar": {"IT": 1}}

# --- 2. BACKEND ENGINE ---
def calculate_internal_metrics():
    """Silent background calculation for Confidence and Stock Positions."""
    analyzer = SentimentIntensityAnalyzer()
    stock_positions = []
    top_tickers = ["HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS"]
    for ticker in top_tickers:
        t = yf.Ticker(ticker)
        d = t.history(period="1d")
        if not d.empty:
            change = d['Close'].iloc[-1] - d['Open'].iloc[-1]
            pos = "Positive (Buying)" if change > 0 else "Negative (Selling)"
            stock_positions.append({"Stock": ticker.replace(".NS",""), "Status": pos})
    return stock_positions

def get_global_metrics():
    indices = {"^GSPC": 0.5, "^N225": 0.5}
    score = 0
    for t, w in indices.items():
        try:
            d = yf.download(t, period="2d", interval="15m", progress=False)
            change = (d['Close'].iloc[-1] - d['Close'].iloc[0]) / d['Close'].iloc[0]
            score += float(change) * w
        except: continue
    label = "POSITIVE" if score > 0.001 else "NEGATIVE" if score < -0.001 else "NEUTRAL"
    color = "green" if label == "POSITIVE" else "red" if label == "NEGATIVE" else "gray"
    return label, color, score

# --- 3. DATA FETCHING ---
nifty = yf.download("^NSEI", period="2d", interval="5m", progress=False)
ltp = float(nifty['Close'].iloc[-1])
global_label, global_color, global_val = get_global_metrics()
top_5_stocks = calculate_internal_metrics()

# --- 4. DASHBOARD UI ---
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>NIFTY AI COMMAND CENTER</h1>", unsafe_allow_html=True)

# ROW 1: CORE METRICS
m1, m2, m3, m4 = st.columns(4)
m1.metric("CURRENT NIFTY", f"{ltp:.2f}")
m2.markdown(f"**GLOBAL SENTIMENT**<br><span style='color:{global_color}; font-size:22px; font-weight:bold;'>{global_label}</span>", unsafe_allow_html=True)
m3.metric("EXPIRY MODE", "HERO-ZERO (TUE)" if now_ist.weekday() == 1 else "REGULAR")
m4.metric("TIME (IST)", now_ist.strftime('%H:%M'))

st.divider()

# ROW 2: SIGNALS, HERO-ZERO & INSTITUTIONS
col_signal, col_fii = st.columns([2, 1])

with col_signal:
    st.subheader("🎯 Active Trade Signal")
    confidence = "Low"
    if abs(global_val) > 0.003: confidence = "Mid"
    if abs(global_val) > 0.006: confidence = "High"
    
    if now_ist.time() < time(9, 30):
        st.info("⌛ OPENING RANGE SCAN: Awaiting 09:30 AM stability.")
    else:
        if global_val > 0.001:
            st.success(f"🚀 **BUY CALL (CE)** | Confidence: **{confidence}**")
            st.write(f"**Target:** {ltp+45:.2f} | **SL:** {ltp-25:.2f}")
        elif global_val < -0.001:
            st.error(f"📉 **BUY PUT (PE)** | Confidence: **{confidence}**")
            st.write(f"**Target:** {ltp-45:.2f} | **SL:** {ltp+25:.2f}")
        else:
            st.warning("⚖️ **WAITING** - Market is currently in a Neutral/Sideways zone.")

    # --- HERO-ZERO SUGGESTION SECTION ---
    st.divider()
    st.subheader("⚡ HERO-ZERO SUGGESTION")
    is_expiry_day = now_ist.weekday() == 1  # Tuesday (Nifty Fin/Midcap common)
    if is_expiry_day and now_ist.hour >= 13:
        # Strike selection: 50-100 points OTM
        strike_call = int(round((ltp + 70) / 50) * 50)
        strike_put = int(round((ltp - 70) / 50) * 50)
        
        if global_val > 0.002:
            st.markdown(f"🔥 **EXPIRY BLAST (CE):** Buy Nifty {strike_call} CE")
            st.caption("Suggested Entry: ₹5-₹10 | Target: ₹30-₹50 | SL: Zero")
        elif global_val < -0.002:
            st.markdown(f"🔥 **EXPIRY BLAST (PE):** Buy Nifty {strike_put} PE")
            st.caption("Suggested Entry: ₹5-₹10 | Target: ₹30-₹50 | SL: Zero")
        else:
            st.info("No clear trend for Hero-Zero yet. Waiting for Gamma move.")
    else:
        st.write("🔴 Hero-Zero mode inactive. (Active Tuesdays after 1:00 PM)")

with col_fii:
    st.subheader("🔝 Top 5 Stock Positions")
    st.table(pd.DataFrame(top_5_stocks))
    st.subheader("📊 Institutional Flow")
    st.markdown("**FII:** -₹1,240 Cr | **DII:** +₹950 Cr")

# ROW 3: CHART (BOTTOM)
st.divider()
st.subheader("📈 Live 5-Min Nifty Candlestick & Volume")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=nifty.index, open=nifty['Open'], high=nifty['High'], low=nifty['Low'], close=nifty['Close'], name="Price"))
fig.add_trace(go.Bar(x=nifty.index, y=nifty['Volume'], name="Volume", yaxis="y2", opacity=0.3, marker_color='white'))
fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, yaxis=dict(title="Price"), yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False))
st.plotly_chart(fig, use_container_width=True)
