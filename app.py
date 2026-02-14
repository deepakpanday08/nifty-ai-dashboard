import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, time
import pytz

# --- 1. SETTINGS ---
st.set_page_config(page_title="NIFTY AI MASTER", layout="wide")
st_autorefresh(interval=60 * 1000, key="live_sync")
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

# --- 2. BACKEND CALCULATIONS ---
def get_global_bias_text():
    # US & Japan weighted analysis
    indices = {"^GSPC": 0.5, "^N225": 0.5}
    score = 0
    for ticker, weight in indices.items():
        try:
            data = yf.download(ticker, period="2d", interval="15m", progress=False)
            change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
            score += float(change) * weight
        except: continue
    
    if score > 0.001: return "POSITIVE (Bullish)", "green", score
    if score < -0.001: return "NEGATIVE (Bearish)", "red", score
    return "NEUTRAL", "gray", score

@st.cache_data(ttl=60)
def fetch_nifty():
    df = yf.download("^NSEI", period="2d", interval="5m", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
    df['MA20'] = df['Close'].rolling(20).mean()
    return df

# --- 3. DASHBOARD UI (The FIX) ---
nifty = fetch_nifty()
ltp = float(nifty['Close'].iloc[-1])
bias_text, bias_color, bias_val = get_global_bias_text()

st.markdown(f"<h1 style='text-align: center; color: #00ffcc;'>NIFTY 50 AI TERMINAL</h1>", unsafe_allow_html=True)

# TOP METRICS (Clearer Labels)
m1, m2, m3, m4 = st.columns(4)
m1.metric("CURRENT NIFTY", f"{ltp:.2f}")
m2.markdown(f"**GLOBAL BIAS** \n<span style='color:{bias_color}; font-size:20px;'>{bias_text}</span>", unsafe_allow_html=True)
m3.metric("EXPIRY MODE", "TUESDAY HERO-ZERO" if now_ist.weekday() == 1 else "NORMAL")
m4.metric("TIME (IST)", now_ist.strftime('%H:%M'))

st.divider()

# SIGNAL AREA (Must be above chart)
sig_col, hero_col = st.columns([2, 1])

with sig_col:
    st.subheader("🎯 Active Trade Signal")
    
    # 9:30 AM Logic
    if now_ist.time() < time(9, 30):
        st.info(f"⏳ **WAITING FOR 9:30 AM** - System is currently analyzing opening volatility. {30 - now_ist.minute if now_ist.minute < 30 else 0} mins left.")
    else:
        # Final Decision Logic
        if bias_val > 0.001 and ltp > nifty['MA20'].iloc[-1]:
            st.success(f"🚀 **BUY CALL (CE) SUGGESTION**\n\n**ENTRY:** {ltp:.2f} | **TARGET:** {ltp+40:.2f} | **SL:** {ltp-20:.2f}")
            st.markdown("---")
            st.caption("✅ Confidence: HIGH (Global Bias + Technical Trend match)")
        elif bias_val < -0.001 and ltp < nifty['MA20'].iloc[-1]:
            st.error(f"📉 **BUY PUT (PE) SUGGESTION**\n\n**ENTRY:** {ltp:.2f} | **TARGET:** {ltp-40:.2f} | **SL:** {ltp+20:.2f}")
            st.markdown("---")
            st.caption("✅ Confidence: HIGH (Global Bias + Technical Trend match)")
        else:
            st.warning("⚖️ **STATUS: NEUTRAL** - Market is sideways or Global/Local signals are conflicting. Do not enter.")

with hero_col:
    st.subheader("⚡ Hero-Zero (Tuesday)")
    if now_ist.weekday() == 1 and now_ist.hour >= 13:
        strike = int(round(ltp / 50) * 50)
        st.error(f"🔥 ACTIVE: {strike} {'CE' if bias_val > 0 else 'PE'}")
        st.write("Target: 3x Returns | SL: 0")
    else:
        st.write("🔴 Inactive (Only on Tuesdays after 1 PM)")

# --- 4. CHART (Bottom) ---
st.divider()
fig = go.Figure(data=[go.Candlestick(x=nifty.index, open=nifty['Open'], high=nifty['High'], low=nifty['Low'], close=nifty['Close'])])
fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, title="Live 5-Min Nifty Chart")
st.plotly_chart(fig, use_container_width=True)
