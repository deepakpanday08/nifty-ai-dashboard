import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz
import base64

# --- 1. SETTINGS & REFRESH ---
st.set_page_config(page_title="NIFTY AI MASTER", layout="wide", page_icon="🎯")
st_autorefresh(interval=60 * 1000, key="live_sync")

IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

# --- 2. AUDIO NOTIFICATION HACK ---
def play_sound():
    # A short "Ding" sound encoded in Base64 to play in the browser
    sound_file = "https://www.soundjay.com/buttons/sounds/button-3.mp3"
    html_string = f"""
        <audio autoplay>
          <source src="{sound_file}" type="audio/mpeg">
        </audio>
    """
    st.components.v1.html(html_string, height=0)

# --- 3. BACKEND DATA ---
@st.cache_data(ttl=300)
def get_global_bias():
    indices = {"^GSPC": 0.5, "^N225": 0.5}
    bias_score = 0.0
    for ticker, weight in indices.items():
        try:
            data = yf.download(ticker, period="2d", interval="15m", progress=False)
            if not data.empty:
                change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
                bias_score += float(change) * weight
        except: continue
    return bias_score

@st.cache_data(ttl=60)
def fetch_nifty_data():
    df = yf.download("^NSEI", period="2d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Std'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (1.5 * df['Std'])
    df['Lower'] = df['MA20'] - (1.5 * df['Std'])
    return df

# --- 4. DATA PROCESSING ---
nifty = fetch_nifty_data()
global_bias = float(get_global_bias())
ltp = float(nifty['Close'].iloc[-1])
upper_band = float(nifty['Upper'].iloc[-1])
lower_band = float(nifty['Lower'].iloc[-1])

# --- 5. TOP PANEL: SIGNALS & METRICS (Moved Up) ---
st.markdown("<h2 style='text-align: center; color: #00ffcc;'>NIFTY 50 EXECUTION TERMINAL</h2>", unsafe_allow_html=True)

# Row 1: High Level Stats
m1, m2, m3, m4 = st.columns(4)
m1.metric("LIVE NIFTY", f"{ltp:.2f}", f"{ltp - nifty['Close'].iloc[0]:+.2f}")
m2.metric("GLOBAL BIAS", "BULLISH" if global_bias > 0 else "BEARISH", f"{global_bias:+.4f}")
m3.metric("EXPIRY MODE", "TUESDAY HERO-ZERO" if now_ist.weekday() == 1 else "NORMAL")
m4.metric("MARKET TIME", now_ist.strftime('%H:%M'))

st.divider()

# Row 2: TRADE SIGNALS (Moved Up)
sig_col, hero_col = st.columns([2, 1])

with sig_col:
    st.subheader("🎯 Active Trade Signal")
    if ltp < lower_band and global_bias > 0:
        st.success(f"🚀 **ACTION: BUY CALL (CE)**\n\n**ENTRY:** {ltp:.2f} | **TARGET:** {ltp+45:.2f} | **EXIT/SL:** {ltp-20:.2f}")
        play_sound() # Triggers sound on Entry
    elif ltp > upper_band and global_bias < 0:
        st.error(f"📉 **ACTION: BUY PUT (PE)**\n\n**ENTRY:** {ltp:.2f} | **TARGET:** {ltp-45:.2f} | **EXIT/SL:** {ltp+20:.2f}")
        play_sound() # Triggers sound on Entry
    else:
        st.info("⚖️ **STATUS: SCANNING MARKET...** (No Entry Yet)")

with hero_col:
    st.subheader("⚡ Hero-Zero")
    # Tuesday Expiry Logic
    if now_ist.weekday() == 1 and now_ist.hour >= 13:
        strike = int(round(ltp / 50) * 50)
        st.warning(f"🔥 HERO STRIKE: {strike} {'CE' if global_bias > 0 else 'PE'}")
        st.write("Target: 3x-5x | SL: Zero")
    else:
        st.write("System: Waiting for Tuesday afternoon.")

# --- 6. BOTTOM PANEL: THE CHART ---
st.divider()
st.subheader("📊 Live 15-Minute Analysis Chart")



fig = go.Figure()
fig.add_trace(go.Candlestick(x=nifty.index, open=nifty['Open'], high=nifty['High'], low=nifty['Low'], close=nifty['Close'], name="Price"))
fig.add_trace(go.Scatter(x=nifty.index, y=nifty['Upper'], line=dict(color='rgba(255,0,0,0.3)', width=1), name="Resistance"))
fig.add_trace(go.Scatter(x=nifty.index, y=nifty['Lower'], line=dict(color='rgba(0,255,0,0.3)', width=1), name="Support"))

fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

st.caption(f"Last Intelligence Sync: {now_ist.strftime('%H:%M:%S')} IST")
