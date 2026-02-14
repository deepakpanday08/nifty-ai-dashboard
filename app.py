import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz
import requests
from bs4 import BeautifulSoup

# --- 1. SETTINGS ---
st.set_page_config(page_title="NIFTY AI MASTER", layout="wide", page_icon="🎯")
st_autorefresh(interval=60 * 1000, key="live_sync")

IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

# --- 2. BACKEND INTELLIGENCE (Hidden from Dashboard) ---

@st.cache_data(ttl=3600) # FII/DII data only updates once a day
def get_fii_dii_bias():
    try:
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = pd.read_html(str(soup))
        # Usually, the first table contains the latest daily activity
        df = tables[0] 
        # Logic: If FII Net is positive, bias +1. If DII Net is positive, bias +0.5.
        fii_net = float(df.iloc[0, 3]) # Net Purchase/Sales
        dii_net = float(df.iloc[1, 3])
        
        bias = 0
        if fii_net > 0: bias += 1
        else: bias -= 1
        if dii_net > 0: bias += 0.5
        
        return bias
    except:
        return 0 # Neutral if data fetch fails

@st.cache_data(ttl=300)
def get_global_bias():
    indices = {"^GSPC": 0.5, "^N225": 0.5}
    bias_score = 0.0
    for ticker, weight in indices.items():
        try:
            data = yf.download(ticker, period="2d", interval="15m", progress=False)
            change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
            bias_score += float(change) * weight
        except: continue
    return bias_score

@st.cache_data(ttl=60)
def fetch_nifty_data():
    df = yf.download("^NSEI", period="2d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Std'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (1.5 * df['Std'])
    df['Lower'] = df['MA20'] - (1.5 * df['Std'])
    return df

# --- 3. EXECUTION LOGIC ---
nifty = fetch_nifty_data()
global_bias = float(get_global_bias())
fii_dii_bias = float(get_fii_dii_bias()) # New Backend Data
ltp = float(nifty['Close'].iloc[-1])
upper_band = float(nifty['Upper'].iloc[-1])
lower_band = float(nifty['Lower'].iloc[-1])

# Combined Analysis Score
total_sentiment_score = global_bias + (fii_dii_bias * 0.1) # FII/DII weighted lightly as it's EOD data

# --- 4. DASHBOARD VIEW (Execution Above, Chart Below) ---
st.markdown("<h2 style='text-align: center; color: #00ffcc;'>NIFTY 50 EXECUTION TERMINAL</h2>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("LIVE NIFTY", f"{ltp:.2f}", f"{ltp - nifty['Close'].iloc[0]:+.2f}")
m2.metric("GLOBAL BIAS", "BULLISH" if global_bias > 0 else "BEARISH", f"{global_bias:+.4f}")
m3.metric("EXPIRY MODE", "TUESDAY HERO-ZERO" if now_ist.weekday() == 1 else "NORMAL")
m4.metric("SMART MONEY", "ACCUMULATING" if fii_dii_bias > 0 else "DISTRIBUTING")

st.divider()

sig_col, hero_col = st.columns([2, 1])

def play_sound():
    sound_url = "https://www.soundjay.com/buttons/sounds/button-3.mp3"
    st.components.v1.html(f'<audio autoplay><source src="{sound_url}"></audio>', height=0)

with sig_col:
    st.subheader("🎯 Active Trade Signal")
    # THE AI DECISION: Now includes FII/DII sentiment in the background
    if ltp < lower_band and total_sentiment_score > 0:
        st.success(f"🚀 **ACTION: BUY CALL (CE)**\n\nEntry: {ltp:.2f} | Target: {ltp+45:.2f} | SL: {ltp-20:.2f}")
        play_sound()
    elif ltp > upper_band and total_sentiment_score < 0:
        st.error(f"📉 **ACTION: BUY PUT (PE)**\n\nEntry: {ltp:.2f} | Target: {ltp-45:.2f} | SL: {ltp+20:.2f}")
        play_sound()
    else:
        st.info("⚖️ **STATUS: SCANNING...** (No high-probability setup)")

with hero_col:
    st.subheader("⚡ Hero-Zero")
    if now_ist.weekday() == 1 and now_ist.hour >= 13:
        strike = int(round(ltp / 50) * 50)
        st.warning(f"🔥 HERO STRIKE: {strike} {'CE' if total_sentiment_score > 0 else 'PE'}")
    else: st.write("Waiting for Expiry Hours...")

st.divider()
st.subheader("📊 Live Technical Chart")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=nifty.index, open=nifty['Open'], high=nifty['High'], low=nifty['Low'], close=nifty['Close'], name="Nifty"))
fig.add_trace(go.Scatter(x=nifty.index, y=nifty['Upper'], line=dict(color='rgba(255,0,0,0.3)', width=1), name="Resistance"))
fig.add_trace(go.Scatter(x=nifty.index, y=nifty['Lower'], line=dict(color='rgba(0,255,0,0.3)', width=1), name="Support"))
fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
