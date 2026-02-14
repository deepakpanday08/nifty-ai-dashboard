import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz
import matplotlib.pyplot as plt

# --- 1. SETTINGS & REFRESH ---
st.set_page_config(page_title="Nifty AI Master", layout="wide", page_icon="🎯")
st_autorefresh(interval=60 * 1000, key="live_sync") # 1-minute fast-track refresh

# Timezone Handling
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

# NSE Heavyweights (Impact Nifty by ~40%)
HEAVYWEIGHTS = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS"]

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=60)
def fetch_market_data():
    # Primary Data
    nifty = yf.download("^NSEI", interval="15m", period="5d", progress=False)
    # Global/Gifty Nifty proxy (using pre-market index)
    sp500 = yf.download("^GSPC", interval="15m", period="5d", progress=False)
    
    # Flatten MultiIndex if yfinance returns it
    for df in [nifty, sp500]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
    return nifty, sp500

def get_weighted_sentiment():
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    # Scrape Nifty + Top 5 heavyweights for higher accuracy
    for t in ["^NSEI"] + HEAVYWEIGHTS:
        try:
            news = yf.Ticker(t).news
            if news:
                scores.append(analyzer.polarity_scores(news[0].get('title', ''))['compound'])
        except: continue
    return np.mean(scores) if scores else 0.0

# --- 3. PATTERN & HERO-ZERO LOGIC ---
def analyze_execution(df):
    # Linear Regression Channel Calculation
    y = df['Close'].values
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    center = slope * x + intercept
    std = np.std(y - center)
    
    upper = center + (1.5 * std)
    lower = center - (1.5 * std)
    
    current_price = df['Close'].iloc[-1]
    
    # Hero-Zero Logic (TUESDAY ONLY)
    hero_signal = "WAITING"
    # 1 = Tuesday
    if now_ist.weekday() == 1 and now_ist.hour >= 13 and now_ist.minute >= 30:
        strike = int(round(current_price / 50) * 50)
        if current_price > upper[-1]:
            hero_signal = f"🚀 HERO CALL: {strike} CE"
        elif current_price < lower[-1]:
            hero_signal = f"🩸 HERO PUT: {strike} PE"
            
    return current_price, upper, lower, hero_signal

# --- 4. DASHBOARD UI ---
st.title("🏹 Nifty AI Execution Engine")
st.write(f"IST Time: {now_ist.strftime('%H:%M:%S')} | Day: {now_ist.strftime('%A')}")

# Run Backend
with st.spinner('Syncing with NSE Servers...'):
    nifty, sp500 = fetch_market_data()
    sentiment = get_weighted_sentiment()
    ltp, up_band, lo_band, hero = analyze_execution(nifty)

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("NIFTY 50", f"{ltp:.2f}")
m2.metric("Heavyweight Sentiment", f"{sentiment:+.2f}")
m3.metric("Channel Status", "Overbought" if ltp > up_band[-1] else "Oversold" if ltp < lo_band[-1] else "Neutral")
m4.metric("Market Bias", "BULLISH" if sentiment > 0.1 else "BEARISH" if sentiment < -0.1 else "SIDEWAYS")

# --- SIGNAL & HERO-ZERO PANEL ---
col_sig, col_hero = st.columns(2)

with col_sig:
    st.subheader("🎯 Trade Execution")
    if ltp > up_band[-1] and sentiment > 0:
        st.success(f"**SIGNAL: BUY/CALL ENTRY**\n\nTarget: {ltp + 45} | SL: {ltp - 20}")
    elif ltp < lo_band[-1] and sentiment < 0:
        st.error(f"**SIGNAL: SELL/PUT ENTRY**\n\nTarget: {ltp - 45} | SL: {ltp + 20}")
    else:
        st.warning("⚖️ **NO TRADE ZONE**: Wait for Channel Breakout + Sentiment Sync.")

with col_hero:
    st.subheader("⚡ Tuesday Hero-Zero")
    if now_ist.weekday() == 1:
        if "WAITING" in hero:
            st.info("Logic active after 01:30 PM on Expiry Day (Tuesday).")
        else:
            st.warning(f"🔥 **{hero}**")
            st.caption("Low capital high-risk trade. Suggested entry ₹5-₹10.")
    else:
        st.write("Inactive (Only active on Tuesdays).")

# --- CHARTING ---
st.subheader("📊 15-Minute Linear Regression Channel")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(nifty.index, nifty['Close'], label="Price", color='#1f77b4', linewidth=2)
ax.plot(nifty.index, up_band, '--', color='red', alpha=0.6, label="Resistance")
ax.plot(nifty.index, lo_band, '--', color='green', alpha=0.6, label="Support")
ax.fill_between(nifty.index, lo_band, up_band, color='gray', alpha=0.1)
plt.legend()
st.pyplot(fig)

st.divider()
st.caption("Disclaimer: AI analysis is for educational purposes. Always verify with your financial advisor before trading.")
