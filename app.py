import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt

# --- CONFIG & HEAVYWEIGHTS ---
HEAVYWEIGHTS = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS"]
st_autorefresh(interval=60 * 1000, key="live_update") # Refresh every 60s for live market

def get_weighted_sentiment():
    """Scrapes news for Nifty + Top 5 Stocks for deeper accuracy."""
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    # Scrape Nifty + Top 5 heavyweights
    for ticker in ["^NSEI"] + HEAVYWEIGHTS:
        try:
            news = yf.Ticker(ticker).news
            for n in news[:3]:
                scores.append(analyzer.polarity_scores(n['title'])['compound'])
        except: continue
    return np.mean(scores) if scores else 0.0

def detect_channel(df):
    """Calculates a Linear Regression Channel for the current day."""
    y = df['Close'].values
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    line = slope * x + intercept
    std = np.std(y - line)
    upper_channel = line + (2 * std)
    lower_channel = line - (2 * std)
    return line, upper_channel, lower_channel

# --- MAIN ENGINE ---
st.title("🏹 Nifty Execution Engine: Live Entry/Exit")

# Data Fetching
nifty_df = yf.download("^NSEI", interval="15m", period="2d", progress=False)
if isinstance(nifty_df.columns, pd.MultiIndex): nifty_df.columns = [c[0] for c in nifty_df.columns]

# Pattern & Sentiment
sentiment = get_weighted_sentiment()
mid, upper, lower = detect_channel(nifty_df)

# AI Signal Logic
current_price = nifty_df['Close'].iloc[-1]
rsi = 100 - (100 / (1 + (nifty_df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                         (-nifty_df['Close'].diff().clip(upper=0).rolling(14).mean() + 1e-9))))

# --- PREDICTION & EXECUTION BOX ---
st.subheader("🎯 Live Execution Signal")
col1, col2 = st.columns([1, 1])

# Logic for Entry/Exit
if current_price >= upper[-1] and sentiment < 0:
    signal = "🔴 SELL/PUT ENTRY"
    entry = current_price
    target = lower[-1]
    stop = current_price + (current_price * 0.002) # 0.2% SL
    confidence = "HIGH (Channel Overbought + Negative Sentiment)"
elif current_price <= lower[-1] and sentiment > 0:
    signal = "🟢 BUY/CALL ENTRY"
    entry = current_price
    target = upper[-1]
    stop = current_price - (current_price * 0.002)
    confidence = "HIGH (Channel Oversold + Positive Sentiment)"
else:
    signal = "🟡 NO TRADE ZONE"
    entry, target, stop, confidence = 0, 0, 0, "WAITING FOR BREAKOUT"

with col1:
    st.info(f"**Action:** {signal}")
    st.write(f"**Confidence:** {confidence}")

with col2:
    if entry > 0:
        st.write(f"✅ **Entry:** {entry:.2f}")
        st.write(f"🎯 **Target:** {target:.2f}")
        st.write(f"🛑 **Stop Loss:** {stop:.2f}")

# Graphical Representation
st.subheader("📊 15-Min Trend Channel")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(nifty_df.index, nifty_df['Close'], label="Price", color="cyan")
ax.plot(nifty_df.index, upper, '--', color="red", alpha=0.5, label="Upper Channel")
ax.plot(nifty_df.index, lower, '--', color="green", alpha=0.5, label="Lower Channel")
ax.fill_between(nifty_df.index, lower, upper, color='gray', alpha=0.1)
plt.legend()
st.pyplot(fig)
