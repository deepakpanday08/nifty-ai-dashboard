import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh
import pytz
import matplotlib.pyplot as plt

# --- 1. CONFIG & REFRESH ---
st.set_page_config(page_title="Nifty Global AI", layout="wide")
st_autorefresh(interval=60 * 1000, key="global_sync")

IST = pytz.timezone('Asia/Kolkata')

# --- 2. GLOBAL DATA ENGINE ---
@st.cache_data(ttl=300)
def get_global_pulse():
    # US, Japan, Hong Kong, and Nifty
    tickers = {
        "NIFTY": "^NSEI",
        "S&P 500": "^GSPC",
        "NIKKEI 225": "^N225",
        "HANG SENG": "^HSI"
    }
    
    global_data = {}
    for name, sym in tickers.items():
        try:
            df = yf.download(sym, period="2d", interval="15m", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
            
            # Calculate % change from previous close
            current = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[0]
            pct_change = ((current - prev_close) / prev_close) * 100
            global_data[name] = {"price": current, "change": pct_change}
        except:
            global_data[name] = {"price": 0, "change": 0}
            
    return global_data

def get_heavyweight_sentiment():
    analyzer = SentimentIntensityAnalyzer()
    weights = {"RELIANCE.NS": 0.10, "HDFCBANK.NS": 0.12, "ICICIBANK.NS": 0.08, "INFY.NS": 0.06, "ITC.NS": 0.05}
    total_sentiment = 0
    for stock, weight in weights.items():
        try:
            news = yf.Ticker(stock).news
            if news:
                score = analyzer.polarity_scores(news[0]['title'])['compound']
                total_sentiment += score * weight
        except: continue
    return total_sentiment

# --- 3. UI LAYOUT ---
st.title("🌏 Nifty Global Multi-Market AI")

# Sidebar: Market Sentiment Meter
global_stats = get_global_pulse()
hw_sentiment = get_heavyweight_sentiment()

# Calculate Global Bias Score
# (Weightage: US 40%, Japan 30%, HK 30%)
global_bias = (global_stats['S&P 500']['change'] * 0.4) + \
              (global_stats['NIKKEI 225']['change'] * 0.3) + \
              (global_stats['HANG SENG']['change'] * 0.3)

# --- DISPLAY SECTION ---
st.subheader("🌐 Global Market Dashboard (Asian & US Lead)")
cols = st.columns(4)
cols[0].metric("S&P 500 (USA)", f"{global_stats['S&P 500']['change']:+.2f}%")
cols[1].metric("NIKKEI 225 (JPN)", f"{global_stats['NIKKEI 225']['change']:+.2f}%")
cols[2].metric("HANG SENG (HKG)", f"{global_stats['HANG SENG']['change']:+.2f}%")
cols[3].metric("GLOBAL BIAS SCORE", f"{global_bias:+.2f}")

# --- AI SIGNAL ENGINE ---
st.divider()
st.subheader("🎯 Nifty 50 Execution Signal")

# Fetch Nifty for Patterns
nifty_df = yf.download("^NSEI", interval="15m", period="5d", progress=False)
if isinstance(nifty_df.columns, pd.MultiIndex): nifty_df.columns = [c[0] for c in nifty_df.columns]

# Signal Logic: Global Bias + Local Heavyweight News + Price Action
ltp = nifty_df['Close'].iloc[-1]
rsi = 100 - (100 / (1 + (nifty_df['Close'].diff().rolling(14).mean() / (-nifty_df['Close'].diff().rolling(14).mean() + 1e-9))))

col_sig, col_details = st.columns([1, 2])

with col_sig:
    if global_bias > 0.3 and hw_sentiment > 0:
        st.success("🔥 SIGNAL: STRONG BUY")
        st.write("**Reason:** Global Asian markets & US are green + Heavyweight news is positive.")
    elif global_bias < -0.3 and hw_sentiment < 0:
        st.error("📉 SIGNAL: STRONG SELL")
        st.write("**Reason:** Global Asian drag + US weakness + Negative local news.")
    else:
        st.warning("⚖️ SIGNAL: NEUTRAL")
        st.write("**Reason:** Mixed global signals. Avoid aggressive entry.")

with col_details:
    st.info(f"**Heavyweight News Impact:** {hw_sentiment:+.2f}")
    st.write(f"**Nifty RSI:** {rsi.iloc[-1]:.1f}")
    if abs(global_bias) > 0.5:
        st.markdown("⚠️ **High Volatility Alert:** Global markets are moving sharply.")

# Simple Trend Chart
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(nifty_df.index, nifty_df['Close'], label="Nifty 15m")
plt.title("Nifty Intraday Trend")
st.pyplot(fig)
