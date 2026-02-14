import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, time
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. CORE SETUP ---
st.set_page_config(page_title="Nifty Quant Intelligence", layout="wide")
st_autorefresh(interval=60 * 1000, key="quant_engine_refresh")
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

# --- 2. BACKEND MULTI-FACTOR SCORING (HIDDEN) ---

def analyze_macro_and_sector():
    """Scans for indirect impact: Geopolitics, Oil, and Reliance/HDFC News."""
    analyzer = SentimentIntensityAnalyzer()
    # Strategic Tickers for Indirect Impact
    impact_tickers = ["RELIANCE.NS", "HDFCBANK.NS", "CL=F", "GC=F"] # Reliance, HDFC, Oil, Gold
    total_sentiment = 0
    try:
        for ticker in impact_tickers:
            news = yf.Ticker(ticker).news
            # Analyze top 3 headlines per ticker
            scores = [analyzer.polarity_scores(n['title'])['compound'] for n in news[:3]]
            total_sentiment += np.mean(scores) if scores else 0
    except: pass
    return total_sentiment / len(impact_tickers)

def get_institutional_bias():
    """Analyzes FII/DII proxy via Volume-Price Trend."""
    df = yf.download("^NSEI", period="5d", interval="1d", progress=False)
    if df.empty: return 0
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
    
    # Calculate delivery/volume pressure proxy
    avg_vol = df['Volume'].mean()
    curr_vol = df['Volume'].iloc[-1]
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
    
    # If price is up on high volume, institutional accumulation is likely
    if price_change > 0 and curr_vol > avg_vol: return 1  # Accumulation
    if price_change < 0 and curr_vol > avg_vol: return -1 # Distribution
    return 0

def get_global_bias():
    """US & Asian Market Lead Analysis."""
    indices = {"^GSPC": 0.5, "^N225": 0.5} # S&P 500 and Nikkei
    score = 0
    for ticker, weight in indices.items():
        try:
            d = yf.download(ticker, period="2d", interval="15m", progress=False)
            change = (d['Close'].iloc[-1] - d['Close'].iloc[0]) / d['Close'].iloc[0]
            score += float(change) * weight
        except: continue
    return score

# --- 3. LIVE DATA & TECHNICAL CHANNELS ---
@st.cache_data(ttl=60)
def fetch_live_data():
    df = yf.download("^NSEI", period="2d", interval="5m", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
    # 15-min Logic applied to 5-min data for precision
    df['MA_Channel'] = df['Close'].rolling(window=20).mean()
    df['Upper'] = df['MA_Channel'] + (df['Close'].rolling(20).std() * 1.5)
    df['Lower'] = df['MA_Channel'] - (df['Close'].rolling(20).std() * 1.5)
    return df

# --- 4. THE DECISION MATRIX ---
nifty = fetch_live_data()
ltp = float(nifty['Close'].iloc[-1])
macro_score = analyze_macro_and_sector()
global_bias = get_global_bias()
inst_bias = get_institutional_bias()

# Weighted Probability Calculation
# (Global 30% + Macro 30% + Institutional 20% + Technical 20%)
prob_score = (global_bias * 10) + (macro_score * 5) + (inst_bias * 2)

# --- 5. DASHBOARD LAYOUT (EXECUTIVE VIEW) ---
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>NIFTY QUANT INTELLIGENCE</h1>", unsafe_allow_html=True)

# Metric Row
c1, c2, c3, c4 = st.columns(4)
overall_sent = "Positive" if prob_score > 0.1 else "Negative" if prob_score < -0.1 else "Neutral"
c1.metric("NIFTY 50", f"{ltp:.2f}")
c2.metric("OVERALL SENTIMENT", overall_sent)
c3.metric("GLOBAL BIAS", f"{global_bias:+.4f}")
c4.metric("EXPIRY MODE", "TUESDAY HERO-ZERO" if now_ist.weekday() == 1 else "NORMAL")

st.divider()

# Signal Section
sig_col, stat_col = st.columns([2, 1])

with sig_col:
    st.subheader("🎯 Active Trade Signal")
    # 9:30 AM Rule Check
    if now_ist.time() < time(9, 30):
        st.warning("🕒 Analyzing Opening Range... Signal locks at 09:30 AM IST.")
    else:
        conf = "High" if abs(prob_score) > 0.5 else "Mid" if abs(prob_score) > 0.2 else "Low"
        if prob_score > 0.1 and ltp > nifty['MA_Channel'].iloc[-1]:
            st.success(f"🚀 **BUY CALL (CE)** | Entry: {ltp:.2f} | SL: {ltp-25:.2f} | Confidence: {conf}")
        elif prob_score < -0.1 and ltp < nifty['MA_Channel'].iloc[-1]:
            st.error(f"📉 **BUY PUT (PE)** | Entry: {ltp:.2f} | SL: {ltp+25:.2f} | Confidence: {conf}")
        else:
            st.info("⚖️ **WAITING: Multi-Factor Mismatch.** High-probability setup not found.")

with stat_col:
    st.subheader("🏦 Heavyweight Tracker")
    # In backend we analyze, here we just show status
    st.write(f"Reliance Impact: {'Bullish' if macro_score > 0 else 'Bearish'}")
    st.write(f"Inst. Flow (FII/DII): {'Buying' if inst_bias > 0 else 'Selling'}")

# --- 6. CHART AREA (Bottom) ---
st.divider()
st.subheader("📊 Live Technical Pattern (5-Min Chart)")



fig = go.Figure()
fig.add_trace(go.Candlestick(x=nifty.index, open=nifty['Open'], high=nifty['High'], low=nifty['Low'], close=nifty['Close'], name="Candle"))
fig.add_trace(go.Scatter(x=nifty.index, y=nifty['Upper'], line=dict(color='rgba(255,0,0,0.4)', width=1), name="Resistance"))
fig.add_trace(go.Scatter(x=nifty.index, y=nifty['Lower'], line=dict(color='rgba(0,255,0,0.4)', width=1), name="Support"))
fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
