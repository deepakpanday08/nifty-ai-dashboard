import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, time
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. SETTINGS & WEIGHTS ---
st.set_page_config(page_title="NIFTY AI MASTER", layout="wide")
st_autorefresh(interval=60 * 1000, key="live_sync")
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

# --- NEW: SECTORAL WEIGHTAGE & MAPPING (Updated 2026 Weights) ---
NIFTY_CONSTITUENTS = {
    "FINANCE": {"HDFCBANK.NS": 13.2, "ICICIBANK.NS": 9.1, "AXISBANK.NS": 3.2},
    "OIL_GAS": {"RELIANCE.NS": 9.4, "ONGC.NS": 1.2, "BPCL.NS": 0.6},
    "IT": {"TCS.NS": 3.9, "INFY.NS": 5.1, "HCLTECH.NS": 1.5},
    "FMCG": {"ITC.NS": 3.8, "HINDUNILVR.NS": 2.2}
}

# Key Macro-to-Sector Relationships (Logic: News -> Target Sector -> Impact)
MACRO_IMPACT_MAP = {
    "crude oil": {"OIL_GAS": -1, "PAINTS": 1, "AVIATION": 1}, 
    "repo rate": {"FINANCE": -1, "AUTO": -1, "REALTY": -1}, 
    "dollar": {"IT": 1, "PHARMA": 1} 
}

# --- 2. BACKEND CALCULATIONS ---

def get_sectoral_sentiment():
    """Calculates weighted sentiment for the entire Nifty 50."""
    analyzer = SentimentIntensityAnalyzer()
    total_score = 0
    sector_report = {}

    for sector, stocks in NIFTY_CONSTITUENTS.items():
        s_score = 0
        for ticker, weight in stocks.items():
            try:
                # Fetch recent news for the specific heavyweight
                news = yf.Ticker(ticker).news
                comp_sentiment = 0
                for n in news[:2]: # Look at top 2 headlines
                    text = n['title'].lower()
                    score = analyzer.polarity_scores(text)['compound']
                    
                    # Apply Macro-Impact logic (The "Context Filter")
                    for keyword, impact_map in MACRO_IMPACT_MAP.items():
                        if keyword in text:
                            score *= impact_map.get(sector, 1)
                    comp_sentiment += score
                
                # Weight the sentiment by the stock's Nifty weightage
                weighted_impact = (comp_sentiment / 2) * (weight / 100)
                s_score += weighted_impact
            except: continue
        sector_report[sector] = s_score
        total_score += s_score
    
    return total_score, sector_report

def get_global_bias_text():
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

# --- 3. DASHBOARD UI ---
nifty = fetch_nifty()
ltp = float(nifty['Close'].iloc[-1])
bias_text, bias_color, bias_val = get_global_bias_text()
n_sentiment, s_report = get_sectoral_sentiment()

st.markdown(f"<h1 style='text-align: center; color: #00ffcc;'>NIFTY 50 AI TERMINAL</h1>", unsafe_allow_html=True)

# TOP METRICS
m1, m2, m3, m4 = st.columns(4)
m1.metric("CURRENT NIFTY", f"{ltp:.2f}")
m2.markdown(f"**GLOBAL BIAS** \n<span style='color:{bias_color}; font-size:20px;'>{bias_text}</span>", unsafe_allow_html=True)
m3.metric("SENTIMENT SCORE", f"{n_sentiment:+.4f}")
m4.metric("TIME (IST)", now_ist.strftime('%H:%M'))

st.divider()

# SIGNAL & SECTOR HEATMAP
sig_col, stat_col = st.columns([2, 1])

with sig_col:
    st.subheader("🎯 Active Trade Signal")
    
    # 9:30 AM Logic + Sentiment Filter
    if now_ist.time() < time(9, 30):
        st.info(f"⏳ **OPENING RANGE SCAN** - Locking entry at 09:30 AM IST.")
    else:
        # CONFLUENCE LOGIC: Technicals + Global + Sectoral News
        if bias_val > 0 and n_sentiment > 0.01 and ltp > nifty['MA20'].iloc[-1]:
            st.success(f"🚀 **BUY CALL (CE)** | Entry: {ltp:.2f} | Confidence: HIGH")
        elif bias_val < 0 and n_sentiment < -0.01 and ltp < nifty['MA20'].iloc[-1]:
            st.error(f"📉 **BUY PUT (PE)** | Entry: {ltp:.2f} | Confidence: HIGH")
        else:
            st.warning("⚖️ **WAITING** - Signals are currently conflicting (e.g., Global Bullish but News Bearish).")

with stat_col:
    st.subheader("🏦 Sectoral Strength")
    # Displays the impact of each sector based on your weightage
    for sector, score in s_report.items():
        color = "green" if score > 0 else "red" if score < 0 else "white"
        st.markdown(f"**{sector}:** <span style='color:{color}'>{score:+.4f}</span>", unsafe_allow_html=True)

# --- 4. CHART ---
st.divider()
fig = go.Figure(data=[go.Candlestick(x=nifty.index, open=nifty['Open'], high=nifty['High'], low=nifty['Low'], close=nifty['Close'])])
fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, title="Live 5-Min Nifty Chart")
st.plotly_chart(fig, use_container_width=True)
