import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import matplotlib.pyplot as plt

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Nifty 50 AI Dashboard", layout="wide", page_icon="📈")

# Auto-refresh every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# 2. DATA FETCHING (with Cache to prevent API blocking)
@st.cache_data(ttl=200)
def fetch_data():
    # Fetch Nifty, S&P 500 (Global), and USDINR (Currency)
    nifty = yf.download("^NSEI", interval="15m", period="5d", auto_adjust=True, progress=False)
    sp500 = yf.download("^GSPC", interval="15m", period="5d", auto_adjust=True, progress=False)
    usdinr = yf.download("USDINR=X", interval="15m", period="5d", auto_adjust=True, progress=False)

    # Clean MultiIndex Columns if they exist
    for df in [nifty, sp500, usdinr]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
    
    # Feature Engineering
    # RSI
    delta = nifty['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    nifty['rsi'] = 100 - (100 / (1 + rs))
    
    # Global Correlations
    nifty['sp500_chg'] = sp500['Close'].pct_change().reindex(nifty.index, method='ffill')
    nifty['usd_chg'] = usdinr['Close'].pct_change().reindex(nifty.index, method='ffill')
    
    return nifty

def get_sentiment():
    analyzer = SentimentIntensityAnalyzer()
    try:
        ticker = yf.Ticker("^NSEI")
        news = getattr(ticker, 'news', [])
        if not news: return 0.0, ["No recent news found."]
        
        scores = []
        titles = []
        for n in news[:5]:
            title = n.get('title', '')
            if title:
                titles.append(title)
                scores.append(analyzer.polarity_scores(title)['compound'])
        
        avg_score = np.mean(scores) if scores else 0.0
        return avg_score, titles
    except Exception:
        return 0.0, ["Sentiment engine offline."]

# 3. DASHBOARD LOGIC
st.title("🚀 Nifty 50 Live AI Dashboard")
st.write(f"Refreshed at: {datetime.now().strftime('%H:%M:%S')} IST")

# Fetch Data
with st.spinner('Analyzing Market Waves...'):
    nifty = fetch_data()
    sentiment_score, headlines = get_sentiment()

# --- AI MODELING SECTION ---
# Prepare Target: Will price be higher in 3 candles (45 mins)?
nifty['target'] = (nifty['Close'].shift(-3) > nifty['Close']).astype(int)
feature_cols = ['rsi', 'sp500_chg', 'usd_chg']

# CLEANING: The Fix for the ValueError (Removing NaNs)
ml_ready = nifty[feature_cols + ['target']].dropna()

if len(ml_ready) > 30:
    X = ml_ready[feature_cols]
    y = ml_ready['target']
    
    # Train (excluding last 3 rows as they have no target yet)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X.iloc[:-3], y.iloc[:-3])
    
    # Predict on latest row
    latest_row = nifty[feature_cols].iloc[-1:].fillna(0)
    prob_up = model.predict_proba(latest_row)[0][1]
else:
    prob_up = 0.5 # Default to neutral if data is sparse

# --- UI LAYOUT ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("LTP", f"{nifty['Close'].iloc[-1]:.2f}")
col2.metric("RSI (14)", f"{nifty['rsi'].iloc[-1]:.1f}")
col3.metric("News Sentiment", f"{sentiment_score:+.2f}")
col4.metric("Bullish Prob", f"{prob_up*100:.1f}%")

# Signal Alert
if prob_up > 0.65:
    st.success("🎯 **SIGNAL: STRONG BUY** (AI is highly confident in an upward move)")
elif prob_up < 0.35:
    st.error("📉 **SIGNAL: STRONG SELL** (AI detects significant downward pressure)")
else:
    st.warning("⚖️ **SIGNAL: NEUTRAL** (Wait for better confirmation)")

# Charting
st.subheader("Interactive Price Chart")
fig, ax = mpf.plot(nifty.tail(50), type='candle', style='charles', 
                   mav=(20, 50), volume=False, returnfig=True, figsize=(12, 6))
st.pyplot(fig)

# News Ticker
st.subheader("📰 Live Market Intelligence")
for h in headlines:
    st.write(f"• {h}")

st.divider()
st.caption("Disclaimer: This is an AI-driven tool for educational purposes. Trading involves risk.")
