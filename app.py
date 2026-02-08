import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# 1. PAGE SETUP
st.set_page_config(page_title="Nifty AI Live", layout="wide")

# This line makes the website REFRESH every 5 minutes automatically
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# 2. DATA FUNCTIONS
@st.cache_data(ttl=240) # Cache for 4 mins to stay fast
def get_analysis_data():
    # Fetch Data
    nifty = yf.download("^NSEI", interval="15m", period="5d", auto_adjust=True)
    sp500 = yf.download("^GSPC", interval="15m", period="5d", auto_adjust=True)
    usdinr = yf.download("USDINR=X", interval="15m", period="5d", auto_adjust=True)
    
    # Fix MultiIndex columns if present
    for df in [nifty, sp500, usdinr]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
    # Calculate RSI
    delta = nifty['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    nifty['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    # Global Changes
    nifty['sp500_chg'] = sp500['Close'].pct_change().reindex(nifty.index, method='ffill')
    nifty['usd_chg'] = usdinr['Close'].pct_change().reindex(nifty.index, method='ffill')
    
    return nifty

def get_sentiment():
    analyzer = SentimentIntensityAnalyzer()
    try:
        news = yf.Ticker("^NSEI").news
        scores = [analyzer.polarity_scores(n.get('title', ''))['compound'] for n in news[:10]]
        headlines = [n.get('title', '') for n in news[:5]]
        return np.mean(scores) if scores else 0.0, headlines
    except:
        return 0.0, ["News currently unavailable"]

# 3. DASHBOARD UI
st.title("📈 NIFTY 50 Live AI Dashboard")
st.write(f"Last Update: {datetime.now().strftime('%H:%M:%S')} (Auto-refreshes every 5m)")

nifty = get_analysis_data()
sentiment, headlines = get_sentiment()

# AI Logic (Simple Random Forest)
nifty['target'] = (nifty['Close'].shift(-3) > nifty['Close']).astype(int)
features = nifty[['rsi', 'sp500_chg', 'usd_chg']].dropna()
target = nifty['target'].loc[features.index]
model = RandomForestClassifier(n_estimators=50).fit(features[:-3], target[:-3])
prob = model.predict_proba(features.iloc[-1:])[0][1]

# Display Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("LTP", f"{nifty['Close'].iloc[-1]:.2f}")
col2.metric("RSI", f"{nifty['rsi'].iloc[-1]:.1f}")
col3.metric("Sentiment", f"{sentiment:+.2f}")
col4.metric("Bullish Prob", f"{prob*100:.1f}%")

# Signal Box
if prob > 0.65: st.success("🔥 SIGNAL: STRONG BUY")
elif prob < 0.35: st.error("❄️ SIGNAL: STRONG SELL")
else: st.warning("⚖️ SIGNAL: NEUTRAL")

# Chart
fig, ax = mpf.plot(nifty, type='candle', style='charles', returnfig=True, figsize=(12, 6))
st.pyplot(fig)

# News Ticker
st.info("📰 Latest Headlines: " + " | ".join(headlines))
