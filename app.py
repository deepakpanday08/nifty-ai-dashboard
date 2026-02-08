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

# 1. PAGE SETUP
st.set_page_config(page_title="Self-Learning Nifty AI", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# 2. DATA ENGINE
@st.cache_data(ttl=200)
def fetch_data():
    nifty = yf.download("^NSEI", interval="15m", period="5d", auto_adjust=True, progress=False)
    sp500 = yf.download("^GSPC", interval="15m", period="5d", auto_adjust=True, progress=False)
    usdinr = yf.download("USDINR=X", interval="15m", period="5d", auto_adjust=True, progress=False)
    
    for df in [nifty, sp500, usdinr]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
    
    # Features
    delta = nifty['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    nifty['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    nifty['sp500_chg'] = sp500['Close'].pct_change().reindex(nifty.index, method='ffill')
    nifty['usd_chg'] = usdinr['Close'].pct_change().reindex(nifty.index, method='ffill')
    
    return nifty

# 3. FEEDBACK & LEARNING LOGIC
def evaluate_performance(df):
    """Checks past predictions vs actual outcomes."""
    # We define a 'Correct' prediction if the AI predicted UP and price went UP 3 candles later
    df['target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
    
    # Simulating backtest on the last 50 candles to see AI accuracy
    feature_cols = ['rsi', 'sp500_chg', 'usd_chg']
    data = df[feature_cols + ['target']].dropna()
    
    if len(data) > 40:
        # Split into training and testing
        train = data.iloc[:-10]
        test = data.iloc[-10:]
        
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(train[feature_cols], train['target'])
        
        preds = clf.predict(test[feature_cols])
        accuracy = (preds == test['target']).mean()
        return accuracy, clf, feature_cols
    return 0.5, None, feature_cols

# 4. DASHBOARD UI
st.title("🤖 Self-Learning Nifty 50 AI")
nifty = fetch_data()
accuracy, trained_model, features = evaluate_performance(nifty)

# Performance Header
col_acc1, col_acc2 = st.columns(2)
with col_acc1:
    st.metric("Model Learning Accuracy", f"{accuracy*100:.1f}%", 
              delta="Learning" if accuracy > 0.5 else "Calibrating")
with col_acc2:
    st.info(f"AI is currently learning from the last {len(nifty)} market intervals.")

# Prediction Section
if trained_model:
    latest_row = nifty[features].iloc[-1:].fillna(0)
    prob = trained_model.predict_proba(latest_row)[0][1]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("LTP", f"{nifty['Close'].iloc[-1]:.2f}")
    m2.metric("Bullish Confidence", f"{prob*100:.1f}%")
    
    if prob > 0.60:
        st.success("🎯 AI Feedback: High confidence in UPWARD move.")
    elif prob < 0.40:
        st.error("📉 AI Feedback: High confidence in DOWNWARD move.")
    else:
        st.warning("⚖️ AI Feedback: Neutral. Market is indecisive.")

# Chart
fig, ax = mpf.plot(nifty.tail(50), type='candle', style='charles', mav=(20, 50), returnfig=True, figsize=(10, 5))
st.pyplot(fig)
