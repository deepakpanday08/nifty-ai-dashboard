import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, time, timedelta
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. SETTINGS & BACKGROUND LOGIC ---
st.set_page_config(page_title="NIFTY AI COMMAND CENTER", layout="wide")
st_autorefresh(interval=60 * 1000, key="live_sync")
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

NIFTY_CONSTITUENTS = {
    "FINANCE": {"HDFCBANK.NS": 13.2, "ICICIBANK.NS": 9.1},
    "OIL_GAS": {"RELIANCE.NS": 9.4},
    "IT": {"TCS.NS": 3.9, "INFY.NS": 5.1}
}

# --- 2. BACKEND CALCULATIONS ---
def get_sr_levels(df):
    """Detects Support & Resistance levels from 30-day data."""
    highs = df['High'].values
    lows = df['Low'].values
    # Logic: Identify local max/min points
    res = df['High'].rolling(window=10, center=True).max().unique()
    sup = df['Low'].rolling(window=10, center=True).min().unique()
    # Clean up and pick top 2 closest to current price
    current_price = df['Close'].iloc[-1]
    resistances = sorted([x for x in res if x > current_price])[:2]
    supports = sorted([x for x in sup if x < current_price], reverse=True)[:2]
    return supports, resistances

def get_option_chain():
    """Fetches Nifty Option Chain data."""
    try:
        nft = yf.Ticker("^NSEI")
        expiry = nft.options[0]  # Get nearest expiry
        chain = nft.option_chain(expiry)
        calls = chain.calls[['strike', 'lastPrice', 'openInterest', 'volume']].rename(columns=lambda x: f"CE_{x}")
        puts = chain.puts[['strike', 'lastPrice', 'openInterest', 'volume']].rename(columns=lambda x: f"PE_{x}")
        merged = pd.merge(calls, puts, left_on='CE_strike', right_on='PE_strike').rename(columns={'CE_strike': 'Strike'})
        # Filter 10 strikes around ATM
        ltp_now = float(yf.download("^NSEI", period="1d", progress=False)['Close'].iloc[-1])
        atm_strike = round(ltp_now / 50) * 50
        return merged[(merged['Strike'] >= atm_strike-250) & (merged['Strike'] <= atm_strike+250)]
    except: return pd.DataFrame()

def calculate_internal_metrics():
    stock_positions = []
    top_tickers = ["HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS", "INFY.NS"]
    for ticker in top_tickers:
        try:
            d = yf.download(ticker, period="1d", progress=False)
            change = d['Close'].iloc[-1] - d['Open'].iloc[-1]
            stock_positions.append({"Stock": ticker.replace(".NS",""), "Status": "Buying" if change > 0 else "Selling"})
        except: continue
    return stock_positions

def get_global_metrics():
    indices = {"^GSPC": 0.5, "^N225": 0.5}
    score = 0
    for t, w in indices.items():
        try:
            d = yf.download(t, period="2d", interval="15m", progress=False)
            change = (d['Close'].iloc[-1] - d['Close'].iloc[0]) / d['Close'].iloc[0]
            score += float(change) * w
        except: continue
    label = "POSITIVE" if score > 0.001 else "NEGATIVE" if score < -0.001 else "NEUTRAL"
    color = "green" if label == "POSITIVE" else "red" if label == "NEGATIVE" else "gray"
    return label, color, score

# --- 3. DATA FETCHING ---
nifty_5m = yf.download("^NSEI", period="2d", interval="5m", progress=False)
nifty_30d = yf.download("^NSEI", period="1mo", interval="1d", progress=False)
ltp = float(nifty_5m['Close'].iloc[-1])
global_label, global_color, global_val = get_global_metrics()
top_5_stocks = calculate_internal_metrics()
sup_levels, res_levels = get_sr_levels(nifty_30d)

# --- 4. DASHBOARD UI ---
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>NIFTY AI PRO TERMINAL</h1>", unsafe_allow_html=True)

# ROW 1: TOP METRICS
m1, m2, m3, m4 = st.columns(4)
m1.metric("CURRENT NIFTY", f"{ltp:.2f}")
m2.markdown(f"**GLOBAL SENTIMENT**<br><span style='color:{global_color}; font-size:22px;'>{global_label}</span>", unsafe_allow_html=True)
m3.metric("EXPIRY MODE", "HERO-ZERO" if now_ist.weekday() == 1 else "REGULAR")
m4.metric("TIME (IST)", now_ist.strftime('%H:%M'))

st.divider()

# ROW 2: SIGNALS & OPTION CHAIN
col_signal, col_fii = st.columns([2, 1])
with col_signal:
    st.subheader("🎯 Active Trade Signal")
    if now_ist.time() < time(9, 30):
        st.info("⌛ OPENING RANGE SCAN: Awaiting 09:30 AM stability.")
    else:
        conf = "High" if abs(global_val) > 0.005 else "Mid"
        if global_val > 0.001:
            st.success(f"🚀 BUY CALL (CE) | Confidence: {conf}")
            st.write(f"Target: {ltp+40:.2f} | SL: {ltp-25:.2f}")
        elif global_val < -0.001:
            st.error(f"📉 BUY PUT (PE) | Confidence: {conf}")
            st.write(f"Target: {ltp-40:.2f} | SL: {ltp+25:.2f}")
        else: st.warning("⚖️ WAITING - No Trend Confluence")

    st.subheader("⛓️ Live Option Chain (ATM ± 250)")
    oc_data = get_option_chain()
    if not oc_data.empty:
        st.dataframe(oc_data, hide_index=True)

with col_fii:
    st.subheader("🔝 Top Stock Positions")
    st.table(pd.DataFrame(top_5_stocks))
    if now_ist.weekday() == 1 and now_ist.hour >= 13:
        st.subheader("⚡ HERO-ZERO")
        st.warning(f"Strike: {round(ltp/50)*50} | Target: 3X-5X")

# ROW 3: CHARTS
st.divider()
c1, c2 = st.columns(2)

with c1:
    st.subheader("📈 Live 5-Min Chart & Volume")
    fig1 = go.Figure(data=[go.Candlestick(x=nifty_5m.index, open=nifty_5m['Open'], high=nifty_5m['High'], low=nifty_5m['Low'], close=nifty_5m['Close'])])
    fig1.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("🗓️ 30-Day Trend + Support/Resistance")
    fig2 = go.Figure(data=[go.Candlestick(x=nifty_30d.index, open=nifty_30d['Open'], high=nifty_30d['High'], low=nifty_30d['Low'], close=nifty_30d['Close'])])
    # Add S&R Lines
    for s in sup_levels:
        fig2.add_hline(y=s, line_dash="dash", line_color="green", annotation_text="Support")
    for r in res_levels:
        fig2.add_hline(y=r, line_dash="dash", line_color="red", annotation_text="Resistance")
    fig2.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig2, use_container_width=True)
