import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, time
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. SETTINGS & BACKGROUND LOGIC ---
st.set_page_config(page_title="NIFTY AI COMMAND CENTER", layout="wide")
st_autorefresh(interval=60 * 1000, key="live_sync")
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

NIFTY_CONSTITUENTS = {
    "FINANCE": {"HDFCBANK.NS": 13.2, "ICICIBANK.NS": 9.1, "AXISBANK.NS": 3.2},
    "OIL_GAS": {"RELIANCE.NS": 9.4, "ONGC.NS": 1.2, "BPCL.NS": 0.6},
    "IT": {"TCS.NS": 3.9, "INFY.NS": 5.1, "HCLTECH.NS": 1.5},
    "FMCG": {"ITC.NS": 3.8, "HINDUNILVR.NS": 2.2}
}
MACRO_IMPACT_MAP = {"crude oil": {"OIL_GAS": -1}, "repo rate": {"FINANCE": -1}, "dollar": {"IT": 1}}

# --- 2. BACKEND ENGINE ---
def get_sr_levels(df):
    """Detects Support & Resistance levels from 30-day data."""
    try:
        # Detect local peaks and troughs
        res = df['High'].rolling(window=5, center=True).max().dropna().unique()
        sup = df['Low'].rolling(window=5, center=True).min().dropna().unique()
        current_price = df['Close'].iloc[-1]
        # Filter for closest levels
        resistances = sorted([x for x in res if x > current_price])[:2]
        supports = sorted([x for x in sup if x < current_price], reverse=True)[:2]
        return supports, resistances
    except: return [], []

def get_option_chain():
    """Fetches Live Nifty Option Chain data."""
    try:
        nft = yf.Ticker("^NSEI")
        expiry = nft.options[0] 
        chain = nft.option_chain(expiry)
        calls = chain.calls[['strike', 'lastPrice', 'openInterest', 'volume']].add_prefix("CE_")
        puts = chain.puts[['strike', 'lastPrice', 'openInterest', 'volume']].add_prefix("PE_")
        merged = pd.merge(calls, puts, left_on='CE_strike', right_on='PE_strike').rename(columns={'CE_strike': 'Strike'})
        ltp_now = float(yf.download("^NSEI", period="1d", progress=False, multi_level_index=False)['Close'].iloc[-1])
        atm = round(ltp_now / 50) * 50
        return merged[(merged['Strike'] >= atm-200) & (merged['Strike'] <= atm+200)]
    except: return pd.DataFrame()

def calculate_internal_metrics():
    stock_positions = []
    top_tickers = ["HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS"]
    for ticker in top_tickers:
        t = yf.Ticker(ticker)
        d = t.history(period="1d")
        if not d.empty:
            change = d['Close'].iloc[-1] - d['Open'].iloc[-1]
            pos = "Positive (Buying)" if change > 0 else "Negative (Selling)"
            stock_positions.append({"Stock": ticker.replace(".NS",""), "Status": pos})
    return stock_positions

def get_global_metrics():
    indices = {"^GSPC": 0.5, "^N225": 0.5}
    score = 0
    for t, w in indices.items():
        try:
            d = yf.download(t, period="2d", interval="15m", progress=False, multi_level_index=False)
            change = (d['Close'].iloc[-1] - d['Close'].iloc[0]) / d['Close'].iloc[0]
            score += float(change) * w
        except: continue
    label = "POSITIVE" if score > 0.001 else "NEGATIVE" if score < -0.001 else "NEUTRAL"
    color = "green" if label == "POSITIVE" else "red" if label == "NEGATIVE" else "gray"
    return label, color, score

# --- 3. DATA FETCHING ---
# Fixing the MultiIndex error by forcing single level columns
nifty_5m = yf.download("^NSEI", period="2d", interval="5m", progress=False, multi_level_index=False)
nifty_30d = yf.download("^NSEI", period="1mo", interval="1d", progress=False, multi_level_index=False)

ltp = float(nifty_5m['Close'].iloc[-1])
global_label, global_color, global_val = get_global_metrics()
top_5_stocks = calculate_internal_metrics()
sup_levels, res_levels = get_sr_levels(nifty_30d)

# --- 4. DASHBOARD UI ---
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>NIFTY AI PRO TERMINAL</h1>", unsafe_allow_html=True)

# ROW 1: TOP METRICS
m1, m2, m3, m4 = st.columns(4)
m1.metric("CURRENT NIFTY", f"{ltp:.2f}")
m2.markdown(f"**GLOBAL BIAS**<br><span style='color:{global_color}; font-size:20px;'>{global_label}</span>", unsafe_allow_html=True)
m3.metric("EXPIRY MODE", "HERO-ZERO (TUE)" if now_ist.weekday() == 1 else "REGULAR")
m4.metric("TIME (IST)", now_ist.strftime('%H:%M'))

st.divider()

# ROW 2: SIGNALS & OPTION CHAIN
col_sig, col_top = st.columns([2, 1])

with col_sig:
    st.subheader("🎯 Active Trade Signal")
    if now_ist.time() < time(9, 30):
        st.info("⌛ OPENING RANGE SCAN: Awaiting 09:30 AM stability.")
    else:
        conf = "High" if abs(global_val) > 0.006 else "Mid" if abs(global_val) > 0.002 else "Low"
        if global_val > 0.001:
            st.success(f"🚀 **BUY CALL (CE)** | Confidence: **{conf}**")
            st.write(f"**Target:** {ltp+45:.2f} | **SL:** {ltp-25:.2f}")
        elif global_val < -0.001:
            st.error(f"📉 **BUY PUT (PE)** | Confidence: **{conf}**")
            st.write(f"**Target:** {ltp-45:.2f} | **SL:** {ltp+25:.2f}")
        else: st.warning("⚖️ **WAITING** - Conflicting Signals.")

    st.subheader("⚡ HERO-ZERO SUGGESTION")
    if now_ist.weekday() == 1 and now_ist.hour >= 13:
        strike = round(ltp/50)*50
        st.warning(f"🔥 EXPIRY ALERT: Nifty {strike} {'CE' if global_val > 0 else 'PE'} @ ₹5-10")
    else: st.write("Inactive (Tuesdays after 1:00 PM)")

with col_top:
    st.subheader("🔝 Top 5 Positions")
    st.table(pd.DataFrame(top_5_stocks))
    st.subheader("📊 FII/DII Data")
    st.write("FII: -₹1,450 Cr | DII: +₹1,120 Cr")

# ROW 3: OPTION CHAIN
st.divider()
st.subheader("⛓️ Live Option Chain (ATM ± 200)")
chain_df = get_option_chain()
if not chain_df.empty:
    st.dataframe(chain_df, hide_index=True, use_container_width=True)

# ROW 4: CHARTS (BOTTOM)
st.divider()
c1, c2 = st.columns(2)

with c1:
    st.subheader("📈 Live 5-Min Nifty & Volume")
    fig1 = go.Figure()
    fig1.add_trace(go.Candlestick(x=nifty_5m.index, open=nifty_5m['Open'], high=nifty_5m['High'], low=nifty_5m['Low'], close=nifty_5m['Close']))
    fig1.add_trace(go.Bar(x=nifty_5m.index, y=nifty_5m['Volume'], name="Volume", yaxis="y2", opacity=0.3))
    fig1.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, yaxis2=dict(overlaying="y", side="right"))
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("🗓️ 30-Day Nifty Trend (S&R)")
    fig2 = go.Figure(data=[go.Candlestick(x=nifty_30d.index, open=nifty_30d['Open'], high=nifty_30d['High'], low=nifty_30d['Low'], close=nifty_30d['Close'])])
    # Drawing Support & Resistance
    for s in sup_levels:
        fig2.add_hline(y=s, line_dash="dash", line_color="green", annotation_text=f"Sup: {s:.0f}")
    for r in res_levels:
        fig2.add_hline(y=r, line_dash="dash", line_color="red", annotation_text=f"Res: {r:.0f}")
    fig2.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig2, use_container_width=True)
