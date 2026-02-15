import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from textblob import TextBlob
import requests
import xml.etree.ElementTree as ET

# --- Page Configuration ---
st.set_page_config(page_title="Philip's Pocket Analyst Pro", layout="wide")

# --- 1. Session State ---
if 'watchlists' not in st.session_state:
    st.session_state.watchlists = {
        'Default': ['NVDA', 'INTC', 'AMD', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'F', 'JPM', 'KO'],
        'High Growth': ['PLTR', 'SOFI', 'COIN', 'MSTR'],
        'Safe Dividend': ['SCHD', 'O', 'VIG', 'JNJ', 'PG']
    }
if 'active_list' not in st.session_state:
    st.session_state.active_list = 'Default'

if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None 

# --- 2. Sector Mapping ---
SECTOR_MAP = {
    'Technology': 'XLK', 'Financial Services': 'XLF', 'Healthcare': 'XLV',
    'Energy': 'XLE', 'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP',
    'Industrials': 'XLI', 'Utilities': 'XLU', 'Real Estate': 'XLRE',
    'Basic Materials': 'XLB', 'Communication Services': 'XLC'
}

# --- 3. Helper Functions ---
def create_chart(ticker):
    """Generates a professional Candlestick + Volume chart."""
    try:
        df = yf.Ticker(ticker).history(period="6mo")
        if df.empty: return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
            subplot_titles=(f"{ticker} Price Action", "Volume"), row_heights=[0.7, 0.3] 
        )
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name="Price"
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'], name="Volume", marker_color='rgba(100, 100, 255, 0.5)'
        ), row=2, col=1)
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, showlegend=False)
        return fig
    except:
        return go.Figure()

def get_sector_performance(sector, period="3mo"):
    etf = SECTOR_MAP.get(sector, 'SPY')
    try:
        hist = yf.Ticker(etf).history(period=period)
        if not hist.empty:
            start = hist['Close'].iloc[0]
            end = hist['Close'].iloc[-1]
            return ((end - start) / start) * 100, etf
    except:
        return 0, 'SPY'
    return 0, 'SPY'

def get_google_news(ticker):
    """Fallback: Fetches news from Google RSS if Yahoo fails."""
    try:
        # Standard Google News RSS Feed
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        
        headlines = []
        for item in root.findall('.//item')[:5]:
            title = item.find('title').text
            link = item.find('link').text
            # Clean up Google's title (often includes " - Publisher Name")
            title = title.split(' - ')[0]
            
            blob = TextBlob(title)
            score = blob.sentiment.polarity
            headlines.append({"title": title, "score": score, "link": link})
            
        return headlines
    except:
        return []

def get_news_sentiment(ticker):
    """Hybrid News Engine: Tries Yahoo, falls back to Google."""
    sentiment_score = 0
    headlines = []
    
    # 1. Try Yahoo Finance First
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news:
            for item in news[:5]:
                title = item.get('title', '')
                link = item.get('link', '#')
                if title:
                    blob = TextBlob(title)
                    score = blob.sentiment.polarity
                    headlines.append({"title": title, "score": score, "link": link})
    except:
        pass # Yahoo failed, move to fallback

    # 2. If Yahoo failed (empty list), use Google News
    if not headlines:
        headlines = get_google_news(ticker)
    
    # 3. Calculate Average Sentiment
    if headlines:
        total_score = sum(h['score'] for h in headlines)
        avg_score = total_score / len(headlines)
        return avg_score, headlines
    
    return 0, []

def analyze_stock(ticker):
    """Buy/Sell/Hold Logic + Sector Comparison."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="3mo")
        if history.empty: return None
            
        sma_50 = history['Close'].tail(50).mean()
        current_price = history['Close'].iloc[-1]
        start_price_3mo = history['Close'].iloc[0]
        stock_return = ((current_price - start_price_3mo) / start_price_3mo) * 100

        pe_ratio = info.get('forwardPE', 100)
        profit_margin = info.get('profitMargins', 0)
        rev_growth = info.get('revenueGrowth', 0)
        debt_to_equity = info.get('debtToEquity', None)
        target_price = info.get('targetMeanPrice', None)
        sector = info.get('sector', 'Unknown')
        
        sector_return, sector_etf = get_sector_performance(sector)
        alpha = stock_return - sector_return 
        
        score = 0
        reasons = []
        
        if pe_ratio < 20: score += 20; reasons.append("✅ Cheap")
        elif pe_ratio < 35: score += 10
        if profit_margin > 0.15: score += 30; reasons.append("✅ Profitable")
        elif profit_margin > 0.05: score += 15
        if rev_growth > 0.10: score += 20; reasons.append("✅ Growing")
        if debt_to_equity and debt_to_equity < 50: score += 10; reasons.append("🛡️ Safe Debt")
        if current_price > sma_50: score += 10; trend_status = "UP"
        else: score -= 10; trend_status = "DOWN"
        if alpha > 5: score += 10; reasons.append(f"🚀 Crushing {sector_etf}")
        elif alpha < -5: score -= 5; reasons.append(f"🐢 Lagging {sector_etf}")

        if score >= 80: verdict = "STRONG BUY"
        elif score >= 60: verdict = "BUY"
        elif score >= 40: verdict = "HOLD"
        else: verdict = "SELL"
        
        upside = 0
        if target_price and current_price > 0:
            upside = ((target_price - current_price) / current_price) * 100
            
        return {
            "Ticker": ticker.upper(), "Price": current_price, "Score": score,
            "Verdict": verdict, "Trend": trend_status, "Sector": sector,
            "Sector ETF": sector_etf, "Stock Return": stock_return,
            "Sector Return": sector_return, "Alpha": alpha,
            "Target Price": target_price, "Upside %": upside,
            "Key Strengths": ", ".join(reasons)
        }
    except Exception: return None

def get_valuation_models(ticker):
    """Calculates Graham Number and Peter Lynch Value."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('currentPrice', 0)
        eps = info.get('trailingEps', 0)
        book_value = info.get('bookValue', 0)
        growth_est = info.get('earningsGrowth', 0) 
        if growth_est is None: growth_est = 0.05 
        
        models = {}
        if eps and eps > 0 and book_value and book_value > 0:
            models['Graham Number'] = np.sqrt(22.5 * eps * book_value)
        else: models['Graham Number'] = None
            
        growth_rate_whole = min(growth_est * 100, 25) 
        if eps and eps > 0 and growth_rate_whole > 0:
            models['Peter Lynch Value'] = eps * growth_rate_whole
        else: models['Peter Lynch Value'] = None
            
        return models, price, growth_est
    except: return {}, 0, 0.05

def calculate_dcf(ticker, growth_rate, discount_rate=0.10, years=5):
    """Standard DCF Calculation."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fcf = info.get('freeCashFlow', info.get('operatingCashflows', None))
        shares = info.get('sharesOutstanding', 1)
        if fcf is None or shares is None: return None
        future_fcf = []
        for i in range(1, years + 1):
            fcf = fcf * (1 + growth_rate)
            future_fcf.append(fcf / ((1 + discount_rate) ** i))
        terminal_value = (fcf * (1 + 0.02)) / (discount_rate - 0.02)
        pv_terminal = terminal_value / ((1 + discount_rate) ** years)
        return (sum(future_fcf) + pv_terminal) / shares
    except: return None

# --- 4. Sidebar ---
with st.sidebar:
    st.title("📂 Portfolio Manager")
    watchlist_names = list(st.session_state.watchlists.keys())
    selected_list_name = st.selectbox("Select Watchlist", watchlist_names, index=watchlist_names.index(st.session_state.active_list))
    st.session_state.active_list = selected_list_name
    current_tickers = st.session_state.watchlists[st.session_state.active_list]
    
    st.code(", ".join(current_tickers))
    st.divider()
    new_ticker = st.text_input("Add Ticker").upper()
    if st.button("Add Stock"):
        if new_ticker and new_ticker not in current_tickers:
            st.session_state.watchlists[st.session_state.active_list].append(new_ticker)
            st.rerun()

# --- 5. Main Dashboard ---
st.title(f"📱 Pocket Analyst: {st.session_state.active_list}")

if not current_tickers:
    st.warning("Empty Watchlist!")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🔗 Risk & Sectors", "💎 Valuation Models", "🔙 Backtest"])

with tab1:
    results = []
    with st.spinner("Analyzing..."):
        for ticker in current_tickers:
            data = analyze_stock(ticker)
            if data: results.append(data)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Score", ascending=False)
        st.info("👇 Select a stock to view details (Selection syncs to Valuation Tab)")
        
        selection = st.dataframe(
            df,
            column_order=("Ticker", "Verdict", "Score", "Trend", "Price", "Sector", "Upside %"),
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # --- ROBUST SELECTION LOGIC ---
        if st.session_state.selected_ticker is None and not df.empty:
            st.session_state.selected_ticker = df.iloc[0]['Ticker']

        if selection.selection.rows:
            idx = selection.selection.rows[0]
            selected_row = df.iloc[idx]
            st.session_state.selected_ticker = selected_row['Ticker']
            
        if st.session_state.selected_ticker in df['Ticker'].values:
            display_row = df[df['Ticker'] == st.session_state.selected_ticker].iloc[0]
            
            ticker = display_row['Ticker']
            sector = display_row['Sector']
            etf = display_row['Sector ETF']
            st.divider()
            st.subheader(f"🏆 Deep Dive: {ticker} vs. {sector} ({etf})")
            c1, c2 = st.columns([2, 1])
            with c1: st.plotly_chart(create_chart(ticker), use_container_width=True)
            with c2:
                # --- NEW: AI SENTIMENT (Hybrid Engine) ---
                st.write("**🤖 AI News Sentiment**")
                
                sentiment, headlines = get_news_sentiment(ticker)
                
                if sentiment > 0.1: st.success(f"😊 Positive ({sentiment:.2f})")
                elif sentiment < -0.1: st.error(f"😡 Negative ({sentiment:.2f})")
                else: st.warning(f"😐 Neutral ({sentiment:.2f})")
                
                if headlines:
                    st.caption(f"Source: {'Google News' if 'google' in headlines[0]['link'] else 'Yahoo Finance'}")
                    st.write("**Latest Headlines:**")
                    for h in headlines[:3]:
                        icon = "🟢" if h['score'] > 0 else "🔴" if h['score'] < 0 else "⚪"
                        st.markdown(f"[{icon} {h['title']}]({h['link']})")
                else:
                    st.info("No recent news found.")

                st.write("---")
                alpha = display_row['Alpha']
                if alpha > 0: st.success(f"🚀 Beating Sector by {alpha:.1f}%")
                else: st.error(f"🐢 Lagging Sector by {abs(alpha):.1f}%")

with tab2: # Risk Tab
    st.subheader("⚠️ Sector Power Rankings")
    if results:
        sector_df = pd.DataFrame(results).groupby("Sector")['Score'].mean().reset_index().sort_values("Score", ascending=False)
        fig_sec = px.bar(
            sector_df, x="Score", y="Sector", orientation='h', color="Score", 
            color_continuous_scale=["red", "yellow", "green"], range_color=[0, 100], text_auto=True
        )
        st.plotly_chart(fig_sec, use_container_width=True)

with tab3: # VALUATION TAB
    val_ticker = st.session_state.selected_ticker
    if not val_ticker and current_tickers: val_ticker = current_tickers[0]

    st.subheader(f"💎 Triangulated Valuation: {val_ticker}")
    if val_ticker:
        models, price, implied_growth = get_valuation_models(val_ticker)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 1. Graham Number"); st.caption("Conservative")
            g = models.get('Graham Number')
            st.metric("Value", f"${g:.2f}", f"{((g-price)/price)*100:.1f}%") if g else st.warning("N/A")
        with c2:
            st.markdown("##### 2. Peter Lynch"); st.caption("Growth")
            l = models.get('Peter Lynch Value')
            st.metric("Value", f"${l:.2f}", f"{((l-price)/price)*100:.1f}%") if l else st.warning("N/A")
        with c3:
            st.markdown("##### 3. DCF"); st.caption("Intrinsic")
            dg = st.slider(f"Growth", 0.0, 0.25, float(implied_growth), 0.01)
            d = calculate_dcf(val_ticker, dg)
            st.metric("Value", f"${d:.2f}", f"{((d-price)/price)*100:.1f}%") if d else st.warning("N/A")

with tab4: # BACKTEST TAB
    st.subheader("📜 1-Year Backtest")
    if current_tickers:
        try:
            data = yf.download(current_tickers, period="1y", progress=False)
            if 'Close' in data.columns: st.line_chart(data['Close'])
            else: st.line_chart(data)
        except: st.error("Data error")