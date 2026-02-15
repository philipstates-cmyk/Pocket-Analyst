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
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        headlines = []
        for item in root.findall('.//item')[:5]:
            title = item.find('title').text
            link = item.find('link').text
            title = title.split(' - ')[0]
            blob = TextBlob(title)
            score = blob.sentiment.polarity
            headlines.append({"title": title, "score": score, "link": link})
        return headlines
    except:
        return []

def get_news_sentiment(ticker):
    headlines = []
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
    except: pass 

    if not headlines: headlines = get_google_news(ticker)
    
    if headlines:
        total_score = sum(h['score'] for h in headlines)
        avg_score = total_score / len(headlines)
        return avg_score, headlines
    return 0, []

def calculate_risk_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: return None

        hist['Returns'] = hist['Close'].pct_change()
        volatility = hist['Returns'].std() * np.sqrt(252) * 100
        
        hist['Cumulative'] = (1 + hist['Returns']).cumprod()
        hist['Peak'] = hist['Cumulative'].cummax()
        hist['Drawdown'] = (hist['Cumulative'] - hist['Peak']) / hist['Peak']
        max_drawdown = hist['Drawdown'].min() * 100
        
        mean_return = hist['Returns'].mean() * 252
        risk_free_rate = 0.04
        if hist['Returns'].std() > 0:
            sharpe_ratio = (mean_return - risk_free_rate) / (hist['Returns'].std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        return {"Volatility": volatility, "Max Drawdown": max_drawdown, "Sharpe Ratio": sharpe_ratio}
    except: return None

def analyze_stock(ticker):
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
        
        risk_data = calculate_risk_metrics(ticker)
        volatility = risk_data['Volatility'] if risk_data else 0
        drawdown = risk_data['Max Drawdown'] if risk_data else 0
        sharpe = risk_data['Sharpe Ratio'] if risk_data else 0

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

        # --- KEY INDICATORS ---
        trend_icon = "📈" if current_price > sma_50 else "📉"
        val_icon = "🏷️" if pe_ratio < 25 else "⚡"
        alpha_icon = "🏆" if alpha > 0 else "🐢"
        key_indicators = f"{trend_icon} {val_icon} {alpha_icon}"

        if score >= 80: verdict = "STRONG BUY"
        elif score >= 60: verdict = "BUY"
        elif score >= 40: verdict = "HOLD"
        else: verdict = "SELL"
        
        upside = 0
        if target_price and current_price > 0:
            upside = ((target_price - current_price) / current_price) * 100
            
        return {
            "Ticker": ticker.upper(), 
            "Price": current_price, 
            "Score": score,
            "Verdict": verdict, 
            "Key Indicators": key_indicators,
            "Trend": trend_status, 
            "Sector": sector,
            "Sector ETF": sector_etf, 
            "Stock Return": stock_return,
            "Sector Return": sector_return, 
            "Alpha": alpha,
            "Target Price": target_price, 
            "Upside %": upside,
            "Key Strengths": ", ".join(reasons),
            "Volatility": volatility, 
            "Max Drawdown": drawdown, 
            "Sharpe Ratio": sharpe
        }
    except Exception: return None

def get_valuation_models(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('currentPrice', 0)
        eps = info.get('trailingEps', 0)
        book_value = info.get('bookValue', 0)
        growth_est = info.get('earningsGrowth', 0) 
        
        target_mean = info.get('targetMeanPrice')
        target_low = info.get('targetLowPrice')
        target_high = info.get('targetHighPrice')
        num_analysts = info.get('numberOfAnalystOpinions')
        
        models = {}
        if eps and eps > 0 and book_value and book_value > 0:
            models['Graham Number'] = np.sqrt(22.5 * eps * book_value)
        else: models['Graham Number'] = None
            
        growth_rate_whole = min((growth_est or 0.05) * 100, 25) 
        if eps and eps > 0 and growth_rate_whole > 0:
            models['Peter Lynch Value'] = eps * growth_rate_whole
        else: models['Peter Lynch Value'] = None
            
        return models, price, growth_est, target_mean, target_low, target_high, num_analysts
    except: return {}, 0, 0.05, None, None, None, None

# --- 4. Sidebar: Portfolio Manager ---
with st.sidebar:
    st.title("📂 Portfolio Manager")

    # SECTION 1: WATCHLIST SELECTOR
    watchlist_names = list(st.session_state.watchlists.keys())
    selected_list_name = st.selectbox("Select Watchlist", watchlist_names, index=watchlist_names.index(st.session_state.active_list))
    st.session_state.active_list = selected_list_name
    current_tickers = st.session_state.watchlists[st.session_state.active_list]
    
    st.write(f"**Current: {len(current_tickers)} Stocks**")
    st.code(", ".join(current_tickers))
    
    st.divider()

    # SECTION 2: EDIT CURRENT LIST
    with st.expander("📝 Edit Current List", expanded=True):
        # ADD
        new_ticker = st.text_input("Add Ticker").upper()
        if st.button("Add ➕"):
            if new_ticker and new_ticker not in current_tickers:
                st.session_state.watchlists[st.session_state.active_list].append(new_ticker)
                st.rerun()
        
        # REMOVE
        if current_tickers:
            remove_ticker = st.selectbox("Remove Ticker", ["Select..."] + current_tickers)
            if st.button("Remove ➖"):
                if remove_ticker != "Select...":
                    st.session_state.watchlists[st.session_state.active_list].remove(remove_ticker)
                    st.rerun()

    # SECTION 3: CREATE NEW LIST
    with st.expander("➕ Create New Watchlist"):
        new_list_name = st.text_input("New List Name (e.g., 'Crypto')")
        if st.button("Create List"):
            if new_list_name and new_list_name not in st.session_state.watchlists:
                st.session_state.watchlists[new_list_name] = []
                st.session_state.active_list = new_list_name
                st.success(f"Created '{new_list_name}'!")
                st.rerun()

# --- 5. Main Dashboard ---
st.title(f"📱 Pocket Analyst: {st.session_state.active_list}")

if not current_tickers:
    st.info("👋 This watchlist is empty! Use the sidebar to add some tickers.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "⚠️ Risk & Sectors", "💎 Valuation", "🔙 Backtest"])

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
            column_order=("Ticker", "Verdict", "Key Indicators", "Score", "Price", "Sector", "Upside %"),
            hide_index=True,
            width='stretch',
            on_select="rerun",
            selection_mode="single-row",
            key="analysis_table"
        )
        
        if selection.selection.rows:
            idx = selection.selection.rows[0]
            selected_row = df.iloc[idx]
            st.session_state.selected_ticker = selected_row['Ticker']
            
        if st.session_state.selected_ticker is None and not df.empty:
            st.session_state.selected_ticker = df.iloc[0]['Ticker']
            
        if st.session_state.selected_ticker in df['Ticker'].values:
            display_row = df[df['Ticker'] == st.session_state.selected_ticker].iloc[0]
            
            ticker = display_row['Ticker']
            sector = display_row['Sector']
            etf = display_row['Sector ETF']
            st.divider()
            st.subheader(f"🏆 Deep Dive: {ticker} vs. {sector} ({etf})")
            c1, c2 = st.columns([2, 1])
            with c1: 
                st.plotly_chart(create_chart(ticker), width='stretch') 
            with c2:
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
                with st.expander("ℹ️ Key Indicators Legend"):
                    st.write("📈/📉: Price Trend")
                    st.write("🏷️/⚡: Valuation (Cheap vs Premium)")
                    st.write("🏆/🐢: Performance (Beating vs Lagging Sector)")
                
                alpha = display_row['Alpha']
                if alpha > 0: st.success(f"🚀 Beating Sector by {alpha:.1f}%")
                else: st.error(f"🐢 Lagging Sector by {abs(alpha):.1f}%")

with tab2: # RISK DASHBOARD
    val_ticker = st.session_state.selected_ticker
    if not val_ticker and current_tickers: val_ticker = current_tickers[0]
    
    st.subheader(f"⚠️ Risk Profile: {val_ticker}")
    if results:
        selected_risk_data = next((item for item in results if item["Ticker"] == val_ticker), None)
        if selected_risk_data:
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Volatility", f"{selected_risk_data['Volatility']:.1f}%")
            with c2: st.metric("Max Drawdown", f"{selected_risk_data['Max Drawdown']:.1f}%")
            with c3: st.metric("Sharpe Ratio", f"{selected_risk_data['Sharpe Ratio']:.2f}")

            if selected_risk_data['Sharpe Ratio'] < 1:
                st.warning("⚠️ Low Sharpe Ratio: Taking high risk for low return.")
            else:
                st.success("✅ Good Risk/Reward Balance.")
        
    st.divider()
    st.subheader("🏭 Portfolio Heatmap")
    if results:
        fig_tree = px.treemap(
            df, path=[px.Constant("Portfolio"), 'Sector', 'Ticker'], 
            values='Score', color='Score', color_continuous_scale='RdYlGn', color_continuous_midpoint=50
        )
        st.plotly_chart(fig_tree, width='stretch')
        
        st.caption("Full Risk Table:")
        risk_df = df[['Ticker', 'Volatility', 'Max Drawdown', 'Sharpe Ratio']].copy()
        risk_df['Volatility'] = risk_df['Volatility'].apply(lambda x: f"{x:.1f}%")
        risk_df['Max Drawdown'] = risk_df['Max Drawdown'].apply(lambda x: f"{x:.1f}%")
        risk_df['Sharpe Ratio'] = risk_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        st.dataframe(risk_df, hide_index=True, width='stretch')

with tab3: # VALUATION TAB
    val_ticker = st.session_state.selected_ticker
    if not val_ticker and current_tickers: val_ticker = current_tickers[0]

    st.subheader(f"💎 Valuation Reality Check: {val_ticker}")
    if val_ticker:
        models, price, implied_growth, t_mean, t_low, t_high, n_analysts = get_valuation_models(val_ticker)
        
        st.write("#### 1. What does Wall Street think?")
        if t_mean:
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Analyst Target", f"${t_mean:.2f}", f"{((t_mean-price)/price)*100:.1f}% Upside")
            with c2: st.metric("Lowest Target", f"${t_low:.2f}")
            with c3: st.metric("Highest Target", f"${t_high:.2f}")
            st.caption(f"Based on {n_analysts} analyst opinions.")
            
            if t_high > t_low:
                progress = (price - t_low) / (t_high - t_low)
                progress = max(0.0, min(1.0, progress)) 
                st.progress(progress)
                st.text(f"Low (${t_low}) <--- Current Price (${price:.2f}) ---> High (${t_high})")
        else:
            st.warning("No Analyst Coverage found for this stock.")

        st.divider()
        st.write("#### 2. What do the Math Models say?")
        m1, m2 = st.columns(2)
        with m1:
            st.info("**Graham Number (Conservative)**")
            g = models.get('Graham Number')
            if g: st.metric("Fair Value", f"${g:.2f}", f"{((g-price)/price)*100:.1f}%")
            else: st.warning("N/A (Unprofitable)")
            st.caption("Best for: Stable, boring companies (Banks, Utilities).")
            
        with m2:
            st.info("**Peter Lynch Value (Growth)**")
            l = models.get('Peter Lynch Value')
            if l: st.metric("Fair Value", f"${l:.2f}", f"{((l-price)/price)*100:.1f}%")
            else: st.warning("N/A (No Growth)")
            st.caption("Best for: Fast growers (Tech, Consumer Discretionary).")

with tab4: # BACKTEST TAB (SELECTED STOCK vs SPY)
    val_ticker = st.session_state.selected_ticker
    if not val_ticker and current_tickers: val_ticker = current_tickers[0]
    
    st.subheader(f"📜 Backtest: {val_ticker} vs. S&P 500 (1 Year)")
    
    if val_ticker:
        with st.spinner("Simulating the Fight..."):
            try:
                tickers_to_download = [val_ticker, 'SPY']
                data = yf.download(tickers_to_download, period="1y", progress=False)['Close']
                
                if not data.empty and val_ticker in data.columns and 'SPY' in data.columns:
                    fight_data = data[[val_ticker, 'SPY']].dropna()
                    
                    if not fight_data.empty:
                        normalized_data = (fight_data / fight_data.iloc[0]) * 10000
                        
                        fig_bt = px.line(
                            normalized_data, 
                            y=[val_ticker, 'SPY'],
                            labels={"value": "Investment Value ($)", "variable": "Ticker"},
                            color_discrete_map={val_ticker: "#00CC96", "SPY": "#EF553B"}
                        )
                        
                        fig_bt.add_hline(y=10000, line_dash="dot", line_color="white", opacity=0.5)
                        st.plotly_chart(fig_bt, width='stretch')
                        
                        stock_end = normalized_data[val_ticker].iloc[-1]
                        spy_end = normalized_data['SPY'].iloc[-1]
                        
                        c1, c2 = st.columns(2)
                        c1.metric(f"{val_ticker} Final Value", f"${stock_end:,.0f}", f"{((stock_end-10000)/10000)*100:.1f}% Return")
                        c2.metric("S&P 500 Final Value", f"${spy_end:,.0f}", f"{((spy_end-10000)/10000)*100:.1f}% Return")
                        
                        if stock_end > spy_end:
                            st.success(f"🚀 {val_ticker} beat the market by ${stock_end - spy_end:,.0f}!")
                        else:
                            st.error(f"🐢 {val_ticker} trailed the market by ${spy_end - stock_end:,.0f}.")
                    else:
                         st.warning(f"Not enough data overlap to compare {val_ticker} and SPY.")
                else:
                    st.warning(f"Could not retrieve backtest data for {val_ticker}. (Ticker might be new or delisted)")
            except Exception as e:
                st.error(f"Backtest Error: {e}")