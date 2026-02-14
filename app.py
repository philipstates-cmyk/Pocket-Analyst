import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

# --- 2. Sector Mapping ---
SECTOR_MAP = {
    'Technology': 'XLK',
    'Financial Services': 'XLF',
    'Healthcare': 'XLV',
    'Energy': 'XLE',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Industrials': 'XLI',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Basic Materials': 'XLB',
    'Communication Services': 'XLC'
}

# --- 3. Helper Functions ---
def create_chart(ticker):
    """Generates a professional Candlestick + Volume chart."""
    df = yf.Ticker(ticker).history(period="6mo")
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        subplot_titles=(f"{ticker} Price Action", "Volume"),
        row_heights=[0.7, 0.3] 
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name="Volume",
        marker_color='rgba(100, 100, 255, 0.5)'
    ), row=2, col=1)

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, showlegend=False)
    return fig

def get_sector_performance(sector, period="3mo"):
    """Fetches the performance of the Sector ETF."""
    etf = SECTOR_MAP.get(sector, 'SPY') # Default to S&P 500 if unknown
    try:
        hist = yf.Ticker(etf).history(period=period)
        if not hist.empty:
            start = hist['Close'].iloc[0]
            end = hist['Close'].iloc[-1]
            return ((end - start) / start) * 100, etf
    except:
        return 0, 'SPY'
    return 0, 'SPY'

def analyze_stock(ticker):
    """Your custom Buy/Sell/Hold logic + Sector Comparison."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Historical Data
        history = stock.history(period="3mo")
        if history.empty: return None
            
        sma_50 = history['Close'].tail(50).mean()
        current_price = history['Close'].iloc[-1]
        start_price_3mo = history['Close'].iloc[0]
        
        # Calculate Stock's 3-Month Return
        stock_return = ((current_price - start_price_3mo) / start_price_3mo) * 100

        # Metrics
        pe_ratio = info.get('forwardPE', 100)
        profit_margin = info.get('profitMargins', 0)
        rev_growth = info.get('revenueGrowth', 0)
        debt_to_equity = info.get('debtToEquity', None)
        target_price = info.get('targetMeanPrice', None)
        sector = info.get('sector', 'Unknown')
        
        # Sector Performance
        sector_return, sector_etf = get_sector_performance(sector)
        alpha = stock_return - sector_return 
        
        score = 0
        reasons = []
        
        # Rule 1: Valuation
        if pe_ratio < 20: 
            score += 20
            reasons.append("✅ Cheap")
        elif pe_ratio < 35:
            score += 10
            
        # Rule 2: Profitability
        if profit_margin > 0.15: 
            score += 30
            reasons.append("✅ Profitable")
        elif profit_margin > 0.05:
            score += 15
        
        # Rule 3: Growth
        if rev_growth > 0.10: 
            score += 20
            reasons.append("✅ Growing")
            
        # Rule 4: Debt Safety
        if debt_to_equity and debt_to_equity < 50:
            score += 10
            reasons.append("🛡️ Safe Debt")
            
        # Rule 5: Trend
        if current_price > sma_50:
            score += 10
            trend_status = "UP"
        else:
            score -= 10
            trend_status = "DOWN"
            
        # Sector Beater Rule
        if alpha > 5: 
            score += 10
            reasons.append(f"🚀 Crushing {sector_etf}")
        elif alpha < -5:
            score -= 5
            reasons.append(f"🐢 Lagging {sector_etf}")

        # Verdict
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
            "Trend": trend_status,
            "Sector": sector,
            "Sector ETF": sector_etf,
            "Stock Return": stock_return,
            "Sector Return": sector_return,
            "Alpha": alpha,
            "Target Price": target_price,
            "Upside %": upside,
            "Key Strengths": ", ".join(reasons)
        }
    except Exception:
        return None

def calculate_dcf(ticker, growth_rate=0.03, discount_rate=0.10, years=5):
    """Automated DCF Calculation"""
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
    except:
        return None

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
    st.warning("This watchlist is empty!")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis Board", "🔗 Risk & Sectors", "💎 Valuation", "🔙 Backtest"])

with tab1:
    results = []
    with st.spinner("Analyzing against Market Sectors..."):
        for ticker in current_tickers:
            data = analyze_stock(ticker)
            if data: results.append(data)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Score", ascending=False)
        
        st.info("👇 Select a stock below. (Defaults to Top Pick if none selected)")
        
        selection = st.dataframe(
            df,
            column_order=("Ticker", "Verdict", "Score", "Trend", "Price", "Sector", "Upside %"),
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # --- THE FIX: AUTO-SELECT TOP PICK ---
        selected_row = None
        
        # 1. Did the user click something?
        if selection.selection.rows:
            idx = selection.selection.rows[0]
            selected_row = df.iloc[idx]
        
        # 2. If not, auto-select the winner (Row 0)
        elif not df.empty:
            selected_row = df.iloc[0]
            
        # --- RENDER DEEP DIVE ---
        if selected_row is not None:
            ticker = selected_row['Ticker']
            sector = selected_row['Sector']
            etf = selected_row['Sector ETF']
            
            st.divider()
            st.subheader(f"🏆 Deep Dive: {ticker} vs. {sector} Sector ({etf})")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**Price Action**")
                st.plotly_chart(create_chart(ticker), use_container_width=True)
            
            with col2:
                st.write(f"**🆚 Performance (3-Month)**")
                
                s_ret = selected_row['Stock Return']
                e_ret = selected_row['Sector Return']
                alpha = selected_row['Alpha']
                
                st.metric(f"{ticker} Return", f"{s_ret:.1f}%")
                st.metric(f"Sector ({etf}) Return", f"{e_ret:.1f}%")
                
                if alpha > 0:
                    st.success(f"🚀 Beating {etf} by {alpha:.1f}%")
                else:
                    st.error(f"🐢 Lagging {etf} by {abs(alpha):.1f}%")
                
                st.write("---")
                if selected_row['Target Price']:
                    st.metric("Analyst Target", f"${selected_row['Target Price']:.2f}", f"{selected_row['Upside %']:.1f}% Upside")

with tab2: # Risk Tab (Preserved)
    st.subheader("⚠️ Sector Power Rankings")
    if results:
        # Group by Sector and calculate average score
        sector_df = pd.DataFrame(results).groupby("Sector")['Score'].mean().reset_index().sort_values("Score", ascending=False)
        fig_sec = px.bar(sector_df, x="Score", y="Sector", orientation='h', color="Score", color_continuous_scale=["red", "yellow", "green"], range_color=[0, 100], text_auto=True)
        st.plotly_chart(fig_sec, use_container_width=True)

with tab3: # DCF Tab (Preserved)
    st.subheader("💰 DCF Calculator")
    c1, c2 = st.columns(2)
    gr = c1.slider("Growth", 0.01, 0.20, 0.05)
    dr = c2.slider("Discount", 0.05, 0.15, 0.09)
    if results:
        dcf_d = []
        for r in results:
            fv = calculate_dcf(r['Ticker'], gr, dr)
            if fv: dcf_d.append({"Ticker": r['Ticker'], "Price": f"${r['Price']:.2f}", "Fair Value": f"${fv:.2f}", "Upside": f"{((fv-r['Price'])/r['Price'])*100:.1f}%"})
        st.dataframe(pd.DataFrame(dcf_d), use_container_width=True)

with tab4: # Backtest Tab (Preserved)
    st.subheader("📜 1-Year Backtest")
    if current_tickers:
        d_bt = yf.download(current_tickers, period="1y", progress=False)['Close']
        st.line_chart(d_bt)