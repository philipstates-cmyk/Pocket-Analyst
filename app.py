import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Philip's Pocket Analyst Pro", layout="wide")

# --- 1. Session State & Watchlist Management ---
# We upgraded this to support multiple lists (e.g., "High Risk", "Safe", "Tech")
if 'watchlists' not in st.session_state:
    st.session_state.watchlists = {
        'Default': ['NVDA', 'INTC', 'MSFT', 'F', 'GOOGL'],
        'High Growth': ['TSLA', 'COIN', 'PLTR'],
        'Safe Dividend': ['KO', 'JNJ', 'PG', 'VZ']
    }
if 'active_list' not in st.session_state:
    st.session_state.active_list = 'Default'

# --- 2. Helper Functions (YOUR CORE LOGIC) ---

def create_chart(ticker):
    """
    Generates a professional Candlestick + Volume chart using Plotly.
    (Preserved from your original code)
    """
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

    fig.update_layout(
        height=600, xaxis_rangeslider_visible=False, 
        showlegend=False, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def analyze_stock(ticker):
    """
    Your custom Buy/Sell/Hold logic.
    (Preserved exactly as you wrote it)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get historical data for Trend Analysis
        history = stock.history(period="3mo")
        if not history.empty:
            sma_50 = history['Close'].tail(50).mean()
            current_price = history['Close'].iloc[-1]
        else:
            sma_50 = 0
            current_price = info.get('currentPrice', 0)

        # Metrics
        pe_ratio = info.get('forwardPE', 100)
        profit_margin = info.get('profitMargins', 0)
        rev_growth = info.get('revenueGrowth', 0)
        debt_to_equity = info.get('debtToEquity', None)
        target_price = info.get('targetMeanPrice', None)
        
        score = 0
        reasons = []
        
        # Rule 1: Valuation
        if pe_ratio < 20: 
            score += 20
            reasons.append("✅ Cheap Valuation")
        elif pe_ratio < 35:
            score += 10
            
        # Rule 2: Profitability
        if profit_margin > 0.15: 
            score += 30
            reasons.append("✅ Cash Machine")
        elif profit_margin > 0.05:
            score += 15
        
        # Rule 3: Growth
        if rev_growth > 0.10: 
            score += 20
            reasons.append("✅ Fast Growing")
            
        # Rule 4: Debt Safety
        if debt_to_equity and debt_to_equity < 50:
            score += 10
            reasons.append("🛡️ Safe Debt")
            
        # Rule 5: The "Trend Scanner"
        if current_price > sma_50:
            score += 20
            reasons.append("📈 Bullish Trend")
            trend_status = "UP"
        else:
            score -= 10
            reasons.append("📉 Bearish Trend")
            trend_status = "DOWN"

        # Verdict
        if score >= 80: verdict = "STRONG BUY"
        elif score >= 60: verdict = "BUY"
        elif score >= 40: verdict = "HOLD"
        else: verdict = "SELL"
        
        # Calculate Upside
        upside = 0
        if target_price and current_price > 0:
            upside = ((target_price - current_price) / current_price) * 100
            
        return {
            "Ticker": ticker.upper(),
            "Price": current_price,
            "Score": score,
            "Verdict": verdict,
            "Trend": trend_status,
            "Target Price": target_price,
            "Upside %": upside,
            "Key Strengths": ", ".join(reasons)
        }
        
    except Exception as e:
        return None

def calculate_dcf(ticker, growth_rate=0.03, discount_rate=0.10, years=5):
    """New Helper: Automated DCF Calculation"""
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
        total_value = sum(future_fcf) + pv_terminal
        return total_value / shares
    except:
        return None

# --- 3. Sidebar: Portfolio Manager ---
with st.sidebar:
    st.title("📂 Portfolio Manager")
    
    # Watchlist Selector
    watchlist_names = list(st.session_state.watchlists.keys())
    selected_list_name = st.selectbox("Select Watchlist", watchlist_names, index=watchlist_names.index(st.session_state.active_list))
    st.session_state.active_list = selected_list_name
    
    current_tickers = st.session_state.watchlists[st.session_state.active_list]
    
    st.write(f"**Current Tickers:**")
    st.code(", ".join(current_tickers))
    
    st.divider()
    
    # Add Ticker
    new_ticker = st.text_input("Add Ticker").upper()
    if st.button("Add Stock"):
        if new_ticker and new_ticker not in current_tickers:
            st.session_state.watchlists[st.session_state.active_list].append(new_ticker)
            st.rerun()
            
    # Create New List
    st.divider()
    new_list_name = st.text_input("New Watchlist Name")
    if st.button("Create List"):
        if new_list_name and new_list_name not in st.session_state.watchlists:
            st.session_state.watchlists[new_list_name] = []
            st.session_state.active_list = new_list_name
            st.rerun()

# --- 4. Main Dashboard ---
st.title(f"📱 Pocket Analyst: {st.session_state.active_list}")

if not current_tickers:
    st.warning("This watchlist is empty! Add stocks from the sidebar.")
    st.stop()

# TABS for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis Board", "🔗 Correlation & Risk", "💎 Valuation (DCF)", "🔙 Backtest"])

# --- TAB 1: MAIN ANALYSIS (The "Master" View) ---
with tab1:
    # Run your custom analysis logic
    results = []
    # We use a spinner so the user knows it's working
    with st.spinner("Crunching the numbers..."):
        for ticker in current_tickers:
            data = analyze_stock(ticker)
            if data:
                results.append(data)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Score", ascending=False)
        
        # Display Table with Selection Enabled
        st.info("👇 Click a row below to see the Deep Dive Charts")
        
        selection = st.dataframe(
            df,
            column_order=("Ticker", "Verdict", "Score", "Trend", "Price", "Upside %", "Key Strengths"),
            hide_index=True,
            use_container_width=True,
            on_select="rerun",  # Triggers the drill-down
            selection_mode="single-row"
        )
        
        # --- THE "DEEP DIVE" SECTION (Drill Down) ---
        if selection.selection.rows:
            selected_index = selection.selection.rows[0]
            # Match the selected row to the DataFrame
            selected_row = df.iloc[selected_index]
            selected_ticker = selected_row['Ticker']
            
            st.divider()
            st.subheader(f"🏆 Deep Dive: {selected_ticker}")
            
            # Use YOUR existing chart function
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Technical Analysis (6 Months)**")
                fig = create_chart(selected_ticker)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.write("**Wall St. Analyst Targets**")
                target = selected_row['Target Price']
                upside = selected_row['Upside %']
                
                if target and target > 0:
                    st.metric(label="Analyst Price Target", value=f"${target:.2f}", delta=f"{upside:.2f}% Upside")
                    st.write(f"Consensus: Is {selected_ticker} worth **${target:.2f}**?")
                else:
                    st.warning("No analyst targets available.")
                
                st.write("---")
                st.write("**Pocket Analyst Verdict**")
                st.metric(label="Score", value=selected_row["Score"], delta=selected_row["Verdict"])
                st.write(f"**Strengths:** {selected_row['Key Strengths']}")

# --- TAB 2: RISK & CORRELATION ---
with tab2:
    st.subheader("⚠️ Risk & Correlation Matrix")
    st.write("Do your stocks move together? (Dark Red = High Correlation/Risk)")
    
    if len(current_tickers) > 1:
        # Fetch simple closing data for correlation
        data_hist = yf.download(current_tickers, period="1y", progress=False)['Close']
        corr_matrix = data_hist.corr()
        
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale='RdBu_r',
            origin='lower'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Add at least 2 stocks to see correlation.")

# --- TAB 3: DCF VALUATION ---
with tab3:
    st.subheader("💰 Automated Discounted Cash Flow (DCF)")
    st.write("Intrinsic Value estimate based on Free Cash Flow.")
    
    col_dcf1, col_dcf2 = st.columns(2)
    with col_dcf1:
        growth_input = st.slider("Assumed Growth Rate", 0.01, 0.20, 0.05)
    with col_dcf2:
        discount_input = st.slider("Discount Rate (Risk)", 0.05, 0.15, 0.09)
    
    dcf_results = []
    for ticker in current_tickers:
        # Fetch current price again for accuracy or reuse from tab1
        stock_price = df[df['Ticker'] == ticker]['Price'].values[0] if results else 0
        fair_value = calculate_dcf(ticker, growth_rate=growth_input, discount_rate=discount_input)
        
        status = "N/A"
        if fair_value:
            dcf_upside = ((fair_value - stock_price) / stock_price) * 100
            status = "Undervalued 🟢" if fair_value > stock_price else "Overvalued 🔴"
            
            dcf_results.append({
                "Ticker": ticker,
                "Current Price": f"${stock_price:.2f}",
                "Est. Fair Value": f"${fair_value:.2f}",
                "DCF Upside": f"{dcf_upside:.1f}%",
                "Verdict": status
            })
    
    if dcf_results:
        st.dataframe(pd.DataFrame(dcf_results))

# --- TAB 4: BACKTEST ---
with tab4:
    st.subheader("📜 1-Year Backtest")
    st.write("Hypothetical Return if you invested $10,000 evenly 1 year ago.")
    
    if len(current_tickers) > 0:
        data_backtest = yf.download(current_tickers, period="1y", progress=False)['Close']
        
        bt_results = []
        total_start = 0
        total_end = 0
        investment_per_stock = 10000 / len(current_tickers)
        
        for ticker in current_tickers:
            # Handle single vs multi-index series
            prices = data_backtest[ticker] if len(current_tickers) > 1 else data_backtest
            
            # Simple check for empty data
            if prices.empty: continue

            start_price = prices.iloc[0]
            end_price = prices.iloc[-1]
            shares = investment_per_stock / start_price
            end_value = shares * end_price
            
            total_start += investment_per_stock
            total_end += end_value
            
            roi = ((end_value - investment_per_stock) / investment_per_stock) * 100
            
            bt_results.append({
                "Ticker": ticker,
                "Start Price": f"${start_price:.2f}",
                "End Price": f"${end_price:.2f}",
                "Return (%)": f"{roi:.1f}%"
            })
            
        st.dataframe(pd.DataFrame(bt_results))
        
        total_roi = ((total_end - total_start) / total_start) * 100
        st.metric(label="Total Portfolio Return", value=f"${total_end:,.2f}", delta=f"{total_roi:.1f}%")