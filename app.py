import streamlit as st
import yfinance as yf
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Philip's Pocket Analyst", layout="wide")

st.title("📱 Philip's Pocket Analyst")
st.markdown("### Buy, Sell, or Hold Decision Engine")

# --- 1. The Scoring Engine ---
def analyze_stock(ticker):
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
        target_price = info.get('targetMeanPrice', None) # NEW: Analyst Target
        
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
            "Target Price": target_price, # Passing this to the UI
            "Upside %": upside,           # Passing this to the UI
            "Key Strengths": ", ".join(reasons)
        }
        
    except Exception as e:
        return None

# --- 2. Watchlist Management ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['NVDA', 'INTC', 'MSFT', 'F', 'GOOGL']

with st.sidebar:
    st.header("My Watchlist")
    new_ticker = st.text_input("Add Ticker (e.g., AMD):").upper()
    if st.button("Add"):
        if new_ticker and new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
            
    if st.button("Clear List"):
        st.session_state.watchlist = []

# --- 3. Main Dashboard ---
if st.button("🔄 Analyze Watchlist Now"):
    results = []
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(st.session_state.watchlist):
        data = analyze_stock(ticker)
        if data:
            results.append(data)
        progress_bar.progress((i + 1) / len(st.session_state.watchlist))
        
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Score", ascending=False)
        
        # Clean up price formatting for the table
        df['Display Price'] = df['Price'].apply(lambda x: f"${x:.2f}")
        
        # Display Table
        st.dataframe(
            df,
            column_order=("Ticker", "Verdict", "Score", "Trend", "Display Price", "Key Strengths"),
            hide_index=True,
            use_container_width=True
        )

        # --- NEW: Top Pick Deep Dive (Chart + Analyst Targets) ---
        st.markdown("---")
        
        top_stock = df.iloc[0]
        top_ticker = top_stock['Ticker']
        
        st.subheader(f"🏆 Deep Dive: {top_ticker}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**6-Month Price Trend**")
            chart_data = yf.Ticker(top_ticker).history(period="6mo")
            st.line_chart(chart_data['Close'])
            
        with col2:
            st.write("**Wall St. Analyst Targets**")
            
            # Use the data we already fetched!
            target = top_stock['Target Price']
            upside = top_stock['Upside %']
            price = top_stock['Price']
            
            if target and target > 0:
                # Color code the upside
                color = "normal"
                if upside > 10: color = "normal" # Streamlit metric handles green automatically for positive delta
                
                st.metric(
                    label="Average Price Target",
                    value=f"${target:.2f}",
                    delta=f"{upside:.2f}% Upside"
                )
                st.write(f"Analysts think {top_ticker} is worth **${target:.2f}**. It is currently trading at **${price:.2f}**.")
            else:
                st.warning("No analyst targets available for this stock.")
                
    else:
        st.warning("No data found.")
else:
    st.info("Click 'Analyze Watchlist Now' to start.")