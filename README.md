# 📱 Philip's Pocket Analyst

A personal financial analysis tool built with Python and Streamlit. This app automates stock research by calculating a custom "Health Score" (0-100) based on fundamental data, technical trends, and Wall Street targets.

**Live Demo:** [Insert your Streamlit Share link here]

## 🚀 Key Features
* **Custom Scoring Engine:** instantly ranks stocks based on 5 key metrics:
    * **Valuation:** (P/E Ratio)
    * **Profitability:** (Profit Margins)
    * **Growth:** (Revenue Growth)
    * **Safety:** (Debt-to-Equity Ratio)
    * **Momentum:** (Price vs. 50-Day Moving Average)
* **Automated Verdict:** Generates a simple **BUY**, **SELL**, or **HOLD** recommendation.
* **Trend Scanner:** Visualizes the 6-month price trend and flags Bullish/Bearish momentum.
* **Wall St. Integration:** Compares current price to Analyst Price Targets to calculate potential upside.
* **Mobile Optimized:** Designed to run in a mobile browser for on-the-go analysis.

## 🛠️ How It Works (The Math)
The "Philip Score" aggregates points across these categories:

| Metric | Condition | Points |
| :--- | :--- | :--- |
| **Valuation** | P/E < 20 | +20 |
| **Profitability** | Margin > 15% | +30 |
| **Growth** | Revenue Growth > 10% | +20 |
| **Safety** | Debt/Equity < 50% | +10 |
| **Trend** | Price > 50-Day Avg | +20 |
| **Penalty** | Price < 50-Day Avg | -10 |

* **Total Score > 80:** 🟢 STRONG BUY
* **Total Score > 60:** 🔵 BUY
* **Total Score > 40:** 🟠 HOLD
* **Total Score < 40:** 🔴 SELL

## 💻 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/pocket-analyst.git](https://github.com/YOUR_USERNAME/pocket-analyst.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## 🔮 Future Roadmap
* [ ] **Phase 1:** Add Candlestick Charts & Volume indicators.
* [ ] **Phase 2:** Competitor Analysis & Sector Heatmaps.
* [ ] **Phase 3:** Dividend Calendar & Reinvestment Calculator.
* [ ] **Phase 4:** AI-powered Earnings Call Summaries.
