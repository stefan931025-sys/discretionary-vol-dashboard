[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/stefan931025-sys/discretionary-vol-dashboard)

# 📈 Discretionary Volatility Trading Dashboard & Option Mispricing Engine

An institutional-grade decision-support engine designed for discretionary options and volatility traders. The system models underlying asset variance using a hybrid **GARCH(1,1)** statistical model and an **LSTM (Long Short-Term Memory)** neural network to forecast realized volatility ($\sigma_{\text{forecast}}$), comparing it in real time against market Implied Volatility ($\sigma_{\text{implied}}$) to identify actionable option mispricing setups.

---

## 💡 Core Discretionary Trading Thesis

In discretionary options trading, models are not used for automated execution; rather, they serve as **quantitative decision-support systems**. 

* **Volatility Spread Metric:**
  $$\text{Vol Spread} = \sigma_{\text{implied}} - \sigma_{\text{forecast}}$$

* **Trader Execution Rules:**
  * **Vol Spread > +5.0% (Rich IV):** Market options are overpriced relative to forecasted realized volatility. The discretionary desk looks to **sell premium** (e.g., Short Straddles, Iron Condors, Credit Spreads).
  * **Vol Spread < -5.0% (Cheap IV):** Market options are underpriced relative to forecasted realized volatility. The discretionary desk looks to **buy volatility / tail risk** (e.g., Long Straddles, Calendar Spreads, Long Puts ahead of catalysts).
  * **Neutral Spread:** Delta-neutral or directional setups with strict risk controls.

---

## 🛠️ Architecture & Tech Stack

* **Language & Data:** Python 3.10+, `yfinance`, `pandas`, `numpy`
* **Quantitative Modeling:** `arch` (GARCH(1,1) Conditional Volatility)
* **Machine Learning:** `tensorflow` / `keras` (LSTM Volatility Regime Forecasting)
* **Interactive UI & Visualization:** `streamlit`, `plotly`

---

## 🚀 Local Setup & Running the Dashboard

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/stefan931025-sys/discretionary-vol-dashboard.git](https://github.com/stefan931025-sys/discretionary-vol-dashboard.git)
   cd discretionary-vol-dashboard
