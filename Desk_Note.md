# 📝 DESK NOTE: Discretionary Volatility Arbitrage & Option Mispricing Engine

**To:** Discretionary Options & Macro Trading Desks  
**From:** Discretionary Trading & Quantitative Analysis  
**Date:** July 2026  
**Subject:** Utilizing Hybrid GARCH(1,1) + LSTM Forecasting Models for Discretionary Volatility Spread Trading  

---

## 1. Executive Summary

In high-volatility regimes, option Implied Volatility ($\sigma_{\text{implied}}$) frequently decouples from short-term Realized Volatility ($\sigma_{\text{realized}}$) due to market panic, structural dealer positioning, or demand for directional hedges. 

This decision-support tool combines a parametric **GARCH(1,1)** statistical model with a non-parametric **LSTM (Long Short-Term Memory)** neural network to predict 1-day to 5-day realized volatility regimes. By measuring the delta between current market Implied Volatility and the ensemble forecast, discretionary traders can systematically exploit overpriced option premium or acquire underpriced tail protection.

---

## 2. Quantitative Model Architecture

The tool executes a two-stage forecast engine before aggregating output into a single signal:

1. **GARCH(1,1) Variance Engine:**
   Captures long-term variance targeting, ARCH effects (recent shock persistence), and GARCH effects (volatility clustering):
   $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

2. **LSTM Neural Network Regime Detector:**
   Processes rolling 30-day sequences of historical absolute returns to detect non-linear volatility regime shifts (e.g., transitions from low-volatility mean reversion to high-volatility momentum).

3. **Ensemble Forecast Aggregation:**
   $$\sigma_{\text{forecast}} = 0.50 \cdot \sigma_{\text{GARCH}} + 0.50 \cdot \sigma_{\text{LSTM}}$$

---

## 3. Discretionary Execution Rules & Strategy Matrix

Instead of automated trade execution, the model feeds real-time alerts into the Streamlit desk interface. The discretionary trader evaluates market catalysts (earnings, CPI, macro announcements) and selects the appropriate strategy:

| Volatility Spread ($\sigma_{\text{implied}} - \sigma_{\text{forecast}}$) | Market State | Discretionary Strategy Selection | Execution Tactics |
| :--- | :--- | :--- | :--- |
| **Spread $> +5.0\%$** | **Rich IV (Overpriced)** | **Short Volatility / Harvest Premium** | Sell Short Straddles/Strangles, Iron Condors, Credit Spreads. |
| **Spread $< -5.0\%$** | **Cheap IV (Underpriced)** | **Long Volatility / Convexity Capture** | Buy Long Straddles/Strangles, Calendar Spreads, Debit Volatility Spreads. |
| **$-5.0\% \le \text{Spread} \le +5.0\%$** | **Fairly Valued** | **Directional / Delta-Neutral** | Execute directional spreads with delta-hedging and strict VaR bounds. |

---

## 4. Risk Controls & Portfolio Guardrails

* **Max Position Sizing:** Volatility trades must adhere to a Maximum Expected Loss constrained by portfolio Value-at-Risk (VaR) parameters.
* **Catalyst Risk:** Never hold naked short volatility positions through binary macro events (e.g., FOMC rate decisions) without defined-risk wing protection.
* **Stop-Loss Protocol:** If $\sigma_{\text{realized}}$ spikes beyond 1.5x the model forecast, close or delta-hedge open premium-selling legs immediately.
