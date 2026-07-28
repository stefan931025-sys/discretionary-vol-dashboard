import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.vol_engine import DiscretionaryVolEngine

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Discretionary Volatility Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- TITLE & DESK OVERVIEW ---
st.title("📈 Discretionary Volatility & Options Mispricing Engine")
st.markdown("""
*Decision-support system for discretionary traders. Compares implied option volatility against quantitative 
forecasts (GARCH + LSTM) to spot mispriced options structures.*
""")

st.divider()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Desk Parameters")

ticker = st.sidebar.text_input("Asset Ticker", value="SPY").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
implied_vol_input = st.sidebar.slider(
    "Current Market Implied Volatility (IV %)", 
    min_value=5.0, 
    max_value=80.0, 
    value=18.5, 
    step=0.5
) / 100.0

run_button = st.sidebar.button("Run Volatility Analysis", type="primary")

# --- MAIN ANALYSIS BLOCK ---
if run_button:
    with st.spinner(f"Fetching market data and running quantitative models for {ticker}..."):
        try:
            # Initialize & run engine
            engine = DiscretionaryVolEngine(ticker=ticker, start_date=str(start_date))
            df = engine.fetch_data()
            signal = engine.generate_trader_signal(current_implied_vol=implied_vol_input)

            # --- TOP METRICS CARDS ---
            st.subheader("📊 Volatility Metrics & Signal")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Current Market IV", f"{signal['Current_IV']:.1%}")
            col2.metric("GARCH(1,1) Forecast", f"{signal['GARCH_Forecast']:.1%}")
            col3.metric("LSTM Regime Forecast", f"{signal['LSTM_Forecast']:.1%}")
            
            # Highlight spread status
            spread_val = signal['Vol_Spread']
            col4.metric(
                "Vol Spread (IV - Model)", 
                f"{spread_val:+.1%}", 
                delta=f"{spread_val:+.1%}",
                delta_color="inverse"
            )

            st.divider()

            # --- TRADER EXECUTION CARD ---
            st.subheader("🎯 Discretionary Trade Execution Signal")
            
            if "SHORT" in signal['Trader_Bias']:
                st.error(f"**TRADER BIAS:** {signal['Trader_Bias']}")
            elif "LONG" in signal['Trader_Bias']:
                st.success(f"**TRADER BIAS:** {signal['Trader_Bias']}")
            else:
                st.info(f"**TRADER BIAS:** {signal['Trader_Bias']}")

            st.write(f"**Thesis / Rationale:** {signal['Rationale']}")
            
            st.markdown("**Suggested Options Structures:**")
            for tactic in signal['Suggested_Structures']:
                st.markdown(f"* 🔹 {tactic}")

            st.divider()

            # --- INTERACTIVE CHARTS ---
            st.subheader("📉 Historical Price & Volatility Trends")
            
            # Plot 1: Closing Price
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='#00F0FF')))
            fig_price.update_layout(
                title=f"{ticker} Asset Price Trajectory",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # Plot 2: Historical Volatility vs GARCH Conditional Volatility
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=df.index, y=df['Hist_Vol_30D'], name='30D Realized Vol (Hist)', line=dict(color='#FF9900')))
            fig_vol.add_trace(go.Scatter(x=engine.garch_vol.index, y=engine.garch_vol, name='GARCH(1,1) Conditional Vol', line=dict(color='#00FF66')))
            
            # Draw line for input Implied Volatility
            fig_vol.add_hline(
                y=implied_vol_input, 
                line_dash="dash", 
                line_color="red", 
                annotation_text=f"Market IV ({implied_vol_input:.1%})"
            )
            
            fig_vol.update_layout(
                title=f"{ticker} Volatility Dynamic Comparison",
                xaxis_title="Date",
                yaxis_title="Annualized Volatility",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while processing data: {str(e)}")

else:
    st.info("👈 Set your parameters in the sidebar and click **Run Volatility Analysis** to generate discretionary trading signals.")
