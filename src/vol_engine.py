import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

class DiscretionaryVolEngine:
    def __init__(self, ticker: str, start_date: str, end_date: str = None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.garch_vol = None
        self.lstm_vol = None
        
    def fetch_data(self):
        """Fetch historical daily price data via Yahoo Finance."""
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        else:
            df = df[['Close']]
            
        df = df.dropna()
        # Calculate annualized log returns
        df['Log_Returns'] = np.log(df / df.shift(1))
        df['Hist_Vol_30D'] = df['Log_Returns'].rolling(window=30).std() * np.sqrt(252)
        
        self.data = df.dropna()
        self.returns = self.data['Log_Returns'] * 100  # Scale returns for GARCH convergence
        return self.data

    def fit_garch(self, p=1, q=1):
        """Fit GARCH(1,1) model to extract conditional volatility forecast."""
        garch = arch_model(self.returns, vol='Garch', p=p, q=q, dist='normal')
        res = garch.fit(disp='off')
        
        # Extract annualized conditional volatility series
        cond_vol = res.conditional_volatility / 100 * np.sqrt(252)
        self.garch_vol = cond_vol
        
        # 1-Day Ahead Forecast
        forecast = res.forecast(horizon=1)
        next_day_vol = np.sqrt(forecast.variance.values[-1, :][0]) / 100 * np.sqrt(252)
        return next_day_vol

    def train_lstm_regime(self, lookback=30, epochs=15, batch_size=32):
        """Train an LSTM network to predict next-day realized volatility regime."""
        vol_series = (self.returns.abs() * np.sqrt(252) / 100).values.reshape(-1, 1)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_vol = scaler.fit_transform(vol_series)
        
        X, y = [], []
        for i in range(lookback, len(scaled_vol)):
            X.append(scaled_vol[i-lookback:i, 0])
            y.append(scaled_vol[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Simple LSTM Architecture
        model = Sequential([
            LSTM(32, return_sequences=False, input_shape=(X.shape[1], 1)),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Predict Next-Day Volatility
        last_sequence = scaled_vol[-lookback:].reshape(1, lookback, 1)
        pred_scaled = model.predict(last_sequence, verbose=0)
        lstm_pred_vol = scaler.inverse_transform(pred_scaled)[0][0]
        
        self.lstm_vol = lstm_pred_vol
        return lstm_pred_vol

    def generate_trader_signal(self, current_implied_vol: float):
        """
        Compare Market Implied Volatility (IV) against Model Forecasted Volatility (RV)
        to generate actionable discretionary options setups.
        """
        garch_forecast = self.fit_garch()
        lstm_forecast = self.train_lstm_regime()
        
        # Ensemble Forecast (weighted average)
        ensemble_forecast = 0.5 * garch_forecast + 0.5 * lstm_forecast
        
        vol_spread = current_implied_vol - ensemble_forecast
        
        # Define Discretionary Setup Logic
        if vol_spread > 0.05:  # IV is >5% overpriced vs Model
            bias = "SHORT VOLATILITY"
            tactics = ["Sell Credit Spread", "Iron Condor", "Short Straddle / Strangle"]
            rationale = (f"Market IV ({current_implied_vol:.1%}) is significantly higher than "
                         f"Model Forecast ({ensemble_forecast:.1%}). Volatility premium is rich.")
            
        elif vol_spread < -0.05:  # IV is >5% underpriced vs Model
            bias = "LONG VOLATILITY"
            tactics = ["Long Straddle / Strangle", "Calendar Spread", "Debit Volatility Spread"]
            rationale = (f"Market IV ({current_implied_vol:.1%}) is cheap relative to "
                         f"Model Forecast ({ensemble_forecast:.1%}). Option premium is underpriced.")
            
        else:
            bias = "NEUTRAL / DIRECTIONAL EXECUTION"
            tactics = ["Defined-Risk Directional Spreads", "Delta-Neutral Position Sizing"]
            rationale = "Implied Volatility is fairly valued relative to quantitative model forecasts."

        return {
            "Ticker": self.ticker,
            "Current_IV": current_implied_vol,
            "GARCH_Forecast": garch_forecast,
            "LSTM_Forecast": lstm_forecast,
            "Ensemble_Forecast": ensemble_forecast,
            "Vol_Spread": vol_spread,
            "Trader_Bias": bias,
            "Suggested_Structures": tactics,
            "Rationale": rationale
        }
