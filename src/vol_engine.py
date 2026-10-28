import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from arch import arch_model

class DiscretionaryVolEngine:
    def __init__(self, ticker: str, start_date: str = "2022-01-01"):
        self.ticker = ticker
        self.start_date = start_date
        self.returns = self._fetch_data()

    def _fetch_data(self) -> pd.Series:
        try:
            df = yf.download(self.ticker, start=self.start_date, progress=False)
            if df.empty or "Close" not in df.columns:
                return None
            # Calculate daily log returns percentage
            prices = df["Close"].squeeze()
            returns = 100 * np.log(prices / prices.shift(1)).dropna()
            return returns
        except Exception:
            return None

    def fit_garch(self) -> float:
        """Fit GARCH(1,1) model to extract conditional volatility forecast."""
        if self.returns is None or self.returns.empty:
            raise RuntimeError("Cannot estimate model without data.")
        
        # Scale returns for numerical stability in GARCH optimization
        scaled_returns = self.returns * 10.0
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
        res = model.fit(disp='off')
        
        # 1-Day Ahead Forecast (rescale back)
        forecast = res.forecast(horizon=1)
        next_day_val = np.sqrt(forecast.variance.values[-1, :][0]) / 10.0 * np.sqrt(252)
        return float(next_day_val)

    def train_lstm_regime(self, lookback: int = 30) -> float:
        """Train an LSTM network to predict next-day realized volatility regime."""
        if self.returns is None or self.returns.empty:
            raise RuntimeError("Cannot estimate model without data.")

        vol_series = (self.returns.abs()).values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_vol = scaler.fit_transform(vol_series)

        X, y = [], []
        for i in range(lookback, len(scaled_vol)):
            X.append(scaled_vol[i-lookback:i, 0])
            y.append(scaled_vol[i, 0])

        X, y = np.array(X), np.array(y)
        if len(X) == 0:
            return 0.20 # Fallback default
            
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(12, return_sequences=False, input_shape=(X.shape[1], 1)),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        last_sequence = scaled_vol[-lookback:].reshape(1, lookback, 1)
        pred_scaled = model.predict(last_sequence, verbose=0)
        lstm_pred_val = scaler.inverse_transform(pred_scaled)[0][0]

        return float(lstm_pred_val)

    def generate_trader_signal(self, current_implied_vol: float) -> dict:
        """Compare Market Implied Volatility (IV) against Model Forecasted Volatility (RV)."""
        garch_forecast = self.fit_garch()
        lstm_forecast = self.train_lstm_regime()

        ensemble_forecast = 0.5 * garch_forecast + 0.5 * lstm_forecast
        vol_spread = (current_implied_vol - ensemble_forecast) * 100

        if vol_spread > 5.0:
            bias = "SHORT VOLATILITY"
            tactics = ["Sell Credit Spread", "Iron Condor", "Short Straddle / Strangle"]
            rationale = (f"Market IV ({current_implied_vol:.1%}) is significantly higher than "
                         f"Model Forecast ({ensemble_forecast:.1%}). Volatility premium is rich.")
        elif vol_spread < -5.0:
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
            "Garch_Forecast": garch_forecast,
            "LSTM_Forecast": lstm_forecast,
            "Ensemble_Forecast": ensemble_forecast,
            "Vol_Spread": vol_spread,
            "Trader_Bias": bias,
            "Suggested_Structures": tactics,
            "Rationale": rationale
        }
