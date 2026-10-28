import pytest
import pandas as pd
import numpy as np
from src.vol_engine import DiscretionaryVolEngine


def test_discretionary_vol_engine_initialization():
    """Validates that DiscretionaryVolEngine initializes correctly with returns and ticker."""
    dates = pd.date_range(start="2023-01-01", periods=150, freq="B")
    mock_returns = pd.Series(np.random.normal(0, 0.01, 150), index=dates)
    
    engine = DiscretionaryVolEngine(returns=mock_returns, ticker="NDAQ")
    assert engine.ticker == "NDAQ"
    assert isinstance(engine.returns, pd.Series)
    assert len(engine.returns) == 150


def test_fit_garch():
    """Validates that fit_garch returns a valid next-day GARCH volatility forecast float."""
    dates = pd.date_range(start="2023-01-01", periods=200, freq="B")
    mock_returns = pd.Series(np.random.normal(0, 0.01, 200), index=dates)
    
    engine = DiscretionaryVolEngine(returns=mock_returns, ticker="NDAQ")
    next_day_vol = engine.fit_garch()
    
    assert isinstance(next_day_vol, float)
    assert next_day_vol > 0.0


def test_generate_trader_signals():
    """Validates that generate_trader_signals returns the expected dictionary structure and keys."""
    dates = pd.date_range(start="2023-01-01", periods=200, freq="B")
    mock_returns = pd.Series(np.random.normal(0, 0.01, 200), index=dates)
    
    engine = DiscretionaryVolEngine(returns=mock_returns, ticker="NDAQ")
    signal_output = engine.generate_trader_signals(current_implied_vol=0.25)
    
    assert isinstance(signal_output, dict)
    assert "Ticker" in signal_output
    assert "CURRENT_IV" in signal_output
    assert "GARCH_Forecast" in signal_output
    assert "LSTM_Forecast" in signal_output
    assert "Ensemble_Forecast" in signal_output
    assert "Vol_Spread" in signal_output
    assert "Trader_Bias" in signal_output
    assert "Suggested_Structures" in signal_output
    assert "Rationale" in signal_output
