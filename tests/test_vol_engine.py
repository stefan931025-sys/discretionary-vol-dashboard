import pytest
import pandas as pd
import numpy as np
from src.vol_engine import DiscretionaryVolEngine


def test_discretionary_vol_engine_initialization():
    """Validates that DiscretionaryVolEngine initializes correctly."""
    engine = DiscretionaryVolEngine(ticker="NDAQ", start_date="2022-01-01")
    assert engine.ticker == "NDAQ"


def test_fit_garch(monkeypatch):
    """Validates that fit_garch returns a valid next-day GARCH volatility forecast float."""
    engine = DiscretionaryVolEngine(ticker="NDAQ", start_date="2022-01-01")
    
    # Inject synthetic returns if yfinance returned None in CI
    dates = pd.date_range(start="2022-01-01", periods=200, freq="B")
    engine.returns = pd.Series(np.random.normal(0, 0.01, 200), index=dates)
    
    next_day_vol = engine.fit_garch()
    assert isinstance(next_day_vol, float)
    assert next_day_vol > 0.0


def test_generate_trader_signal():
    """Validates that generate_trader_signal returns the expected dictionary structure and keys."""
    engine = DiscretionaryVolEngine(ticker="NDAQ", start_date="2022-01-01")
    
    # Inject synthetic returns for robust CI execution
    dates = pd.date_range(start="2022-01-01", periods=200, freq="B")
    engine.returns = pd.Series(np.random.normal(0, 0.01, 200), index=dates)
    
    signal_output = engine.generate_trader_signal(current_implied_vol=0.25)
    
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
