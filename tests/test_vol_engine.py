import pytest
import pandas as pd
import numpy as np
from src.vol_engine import DiscretionaryVolEngine


def test_discretionary_vol_engine_initialization():
    """Validates that DiscretionaryVolEngine initializes correctly with ticker and start_date."""
    engine = DiscretionaryVolEngine(ticker="NDAQ", start_date="2022-01-01")
    assert engine.ticker == "NDAQ"
    assert isinstance(engine.returns, pd.Series)
    assert not engine.returns.empty


def test_fit_garch():
    """Validates that fit_garch returns a valid next-day GARCH volatility forecast float."""
    engine = DiscretionaryVolEngine(ticker="NDAQ", start_date="2022-01-01")
    next_day_vol = engine.fit_garch()
    
    assert isinstance(next_day_vol, float)
    assert next_day_vol > 0.0


def test_generate_trader_signals():
    """Validates that generate_trader_signals returns the expected dictionary structure and keys."""
    engine = DiscretionaryVolEngine(ticker="NDAQ", start_date="2022-01-01")
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
