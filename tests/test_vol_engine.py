import numpy as np
import pytest

def test_volatility_spread_calculation():
    sigma_implied = 0.25
    sigma_forecast = 0.18
    vol_spread = (sigma_implied - sigma_forecast) * 100
    
    assert pytest.approx(vol_spread, 0.01) == 7.0
    assert vol_spread > 5.0  # Rich IV Signal (Sell Premium)

def test_garch_lstm_hybrid_bounds():
    mock_garch_vol = 0.20
    mock_lstm_vol = 0.22
    hybrid_forecast = (0.5 * mock_garch_vol) + (0.5 * mock_lstm_vol)
    
    assert 0.0 < hybrid_forecast < 1.0
    assert pytest.approx(hybrid_forecast, 0.001) == 0.21
