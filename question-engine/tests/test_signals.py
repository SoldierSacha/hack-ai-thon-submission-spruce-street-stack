from src.signals import ema_series

def test_ema_empty():
    assert ema_series([], half_life=5) is None

def test_ema_matches_closed_form():
    # constant stream of 1.0 → EMA converges to 1.0
    result = ema_series([1.0] * 100, half_life=5)
    assert abs(result - 1.0) < 1e-6

def test_ema_recency_weighted():
    # 10 negatives followed by 3 positives — short-half-life EMA should skew positive
    vals = [-1.0]*10 + [1.0]*3
    short = ema_series(vals, half_life=2)
    long = ema_series(vals, half_life=20)
    assert short > long
