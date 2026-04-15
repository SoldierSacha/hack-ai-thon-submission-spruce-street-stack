from datetime import date
from src.signals import ema_series, build_rating_field_states
from src.models import Review, RatingBreakdown, FieldState, SUB_RATING_KEYS

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


def _mkreview(pid, idx, when, **ratings):
    return Review(
        review_id=f"{pid}:{idx}",
        eg_property_id=pid,
        acquisition_date=when,
        rating=RatingBreakdown(**ratings),
    )

def test_rating_states_100pct_null_field_is_unknown():
    """A property whose reviews never filled 'checkin' must produce value_known=False."""
    reviews = [
        _mkreview("p1", 0, date(2024, 1, 1), overall=5),
        _mkreview("p1", 1, date(2024, 6, 1), overall=4),
    ]
    states = build_rating_field_states(reviews)
    checkin = next(s for s in states if s.field_id == "rating:checkin" and s.eg_property_id == "p1")
    assert checkin.value_known is False
    assert checkin.mention_count == 0
    assert checkin.short_ema is None
    assert checkin.long_ema is None
    assert checkin.last_confirmed_date is None

def test_rating_states_ema_for_known_field():
    """A property with some filled 'overall' ratings gets EMA + mention_count."""
    reviews = [
        _mkreview("p1", 0, date(2024, 1, 1), overall=5),
        _mkreview("p1", 1, date(2024, 2, 1), overall=4),
        _mkreview("p1", 2, date(2024, 3, 1), overall=5),
    ]
    states = build_rating_field_states(reviews)
    overall = next(s for s in states if s.field_id == "rating:overall" and s.eg_property_id == "p1")
    assert overall.value_known is True
    assert overall.mention_count == 3
    assert overall.short_ema is not None
    assert 4.0 <= overall.short_ema <= 5.0
    assert overall.last_confirmed_date == date(2024, 3, 1)

def test_rating_states_partition_properties_and_keys():
    """Two properties × 15 sub-rating keys = 30 states."""
    reviews = [
        _mkreview("pA", 0, date(2024, 1, 1), overall=5),
        _mkreview("pB", 0, date(2024, 1, 1), overall=4),
    ]
    states = build_rating_field_states(reviews)
    assert len(states) == 2 * len(SUB_RATING_KEYS)
    pids = {s.eg_property_id for s in states}
    assert pids == {"pA", "pB"}

def test_rating_states_last_confirmed_is_most_recent_nonnull_for_that_key():
    """Only rows with a non-None value for this key count toward last_confirmed_date."""
    reviews = [
        _mkreview("p1", 0, date(2024, 1, 1), overall=5, service=4),
        _mkreview("p1", 1, date(2024, 6, 1), overall=4),                  # no service
        _mkreview("p1", 2, date(2024, 12, 1), overall=5),                 # no service
    ]
    states = build_rating_field_states(reviews)
    overall = next(s for s in states if s.field_id == "rating:overall" and s.eg_property_id == "p1")
    service = next(s for s in states if s.field_id == "rating:service" and s.eg_property_id == "p1")
    assert overall.last_confirmed_date == date(2024, 12, 1)
    assert service.last_confirmed_date == date(2024, 1, 1)  # last time it was non-None
    assert service.mention_count == 1
