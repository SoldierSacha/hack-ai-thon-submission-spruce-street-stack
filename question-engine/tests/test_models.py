from datetime import date
from src.models import Property, Review, FieldState, RatingBreakdown, TaxonomyTopic

def test_rating_breakdown_zero_is_none():
    r = RatingBreakdown.from_raw({"overall": 5, "checkin": 0, "service": 4})
    assert r.overall == 5
    assert r.checkin is None          # 0 in data means NULL
    assert r.service == 4

def test_property_accepts_missing_star_rating():
    p = Property(eg_property_id="abc", city="Pompei", country="Italy",
                 star_rating=None, guestrating_avg_expedia=8.4)
    assert p.star_rating is None

def test_field_state_ema_can_be_none_when_sparse():
    fs = FieldState(eg_property_id="abc", field_id="topic:wifi",
                    value_known=True, mention_count=2)
    assert fs.short_ema is None
    assert fs.long_ema is None
