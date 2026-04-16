from datetime import date
from src.models import FieldState
from src.scoring import missing_score, stale_score, coverage_gap_score, find_contradictions

def test_missing_score_unknown_is_1():
    fs = FieldState(eg_property_id="p", field_id="rating:checkin",
                    value_known=False, mention_count=0)
    assert missing_score(fs) == 1.0

def test_missing_score_known_is_0():
    fs = FieldState(eg_property_id="p", field_id="rating:overall",
                    value_known=True, mention_count=100)
    assert missing_score(fs) == 0.0

def test_stale_score_clips_at_1():
    fs = FieldState(eg_property_id="p", field_id="topic:wifi",
                    value_known=True, mention_count=20,
                    last_confirmed_date=date(2024, 1, 1),
                    short_ema=0.5, long_ema=0.5)
    today = date(2025, 9, 9)
    assert stale_score(fs, today=today) == 1.0

def test_stale_score_sentiment_drift_bonus():
    fs = FieldState(eg_property_id="p", field_id="topic:wifi",
                    value_known=True, mention_count=20,
                    last_confirmed_date=date(2025, 9, 1),
                    short_ema=-0.8, long_ema=0.8)   # delta = 1.6 > 0.5
    today = date(2025, 9, 9)
    assert stale_score(fs, today=today) >= 0.5     # drift term kicks in

def test_coverage_gap_decay():
    fs0 = FieldState(eg_property_id="p", field_id="topic:wifi", mention_count=0)
    fs10 = FieldState(eg_property_id="p", field_id="topic:wifi", mention_count=10)
    assert coverage_gap_score(fs0) > coverage_gap_score(fs10)

def test_find_contradictions_detects_negative_topic_with_populated_schema():
    peer = {
        "schema:property_amenity_food_and_drink": FieldState(
            eg_property_id="p", field_id="schema:property_amenity_food_and_drink",
            value_known=True),
        "topic:breakfast": FieldState(
            eg_property_id="p", field_id="topic:breakfast",
            value_known=True, short_ema=-0.7, mention_count=10),
    }
    results = find_contradictions(peer, {"property_amenity_food_and_drink": ["breakfast"]})
    assert len(results) == 1
    assert results[0] == ("property_amenity_food_and_drink", "breakfast")

def test_find_contradictions_ignores_mildly_negative():
    """Sentiment of -0.4 is mixed, not a contradiction (threshold is -0.5)."""
    peer = {
        "schema:property_amenity_food_and_drink": FieldState(
            eg_property_id="p", field_id="schema:property_amenity_food_and_drink",
            value_known=True),
        "topic:breakfast": FieldState(
            eg_property_id="p", field_id="topic:breakfast",
            value_known=True, short_ema=-0.4, mention_count=10),
    }
    results = find_contradictions(peer, {"property_amenity_food_and_drink": ["breakfast"]})
    assert len(results) == 0

def test_find_contradictions_ignores_positive_sentiment():
    peer = {
        "schema:property_amenity_food_and_drink": FieldState(
            eg_property_id="p", field_id="schema:property_amenity_food_and_drink",
            value_known=True),
        "topic:breakfast": FieldState(
            eg_property_id="p", field_id="topic:breakfast",
            value_known=True, short_ema=0.5, mention_count=10),
    }
    results = find_contradictions(peer, {"property_amenity_food_and_drink": ["breakfast"]})
    assert len(results) == 0

def test_find_contradictions_ignores_empty_schema():
    peer = {
        "schema:property_amenity_outdoor": FieldState(
            eg_property_id="p", field_id="schema:property_amenity_outdoor",
            value_known=False),
        "topic:pool": FieldState(
            eg_property_id="p", field_id="topic:pool",
            value_known=True, short_ema=-0.8, mention_count=5),
    }
    results = find_contradictions(peer, {"property_amenity_outdoor": ["pool"]})
    assert len(results) == 0
