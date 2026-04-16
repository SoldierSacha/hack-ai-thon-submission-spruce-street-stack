from datetime import date
from pathlib import Path
from src.models import FieldState, TaxonomyTopic, ScoredField
from src.ranker import rank_fields, pick_k, build_field_cluster_map

WEIGHTS = "config/weights.yaml"
XREF = "config/cross_ref.yaml"

def _fs(property_id="p1", field_id="topic:wifi", value_known=False,
        mention_count=0, short_ema=None, long_ema=None, last_confirmed_date=None):
    return FieldState(
        eg_property_id=property_id, field_id=field_id,
        value_known=value_known, mention_count=mention_count,
        short_ema=short_ema, long_ema=long_ema,
        last_confirmed_date=last_confirmed_date,
    )

def test_rank_picks_missing_over_known():
    states = [
        _fs(field_id="rating:checkin", value_known=False),
        _fs(field_id="rating:overall", value_known=True, mention_count=100),
    ]
    ranked = rank_fields(
        property_id="p1", field_states=states,
        today=date(2025, 9, 1),
        field_cluster={}, weights_path=WEIGHTS, cross_ref_path=XREF,
    )
    assert ranked[0].field_state.field_id == "rating:checkin"
    assert ranked[0].composite > ranked[1].composite

def test_rank_filters_to_requested_property():
    states = [
        _fs(property_id="p1", field_id="rating:checkin", value_known=False),
        _fs(property_id="p2", field_id="rating:checkin", value_known=False),
    ]
    ranked = rank_fields(
        property_id="p1", field_states=states,
        today=date(2025, 9, 1),
        field_cluster={}, weights_path=WEIGHTS, cross_ref_path=XREF,
    )
    assert len(ranked) == 1
    assert ranked[0].field_state.eg_property_id == "p1"

def test_cross_ref_boosts_topic_when_schema_populated():
    """Scenario 1: schema has data but the mapped topic is unknown -> cross_ref fires."""
    states = [
        # Schema field for food_and_drink is populated
        _fs(field_id="schema:property_amenity_food_and_drink", value_known=True),
        # Topic breakfast is unknown (no reviews mention it)
        _fs(field_id="topic:breakfast", value_known=False),
        # Another missing topic with no cross-ref mapping
        _fs(field_id="topic:wifi", value_known=False),
    ]
    ranked = rank_fields(
        property_id="p1", field_states=states,
        today=date(2025, 9, 1),
        field_cluster={}, weights_path=WEIGHTS, cross_ref_path=XREF,
    )
    breakfast = next(sf for sf in ranked if sf.field_state.field_id == "topic:breakfast")
    wifi = next(sf for sf in ranked if sf.field_state.field_id == "topic:wifi")
    # Breakfast should score higher due to cross-ref boost
    assert breakfast.cross_ref == 1.0
    assert wifi.cross_ref == 0.0
    assert breakfast.composite > wifi.composite

def test_cross_ref_boosts_schema_when_topic_mentioned():
    """Scenario 2: schema is empty but related topic has positive mentions -> listing gap."""
    states = [
        # Schema food_and_drink is empty
        _fs(field_id="schema:property_amenity_food_and_drink", value_known=False),
        # But guests mention breakfast positively
        _fs(field_id="topic:breakfast", value_known=True, mention_count=10, short_ema=0.7),
    ]
    ranked = rank_fields(
        property_id="p1", field_states=states,
        today=date(2025, 9, 1),
        field_cluster={}, weights_path=WEIGHTS, cross_ref_path=XREF,
    )
    schema = next(sf for sf in ranked if sf.field_state.field_id == "schema:property_amenity_food_and_drink")
    assert schema.cross_ref > 0

def test_cross_ref_ignores_negative_sentiment():
    """Scenario 2 edge case: negative sentiment ('no pool') should not trigger listing gap."""
    states = [
        _fs(field_id="schema:property_amenity_outdoor", value_known=False),
        _fs(field_id="topic:pool", value_known=True, mention_count=5, short_ema=-0.8),
    ]
    ranked = rank_fields(
        property_id="p1", field_states=states,
        today=date(2025, 9, 1),
        field_cluster={}, weights_path=WEIGHTS, cross_ref_path=XREF,
    )
    schema = next(sf for sf in ranked if sf.field_state.field_id == "schema:property_amenity_outdoor")
    assert schema.cross_ref == 0.0

def test_cross_ref_no_boost_when_schema_populated_and_topic_known():
    """Contradictions are alerts, not score boosters — cross_ref stays 0.0."""
    states = [
        _fs(field_id="schema:property_amenity_outdoor", value_known=True),
        _fs(field_id="topic:pool", value_known=True, mention_count=10, short_ema=-0.6),
    ]
    ranked = rank_fields(
        property_id="p1", field_states=states,
        today=date(2025, 9, 1),
        field_cluster={}, weights_path=WEIGHTS, cross_ref_path=XREF,
    )
    pool = next(sf for sf in ranked if sf.field_state.field_id == "topic:pool")
    assert pool.cross_ref == 0.0  # contradiction detected separately, not via scoring

def test_rank_fields_returns_scored_field_objects():
    states = [
        _fs(field_id="rating:checkin", value_known=False),
        _fs(field_id="rating:overall", value_known=True, mention_count=100),
    ]
    ranked = rank_fields(
        property_id="p1", field_states=states,
        today=date(2025, 9, 1),
        field_cluster={"rating:checkin": "service", "rating:overall": "overall"},
        weights_path=WEIGHTS, cross_ref_path=XREF,
    )
    assert len(ranked) == 2
    sf = ranked[0]
    assert isinstance(sf, ScoredField)
    assert sf.field_state.field_id == "rating:checkin"
    assert sf.composite > 0
    assert sf.missing == 1.0
    assert sf.rank == 1
    assert sf.cluster == "service"

def test_pick_k_returns_one_when_second_score_too_low():
    ranked = [
        ScoredField(field_state=_fs(field_id="rating:checkin"), composite=0.6,
                     missing=1.0, stale=0.0, coverage=0.0, cross_ref=0.0, cluster="service"),
        ScoredField(field_state=_fs(field_id="rating:service"), composite=0.2,
                     missing=0.0, stale=0.0, coverage=0.0, cross_ref=0.0, cluster="service"),
    ]
    picks = pick_k(ranked, field_cluster={"rating:checkin": "service", "rating:service": "service"})
    assert len(picks) == 1
    assert picks[0].field_state.field_id == "rating:checkin"

def test_pick_k_dedupes_same_cluster():
    """If top 2 candidates are in the same cluster, the 2nd one is skipped."""
    ranked = [
        ScoredField(field_state=_fs(field_id="rating:roomcleanliness"), composite=0.8,
                     missing=1.0, stale=0.0, coverage=0.0, cross_ref=0.0, cluster="room"),
        ScoredField(field_state=_fs(field_id="rating:roomcomfort"), composite=0.7,
                     missing=1.0, stale=0.0, coverage=0.0, cross_ref=0.0, cluster="room"),
        ScoredField(field_state=_fs(field_id="rating:service"), composite=0.6,
                     missing=1.0, stale=0.0, coverage=0.0, cross_ref=0.0, cluster="service"),
    ]
    cluster = {
        "rating:roomcleanliness": "room",
        "rating:roomcomfort": "room",
        "rating:service": "service",
    }
    picks = pick_k(ranked, field_cluster=cluster, min_score_for_k2=0.4)
    assert len(picks) == 2
    ids = [p.field_state.field_id for p in picks]
    assert ids == ["rating:roomcleanliness", "rating:service"]

def test_pick_k_empty_input():
    assert pick_k([], field_cluster={}) == []

def test_build_field_cluster_map_covers_all_three_types():
    topics = [
        TaxonomyTopic(topic_id="wifi", label="WiFi", cluster_id="connectivity", question_hint="wifi"),
    ]
    m = build_field_cluster_map(topics)
    assert m["topic:wifi"] == "connectivity"
    assert m["rating:checkin"] == "service"
    assert m["schema:pet_policy"] == "policies"
