import numpy as np
import yaml
from datetime import date
from pathlib import Path
from src.models import FieldState, TaxonomyTopic, ScoredField
from src.ranker import rank_fields, pick_k, build_field_cluster_map

WEIGHTS = "config/weights.yaml"

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
        today=date(2025, 9, 1), topic_embeddings={}, review_embedding=None,
        field_cluster={}, weights_path=WEIGHTS,
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
        today=date(2025, 9, 1), topic_embeddings={}, review_embedding=None,
        field_cluster={}, weights_path=WEIGHTS,
    )
    assert len(ranked) == 1
    assert ranked[0].field_state.eg_property_id == "p1"

def test_rank_redundancy_penalty_for_similar_topic():
    """A topic with an embedding similar to the review gets penalized."""
    states = [
        _fs(field_id="topic:wifi", value_known=False),
        _fs(field_id="topic:pool", value_known=False),
    ]
    # Same base score for both; but topic:wifi's embedding is much more similar to the review.
    rev_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    topic_embs = {
        "wifi": np.array([1.0, 0.0, 0.0], dtype=np.float32),   # cos = 1.0
        "pool": np.array([0.0, 1.0, 0.0], dtype=np.float32),   # cos = 0.0
    }
    ranked = rank_fields(
        property_id="p1", field_states=states,
        today=date(2025, 9, 1), topic_embeddings=topic_embs,
        review_embedding=rev_emb, field_cluster={}, weights_path=WEIGHTS,
    )
    # pool should beat wifi because wifi is redundant with the review
    assert ranked[0].field_state.field_id == "topic:pool"

def test_rank_fields_returns_scored_field_objects():
    states = [
        _fs(field_id="rating:checkin", value_known=False),
        _fs(field_id="rating:overall", value_known=True, mention_count=100),
    ]
    ranked = rank_fields(
        property_id="p1", field_states=states,
        today=date(2025, 9, 1), topic_embeddings={}, review_embedding=None,
        field_cluster={"rating:checkin": "service", "rating:overall": "overall"},
        weights_path=WEIGHTS,
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
                     missing=1.0, stale=0.0, coverage=0.0, redundancy=0.0, cluster="service"),
        ScoredField(field_state=_fs(field_id="rating:service"), composite=0.2,
                     missing=0.0, stale=0.0, coverage=0.0, redundancy=0.0, cluster="service"),
    ]
    picks = pick_k(ranked, field_cluster={"rating:checkin": "service", "rating:service": "service"})
    assert len(picks) == 1
    assert picks[0].field_state.field_id == "rating:checkin"

def test_pick_k_dedupes_same_cluster():
    """If top 2 candidates are in the same cluster, the 2nd one is skipped."""
    ranked = [
        ScoredField(field_state=_fs(field_id="rating:roomcleanliness"), composite=0.8,
                     missing=1.0, stale=0.0, coverage=0.0, redundancy=0.0, cluster="room"),
        ScoredField(field_state=_fs(field_id="rating:roomcomfort"), composite=0.7,
                     missing=1.0, stale=0.0, coverage=0.0, redundancy=0.0, cluster="room"),
        ScoredField(field_state=_fs(field_id="rating:service"), composite=0.6,
                     missing=1.0, stale=0.0, coverage=0.0, redundancy=0.0, cluster="service"),
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
