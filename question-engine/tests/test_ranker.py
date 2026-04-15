import numpy as np
import yaml
from datetime import date
from pathlib import Path
from src.models import FieldState, TaxonomyTopic
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
    assert ranked[0][0].field_id == "rating:checkin"
    assert ranked[0][1] > ranked[1][1]

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
    assert ranked[0][0].eg_property_id == "p1"

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
    assert ranked[0][0].field_id == "topic:pool"

def test_pick_k_returns_one_when_second_score_too_low():
    states = [
        (_fs(field_id="rating:checkin"), 0.6),
        (_fs(field_id="rating:service"), 0.2),   # below min_score
    ]
    picks = pick_k(states, field_cluster={"rating:checkin": "service", "rating:service": "service"})
    assert len(picks) == 1
    assert picks[0].field_id == "rating:checkin"

def test_pick_k_dedupes_same_cluster():
    """If top 2 candidates are in the same cluster, the 2nd one is skipped."""
    states = [
        (_fs(field_id="rating:roomcleanliness"), 0.8),
        (_fs(field_id="rating:roomcomfort"), 0.7),   # same cluster 'room' — skipped
        (_fs(field_id="rating:service"), 0.6),       # different cluster 'service' — picked
    ]
    cluster = {
        "rating:roomcleanliness": "room",
        "rating:roomcomfort": "room",
        "rating:service": "service",
    }
    picks = pick_k(states, field_cluster=cluster, min_score_for_k2=0.4)
    assert len(picks) == 2
    ids = [p.field_id for p in picks]
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
