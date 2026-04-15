from __future__ import annotations
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from src.models import FieldState, TaxonomyTopic
from src.scoring import missing_score, stale_score, coverage_gap_score


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def rank_fields(
    *,
    property_id: str,
    field_states: list[FieldState],
    today: date,
    topic_embeddings: dict[str, np.ndarray],
    review_embedding: np.ndarray | None,
    field_cluster: dict[str, str],
    weights_path: str = "config/weights.yaml",
) -> list[tuple[FieldState, float]]:
    """Score each field_state for `property_id` and return it sorted by descending score."""
    W = yaml.safe_load(Path(weights_path).read_text())
    scored: list[tuple[FieldState, float]] = []
    for fs in field_states:
        if fs.eg_property_id != property_id:
            continue
        m = missing_score(fs)
        s = stale_score(fs, today=today, time_horizon_days=W["time_horizon_days"])
        c = coverage_gap_score(fs)
        red = 0.0
        if review_embedding is not None and fs.field_id.startswith("topic:"):
            topic_id = fs.field_id.split(":", 1)[1]
            if topic_id in topic_embeddings:
                red = max(0.0, _cosine(review_embedding, topic_embeddings[topic_id]))
        score = (
            W["w_missing"] * m
            + W["w_stale"] * s
            + W["w_coverage"] * c
            - W["w_redundancy"] * red
            + W["epsilon"]
        )
        scored.append((fs, score))
    return sorted(scored, key=lambda x: -x[1])


def pick_k(
    ranked: list[tuple[FieldState, float]],
    field_cluster: dict[str, str],
    min_score_for_k2: float = 0.40,
) -> list[FieldState]:
    """Pick at most 2 fields from a ranked list, skipping same-cluster duplicates for slot 2."""
    if not ranked:
        return []
    picks = [ranked[0][0]]
    first_cluster = field_cluster.get(ranked[0][0].field_id)
    for fs, score in ranked[1:]:
        if score < min_score_for_k2:
            break
        if field_cluster.get(fs.field_id) == first_cluster:
            continue
        picks.append(fs)
        break
    return picks


def build_field_cluster_map(topics: list[TaxonomyTopic]) -> dict[str, str]:
    """Map every field_id in the registry to its cluster.

    - topic fields: cluster from the TaxonomyTopic
    - rating fields: static map
    - schema fields: static map
    """
    # Static rating → cluster map (grouping the 15 sub-rating keys into coherent clusters).
    rating_cluster = {
        "rating:overall": "overall",
        "rating:roomcleanliness": "room",
        "rating:roomcomfort": "room",
        "rating:roomquality": "room",
        "rating:roomamenitiesscore": "room",
        "rating:hotelcondition": "building",
        "rating:service": "service",
        "rating:checkin": "service",
        "rating:communication": "service",
        "rating:convenienceoflocation": "location",
        "rating:neighborhoodsatisfaction": "location",
        "rating:location": "location",
        "rating:valueformoney": "value",
        "rating:ecofriendliness": "meta",
        "rating:onlinelisting": "meta",
    }
    # Static schema → cluster map.
    schema_cluster = {
        "schema:pet_policy": "policies",
        "schema:children_and_extra_bed_policy": "policies",
        "schema:check_in_instructions": "service",
        "schema:know_before_you_go": "meta",
        "schema:check_out_policy": "service",
        "schema:property_amenity_accessibility": "building",
        "schema:property_amenity_activities_nearby": "location",
        "schema:property_amenity_business_services": "amenities",
        "schema:property_amenity_food_and_drink": "food",
        "schema:property_amenity_outdoor": "amenities",
        "schema:property_amenity_spa": "amenities",
        "schema:property_amenity_things_to_do": "location",
    }
    out: dict[str, str] = dict(rating_cluster)
    out.update(schema_cluster)
    for t in topics:
        out[f"topic:{t.topic_id}"] = t.cluster_id
    return out
