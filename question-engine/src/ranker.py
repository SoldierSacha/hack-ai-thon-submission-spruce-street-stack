from __future__ import annotations
from datetime import date
from pathlib import Path

import yaml

from src.models import FieldState, TaxonomyTopic, ScoredField
from src.scoring import missing_score, stale_score, stale_score_detail, coverage_gap_score, coverage_gap_detail, cross_ref_score


def rank_fields(
    *,
    property_id: str,
    field_states: list[FieldState],
    today: date,
    field_cluster: dict[str, str],
    weights_path: str = "config/weights.yaml",
    cross_ref_path: str = "config/cross_ref.yaml",
    total_reviews: int = 0,
) -> list[ScoredField]:
    """Score each field_state for `property_id` and return ScoredField objects sorted by descending score."""
    W = yaml.safe_load(Path(weights_path).read_text())
    xref = yaml.safe_load(Path(cross_ref_path).read_text())
    schema_to_topics = xref.get("schema_to_topics", {})
    high_velocity = xref.get("high_velocity_topics", [])

    # Build peer lookup for cross-ref scoring
    peer_states: dict[str, FieldState] = {}
    for fs in field_states:
        if fs.eg_property_id == property_id:
            peer_states[fs.field_id] = fs

    scored: list[ScoredField] = []
    for fs in field_states:
        if fs.eg_property_id != property_id:
            continue
        m = missing_score(fs)
        s_d = stale_score_detail(fs, today=today,
                                 time_horizon_days=W["time_horizon_days"],
                                 high_velocity_topics=high_velocity)
        s = s_d["score"]
        c_d = coverage_gap_detail(fs, total_reviews=total_reviews)
        c = c_d["score"]
        xr = cross_ref_score(fs, peer_states, schema_to_topics)
        composite = (
            W["w_missing"] * m
            + W["w_stale"] * s
            + W["w_coverage"] * c
            + W["w_cross_ref"] * xr
        )
        scored.append(ScoredField(
            field_state=fs, composite=composite,
            missing=m, stale=s, coverage=c, cross_ref=xr,
            cluster=field_cluster.get(fs.field_id, ""),
            stale_age_days=s_d["age_days"],
            stale_time_term=s_d["time_term"],
            stale_drift_term=s_d["drift_term"],
            coverage_response_rate=c_d["response_rate"],
        ))
    scored.sort(key=lambda x: -x.composite)
    for i, sf in enumerate(scored):
        sf.rank = i + 1
    return scored


def pick_k(
    ranked: list[ScoredField],
    field_cluster: dict[str, str],
    min_score_for_k2: float = 0.40,
) -> list[ScoredField]:
    """Pick at most 2 fields from a ranked list, skipping same-cluster duplicates for slot 2."""
    if not ranked:
        return []
    picks = [ranked[0]]
    first_cluster = ranked[0].cluster or field_cluster.get(ranked[0].field_state.field_id)
    for sf in ranked[1:]:
        if sf.composite < min_score_for_k2:
            break
        sf_cluster = sf.cluster or field_cluster.get(sf.field_state.field_id)
        if sf_cluster == first_cluster:
            continue
        picks.append(sf)
        break
    return picks


def build_field_cluster_map(topics: list[TaxonomyTopic]) -> dict[str, str]:
    """Map every field_id in the registry to its cluster.

    - topic fields: cluster from the TaxonomyTopic
    - rating fields: static map
    - schema fields: static map
    """
    # Static rating -> cluster map (grouping the 15 sub-rating keys into coherent clusters).
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
    # Static schema -> cluster map.
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
