from __future__ import annotations
from datetime import date
from src.models import FieldState


def missing_score(fs) -> float:
    return 0.0 if fs.value_known else 1.0


def stale_score(fs, today: date, time_horizon_days: int = 180,
                high_velocity_topics: list[str] | None = None) -> float:
    if not fs.value_known:
        return 0.0
    # High-velocity topics use a shorter horizon (things change fast)
    effective_horizon = time_horizon_days
    if high_velocity_topics and fs.field_id.startswith("topic:"):
        topic_id = fs.field_id.split(":", 1)[1]
        if topic_id in high_velocity_topics:
            effective_horizon = time_horizon_days // 3
    time_term = 0.0
    if fs.last_confirmed_date:
        age = (today - fs.last_confirmed_date).days
        time_term = max(0.0, min(1.0, age / effective_horizon))
    drift_term = 0.0
    if (fs.short_ema is not None and fs.long_ema is not None
            and fs.mention_count >= 5
            and abs(fs.short_ema - fs.long_ema) > 0.5):
        drift_term = 0.5
    return min(1.0, time_term + drift_term)


def stale_score_detail(fs, today: date, time_horizon_days: int = 180,
                       high_velocity_topics: list[str] | None = None) -> dict:
    if not fs.value_known:
        return {"score": 0.0, "age_days": 0, "time_term": 0.0, "drift_term": 0.0}
    effective_horizon = time_horizon_days
    if high_velocity_topics and fs.field_id.startswith("topic:"):
        topic_id = fs.field_id.split(":", 1)[1]
        if topic_id in high_velocity_topics:
            effective_horizon = time_horizon_days // 3
    age_days = 0
    time_term = 0.0
    if fs.last_confirmed_date:
        age_days = (today - fs.last_confirmed_date).days
        time_term = max(0.0, min(1.0, age_days / effective_horizon))
    drift_term = 0.0
    if (fs.short_ema is not None and fs.long_ema is not None
            and fs.mention_count >= 5
            and abs(fs.short_ema - fs.long_ema) > 0.5):
        drift_term = 0.5
    return {
        "score": min(1.0, time_term + drift_term),
        "age_days": age_days,
        "time_term": time_term,
        "drift_term": drift_term,
    }


def coverage_gap_detail(fs, total_reviews: int = 0) -> dict:
    base = 1.0 / (1.0 + fs.mention_count)
    response_rate = 0.0
    if (fs.field_id.startswith("rating:")
            and fs.value_known
            and total_reviews >= 20
            and fs.mention_count > 0):
        response_rate = fs.mention_count / total_reviews
        if response_rate < 0.05:
            base = max(base, 0.85)
    return {"score": base, "response_rate": response_rate}


def coverage_gap_score(fs, total_reviews: int = 0) -> float:
    base = 1.0 / (1.0 + fs.mention_count)
    # Rating fields with very low response rates are effectively unknown
    if (fs.field_id.startswith("rating:")
            and fs.value_known
            and total_reviews >= 20
            and fs.mention_count > 0):
        response_rate = fs.mention_count / total_reviews
        if response_rate < 0.05:
            base = max(base, 0.85)
    return base


def cross_ref_score(fs, peer_states: dict[str, FieldState],
                    schema_to_topics: dict[str, list[str]]) -> float:
    """Cross-reference between description fields and review topics.

    Scenario 1 (verification gap): topic is unknown, but a related schema
    field has data — the listing claims it, nobody verified.
    Scenario 2 (listing gap): schema is empty, but related topics have
    positive guest mentions — guests say it exists, listing doesn't document it.
    Scenario 3 (contradiction): schema is populated AND a related topic has
    negative sentiment — the listing claims it, guests say it's wrong/gone/changed.
    """
    fid = fs.field_id

    # Scenario 1: topic fields check their related schema
    if fid.startswith("topic:"):
        topic_id = fid.split(":", 1)[1]
        for schema_key, topic_ids in schema_to_topics.items():
            if topic_id in topic_ids:
                schema_fs = peer_states.get(f"schema:{schema_key}")
                if schema_fs and schema_fs.value_known and not fs.value_known:
                    return 1.0
        return 0.0

    # Scenario 2: schema empty, related topics have positive mentions
    if fid.startswith("schema:"):
        schema_key = fid.split(":", 1)[1]
        topic_ids = schema_to_topics.get(schema_key, [])
        if not topic_ids or fs.value_known:
            return 0.0
        positive = 0
        for tid in topic_ids:
            tfs = peer_states.get(f"topic:{tid}")
            if tfs and tfs.value_known and tfs.mention_count > 0:
                # Negative sentiment ("no pool") doesn't count as evidence
                if tfs.short_ema is None or tfs.short_ema > -0.5:
                    positive += 1
        return min(1.0, positive / len(topic_ids)) if positive else 0.0

    return 0.0


def find_contradictions(
    peer_states: dict[str, FieldState],
    schema_to_topics: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Find (schema_key, topic_id) pairs where the listing is contradicted by reviews.

    A contradiction: schema field is populated but the mapped topic has
    clearly negative sentiment (short_ema < -0.5) from 3+ reviews.
    """
    results = []
    for schema_key, topic_ids in schema_to_topics.items():
        schema_fs = peer_states.get(f"schema:{schema_key}")
        if not schema_fs or not schema_fs.value_known:
            continue
        for tid in topic_ids:
            tfs = peer_states.get(f"topic:{tid}")
            if (tfs and tfs.value_known
                    and tfs.short_ema is not None
                    and tfs.short_ema < -0.5
                    and tfs.mention_count >= 3):
                results.append((schema_key, tid))
    return results
