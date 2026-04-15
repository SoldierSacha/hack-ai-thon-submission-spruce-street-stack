import math
from datetime import date
from pathlib import Path
from src.db import Repo
from src.models import Property, Review, FieldState, TaxonomyTopic, SUB_RATING_KEYS
from src.taxonomy import SCHEMA_DESCRIPTION_FIELDS, load_taxonomy

def ema_series(values: list[float], half_life: float) -> float | None:
    if not values: return None
    alpha = 1 - math.exp(-math.log(2) / half_life)
    s = values[0]
    for v in values[1:]:
        s = alpha * v + (1 - alpha) * s
    return s

def build_rating_field_states(reviews: list[Review]) -> list[FieldState]:
    """Emit one FieldState per (property_id, SUB_RATING_KEY) pair."""
    property_ids = sorted({r.eg_property_id for r in reviews})
    by_pid: dict[str, list[Review]] = {}
    for r in reviews:
        by_pid.setdefault(r.eg_property_id, []).append(r)
    for pid in by_pid:
        by_pid[pid].sort(key=lambda r: r.acquisition_date)

    out: list[FieldState] = []
    for pid in property_ids:
        prop_reviews = by_pid.get(pid, [])
        for key in SUB_RATING_KEYS:
            pairs: list[tuple[date, float]] = []
            for r in prop_reviews:
                v = getattr(r.rating, key)
                if v is not None:
                    pairs.append((r.acquisition_date, float(v)))
            count = len(pairs)
            if count == 0:
                out.append(FieldState(
                    eg_property_id=pid,
                    field_id=f"rating:{key}",
                    value_known=False,
                    mention_count=0,
                ))
            else:
                values = [v for _, v in pairs]
                last_confirmed = max(d for d, _ in pairs)
                out.append(FieldState(
                    eg_property_id=pid,
                    field_id=f"rating:{key}",
                    value_known=True,
                    last_confirmed_date=last_confirmed,
                    short_ema=ema_series(values, half_life=5),
                    long_ema=ema_series(values, half_life=30),
                    mention_count=count,
                ))
    return out


def build_schema_field_states(
    properties: list[Property],
    reviews: list[Review] | None = None,
) -> list[FieldState]:
    """For each property, emit one FieldState per SCHEMA_DESCRIPTION_FIELD."""
    max_date_by_pid: dict[str, date] = {}
    if reviews is not None:
        for r in reviews:
            cur = max_date_by_pid.get(r.eg_property_id)
            if cur is None or r.acquisition_date > cur:
                max_date_by_pid[r.eg_property_id] = r.acquisition_date

    out: list[FieldState] = []
    for prop in properties:
        last_confirmed = max_date_by_pid.get(prop.eg_property_id) if reviews is not None else None
        for schema_field in SCHEMA_DESCRIPTION_FIELDS:
            if schema_field.startswith("property_amenity_"):
                subkey = schema_field[len("property_amenity_"):]
                value = prop.amenities.get(subkey, [])
                value_known = bool(value)
            else:
                value = getattr(prop, schema_field, None)
                if value is None:
                    value_known = False
                elif isinstance(value, str):
                    value_known = bool(value and value.strip())
                else:
                    value_known = bool(value)
            out.append(FieldState(
                eg_property_id=prop.eg_property_id,
                field_id=f"schema:{schema_field}",
                value_known=value_known,
                last_confirmed_date=last_confirmed,
                short_ema=None,
                long_ema=None,
                mention_count=1 if value_known else 0,
                last_asked_date=None,
            ))
    return out


def build_topic_field_states(
    reviews: list[Review],
    review_tags: dict[str, list[dict]],
    topics: list[TaxonomyTopic],
) -> list[FieldState]:
    """For every (property_id, topic_id) pair, produce a FieldState from review tags."""
    if not reviews:
        return []

    review_by_id = {r.review_id: r for r in reviews}

    # (pid, topic_id) -> list of (acquisition_date, sentiment_float)
    groups: dict[tuple[str, str], list[tuple[date, float]]] = {}

    for review_id, tags in review_tags.items():
        review = review_by_id.get(review_id)
        if review is None:
            continue
        for tag in tags:
            field_id = tag.get("field_id", "")
            if not field_id.startswith("topic:"):
                continue
            if not tag.get("mentioned"):
                continue
            sentiment = tag.get("sentiment")
            if sentiment is None:
                continue
            topic_id = field_id[len("topic:"):]
            key = (review.eg_property_id, topic_id)
            groups.setdefault(key, []).append((review.acquisition_date, float(sentiment)))

    property_ids = sorted({r.eg_property_id for r in reviews})

    out: list[FieldState] = []
    for pid in property_ids:
        for topic in topics:
            key = (pid, topic.topic_id)
            pairs = groups.get(key)
            field_id = f"topic:{topic.topic_id}"
            if not pairs:
                out.append(FieldState(
                    eg_property_id=pid,
                    field_id=field_id,
                    value_known=False,
                    mention_count=0,
                    short_ema=None,
                    long_ema=None,
                    last_confirmed_date=None,
                ))
            else:
                pairs_sorted = sorted(pairs, key=lambda p: p[0])
                values = [v for _, v in pairs_sorted]
                last_confirmed = max(d for d, _ in pairs_sorted)
                out.append(FieldState(
                    eg_property_id=pid,
                    field_id=field_id,
                    value_known=True,
                    last_confirmed_date=last_confirmed,
                    short_ema=ema_series(values, half_life=5),
                    long_ema=ema_series(values, half_life=30),
                    mention_count=len(pairs_sorted),
                ))
    return out


def build_all_field_states(repo: Repo, taxonomy_yaml: str | Path) -> int:
    """
    Read properties, reviews, and review_tags from the repo; run the three
    field_state builders; upsert every resulting FieldState back to the repo.

    Every property in the repo receives a FieldState for every known field id
    (rating, schema, and topic) — even properties with no reviews yet. The
    rating and topic builders key off the review set, so we pad their output
    with "unknown" states for properties that had no reviews.

    Returns the total number of FieldStates written.
    """
    properties = repo.list_properties()
    reviews = [r for p in properties for r in repo.list_reviews_for(p.eg_property_id)]
    review_tags = repo.list_review_tags_for_all()
    topics = load_taxonomy(taxonomy_yaml)

    rating_states = build_rating_field_states(reviews)
    schema_states = build_schema_field_states(properties, reviews=reviews)
    topic_states = build_topic_field_states(reviews, review_tags, topics)

    # Pad ratings with unknown states for properties that had no reviews.
    rating_pids = {s.eg_property_id for s in rating_states}
    for prop in properties:
        if prop.eg_property_id in rating_pids:
            continue
        for key in SUB_RATING_KEYS:
            rating_states.append(FieldState(
                eg_property_id=prop.eg_property_id,
                field_id=f"rating:{key}",
                value_known=False,
                mention_count=0,
            ))

    # Pad topics with unknown states for properties that had no reviews.
    topic_pids = {s.eg_property_id for s in topic_states}
    for prop in properties:
        if prop.eg_property_id in topic_pids:
            continue
        for topic in topics:
            topic_states.append(FieldState(
                eg_property_id=prop.eg_property_id,
                field_id=f"topic:{topic.topic_id}",
                value_known=False,
                mention_count=0,
            ))

    total = 0
    for state in rating_states + schema_states + topic_states:
        repo.upsert_field_state(state)
        total += 1
    return total
