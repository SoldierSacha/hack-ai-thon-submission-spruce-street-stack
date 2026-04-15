import math
from datetime import date
from src.models import Property, Review, FieldState, SUB_RATING_KEYS
from src.taxonomy import SCHEMA_DESCRIPTION_FIELDS

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
