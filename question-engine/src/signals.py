import math
from datetime import date
from src.models import Review, FieldState, SUB_RATING_KEYS

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
