import math
def ema_series(values: list[float], half_life: float) -> float | None:
    if not values: return None
    alpha = 1 - math.exp(-math.log(2) / half_life)
    s = values[0]
    for v in values[1:]:
        s = alpha * v + (1 - alpha) * s
    return s
