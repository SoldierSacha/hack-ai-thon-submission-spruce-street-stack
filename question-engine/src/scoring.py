from datetime import date

def missing_score(fs) -> float:
    return 0.0 if fs.value_known else 1.0

def stale_score(fs, today: date, time_horizon_days: int = 180) -> float:
    if not fs.value_known:
        return 0.0  # unknown is handled by missing_score, not here
    time_term = 0.0
    if fs.last_confirmed_date:
        age = (today - fs.last_confirmed_date).days
        time_term = max(0.0, min(1.0, age / time_horizon_days))
    drift_term = 0.0
    if (fs.short_ema is not None and fs.long_ema is not None
            and fs.mention_count >= 5
            and abs(fs.short_ema - fs.long_ema) > 0.5):
        drift_term = 0.5
    return min(1.0, time_term + drift_term)

def coverage_gap_score(fs) -> float:
    return 1.0 / (1.0 + fs.mention_count)
