import json, math, re
from pathlib import Path
import pandas as pd
from src.models import Property

AMENITY_COLS = [c for c in [
    "accessibility", "activities_nearby", "business_services", "conveniences",
    "family_friendly", "food_and_drink", "guest_services", "internet",
    "langs_spoken", "more", "outdoor", "parking", "spa", "things_to_do",
]]

def _parse_amenity_list(v):
    if not isinstance(v, str) or not v.strip():
        return []
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        # Some cells use single quotes; swap carefully.
        try: return json.loads(v.replace("'", '"'))
        except json.JSONDecodeError: return []

def _clean_text(v):
    if not isinstance(v, str): return None
    v = re.sub(r"\|MASK\|", "", v).strip()
    return v or None

def _nan_to_none(v):
    if v is None: return None
    if isinstance(v, float) and math.isnan(v): return None
    return v

def load_properties(path: str | Path) -> list[Property]:
    df = pd.read_csv(path)
    props = []
    for _, row in df.iterrows():
        amenities = {c: _parse_amenity_list(row.get(f"property_amenity_{c}"))
                     for c in AMENITY_COLS}
        props.append(Property(
            eg_property_id=row["eg_property_id"],
            city=_nan_to_none(row.get("city")),
            province=_nan_to_none(row.get("province")),
            country=_nan_to_none(row.get("country")),
            star_rating=_nan_to_none(row.get("star_rating")),
            guestrating_avg_expedia=_nan_to_none(row.get("guestrating_avg_expedia")),
            area_description=_clean_text(row.get("area_description")),
            property_description=_clean_text(row.get("property_description")),
            popular_amenities_list=_parse_amenity_list(row.get("popular_amenities_list")),
            amenities=amenities,
            check_in_start_time=_nan_to_none(row.get("check_in_start_time")),
            check_in_end_time=_nan_to_none(row.get("check_in_end_time")),
            check_out_time=_nan_to_none(row.get("check_out_time")),
            check_out_policy=_clean_text(row.get("check_out_policy")),
            pet_policy=_clean_text(row.get("pet_policy")),
            children_and_extra_bed_policy=_clean_text(row.get("children_and_extra_bed_policy")),
            check_in_instructions=_clean_text(row.get("check_in_instructions")),
            know_before_you_go=_clean_text(row.get("know_before_you_go")),
        ))
    return props
