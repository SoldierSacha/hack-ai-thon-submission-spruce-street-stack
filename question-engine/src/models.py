from __future__ import annotations
from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field

SUB_RATING_KEYS = (
    "overall", "roomcleanliness", "service", "roomcomfort", "hotelcondition",
    "roomquality", "convenienceoflocation", "neighborhoodsatisfaction",
    "valueformoney", "roomamenitiesscore", "communication", "ecofriendliness",
    "checkin", "onlinelisting", "location",
)

class RatingBreakdown(BaseModel):
    overall: Optional[int] = None
    roomcleanliness: Optional[int] = None
    service: Optional[int] = None
    roomcomfort: Optional[int] = None
    hotelcondition: Optional[int] = None
    roomquality: Optional[int] = None
    convenienceoflocation: Optional[int] = None
    neighborhoodsatisfaction: Optional[int] = None
    valueformoney: Optional[int] = None
    roomamenitiesscore: Optional[int] = None
    communication: Optional[int] = None
    ecofriendliness: Optional[int] = None
    checkin: Optional[int] = None
    onlinelisting: Optional[int] = None
    location: Optional[int] = None

    @classmethod
    def from_raw(cls, d: dict | None) -> "RatingBreakdown":
        d = d or {}
        cleaned = {k: (v if isinstance(v, int) and v > 0 else None)
                   for k, v in d.items() if k in SUB_RATING_KEYS}
        return cls(**cleaned)

class Property(BaseModel):
    eg_property_id: str
    city: Optional[str] = None
    province: Optional[str] = None
    country: Optional[str] = None
    star_rating: Optional[float] = None
    guestrating_avg_expedia: Optional[float] = None
    area_description: Optional[str] = None
    property_description: Optional[str] = None
    amenities: dict[str, list[str]] = Field(default_factory=dict)  # subcategory -> items
    popular_amenities_list: list[str] = Field(default_factory=list)
    check_in_start_time: Optional[str] = None
    check_in_end_time: Optional[str] = None
    check_out_time: Optional[str] = None
    check_out_policy: Optional[str] = None
    pet_policy: Optional[str] = None
    children_and_extra_bed_policy: Optional[str] = None
    check_in_instructions: Optional[str] = None
    know_before_you_go: Optional[str] = None

class Review(BaseModel):
    review_id: str                        # synthetic: f"{property_id}:{row_index}"
    eg_property_id: str
    acquisition_date: date
    lob: Optional[str] = None
    rating: RatingBreakdown
    review_title: Optional[str] = None
    review_text_orig: Optional[str] = None
    review_text_en: Optional[str] = None  # filled by enrichment
    lang: Optional[str] = None
    source: Literal["csv", "live"] = "csv"

class TaxonomyTopic(BaseModel):
    topic_id: str                         # slug, e.g. "wifi"
    label: str                            # display, e.g. "WiFi"
    cluster_id: str                       # coarse group, e.g. "connectivity"
    question_hint: str                    # one-line hint the renderer uses

class FieldState(BaseModel):
    eg_property_id: str
    field_id: str                         # "schema:pet_policy" or "topic:wifi" or "rating:checkin"
    value_known: bool = False
    last_confirmed_date: Optional[date] = None
    short_ema: Optional[float] = None     # sentiment in [-1, 1] or rating mean in [1,5]
    long_ema: Optional[float] = None
    mention_count: int = 0
    last_asked_date: Optional[date] = None

class Question(BaseModel):
    field_id: str
    question_text: str
    input_type: Literal["rating_1_5", "yes_no", "short_text"]
    reason: str                           # "why we asked" — shown in UI

class Answer(BaseModel):
    field_id: str
    question_text: str
    answer_text: str
    parsed_value: Optional[str | int | float] = None
    status: Literal["scored", "unscorable", "skipped"]
