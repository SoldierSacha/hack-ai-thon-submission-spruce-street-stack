from __future__ import annotations

from src.models import FieldState, Property, TaxonomyTopic, Question
from src.llm import LlmClient


_SYSTEM_PROMPT = (
    "You write ONE short follow-up question (\u226420 words) for a hotel "
    "reviewer. Plain language. No compound questions. No pleasantries. "
    "Output ONLY the question."
)

# Short human-readable phrases for each sub-rating.
_RATING_HINTS = {
    "rating:overall": "overall stay experience",
    "rating:roomcleanliness": "room cleanliness",
    "rating:service": "service quality",
    "rating:roomcomfort": "room comfort",
    "rating:hotelcondition": "hotel condition and upkeep",
    "rating:roomquality": "room quality",
    "rating:convenienceoflocation": "location convenience",
    "rating:neighborhoodsatisfaction": "neighborhood satisfaction",
    "rating:valueformoney": "value for money",
    "rating:roomamenitiesscore": "in-room amenities",
    "rating:communication": "communication with staff",
    "rating:ecofriendliness": "eco-friendliness",
    "rating:checkin": "check-in experience",
    "rating:onlinelisting": "online listing accuracy",
    "rating:location": "location quality",
}

# Short human-readable phrases for each schema field.
_SCHEMA_HINTS = {
    "schema:pet_policy": "pet policy",
    "schema:children_and_extra_bed_policy": "children and extra bed policy",
    "schema:check_in_instructions": "check-in instructions",
    "schema:know_before_you_go": "things to know before you go",
    "schema:check_out_policy": "check-out policy",
    "schema:property_amenity_accessibility": "accessibility amenities",
    "schema:property_amenity_activities_nearby": "nearby activities",
    "schema:property_amenity_business_services": "business services",
    "schema:property_amenity_food_and_drink": "food and drink options",
    "schema:property_amenity_outdoor": "outdoor amenities",
    "schema:property_amenity_spa": "spa amenities",
    "schema:property_amenity_things_to_do": "things to do at the property",
}


def _current_value_for(field_state: FieldState, property_: Property) -> str:
    if field_state.value_known is False:
        return "unknown (no data)"
    fid = field_state.field_id
    if fid.startswith("rating:"):
        if field_state.short_ema is not None:
            return (
                f"rated in {field_state.mention_count} reviews "
                f"(short-EMA={field_state.short_ema:.2f})"
            )
        return "rated but no EMA"
    if fid.startswith("schema:property_amenity_"):
        subkey = fid[len("schema:property_amenity_"):]
        items = property_.amenities.get(subkey) if property_.amenities else None
        if items:
            joined = ", ".join(items)
            return f"known amenities: {joined[:120]}"
        return "empty list"
    if fid.startswith("schema:"):
        attr = fid[len("schema:"):]
        val = getattr(property_, attr, None)
        if not val:
            return "empty"
        return str(val)[:120]
    if fid.startswith("topic:"):
        if field_state.short_ema is not None:
            return (
                f"mentioned in {field_state.mention_count} reviews, "
                f"short-EMA={field_state.short_ema:.2f}"
            )
        return "no mentions"
    return "unknown"


def _hint_for(field_state: FieldState, topic: TaxonomyTopic | None) -> str:
    if topic is not None:
        return topic.question_hint
    fid = field_state.field_id
    if fid.startswith("rating:"):
        if fid in _RATING_HINTS:
            return _RATING_HINTS[fid]
        return fid.split(":", 1)[1].replace("_", " ")
    if fid.startswith("schema:"):
        if fid in _SCHEMA_HINTS:
            return _SCHEMA_HINTS[fid]
        return fid.split(":", 1)[1].replace("_", " ")
    # topic without TaxonomyTopic — fall back to the slug.
    return fid.split(":", 1)[1].replace("_", " ") if ":" in fid else fid


def _input_type_for(field_id: str):
    if field_id.startswith("rating:"):
        return "rating_1_5"
    if field_id.startswith("schema:property_amenity_"):
        return "yes_no"
    return "short_text"


def _reason_for(field_state: FieldState) -> str:
    fid = field_state.field_id
    key = fid.split(":", 1)[1].replace("_", " ") if ":" in fid else fid
    if not field_state.value_known:
        if fid.startswith("rating:"):
            return f"No rating data for '{key}' on this property."
        if fid.startswith("schema:"):
            return f"Description field '{key}' is empty."
        if fid.startswith("topic:"):
            return f"No reviews mention '{key}' yet."
        return f"No data for '{key}'."
    # value_known is True: detect sentiment drift, otherwise say it may be stale.
    if (
        field_state.short_ema is not None
        and field_state.long_ema is not None
        and abs(field_state.short_ema - field_state.long_ema) >= 0.3
    ):
        return "Recent reviews suggest this may have changed."
    return "Existing data may be stale."


def render_question(
    *,
    field_state: FieldState,
    property_: Property,
    topic: TaxonomyTopic | None,
    llm: LlmClient,
    model: str = "gpt-4.1-mini",
) -> Question:
    """Use an LLM to phrase a single follow-up question targeted at field_state's gap."""
    input_type = _input_type_for(field_state.field_id)
    current = _current_value_for(field_state, property_)
    hint = _hint_for(field_state, topic)
    cluster_context = (
        f"Property: {property_.city or '?'}, "
        f"{property_.country or '?'}, "
        f"{property_.star_rating if property_.star_rating is not None else '?'}-star"
    )
    user_prompt = (
        f"{cluster_context}\n"
        f"Topic: {hint}\n"
        f"Current knowledge: {current}\n"
        f"Required answer type: {input_type}\n"
        f"Question:"
    )
    raw = llm.chat_text(
        system=_SYSTEM_PROMPT, user=user_prompt, model=model, temperature=0.3
    )
    text = (raw or "").strip()
    if len(text) > 200:
        # Keep just the first non-empty line — brevity already enforced by prompt.
        first_line = text.split("\n", 1)[0].strip()
        text = first_line or text[:200]
    return Question(
        field_id=field_state.field_id,
        question_text=text,
        input_type=input_type,
        reason=_reason_for(field_state),
    )
