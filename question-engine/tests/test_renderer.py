from unittest.mock import MagicMock

from src.models import FieldState, Property, TaxonomyTopic, Question
from src.renderer import render_question


def test_render_question_rating_field_uses_rating_1_5():
    fs = FieldState(eg_property_id="p1", field_id="rating:checkin", value_known=False)
    p = Property(eg_property_id="p1", city="Pompei", country="Italy", star_rating=3.5)
    llm = MagicMock()
    llm.chat_text.return_value = "How smooth was your check-in?"
    q = render_question(field_state=fs, property_=p, topic=None, llm=llm)
    assert q.field_id == "rating:checkin"
    assert q.input_type == "rating_1_5"
    assert q.question_text == "How smooth was your check-in?"
    assert "No rating data" in q.reason or "rating" in q.reason.lower()


def test_render_question_amenity_schema_is_yes_no():
    fs = FieldState(eg_property_id="p1", field_id="schema:property_amenity_spa", value_known=False)
    p = Property(eg_property_id="p1", city="Pompei", country="Italy",
                 star_rating=3.5, amenities={"spa": []})
    llm = MagicMock()
    llm.chat_text.return_value = "Does the hotel have a working spa?"
    q = render_question(field_state=fs, property_=p, topic=None, llm=llm)
    assert q.input_type == "yes_no"


def test_render_question_topic_is_short_text():
    fs = FieldState(eg_property_id="p1", field_id="topic:wifi", value_known=False)
    p = Property(eg_property_id="p1", city="Pompei", country="Italy", star_rating=3.5)
    topic = TaxonomyTopic(topic_id="wifi", label="WiFi", cluster_id="connectivity",
                          question_hint="speed and coverage of WiFi")
    llm = MagicMock()
    llm.chat_text.return_value = "How was the WiFi?"
    q = render_question(field_state=fs, property_=p, topic=topic, llm=llm)
    assert q.input_type == "short_text"


def test_render_question_prompt_includes_property_and_hint():
    fs = FieldState(eg_property_id="p1", field_id="topic:wifi", value_known=False)
    p = Property(eg_property_id="p1", city="Pompei", country="Italy", star_rating=3.5)
    topic = TaxonomyTopic(topic_id="wifi", label="WiFi", cluster_id="connectivity",
                          question_hint="speed and coverage of WiFi")
    llm = MagicMock()
    llm.chat_text.return_value = "How was the WiFi?"
    render_question(field_state=fs, property_=p, topic=topic, llm=llm)
    call_kwargs = llm.chat_text.call_args.kwargs
    assert "Pompei" in call_kwargs["user"]
    assert "Italy" in call_kwargs["user"]
    assert "3.5" in call_kwargs["user"]
    assert "speed and coverage of WiFi" in call_kwargs["user"]
    assert "unknown" in call_kwargs["user"].lower()  # current knowledge is unknown
    assert "short_text" in call_kwargs["user"]
    assert call_kwargs["temperature"] == 0.3
    # System prompt contains constraint
    assert "20 words" in call_kwargs["system"] or "\u226420" in call_kwargs["system"]


def test_render_question_known_value_reason_is_stale_language():
    fs = FieldState(eg_property_id="p1", field_id="rating:overall",
                    value_known=True, mention_count=100, short_ema=4.5, long_ema=4.6)
    p = Property(eg_property_id="p1", city="Pompei", country="Italy", star_rating=3.5)
    llm = MagicMock()
    llm.chat_text.return_value = "Rate your overall stay"
    q = render_question(field_state=fs, property_=p, topic=None, llm=llm)
    # Reason should be something about stale / existing data, not "no data"
    assert "no data" not in q.reason.lower()


def test_render_question_strips_whitespace_from_response():
    fs = FieldState(eg_property_id="p1", field_id="rating:checkin", value_known=False)
    p = Property(eg_property_id="p1")
    llm = MagicMock()
    llm.chat_text.return_value = "  How was check-in?  \n"
    q = render_question(field_state=fs, property_=p, topic=None, llm=llm)
    assert q.question_text == "How was check-in?"
