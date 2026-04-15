from datetime import date
from unittest.mock import MagicMock

import numpy as np

from src.enrich import detect_language, enrich_review, tag_review, translate_to_english
from src.models import RatingBreakdown, Review, TaxonomyTopic


def test_detect_language_basic():
    assert detect_language("The room was clean and staff friendly") == "en"
    assert detect_language("Das Zimmer war sehr sauber") == "de"
    assert detect_language("") == "unknown"
    assert detect_language(None) == "unknown"


def test_translate_english_is_noop():
    llm = MagicMock()
    out = translate_to_english("Great stay", "en", llm)
    assert out == "Great stay"
    llm.chat_text.assert_not_called()


def test_translate_unknown_is_noop():
    llm = MagicMock()
    out = translate_to_english("Mmm", "unknown", llm)
    assert out == "Mmm"
    llm.chat_text.assert_not_called()


def test_translate_non_english_calls_llm_with_correct_prompt():
    llm = MagicMock()
    llm.chat_text.return_value = "The room was very clean"
    out = translate_to_english("Das Zimmer war sehr sauber", "de", llm)
    assert out == "The room was very clean"
    llm.chat_text.assert_called_once()
    kwargs = llm.chat_text.call_args.kwargs
    assert "Translate to English" in kwargs["system"]
    assert kwargs["user"] == "Das Zimmer war sehr sauber"
    assert kwargs["temperature"] == 0.0
    assert kwargs["model"] == "gpt-4.1-mini"


def test_translate_none_or_empty_is_noop():
    llm = MagicMock()
    assert translate_to_english(None, "de", llm) is None
    assert translate_to_english("", "de", llm) == ""
    assert translate_to_english("   ", "de", llm) == "   "
    llm.chat_text.assert_not_called()


def _sample_topics():
    return [
        TaxonomyTopic(topic_id="wifi", label="WiFi", cluster_id="connectivity", question_hint="wifi quality"),
        TaxonomyTopic(topic_id="pool", label="Pool", cluster_id="amenities", question_hint="pool availability"),
        TaxonomyTopic(topic_id="noise", label="Noise", cluster_id="room", question_hint="noise level"),
    ]


def test_tag_review_empty_returns_not_mentioned_defaults():
    topics = _sample_topics()
    llm = MagicMock()
    out = tag_review("", topics, llm)
    assert set(out.keys()) == {"wifi", "pool", "noise"}
    for v in out.values():
        assert v == {"mentioned": False, "sentiment": None, "assertion": None}
    llm.chat_json.assert_not_called()


def test_tag_review_normalizes_llm_response():
    topics = _sample_topics()
    llm = MagicMock()
    # LLM returns only the topics it thinks were discussed; pool missing.
    # Also returns a hallucinated topic "spa" that's not in our taxonomy.
    llm.chat_json.return_value = {
        "wifi": {"mentioned": True, "sentiment": -1, "assertion": "slow in the room"},
        "noise": {"mentioned": True, "sentiment": -1, "assertion": None},
        "spa": {"mentioned": True, "sentiment": 1, "assertion": "nice spa"},
    }
    out = tag_review("WiFi was slow and it was noisy", topics, llm)
    assert set(out.keys()) == {"wifi", "pool", "noise"}  # spa dropped; pool filled
    assert out["wifi"]["mentioned"] is True
    assert out["wifi"]["sentiment"] == -1
    assert out["wifi"]["assertion"] == "slow in the room"
    assert out["pool"] == {"mentioned": False, "sentiment": None, "assertion": None}
    assert out["noise"]["mentioned"] is True
    assert out["noise"]["assertion"] is None
    llm.chat_json.assert_called_once()


def test_tag_review_coerces_malformed_sentiment():
    """LLM occasionally returns strings for sentiment; normalize."""
    topics = _sample_topics()
    llm = MagicMock()
    llm.chat_json.return_value = {
        "wifi": {"mentioned": True, "sentiment": "bad", "assertion": None},   # string -> None
        "pool": {"mentioned": True, "sentiment": 5, "assertion": None},        # out of range -> None
        "noise": {"mentioned": True, "sentiment": 1, "assertion": ""},         # empty string -> None
    }
    out = tag_review("any review", topics, llm)
    assert out["wifi"]["sentiment"] is None
    assert out["pool"]["sentiment"] is None
    assert out["noise"]["sentiment"] == 1
    assert out["noise"]["assertion"] is None   # empty string coerced


def test_tag_review_prompt_contains_all_topic_ids_and_hints():
    topics = _sample_topics()
    llm = MagicMock()
    llm.chat_json.return_value = {}
    tag_review("x", topics, llm)
    call_kwargs = llm.chat_json.call_args.kwargs
    system = call_kwargs["system"]
    assert "wifi: wifi quality" in system
    assert "pool: pool availability" in system
    assert "noise: noise level" in system
    assert call_kwargs["user"] == "x"
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["model"] == "gpt-4.1-mini"


def test_enrich_review_happy_path():
    topics = [
        TaxonomyTopic(topic_id="wifi", label="WiFi", cluster_id="connectivity", question_hint="wifi"),
    ]
    llm = MagicMock()
    llm.chat_text.return_value = "The room was clean"   # translation
    llm.embed.return_value = [0.1, 0.2, 0.3]
    llm.chat_json.return_value = {"wifi": {"mentioned": False, "sentiment": None, "assertion": None}}

    review = Review(
        review_id="p1:0", eg_property_id="p1",
        acquisition_date=date(2025, 9, 1),
        rating=RatingBreakdown(),
        review_text_orig="Das Zimmer war sauber",  # German, will trigger translation
    )
    text_en, lang, emb, tags = enrich_review(review, topics, llm)
    assert text_en == "The room was clean"
    assert lang == "de"
    assert emb is not None and emb.dtype == np.float32
    assert "wifi" in tags


def test_enrich_review_empty_text_skips_llm():
    topics = [TaxonomyTopic(topic_id="wifi", label="WiFi", cluster_id="connectivity", question_hint="wifi")]
    llm = MagicMock()
    review = Review(
        review_id="p1:1", eg_property_id="p1",
        acquisition_date=date(2025, 9, 1),
        rating=RatingBreakdown(),
        review_text_orig=None,
    )
    text_en, lang, emb, tags = enrich_review(review, topics, llm)
    assert text_en is None
    assert lang == "unknown"
    assert emb is None
    assert tags["wifi"] == {"mentioned": False, "sentiment": None, "assertion": None}
    llm.chat_text.assert_not_called()
    llm.embed.assert_not_called()
    # tag_review IS called with None, which internally short-circuits (no LLM call).
    llm.chat_json.assert_not_called()
