from unittest.mock import MagicMock

from src.enrich import detect_language, translate_to_english


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
