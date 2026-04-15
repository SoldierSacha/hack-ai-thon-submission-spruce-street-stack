from src.enrich import detect_language


def test_detect_language_basic():
    assert detect_language("The room was clean and staff friendly") == "en"
    assert detect_language("Das Zimmer war sehr sauber") == "de"
    assert detect_language("") == "unknown"
    assert detect_language(None) == "unknown"
