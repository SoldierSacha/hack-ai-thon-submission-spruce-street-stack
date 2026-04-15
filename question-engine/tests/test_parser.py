from unittest.mock import MagicMock
from src.models import Question, Answer
from src.parser import parse_answer

def _q(input_type="rating_1_5"):
    return Question(field_id="rating:checkin", question_text="How was check-in?",
                    input_type=input_type, reason="no data")

def test_parse_skip_sentinel():
    q = _q()
    llm = MagicMock()
    a = parse_answer(q, None, llm)
    assert a.status == "skipped"
    assert a.parsed_value is None
    llm.chat_json.assert_not_called()

def test_parse_empty_is_unscorable():
    q = _q()
    llm = MagicMock()
    a = parse_answer(q, "", llm)
    assert a.status == "unscorable"
    llm.chat_json.assert_not_called()

def test_parse_filler_is_unscorable():
    q = _q()
    llm = MagicMock()
    for filler in ["ok", "idk", "na", "dunno"]:
        a = parse_answer(q, filler, llm)
        assert a.status == "unscorable", f"failed on '{filler}'"
    llm.chat_json.assert_not_called()

def test_parse_rating_regex_digit():
    q = _q()
    llm = MagicMock()
    a = parse_answer(q, "I'd give it a 4 out of 5", llm)
    assert a.status == "scored"
    assert a.parsed_value == 4
    llm.chat_json.assert_not_called()

def test_parse_rating_word_to_digit():
    q = _q()
    llm = MagicMock()
    a = parse_answer(q, "five stars easily", llm)
    assert a.status == "scored"
    assert a.parsed_value == 5
    llm.chat_json.assert_not_called()

def test_parse_rating_llm_fallback():
    q = _q()
    llm = MagicMock()
    llm.chat_json.return_value = {"rating": 3, "abstain": False}
    a = parse_answer(q, "It was middling, nothing special", llm)
    assert a.status == "scored"
    assert a.parsed_value == 3
    llm.chat_json.assert_called_once()

def test_parse_rating_llm_abstain_is_unscorable():
    q = _q()
    llm = MagicMock()
    llm.chat_json.return_value = {"rating": None, "abstain": True}
    a = parse_answer(q, "I don't remember to be honest", llm)
    assert a.status == "unscorable"

def test_parse_yes_no_regex_yes():
    q = _q(input_type="yes_no")
    llm = MagicMock()
    a = parse_answer(q, "Yes, there is a spa", llm)
    assert a.status == "scored"
    assert a.parsed_value == "yes"
    llm.chat_json.assert_not_called()

def test_parse_yes_no_regex_no():
    q = _q(input_type="yes_no")
    llm = MagicMock()
    a = parse_answer(q, "Nope, never saw one", llm)
    assert a.status == "scored"
    assert a.parsed_value == "no"

def test_parse_yes_no_llm_fallback():
    q = _q(input_type="yes_no")
    llm = MagicMock()
    llm.chat_json.return_value = {"answer": "yes", "abstain": False}
    a = parse_answer(q, "there was indeed one", llm)
    assert a.status == "scored"
    assert a.parsed_value == "yes"
    llm.chat_json.assert_called_once()

def test_parse_short_text_calls_llm():
    q = _q(input_type="short_text")
    llm = MagicMock()
    llm.chat_json.return_value = {"value": "Fast WiFi in room, spotty in lobby", "abstain": False}
    a = parse_answer(q, "The WiFi was really fast in the room but spotty in the lobby", llm)
    assert a.status == "scored"
    assert a.parsed_value == "Fast WiFi in room, spotty in lobby"

def test_parse_short_text_llm_abstain_is_unscorable():
    q = _q(input_type="short_text")
    llm = MagicMock()
    llm.chat_json.return_value = {"value": None, "abstain": True}
    a = parse_answer(q, "blah blah blah", llm)
    assert a.status == "unscorable"
