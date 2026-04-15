from unittest.mock import patch, MagicMock
from src.llm import LlmClient


def test_cache_hit_skips_network(tmp_path):
    client = LlmClient(cache_dir=tmp_path, api_key="fake")
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content='{"x": 1}'))]
    with patch.object(client._client.chat.completions, "create", return_value=fake_resp) as m:
        a = client.chat_json(system="sys", user="u", model="gpt-4.1-mini")
        b = client.chat_json(system="sys", user="u", model="gpt-4.1-mini")
    assert a == b == {"x": 1}
    assert m.call_count == 1  # second call hit cache


def test_distinct_prompts_distinct_cache(tmp_path):
    client = LlmClient(cache_dir=tmp_path, api_key="fake")
    fake_resp_1 = MagicMock()
    fake_resp_1.choices = [MagicMock(message=MagicMock(content='{"ans": "first"}'))]
    fake_resp_2 = MagicMock()
    fake_resp_2.choices = [MagicMock(message=MagicMock(content='{"ans": "second"}'))]
    with patch.object(client._client.chat.completions, "create",
                      side_effect=[fake_resp_1, fake_resp_2]) as m:
        a = client.chat_json(system="sys", user="prompt A", model="gpt-4.1-mini")
        b = client.chat_json(system="sys", user="prompt B", model="gpt-4.1-mini")
    assert a == {"ans": "first"}
    assert b == {"ans": "second"}
    assert m.call_count == 2  # both prompts hit the network (no cache collision)
