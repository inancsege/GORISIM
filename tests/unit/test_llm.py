from unittest.mock import MagicMock, patch

from gorisim.llm import grammar_clean


def test_grammar_clean_passes_through_when_disabled(monkeypatch):
    monkeypatch.setenv("GORISIM_USE_LLM", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from gorisim.config import reset_settings_for_test
    reset_settings_for_test()

    assert grammar_clean("merhaba sen iyi") == "merhaba sen iyi"


def test_grammar_clean_uses_openai_when_enabled(monkeypatch):
    monkeypatch.setenv("GORISIM_USE_LLM", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from gorisim.config import reset_settings_for_test
    reset_settings_for_test()

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Merhaba, sen iyi misin?"))
    ]
    with patch("gorisim.llm._client", return_value=fake_client):
        out = grammar_clean("merhaba sen iyi")
        assert out == "Merhaba, sen iyi misin?"
        fake_client.chat.completions.create.assert_called_once()
