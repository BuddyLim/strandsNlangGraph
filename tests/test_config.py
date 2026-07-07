import pytest
from common.config import Settings, require_api_key


def test_settings_have_gemini_defaults():
    s = Settings(google_api_key="x")
    assert s.model_id == "gemini-2.5-flash"
    assert s.n_subtopics == 3


def test_require_api_key_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.setattr("common.config.settings", Settings(google_api_key=""))
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        require_api_key()


def test_require_api_key_returns_key_when_present(monkeypatch):
    monkeypatch.setattr("common.config.settings", Settings(google_api_key="secret"))
    assert require_api_key() == "secret"
