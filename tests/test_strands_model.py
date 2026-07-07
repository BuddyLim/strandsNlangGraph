import pytest
from common.config import Settings


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setattr("common.config.settings", Settings(google_api_key="test-key"))
    monkeypatch.setattr("strands_app.model.settings", Settings(google_api_key="test-key"))


def test_build_model_uses_configured_model_id():
    from strands_app.model import build_gemini_model
    model = build_gemini_model()
    # GeminiModel exposes its config; model_id must match settings.
    assert model.get_config()["model_id"] == "gemini-2.5-flash"


def test_build_model_grounded_attaches_search_tool():
    from strands_app.model import build_gemini_model
    grounded = build_gemini_model(grounded=True)
    plain = build_gemini_model(grounded=False)
    # Grounded model carries at least one native gemini tool; plain carries none.
    assert grounded.get_config().get("gemini_tools")
    assert not plain.get_config().get("gemini_tools")


def test_build_model_raises_without_key(monkeypatch):
    monkeypatch.setattr("common.config.settings", Settings(google_api_key=""))
    monkeypatch.setattr("strands_app.model.settings", Settings(google_api_key=""))
    from strands_app.model import build_gemini_model
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        build_gemini_model()
