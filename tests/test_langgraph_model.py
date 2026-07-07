import pytest
from common.config import Settings


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setattr("common.config.settings", Settings(google_api_key="test-key"))
    monkeypatch.setattr("langgraph_app.model.settings", Settings(google_api_key="test-key"))


def test_build_model_uses_configured_model_id():
    from langgraph_app.model import build_gemini_model
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = build_gemini_model()
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model.endswith("gemini-2.5-flash")


def test_build_model_grounded_returns_a_bound_runnable():
    from langgraph_app.model import build_gemini_model
    from langchain_google_genai import ChatGoogleGenerativeAI
    grounded = build_gemini_model(grounded=True)
    plain = build_gemini_model(grounded=False)
    # Grounded is the model with google_search bound -> no longer the bare chat model.
    assert isinstance(plain, ChatGoogleGenerativeAI)
    assert grounded is not plain
    assert not isinstance(grounded, ChatGoogleGenerativeAI)


def test_build_model_raises_without_key(monkeypatch):
    monkeypatch.setattr("common.config.settings", Settings(google_api_key=""))
    monkeypatch.setattr("langgraph_app.model.settings", Settings(google_api_key=""))
    from langgraph_app.model import build_gemini_model
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        build_gemini_model()
