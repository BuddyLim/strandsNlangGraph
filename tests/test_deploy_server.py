import importlib
import sys

import pytest

# deploy.server imports bedrock_agentcore (deploy group). In the default env this
# skips cleanly; run the deploy tests with: uv run --group deploy pytest
pytest.importorskip("bedrock_agentcore")

from common.types import ResearchReport, SubFinding


def _load_server(monkeypatch, app="strands"):
    """Import deploy.server fresh with APP set and the key check stubbed out."""
    monkeypatch.setenv("APP", app)
    monkeypatch.setattr("common.config.require_api_key", lambda: "test-key")
    sys.modules.pop("deploy.server", None)
    return importlib.import_module("deploy.server")


def _fake_report(**kw):
    def fake(request, grounded=False, model=None, verbose=False):
        fake.calls.append({"request": request, "grounded": grounded})
        return ResearchReport(question=request.question, summary="ok", findings=[], **kw)
    fake.calls = []
    return fake


def test_handle_maps_question_and_defaults_subtopics_to_3(monkeypatch):
    server = _load_server(monkeypatch)
    fake = _fake_report()
    monkeypatch.setattr(server, "run_research", fake)

    server._handle({"question": "What is X?"})

    assert fake.calls[0]["request"].question == "What is X?"
    assert fake.calls[0]["request"].n_subtopics == 3


def test_handle_honours_explicit_subtopics(monkeypatch):
    server = _load_server(monkeypatch)
    fake = _fake_report()
    monkeypatch.setattr(server, "run_research", fake)

    server._handle({"question": "Q", "n_subtopics": 5})

    assert fake.calls[0]["request"].n_subtopics == 5


def test_handle_defaults_grounded_false_and_passes_true_through(monkeypatch):
    server = _load_server(monkeypatch)
    fake = _fake_report()
    monkeypatch.setattr(server, "run_research", fake)

    server._handle({"question": "Q"})
    server._handle({"question": "Q", "grounded": True})

    assert fake.calls[0]["grounded"] is False
    assert fake.calls[1]["grounded"] is True


def test_handle_returns_report_model_dump_dict(monkeypatch):
    server = _load_server(monkeypatch)

    def fake(request, grounded=False, model=None, verbose=False):
        return ResearchReport(
            question=request.question,
            summary="the summary",
            findings=[SubFinding(subtopic="alpha", findings="fa")],
        )

    monkeypatch.setattr(server, "run_research", fake)

    result = server._handle({"question": "Q"})

    assert isinstance(result, dict)
    assert result == {
        "question": "Q",
        "summary": "the summary",
        "findings": [{"subtopic": "alpha", "findings": "fa", "ok": True}],
    }


@pytest.mark.parametrize("bad", ["", "STRANDS", "both", "gpt"])
def test_invalid_app_raises_at_import(monkeypatch, bad):
    monkeypatch.setenv("APP", bad)
    monkeypatch.setattr("common.config.require_api_key", lambda: "test-key")
    sys.modules.pop("deploy.server", None)
    with pytest.raises(RuntimeError, match="APP must be"):
        importlib.import_module("deploy.server")


def test_unset_app_raises_at_import(monkeypatch):
    monkeypatch.delenv("APP", raising=False)
    monkeypatch.setattr("common.config.require_api_key", lambda: "test-key")
    sys.modules.pop("deploy.server", None)
    with pytest.raises(RuntimeError, match="APP must be"):
        importlib.import_module("deploy.server")
