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


def test_handle_accepts_agentcore_prompt_key(monkeypatch):
    server = _load_server(monkeypatch)
    fake = _fake_report()
    monkeypatch.setattr(server, "run_research", fake)

    server._handle({"prompt": "What is X?"})

    assert fake.calls[0]["request"].question == "What is X?"


def test_handle_prefers_explicit_question_over_prompt(monkeypatch):
    server = _load_server(monkeypatch)
    fake = _fake_report()
    monkeypatch.setattr(server, "run_research", fake)

    server._handle({"question": "explicit", "prompt": "generic"})

    assert fake.calls[0]["request"].question == "explicit"


def test_handle_extracts_question_from_messages(monkeypatch):
    server = _load_server(monkeypatch)
    fake = _fake_report()
    monkeypatch.setattr(server, "run_research", fake)

    server._handle(
        {"messages": [{"role": "user", "content": [{"text": "What is X?"}]}]}
    )

    assert fake.calls[0]["request"].question == "What is X?"


def test_handle_accepts_string_message_content(monkeypatch):
    server = _load_server(monkeypatch)
    fake = _fake_report()
    monkeypatch.setattr(server, "run_research", fake)

    server._handle({"messages": [{"role": "user", "content": "plain text"}]})

    assert fake.calls[0]["request"].question == "plain text"


def test_handle_without_any_question_raises_value_error(monkeypatch):
    server = _load_server(monkeypatch)
    monkeypatch.setattr(server, "run_research", _fake_report())

    # A bare KeyError leaked a stack trace to the caller; this must be actionable.
    with pytest.raises(ValueError, match="question|prompt|messages"):
        server._handle({"unexpected": "shape"})


@pytest.mark.parametrize("bad", ["", "STRANDS", "both", "gpt"])
def test_invalid_app_raises_at_import(monkeypatch, tmp_path, bad):
    # chdir to an empty dir so the repo's own .env can't supply a fallback APP.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP", bad)
    monkeypatch.setattr("common.config.require_api_key", lambda: "test-key")
    sys.modules.pop("deploy.server", None)
    with pytest.raises(RuntimeError, match="APP must be"):
        importlib.import_module("deploy.server")


def test_unset_app_raises_at_import(monkeypatch, tmp_path):
    # No process env APP and no .env in this cwd -> nothing supplies it -> raise.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("APP", raising=False)
    monkeypatch.setattr("common.config.require_api_key", lambda: "test-key")
    sys.modules.pop("deploy.server", None)
    with pytest.raises(RuntimeError, match="APP must be"):
        importlib.import_module("deploy.server")


def test_app_from_dotenv_file_is_honoured(monkeypatch, tmp_path):
    """Local dev: APP set only in a .env file (not the process env) is picked up.

    Regression guard — a raw os.environ read would miss .env entirely.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("APP=langgraph\n")
    monkeypatch.delenv("APP", raising=False)
    monkeypatch.setattr("common.config.require_api_key", lambda: "test-key")
    sys.modules.pop("deploy.server", None)

    server = importlib.import_module("deploy.server")

    assert server.APP == "langgraph"


def test_process_env_app_overrides_dotenv(monkeypatch, tmp_path):
    """Deployed runtime: an injected APP env var wins over any .env fallback."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("APP=langgraph\n")
    monkeypatch.setenv("APP", "strands")
    monkeypatch.setattr("common.config.require_api_key", lambda: "test-key")
    sys.modules.pop("deploy.server", None)

    server = importlib.import_module("deploy.server")

    assert server.APP == "strands"
