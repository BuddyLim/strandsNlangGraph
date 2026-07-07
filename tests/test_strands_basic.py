from strands.models import Model
from strands_app import basic


class _StubAgent:
    def __init__(self):
        self.called_with = None

    def __call__(self, prompt):
        self.called_with = prompt
        return "canned answer"


class _MinimalMockModel(Model):
    """Minimal mock model for testing Agent construction without API calls."""

    @property
    def stateful(self) -> bool:
        return False

    def stream(self, messages, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    @property
    def structured_output(self):
        return False


def test_answer_question_delegates_to_agent_and_returns_text(monkeypatch):
    stub = _StubAgent()
    monkeypatch.setattr(basic, "build_basic_agent", lambda model=None: stub)
    result = basic.answer_question("What is X?")
    assert result == "canned answer"
    assert stub.called_with == "What is X?"


def test_basic_agent_registers_mock_search_as_a_tool():
    # Construct with a minimal mock model so no API key is needed.
    agent = basic.build_basic_agent(model=_MinimalMockModel())
    assert "mock_search" in agent.tool_names
