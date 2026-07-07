from langchain_core.messages import AIMessage
from langgraph_app import basic


class _StubAgent:
    def __init__(self):
        self.called_with = None

    def invoke(self, inputs):
        self.called_with = inputs
        return {"messages": [AIMessage(content="canned answer")]}


def test_answer_question_delegates_to_agent_and_returns_text(monkeypatch):
    stub = _StubAgent()
    monkeypatch.setattr(basic, "build_basic_agent", lambda model=None: stub)
    result = basic.answer_question("What is X?")
    assert result == "canned answer"
    assert stub.called_with["messages"][0]["content"] == "What is X?"


def test_mock_search_tool_wraps_common_tool():
    # The @tool exposes a stable name and delegates to common.tools.mock_search.
    assert basic.mock_search.name == "mock_search"
    out = basic.mock_search.invoke({"subtopic": "photosynthesis"})
    assert isinstance(out, str) and out != ""
