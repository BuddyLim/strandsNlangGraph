from langchain_core.messages import AIMessage

from common.types import SubFinding
from langgraph_app import research


class _StubResearcher:
    """Stands in for a researcher prebuilt agent: returns a canned final message."""

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, inputs):
        return {"messages": [AIMessage(content="canned researcher findings")]}


def test_research_impl_spawns_subagent_and_records_finding(monkeypatch):
    monkeypatch.setattr(research, "create_agent", lambda *a, **k: _StubResearcher())
    findings: list[SubFinding] = []

    result = research._research_impl("photosynthesis", findings, model=object())

    assert result == "canned researcher findings"
    assert len(findings) == 1
    assert findings[0].subtopic == "photosynthesis"
    assert findings[0].ok is True


def test_research_impl_degrades_gracefully_on_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(research, "create_agent", _boom)
    findings: list[SubFinding] = []

    result = research._research_impl("doomed", findings, model=object())

    assert result.startswith("failed:")
    assert "model exploded" in result
    assert len(findings) == 1
    assert findings[0].ok is False
    assert findings[0].subtopic == "doomed"


def test_research_topic_tool_is_a_closure_over_findings(monkeypatch):
    monkeypatch.setattr(research, "create_agent", lambda *a, **k: _StubResearcher())
    findings: list[SubFinding] = []
    tool = research.make_research_tool(findings, model=object())

    assert tool.name == "research_topic"
    out = tool.invoke({"subtopic": "alpha"})

    assert out == "canned researcher findings"
    assert [f.subtopic for f in findings] == ["alpha"]


def test_build_coordinator_registers_research_topic_tool(monkeypatch):
    captured = {}

    def fake_create_agent(model=None, tools=None, **kwargs):
        captured["tools"] = tools
        return object()

    monkeypatch.setattr(research, "create_agent", fake_create_agent)
    research.build_coordinator([], model=object())

    assert "research_topic" in [t.name for t in captured["tools"]]
