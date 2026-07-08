from langchain_core.messages import AIMessage
from langgraph.graph import END

from common.types import ResearchReport, ResearchRequest, SubFinding
from langgraph_app import research_graph


class _StubResearcher:
    """Stands in for a researcher agent: returns a canned final message."""

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, inputs):
        return {"messages": [AIMessage(content="canned findings")]}


def _coordinator_message_with_calls(*subtopics):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "research_topic", "args": {"subtopic": s}, "id": f"c{i}"}
            for i, s in enumerate(subtopics)
        ],
    )


def test_should_continue_routes_to_research_when_coordinator_calls_tools():
    state = {"messages": [_coordinator_message_with_calls("x")], "findings": []}
    assert research_graph._should_continue(state) == "research"


def test_should_continue_ends_when_coordinator_stops_calling_tools():
    state = {"messages": [AIMessage(content="final synthesis")], "findings": []}
    assert research_graph._should_continue(state) == END


def test_research_node_writes_tool_messages_and_findings_to_state(monkeypatch):
    monkeypatch.setattr(research_graph, "create_agent", lambda *a, **k: _StubResearcher())
    state = {"messages": [_coordinator_message_with_calls("alpha", "beta")], "findings": []}

    update = research_graph._research_node(state, model=object())

    # The node writes to BOTH state channels — the thing a closure can't express.
    assert [f.subtopic for f in update["findings"]] == ["alpha", "beta"]
    assert all(f.ok for f in update["findings"])
    assert [m.tool_call_id for m in update["messages"]] == ["c0", "c1"]
    assert update["messages"][0].content == "canned findings"


def test_research_node_degrades_gracefully_on_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(research_graph, "create_agent", _boom)
    state = {"messages": [_coordinator_message_with_calls("doomed")], "findings": []}

    update = research_graph._research_node(state, model=object())

    assert update["findings"][0].ok is False
    assert "model exploded" in update["findings"][0].findings
    # A ToolMessage is still emitted so the coordinator isn't left with a dangling tool call.
    assert update["messages"][0].tool_call_id == "c0"


def test_run_research_graph_reads_findings_from_final_state(monkeypatch):
    class _FakeGraph:
        def invoke(self, initial):
            return {
                "messages": [*initial["messages"], AIMessage(content="synthesized summary")],
                "findings": [
                    SubFinding(subtopic="alpha", findings="fa"),
                    SubFinding(subtopic="beta", findings="fb"),
                ],
            }

    monkeypatch.setattr(research_graph, "build_research_graph", lambda **k: _FakeGraph())

    report = research_graph.run_research_graph(ResearchRequest(question="Q", n_subtopics=3))

    assert isinstance(report, ResearchReport)
    assert report.summary == "synthesized summary"
    assert [f.subtopic for f in report.findings] == ["alpha", "beta"]
    assert report.question == "Q"
