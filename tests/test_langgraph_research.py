from langchain_core.messages import AIMessage, AIMessageChunk

from common.types import ResearchRequest, SubFinding
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


class _FakeCoordinatorStream:
    """Yields canned stream output per stream_mode, like a compiled agent graph."""

    def __init__(self, by_mode):
        self._by_mode = by_mode

    def stream(self, inputs, stream_mode=None):
        return iter(self._by_mode[stream_mode])


def test_run_and_stream_default_prints_and_returns_coordinator_tokens(capsys):
    node = research._COORDINATOR_NODE
    coord = _FakeCoordinatorStream({
        "messages": [
            (AIMessageChunk(content="hello "), {"langgraph_node": node}),
            (AIMessageChunk(content="world"), {"langgraph_node": node}),
            (AIMessageChunk(content="IGNORED"), {"langgraph_node": "other"}),
        ]
    })
    summary = research._run_and_stream(coord, "prompt", verbose=False)
    out = capsys.readouterr().out
    assert summary == "hello world"      # only coordinator-node tokens accumulate
    assert "hello world" in out
    assert "IGNORED" not in out          # other nodes are filtered out


def test_run_and_stream_verbose_prints_updates_and_returns_final_ai_message(capsys):
    coord = _FakeCoordinatorStream({
        "updates": [
            {"tools": {"messages": [AIMessage(content="tool ran")]}},
            {"agent": {"messages": [AIMessage(content="final synthesis")]}},
        ]
    })
    summary = research._run_and_stream(coord, "prompt", verbose=True)
    out = capsys.readouterr().out
    assert summary == "final synthesis"  # last AI message content wins
    assert "agent" in out                # state transitions are printed


def test_run_research_collects_findings_and_uses_stream_output_as_summary(monkeypatch):
    holder = {}

    def fake_build_coordinator(findings, grounded=False, model=None, verbose=False):
        holder["findings"] = findings
        return object()

    def fake_run_and_stream(coordinator, prompt, verbose):
        holder["prompt"] = prompt
        holder["findings"].append(SubFinding(subtopic="alpha", findings="fa"))
        holder["findings"].append(SubFinding(subtopic="beta", findings="fb"))
        return "synthesized summary"

    monkeypatch.setattr(research, "build_coordinator", fake_build_coordinator)
    monkeypatch.setattr(research, "_run_and_stream", fake_run_and_stream)

    report = research.run_research(ResearchRequest(question="Q", n_subtopics=5))

    assert report.summary == "synthesized summary"
    assert [f.subtopic for f in report.findings] == ["alpha", "beta"]
    assert report.question == "Q"
    assert "5" in holder["prompt"] and "Q" in holder["prompt"]


def test_ai_text_flattens_str_and_list_content():
    # Gemini emits content as a str normally, but can emit a list of blocks.
    assert research._ai_text("plain") == "plain"
    assert research._ai_text([{"text": "a"}, {"text": "b"}]) == "ab"
    assert research._ai_text(None) == ""
