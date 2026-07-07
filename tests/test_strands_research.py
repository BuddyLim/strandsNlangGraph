from common.types import ResearchRequest, SubFinding
from strands_app import research


class _StubResearcher:
    """Stands in for a researcher sub-agent Agent: returns canned text on call."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, prompt):
        return "canned researcher findings"


def test_research_impl_spawns_subagent_and_records_finding(monkeypatch):
    monkeypatch.setattr(research, "Agent", _StubResearcher)
    findings: list[SubFinding] = []

    result = research._research_impl("photosynthesis", findings, model=object())

    assert result == "canned researcher findings"
    assert len(findings) == 1
    assert findings[0].subtopic == "photosynthesis"
    assert findings[0].ok is True


def test_research_impl_degrades_gracefully_on_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(research, "Agent", _boom)
    findings: list[SubFinding] = []

    result = research._research_impl("doomed", findings, model=object())

    assert result.startswith("failed:")
    assert "model exploded" in result
    assert len(findings) == 1
    assert findings[0].ok is False
    assert findings[0].subtopic == "doomed"


def test_run_research_collects_findings_and_uses_coordinator_output_as_summary(monkeypatch):
    def fake_build_coordinator(findings, grounded=False, model=None):
        class FakeCoordinator:
            def __call__(self, question):
                # simulate the coordinator LLM deciding two subtopics and delegating
                findings.append(SubFinding(subtopic="alpha", findings="fa"))
                findings.append(SubFinding(subtopic="beta", findings="fb"))
                return "synthesized summary"
        return FakeCoordinator()

    monkeypatch.setattr(research, "build_coordinator", fake_build_coordinator)

    report = research.run_research(ResearchRequest(question="Q"))

    assert report.summary == "synthesized summary"
    assert [f.subtopic for f in report.findings] == ["alpha", "beta"]
    assert report.question == "Q"


def test_coordinator_registers_research_topic_tool(fake_model):
    findings: list[SubFinding] = []
    coordinator = research.build_coordinator(findings, model=fake_model)
    assert "research_topic" in coordinator.tool_names
