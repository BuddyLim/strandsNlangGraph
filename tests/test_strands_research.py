from common.types import ResearchRequest, SubFinding
from strands_app import research


def test_run_research_fans_out_one_finding_per_subtopic(monkeypatch):
    monkeypatch.setattr(research, "plan_subtopics",
                        lambda q, n, model=None: ["a", "b", "c"])
    monkeypatch.setattr(research, "research_subtopic",
                        lambda st, grounded=False, model=None:
                        SubFinding(subtopic=st, findings=f"found {st}"))
    monkeypatch.setattr(research, "synthesize",
                        lambda q, findings, model=None: "final summary")

    report = research.run_research(ResearchRequest(question="Q", n_subtopics=3))

    assert report.summary == "final summary"
    assert [f.subtopic for f in report.findings] == ["a", "b", "c"]


def test_research_subtopic_degrades_gracefully_on_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("model exploded")
    # Agent construction inside research_subtopic raises.
    monkeypatch.setattr(research, "Agent", _boom)

    finding = research.research_subtopic("doomed subtopic", model=object())

    assert finding.ok is False
    assert "doomed subtopic" == finding.subtopic
    assert "model exploded" in finding.findings


def test_synthesize_includes_each_subtopic_in_the_prompt(monkeypatch):
    seen = {}

    class _StubAgent:
        def __init__(self, **kwargs): ...
        def __call__(self, prompt):
            seen["prompt"] = prompt
            return "synthesized"

    monkeypatch.setattr(research, "Agent", _StubAgent)
    findings = [SubFinding(subtopic="alpha", findings="x"),
                SubFinding(subtopic="beta", findings="y")]

    out = research.synthesize("Q", findings, model=object())

    assert out == "synthesized"
    assert "alpha" in seen["prompt"] and "beta" in seen["prompt"]
