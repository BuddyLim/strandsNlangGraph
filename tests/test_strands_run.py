import os
import pytest
from common.types import ResearchReport, SubFinding, ResearchRequest
from strands_app import run


def test_format_report_shows_summary_and_each_subtopic():
    report = ResearchReport(
        question="Q", summary="the summary",
        findings=[SubFinding(subtopic="alpha", findings="x"),
                  SubFinding(subtopic="beta", findings="y")],
    )
    text = run.format_report(report)
    assert "the summary" in text
    assert "alpha" in text and "beta" in text


def test_main_wires_args_into_run_research(monkeypatch, capsys):
    captured = {}

    def _fake_run_research(request, grounded=False, model=None):
        captured["request"] = request
        captured["grounded"] = grounded
        return ResearchReport(question=request.question, summary="ok", findings=[])

    monkeypatch.setattr(run, "run_research", _fake_run_research)
    code = run.main(["What is X?", "--subtopics", "2", "--grounded"])

    assert code == 0
    assert captured["request"].question == "What is X?"
    assert captured["request"].n_subtopics == 2
    assert captured["grounded"] is True
    assert "ok" in capsys.readouterr().out


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="needs GOOGLE_API_KEY")
def test_live_end_to_end_smoke():
    report = run_research_live()
    assert report.summary.strip() != ""
    assert len(report.findings) == 2


def run_research_live():
    from strands_app.research import run_research
    return run_research(ResearchRequest(question="What is photosynthesis?",
                                        n_subtopics=2))
