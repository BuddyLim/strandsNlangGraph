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


def test_format_report_labels_findings_section_with_subagent_count():
    report = ResearchReport(
        question="Q", summary="s",
        findings=[SubFinding(subtopic="alpha", findings="x"),
                  SubFinding(subtopic="beta", findings="y")],
    )
    text = run.format_report(report)
    assert "2 spawned" in text      # section labelled with the sub-agent count
    assert "### alpha" in text      # subtopics demoted to h3 under that section


def test_format_report_shows_empty_state_when_no_subagents_ran():
    report = ResearchReport(question="Q", summary="direct answer", findings=[])
    text = run.format_report(report)
    assert "direct answer" in text
    assert "without delegating" in text


def test_main_wires_args_into_run_research(monkeypatch, capsys):
    captured = {}

    def _fake_run_research(request, grounded=False, model=None, verbose=False):
        captured["request"] = request
        captured["grounded"] = grounded
        captured["verbose"] = verbose
        return ResearchReport(question=request.question, summary="ok", findings=[])

    monkeypatch.setattr(run, "run_research", _fake_run_research)
    code = run.main(["What is X?", "--subtopics", "2", "--grounded", "--verbose"])

    assert code == 0
    assert captured["request"].question == "What is X?"
    assert captured["request"].n_subtopics == 2
    assert captured["grounded"] is True
    assert captured["verbose"] is True
    assert "ok" in capsys.readouterr().out


def test_main_defaults_to_streaming_and_prints_findings_recap(monkeypatch, capsys):
    captured = {}

    def _fake_run_research(request, grounded=False, model=None, verbose=False):
        captured["verbose"] = verbose
        return ResearchReport(
            question=request.question, summary="STREAMED-SUMMARY",
            findings=[SubFinding(subtopic="alpha", findings="fa")],
        )

    monkeypatch.setattr(run, "run_research", _fake_run_research)
    code = run.main(["What is X?"])
    out = capsys.readouterr().out

    assert code == 0
    assert captured["verbose"] is False
    assert "Sub-agent findings" in out and "alpha" in out   # recap prints
    # Summary streams live inside run_research; main must not re-print it here.
    assert "STREAMED-SUMMARY" not in out


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="needs GOOGLE_API_KEY")
def test_live_end_to_end_smoke():
    report = run_research_live()
    assert report.summary.strip() != ""
    # Fan-out is LLM-driven, so the count is not guaranteed — only bounded by the
    # soft target. The real signal is a non-empty synthesized summary.
    assert 0 <= len(report.findings) <= 2


def run_research_live():
    from strands_app.research import run_research
    return run_research(ResearchRequest(question="What is photosynthesis?",
                                        n_subtopics=2))
