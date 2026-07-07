from common.types import ResearchRequest, SubFinding, ResearchReport


def test_research_request_defaults_to_three_subtopics():
    req = ResearchRequest(question="What is X?")
    assert req.n_subtopics == 3


def test_sub_finding_defaults_to_ok():
    f = SubFinding(subtopic="a", findings="text")
    assert f.ok is True


def test_research_report_holds_findings():
    report = ResearchReport(
        question="Q", summary="S",
        findings=[SubFinding(subtopic="a", findings="t")],
    )
    assert len(report.findings) == 1
    assert report.summary == "S"
