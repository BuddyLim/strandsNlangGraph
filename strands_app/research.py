from strands import Agent, tool

from common.prompts import PLANNER_PROMPT, SUB_AGENT_PROMPT, SYNTHESIS_PROMPT
from common.tools import mock_search as _mock_search
from common.types import ResearchReport, ResearchRequest, SubFinding, SubtopicPlan
from strands_app.model import build_gemini_model


@tool
def mock_search(subtopic: str) -> str:
    """Search for information about a subtopic and return findings."""
    return _mock_search(subtopic)


def plan_subtopics(question: str, n: int, model=None) -> list[str]:
    """Split the question into n distinct subtopics using structured output."""
    planner = Agent(model=model or build_gemini_model(),
                    system_prompt=PLANNER_PROMPT)
    plan: SubtopicPlan = planner.structured_output(
        SubtopicPlan,
        f"Question: {question}\nProduce exactly {n} subtopics.",
    )
    return plan.subtopics[:n]


def research_subtopic(subtopic: str, grounded: bool = False, model=None) -> SubFinding:
    """Spawn a fresh sub-agent to research one subtopic.

    Each call constructs its own Agent — that construction IS the subagent
    spawn. Failures are caught and returned as a degraded SubFinding so one bad
    subtopic never aborts the whole report.
    """
    try:
        sub_model = model or build_gemini_model(grounded=grounded)
        tools = None if grounded else [mock_search]
        researcher = Agent(model=sub_model, tools=tools,
                           system_prompt=SUB_AGENT_PROMPT)
        answer = str(researcher(f"Research this subtopic: {subtopic}"))
        return SubFinding(subtopic=subtopic, findings=answer, ok=True)
    except Exception as exc:  # noqa: BLE001 — deliberate graceful degradation boundary
        return SubFinding(subtopic=subtopic, findings=f"failed: {exc}", ok=False)


def synthesize(question: str, findings: list[SubFinding], model=None) -> str:
    """Merge per-subtopic findings into one answer to the original question."""
    synthesizer = Agent(model=model or build_gemini_model(),
                        system_prompt=SYNTHESIS_PROMPT)
    joined = "\n\n".join(f"## {f.subtopic}\n{f.findings}" for f in findings)
    return str(synthesizer(
        f"Question: {question}\n\nFindings:\n{joined}\n\nWrite a synthesis."
    ))


def run_research(request: ResearchRequest, grounded: bool = False,
                 model=None) -> ResearchReport:
    """Plan subtopics, spawn one sub-agent per subtopic, then synthesize."""
    subtopics = plan_subtopics(request.question, request.n_subtopics, model=model)
    findings = [research_subtopic(st, grounded=grounded, model=model)
                for st in subtopics]
    summary = synthesize(request.question, findings, model=model)
    return ResearchReport(question=request.question, summary=summary,
                          findings=findings)
