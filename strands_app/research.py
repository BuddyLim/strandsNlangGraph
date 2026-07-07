from strands import Agent, tool

from common.prompts import COORDINATOR_PROMPT, SUB_AGENT_PROMPT
from common.types import ResearchReport, ResearchRequest, SubFinding
from strands_app.basic import mock_search
from strands_app.model import build_gemini_model


def _research_impl(subtopic: str, findings: list[SubFinding],
                   grounded: bool = False, model=None) -> str:
    """Spawn a fresh researcher sub-agent for one subtopic.

    Constructing the Agent IS the subagent spawn. Records a SubFinding into
    `findings` (shared with the coordinator run) and returns the researcher's
    text for the coordinator to synthesize. Degrades gracefully: on any error
    it records ok=False and returns a failure string rather than raising, so one
    bad subtopic never aborts the whole run.
    """
    try:
        sub_model = model or build_gemini_model(grounded=grounded)
        tools = None if grounded else [mock_search]
        researcher = Agent(model=sub_model, tools=tools,
                           system_prompt=SUB_AGENT_PROMPT)
        text = str(researcher(f"Research this subtopic: {subtopic}"))
        findings.append(SubFinding(subtopic=subtopic, findings=text, ok=True))
        return text
    except Exception as exc:  # noqa: BLE001 — deliberate graceful-degradation boundary
        message = f"failed: {exc}"
        findings.append(SubFinding(subtopic=subtopic, findings=message, ok=False))
        return message


def make_research_tool(findings: list[SubFinding], grounded: bool = False, model=None):
    """Build the research_topic @tool bound to a findings collector for one run.

    The tool is a closure so each coordinator run gets its own findings list and
    grounded/model settings, while the LLM sees a stable single-arg tool.
    """

    @tool
    def research_topic(subtopic: str) -> str:
        """Research one subtopic and return factual findings about it."""
        return _research_impl(subtopic, findings, grounded=grounded, model=model)

    return research_topic


def build_coordinator(findings: list[SubFinding], grounded: bool = False, model=None) -> Agent:
    """Coordinator agent that decomposes the question and delegates via research_topic."""
    return Agent(
        model=model or build_gemini_model(),
        tools=[make_research_tool(findings, grounded=grounded, model=model)],
        system_prompt=COORDINATOR_PROMPT,
    )


def run_research(request: ResearchRequest, grounded: bool = False, model=None) -> ResearchReport:
    """Run the coordinator; it decides subtopics and spawns a sub-agent per subtopic.

    Unlike a Python fan-out, the coordinator LLM drives the fan-out: it calls
    research_topic once per subtopic it identifies. We collect each SubFinding as
    a side effect and use the coordinator's own output as the synthesized summary.
    """
    findings: list[SubFinding] = []
    coordinator = build_coordinator(findings, grounded=grounded, model=model)
    summary = str(coordinator(request.question))
    return ResearchReport(question=request.question, summary=summary, findings=findings)
