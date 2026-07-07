from strands import Agent, tool

from common.prompts import COORDINATOR_PROMPT, SUB_AGENT_PROMPT
from common.types import ResearchReport, ResearchRequest, SubFinding
from strands_app.basic import mock_search
from strands_app.model import build_gemini_model


def _callback_kwargs(verbose: bool) -> dict:
    """Agent kwargs controlling the live tool-call/token trace.

    Omitting `callback_handler` lets Strands use its default PrintingCallbackHandler,
    which streams the trace to stdout; passing None installs a no-op handler that
    suppresses it. Default is quiet — the CLI's own formatted report is the output;
    the trace is opt-in (`--verbose`) because watching the fan-out is a learning aid.
    """
    return {} if verbose else {"callback_handler": None}


def _research_impl(
    subtopic: str,
    findings: list[SubFinding],
    grounded: bool = False,
    model=None,
    verbose: bool = False,
) -> str:
    """Spawn a fresh researcher sub-agent for one subtopic.

    Constructing the Agent IS the subagent spawn. Returns the researcher's text,
    which is what the coordinator actually receives (as the tool result) and
    synthesizes. Separately appends a SubFinding to `findings` — an out-of-band
    collector the run uses to reconstruct the per-subtopic breakdown for the
    report; the coordinator never reads this list. Degrades gracefully: on any
    error it records ok=False and returns a failure string rather than raising,
    so one bad subtopic never aborts the whole run.
    """
    try:
        sub_model = model or build_gemini_model(grounded=grounded)
        tools = None if grounded else [mock_search]
        researcher = Agent(
            model=sub_model, tools=tools, system_prompt=SUB_AGENT_PROMPT,
            **_callback_kwargs(verbose),
        )
        text = str(researcher(f"Research this subtopic: {subtopic}"))
        findings.append(SubFinding(subtopic=subtopic, findings=text, ok=True))
        return text
    except Exception as exc:  # noqa: BLE001 — deliberate graceful-degradation boundary
        message = f"failed: {exc}"
        findings.append(SubFinding(subtopic=subtopic, findings=message, ok=False))
        return message


def make_research_tool(
    findings: list[SubFinding], grounded: bool = False, model=None, verbose: bool = False
):
    """Build the research_topic @tool bound to a findings collector for one run.

    The tool is a closure so each coordinator run gets its own findings list and
    grounded/model/verbose settings, while the LLM sees a stable single-arg tool.
    """

    @tool
    def research_topic(subtopic: str) -> str:
        """Research one subtopic and return factual findings about it."""
        return _research_impl(subtopic, findings, grounded=grounded, model=model,
                              verbose=verbose)

    return research_topic


def build_coordinator(
    findings: list[SubFinding], grounded: bool = False, model=None, verbose: bool = False
) -> Agent:
    """Coordinator agent that decomposes the question and delegates via research_topic."""
    return Agent(
        model=model or build_gemini_model(),
        tools=[make_research_tool(findings, grounded=grounded, model=model, verbose=verbose)],
        system_prompt=COORDINATOR_PROMPT,
        **_callback_kwargs(verbose),
    )


def run_research(
    request: ResearchRequest, grounded: bool = False, model=None, verbose: bool = False
) -> ResearchReport:
    """Run the coordinator; it decides subtopics and spawns a sub-agent per subtopic.

    Unlike a Python fan-out, the coordinator LLM drives the fan-out: it calls
    research_topic once per subtopic it identifies. We collect each SubFinding as
    a side effect and use the coordinator's own output as the synthesized summary.
    `verbose=True` streams Strands' live tool-call trace; the default is quiet.
    """
    findings: list[SubFinding] = []
    coordinator = build_coordinator(findings, grounded=grounded, model=model, verbose=verbose)
    prompt = f"{request.question}\n\nAim for about {request.n_subtopics} subtopics."
    summary = str(coordinator(prompt))
    return ResearchReport(question=request.question, summary=summary, findings=findings)
