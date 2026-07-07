from langchain.agents import create_agent  # Task 0 pin: fallback alias as in basic.py
from langchain_core.tools import tool

from common.prompts import COORDINATOR_PROMPT, SUB_AGENT_PROMPT
from common.types import SubFinding
from langgraph_app.basic import mock_search
from langgraph_app.model import build_gemini_model

# The prebuilt agent's model node — the stream node whose tokens we surface.
# Confirmed static-from-graph in Task 0 (agent.get_graph().nodes on langchain
# 1.3.11's create_agent): the model node is named "model", not "agent".
_COORDINATOR_NODE = "model"


def _research_impl(
    subtopic: str,
    findings: list[SubFinding],
    grounded: bool = False,
    model=None,
    verbose: bool = False,
) -> str:
    """Spawn a fresh researcher sub-agent for one subtopic.

    Constructing the agent IS the subagent spawn. Returns the researcher's final
    text — what the coordinator receives as the tool result and synthesizes.
    Separately appends a SubFinding to `findings`, an out-of-band collector the run
    uses to reconstruct the per-subtopic breakdown; the coordinator never reads it.
    Degrades gracefully: on any error it records ok=False and returns a failure
    string rather than raising, so one bad subtopic never aborts the run.
    """
    if not verbose:
        print(f"▸ researching: {subtopic}\n", flush=True)
    try:
        sub_model = model or build_gemini_model(grounded=grounded)
        tools = [] if grounded else [mock_search]
        researcher = create_agent(sub_model, tools=tools, system_prompt=SUB_AGENT_PROMPT)
        result = researcher.invoke(
            {"messages": [{"role": "user", "content": f"Research this subtopic: {subtopic}"}]}
        )
        text = result["messages"][-1].content
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
        return _research_impl(subtopic, findings, grounded=grounded, model=model, verbose=verbose)

    return research_topic


def build_coordinator(
    findings: list[SubFinding], grounded: bool = False, model=None, verbose: bool = False
):
    """Coordinator agent that decomposes the question and delegates via research_topic."""
    return create_agent(
        model or build_gemini_model(),
        tools=[make_research_tool(findings, grounded=grounded, model=model, verbose=verbose)],
        system_prompt=COORDINATOR_PROMPT,  # Task 0 pin: `prompt=` on the fallback API
    )
