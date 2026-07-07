import sys

from langchain.agents import create_agent  # Task 0 pin: fallback alias as in basic.py
from langchain_core.tools import tool

from common.prompts import COORDINATOR_PROMPT, SUB_AGENT_PROMPT
from common.types import ResearchReport, ResearchRequest, SubFinding
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


def _ai_text(content) -> str:
    """Flatten a message's content to text (Gemini emits str; guard the list form)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return ""


def _stream_tokens(coordinator, inputs) -> str:
    """Default mode: stream only the coordinator node's tokens; return the synthesis.

    Researcher sub-agents are invoked inside the research_topic tool (separate graph
    invocations), so their tokens never enter this stream; filtering by
    _COORDINATOR_NODE additionally guards against any other node's chatter.
    """
    chunks: list[str] = []
    for message_chunk, metadata in coordinator.stream(inputs, stream_mode="messages"):
        if metadata.get("langgraph_node") != _COORDINATOR_NODE:
            continue
        text = _ai_text(message_chunk.content)
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
            chunks.append(text)
    return "".join(chunks)


def _stream_updates(coordinator, inputs) -> str:
    """Verbose mode: print each node's state update; return the final AI message text."""
    summary = ""
    for update in coordinator.stream(inputs, stream_mode="updates"):
        for node, state in update.items():
            print(f"[{node}] {state}", flush=True)
            for msg in (state or {}).get("messages", []) or []:
                if getattr(msg, "type", "") == "ai":
                    text = _ai_text(getattr(msg, "content", ""))
                    if text:
                        summary = text
    return summary


def _run_and_stream(coordinator, prompt: str, verbose: bool) -> str:
    inputs = {"messages": [{"role": "user", "content": prompt}]}
    return _stream_updates(coordinator, inputs) if verbose else _stream_tokens(coordinator, inputs)


def run_research(
    request: ResearchRequest, grounded: bool = False, model=None, verbose: bool = False
) -> ResearchReport:
    """Run the coordinator; it decides subtopics and spawns a sub-agent per subtopic.

    The coordinator LLM drives the fan-out: it calls research_topic once per subtopic
    it identifies. We collect each SubFinding as a side effect (closure) and use the
    coordinator's streamed synthesis as the summary. By default the run streams to
    stdout — a progress line per subtopic and the synthesis token by token;
    verbose=True prints the graph's state transitions instead.
    """
    findings: list[SubFinding] = []
    coordinator = build_coordinator(findings, grounded=grounded, model=model, verbose=verbose)
    prompt = f"{request.question}\n\nAim for about {request.n_subtopics} subtopics."
    summary = _run_and_stream(coordinator, prompt, verbose=verbose)
    return ResearchReport(question=request.question, summary=summary, findings=findings)
