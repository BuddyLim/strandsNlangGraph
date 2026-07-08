"""Pedagogical side-by-side: Stage B coordinator hand-built as an explicit StateGraph.

This is what `langgraph_app/research.py`'s `create_agent` coordinator *desugars to*. It is
deliberately NOT wired into the CLI — it exists to be diffed against the prebuilt version so
two things become visible:

1. **The ReAct loop the prebuilt hides.** `create_agent` builds exactly this graph for you:
   a coordinator (model) node, a conditional edge (`tools_condition`), a tool node, and an
   edge back to the model. Here it is spelled out.
2. **LangGraph's distinctive state model.** In `research.py`, findings are gathered in an
   out-of-band closure because the tool can't easily write to graph state. Here, findings
   live in a typed state channel with a **reducer** (`operator.add`), and the tool node
   writes to *two* channels at once (`messages` and `findings`) — something the closure
   trick can't express. This is the part of LangGraph that has no Strands analogue.

Streaming is intentionally omitted to keep the state lesson uncluttered; see `research.py`
for the `stream_mode="messages"` / `"updates"` version.

Run it: `python -m langgraph_app.research_graph "your question"` (needs GOOGLE_API_KEY).
"""

from __future__ import annotations

import operator
import sys
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from common.prompts import COORDINATOR_PROMPT, SUB_AGENT_PROMPT
from common.types import ResearchReport, ResearchRequest, SubFinding
from langgraph_app.basic import mock_search
from langgraph_app.model import build_gemini_model


class ResearchState(TypedDict):
    """The graph's typed state. Each channel names a reducer that merges node outputs.

    `add_messages` is the same append/merge reducer `create_agent` uses under the hood.
    `operator.add` makes `findings` accumulate across research-node runs — the LangGraph
    state model doing explicitly what research.py's closure does implicitly.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    findings: Annotated[list[SubFinding], operator.add]


@tool
def research_topic(subtopic: str) -> str:
    """Research one subtopic and return factual findings about it."""
    # Exists mainly as a schema for the coordinator to call; in the graph the research
    # node (below) drives execution so it can also write to the `findings` channel.
    text, _ok = _run_researcher(subtopic)
    return text


def _run_researcher(
    subtopic: str, grounded: bool = False, model=None, verbose: bool = False
) -> tuple[str, bool]:
    """Spawn a fresh researcher sub-agent for one subtopic; return (text, ok).

    Never raises: on any error it returns a failure string and ok=False, so one bad
    subtopic degrades gracefully instead of aborting the graph run.
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
        return result["messages"][-1].content, True
    except Exception as exc:  # noqa: BLE001 — deliberate graceful-degradation boundary
        return f"failed: {exc}", False


def _coordinator_node(state: ResearchState, model) -> dict:
    """Call the coordinator LLM (bound to research_topic). Its reply drives routing:
    a reply with tool_calls means "keep researching"; a plain reply is the synthesis.
    """
    return {"messages": [model.invoke(state["messages"])]}


def _research_node(
    state: ResearchState, grounded: bool = False, model=None, verbose: bool = False
) -> dict:
    """Execute the coordinator's research_topic tool calls.

    For each call it spawns a researcher and emits BOTH a ToolMessage back to `messages`
    (so the coordinator sees the result) AND a SubFinding to `findings` (the structured
    record). One node writing to two state channels is the thing the closure can't do —
    the reducers on ResearchState merge each channel independently.
    """
    last = state["messages"][-1]
    tool_messages: list[ToolMessage] = []
    findings: list[SubFinding] = []
    for call in last.tool_calls:
        subtopic = call["args"]["subtopic"]
        text, ok = _run_researcher(subtopic, grounded=grounded, model=model, verbose=verbose)
        tool_messages.append(ToolMessage(content=text, tool_call_id=call["id"]))
        findings.append(SubFinding(subtopic=subtopic, findings=text, ok=ok))
    return {"messages": tool_messages, "findings": findings}


def _should_continue(state: ResearchState) -> str:
    """Hand-rolled equivalent of the prebuilt `tools_condition` edge: route to the research
    node while the coordinator is still calling tools, otherwise finish.
    """
    last = state["messages"][-1]
    return "research" if getattr(last, "tool_calls", None) else END


def build_research_graph(grounded: bool = False, model=None, verbose: bool = False):
    """Compile the explicit coordinator↔research loop.

    The coordinator is ungrounded (it only delegates); `grounded` is threaded to the
    researchers, mirroring research.py. Returns a compiled graph with the same runtime
    contract as create_agent's output (`.invoke`/`.stream` over a messages-bearing state).
    """
    coordinator = (model or build_gemini_model(grounded=False)).bind_tools([research_topic])

    graph = StateGraph(ResearchState)
    graph.add_node("coordinator", lambda s: _coordinator_node(s, coordinator))
    graph.add_node(
        "research", lambda s: _research_node(s, grounded=grounded, model=model, verbose=verbose)
    )
    graph.add_edge(START, "coordinator")
    graph.add_conditional_edges("coordinator", _should_continue, {"research": "research", END: END})
    graph.add_edge("research", "coordinator")
    return graph.compile()


def run_research_graph(
    request: ResearchRequest, grounded: bool = False, model=None, verbose: bool = False
) -> ResearchReport:
    """Same public shape as research.run_research — but findings come from graph STATE.

    Contrast with research.run_research, which reads findings from a closure list. Here the
    final state carries them, collected by the `findings` reducer as the graph ran.
    """
    graph = build_research_graph(grounded=grounded, model=model, verbose=verbose)
    initial: ResearchState = {
        "messages": [
            SystemMessage(content=COORDINATOR_PROMPT),
            HumanMessage(content=f"{request.question}\n\nAim for about {request.n_subtopics} subtopics."),
        ],
        "findings": [],
    }
    final = graph.invoke(initial)
    return ResearchReport(
        question=request.question,
        summary=final["messages"][-1].content,
        findings=final["findings"],
    )


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What is photosynthesis?"
    report = run_research_graph(ResearchRequest(question=question))
    print(f"\n## Summary\n{report.summary}")
    print(f"\n## Findings ({len(report.findings)} subtopics)")
    for finding in report.findings:
        print(f"- {finding.subtopic}{'' if finding.ok else ' (failed)'}")
