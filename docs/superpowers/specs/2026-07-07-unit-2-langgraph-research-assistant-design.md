# Unit 2 — LangGraph Research Assistant

**Status:** Approved design, pre-implementation
**Date:** 2026-07-07
**Repo purpose:** A comparative learning repo. The same research-assistant task built in
Strands (Unit 1, done + merged) and LangGraph (Unit 2, this spec), reusing all of `common/`
verbatim, so the *only* thing that differs when reading the two implementations is the
framework itself. Unit 3 later deploys to AWS Bedrock AgentCore.

This spec covers **Unit 2 only**. It supersedes the one-line Unit 2 sketch in the Unit 1
spec (which guessed `StateGraph` + `Send` fan-out) — see Decision log for why that guess was
wrong.

---

## Blast radius

`common/` is reused verbatim — **no changes**. New files, all under `langgraph_app/` + `tests/`,
plus a deps bump to `pyproject.toml`:

| File | Rough LOC | Purpose |
|---|---|---|
| `pyproject.toml` | ~+3 | add `langgraph`, `langchain-google-genai`, `langchain-core` |
| `langgraph_app/__init__.py` | ~1 | package marker |
| `langgraph_app/model.py` | ~15 | `build_gemini_model(grounded=False)` → `ChatGoogleGenerativeAI` |
| `langgraph_app/basic.py` | ~30 | Stage A: single prebuilt agent + one tool |
| `langgraph_app/research.py` | ~90 | Stage B: coordinator + agents-as-tools fan-out + streaming |
| `langgraph_app/run.py` | ~45 | CLI entrypoint, `--subtopics/--grounded/--verbose` |
| `tests/conftest.py` | ~+15 | add a LangChain `GenericFakeChatModel` fixture (append to existing) |
| `tests/test_langgraph_model.py` | ~30 | model factory attrs, grounded binding |
| `tests/test_langgraph_basic.py` | ~40 | Stage A reaches the tool |
| `tests/test_langgraph_research.py` | ~70 | fan-out, graceful degradation, findings collection |
| `tests/test_langgraph_run.py` | ~50 | CLI formatting, default-vs-verbose, error paths |

**~11 new files + 2 edits, well under the 15-file ceiling.** Mirrors the ~7 Strands files + tests.

## Problem

Unit 1 delivered `common/` + a complete Strands half. We now need the LangGraph half so a
reader can diff Strands vs LangGraph idioms side by side. `langgraph_app/` must produce the
*same* `ResearchReport` and the *same* CLI UX as `strands_app/`, reusing `common/` untouched,
so any difference in the two trees is attributable to the framework and nothing else.

## Non-goals

- **Not** changing `common/`, `strands_app/`, or any Unit 1 behaviour.
- **Not** deploying to AgentCore (Unit 3).
- **Not** a hand-built `StateGraph`, `Send` map-reduce, or the `langgraph-supervisor` package —
  all three are the wrong pattern here (see Decision log).
- **Not** async. The public `run_research(...) -> ResearchReport` surface stays **sync**, matching
  Unit 1 and the CLI.
- **Not** production hardening beyond the same graceful-degradation Unit 1 has.

## Approach

Mirror the **current official LangChain/LangGraph "Subagents" pattern** (agents-as-tools): a
coordinator agent whose tools are researcher agents wrapped with `@tool`. This is simultaneously
(a) the framework's currently-recommended multi-agent baseline and (b) the honest one-to-one
mirror of Strands' "agents as tools" — the two coincide, which is exactly what a comparative repo
wants.

### Task 0 (do first): add deps, pin, and verify the API

Nothing LangGraph-related is installed yet. The prebuilt-agent API renamed recently and the exact
import churned across versions, so **verify against what actually resolves before writing agent
code**:

```
uv add langgraph langchain-google-genai langchain-core
```

Then confirm, against the resolved versions, and bake the answers into the plan:

1. **Prebuilt agent entry point.** Prefer the current-canon `from langchain.agents import create_agent`
   (system-prompt kwarg = `system_prompt`). Fallback if unavailable: `from langgraph.prebuilt import
   create_react_agent` (kwarg = `prompt`; deprecated but functional). Pick one, pin the version, use
   it everywhere.
2. **Model kwarg** for `ChatGoogleGenerativeAI`: `streaming=True` present? API key as `google_api_key=`.
3. **Streaming** `graph.stream(inputs, stream_mode="messages")` yields `(chunk, metadata)` and
   `metadata["langgraph_node"]` names the node.
4. **Grounding** `model.bind_tools([{"google_search": {}}])`.
5. **Fake model** `from langchain_core.language_models.fake_chat_models import GenericFakeChatModel`,
   able to emit `AIMessage(tool_calls=[...])`.

### Layout (mirrors `strands_app/` file-for-file)

```
langgraph_app/
├─ __init__.py
├─ model.py       # build_gemini_model(grounded=False) -> ChatGoogleGenerativeAI
├─ basic.py       # Stage A: single prebuilt agent + one tool
├─ research.py    # Stage B: coordinator + agents-as-tools fan-out + streaming
└─ run.py         # CLI: python -m langgraph_app.run "question" [--subtopics N] [--grounded] [--verbose]
```

### Model factory (`langgraph_app/model.py`)

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from common.config import settings, require_api_key

def build_gemini_model(grounded: bool = False) -> ChatGoogleGenerativeAI:
    model = ChatGoogleGenerativeAI(
        model=settings.model_id,                 # gemini-2.5-flash
        google_api_key=require_api_key(),
        streaming=True,                          # REQUIRED for token streaming
    )
    return model.bind_tools([{"google_search": {}}]) if grounded else model
```

The single provider-wiring difference from `strands_app/model.py`. `streaming=True` is non-optional:
without it, `stream_mode="messages"` yields no token events.

### Stage A — basic API (`langgraph_app/basic.py`)

Smallest complete agent, to learn the prebuilt surface (agent construction, `@tool` registration,
the tool-call loop, `.invoke`/final-message reading):

```python
from langchain_core.tools import tool
from common.tools import mock_search as _mock_search
from common.prompts import SINGLE_AGENT_PROMPT
# from langchain.agents import create_agent  (or create_react_agent — decided in Task 0)

@tool
def mock_search(subtopic: str) -> str:
    """Search for information about a subtopic."""
    return _mock_search(subtopic)

def build_basic_agent(model=None):
    return create_agent(model or build_gemini_model(),
                        tools=[mock_search], system_prompt=SINGLE_AGENT_PROMPT)

def answer_question(question: str, model=None) -> str:
    result = build_basic_agent(model).invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
```

### Stage B — subagent spawning (`langgraph_app/research.py`)

The agents-as-tools fan-out. A `research_topic(subtopic)` tool wraps a fresh researcher agent; the
coordinator LLM decides the subtopics and calls that tool once per subtopic, then synthesizes.

```
route: run.py ─────────────► service: research.py ──────────► common/
  ResearchRequest              coordinator agent (prebuilt)      mock_search
                                 │ LLM decides subtopics
                                 ├─ research_topic(subtopic) ─┐  each call spins up a
                                 ├─ research_topic(subtopic) ─┤  FRESH researcher agent
                                 └─ synthesize (streamed) ────┘  (own model + mock_search)
                                       │
                                       ▼
                                 ResearchReport(question, summary, findings[])
```

**Findings collection — closure side-channel (mirror Unit 1).** `make_research_tool(findings, ...)`
returns a `@tool def research_topic(subtopic)` closed over that run's `list[SubFinding]`. The tool
body: prints `▸ researching: {subtopic}` (non-verbose only), invokes a fresh researcher agent,
appends `SubFinding(subtopic, findings=text, ok=True)`, and **returns the text string to the
coordinator** (which becomes the ToolMessage it synthesizes from). This is both the canonical
agents-as-tools mechanism (results flow back through messages) *and* Unit 1's exact trick — so
structured per-subtopic capture stays an implementation detail of the tool body, and `run_research`
needs no custom graph state.

**Graceful degradation:** the tool catches its own exceptions, appends `SubFinding(ok=False,
findings=f"failed: {exc}")`, and returns that string — one bad subtopic never aborts the run.

**Public seam (identical to Unit 1):**
```python
def run_research(request: ResearchRequest, grounded=False, model=None, verbose=False) -> ResearchReport:
    findings: list[SubFinding] = []
    coordinator = build_coordinator(findings, grounded=grounded, model=model, verbose=verbose)
    prompt = f"{request.question}\n\nAim for about {request.n_subtopics} subtopics."
    summary = _run_and_stream(coordinator, prompt, verbose=verbose)
    return ResearchReport(question=request.question, summary=summary, findings=findings)
```

**Streaming (the framework-idiom lesson).** `_run_and_stream` drives the coordinator via
`graph.stream(..., stream_mode="messages")`:
- **Default:** print only tokens whose `metadata["langgraph_node"]` is the coordinator's node, so
  the synthesis streams token-by-token while researcher sub-agent tokens stay quiet — the analog of
  Unit 1's `callback_handler` on the coordinator + `callback_handler=None` on sub-agents.
  Accumulate the printed tokens into the returned `summary`.
- **`--verbose`:** stream `stream_mode="updates"` instead, printing each node's state update (the
  canonical "watch the graph think" view) — the analog of Unit 1's raw framework trace. Return the
  final message content as `summary`.

`--subtopics N` is passed as a **soft hint** in the prompt (the LLM still decides the count).
`--grounded` builds the model with `bind_tools([{"google_search": {}}])` and gives the researcher no
`mock_search` tool (Gemini grounds internally), mirroring Unit 1.

### CLI (`langgraph_app/run.py`)

Byte-for-byte the same output contract as `strands_app/run.py`:

- `format_findings(report)` — 0 findings → `## Sub-agent findings\n_(coordinator answered directly
  without delegating to sub-agents)_`; else `## Sub-agent findings ({n} spawned)` then per finding
  `### {subtopic}{" (failed)" if not ok}` + body.
- `format_report(report)` — `# Research: {question}`, `## Summary`, summary, then `format_findings`.
- `main(argv=None) -> int` — argparse: positional `question`; `--subtopics` (int, default
  `settings.n_subtopics`); `--grounded`, `--verbose` (store_true).
  - **default:** summary already streamed live inside `run_research`; print `"\n"` then
    `format_findings(report)` only (must NOT re-print the summary).
  - **verbose:** print full `format_report(report)`.
  - `RuntimeError` → `error: {exc}` to stderr, return 1; other `Exception` → `research failed: {exc}`
    to stderr, return 1.

### Data flow

```
CLI (run.py)
  → ResearchRequest(question, n_subtopics)             [common/types — reused]
  → coordinator prebuilt agent (research.py)           [framework layer]
      → research_topic @tool ×N  → researcher agent    [spawned per subtopic]
          → mock_search(subtopic)                      [common/tools — reused, deterministic]
      → synthesize (streamed via stream_mode)
  → findings[] captured out-of-band in the tool closure
  → ResearchReport(question, summary, findings[])      [common/types — reused]
  → printed to stdout
```

### Error handling

- **Config:** `require_api_key()` (reused) fails fast with a clear message if `GOOGLE_API_KEY` is
  missing.
- **Sub-agent failure:** caught inside `research_topic`, recorded as `SubFinding(ok=False)`, never
  raised — graceful degradation.
- **Model/other errors:** surfaced at the `run.py` boundary as a clean CLI message, not a stack
  trace (same two-tier `RuntimeError` vs `Exception` handling as Unit 1).

### Testing (behaviour, not internals — no live calls by default)

Mirror Unit 1's strategy: monkeypatch the agent factory with stubs; no model needed for the
fan-out/CLI tests. Where a real model object is needed, inject `GenericFakeChatModel` emitting
canned `AIMessage`s (with `tool_calls` when a tool route must fire).

- `test_langgraph_model.py` — `build_gemini_model()` sets `.model == settings.model_id` and streaming;
  `grounded=True` returns a bound model; API key sourced from a monkeypatched `settings`. No call.
- `test_langgraph_basic.py` — Stage A agent reaches `mock_search` and returns a non-empty answer
  (fake model emits a tool call then a final message). Asserts the tool was reached, not how.
- `test_langgraph_research.py` — `research_topic` spawns a researcher and records a `SubFinding`;
  a forced-error case proves graceful degradation (`ok=False`, run continues); `run_research`
  collects findings via the closure and uses coordinator output as `summary`; coordinator registers
  `research_topic`. Factory faked at the boundary → fast, free, deterministic.
- `test_langgraph_run.py` — `format_findings`/`format_report` for 0/N/failed findings; default path
  prints only findings (not the summary); verbose prints full report; error paths return 1 with the
  right stderr prefix. `run_research` monkeypatched.
- **One opt-in live smoke test** (`@pytest.mark.live`, skipped without `GOOGLE_API_KEY`) exercises
  the real LLM-driven fan-out end-to-end — faking the coordinator's tool-calling loop would assert
  nothing real.

## Alternatives considered

- **Hand-built `StateGraph` (model node + `ToolNode` + `tools_condition`), findings via a custom
  state reducer.** Rejected: current official docs treat hand-building the ReAct loop as the
  *legacy/lower-level* path, recommended only for parallel fan-out or custom retry — and its
  parallel primitive is `Send`, which is code-orchestration (the pattern we deliberately moved away
  from). It would also make the LangGraph tree diverge structurally from `strands_app/`, defeating
  the side-by-side goal. (Considered specifically because it teaches more machinery; the machinery
  it teaches is now non-canonical for this use case.)
- **`langgraph-supervisor` / `create_supervisor`.** Rejected: the package's own README self-deprecates
  it and points users to exactly the tool-calling (agents-as-tools) pattern we're using.
- **`astream_events()` for streaming.** Rejected: forces `run_research` async (breaking the sync
  surface Unit 1 established) and is the lower-level event API; `stream_mode="messages"` was
  introduced so token UIs rarely need it.
- **State reducer for findings instead of the closure.** Rejected: non-canonical for agents-as-tools
  (results are meant to flow through messages), and it complicates the sync `run_research` seam and
  the no-live-call test strategy for no learning gain that the streaming modes don't already provide.

## Decision log

- Chose prebuilt **agents-as-tools** over a hand-built `StateGraph`+`Send` because the current
  LangChain docs make agents-as-tools the recommended multi-agent baseline *and* it is the honest
  one-to-one mirror of Strands — the boring-standard and the honest-mirror coincide. (This reverses
  the Unit 1 spec's Unit 2 sketch, which guessed `StateGraph`/`Send` before the API was researched.)
- Chose to target `langchain.agents.create_agent` over the deprecated `langgraph.prebuilt.create_react_agent`,
  with a version-gated fallback, because the former is current canon — but the exact import churned
  recently, so the choice is pinned in Task 0 against the resolved version rather than assumed.
- Chose the **closure side-channel** for findings over a state reducer because it is simultaneously
  Unit 1's mechanism and the canonical agents-as-tools mechanism (tool returns a string to the
  coordinator; structured capture is a tool-body detail), keeping `run_research` sync and testable.
- Chose `stream_mode="messages"` filtered by `metadata["langgraph_node"]` for token streaming, and
  `stream_mode="updates"` for `--verbose`, over `astream_events`, to keep the sync surface and to
  teach the two boring-standard streaming modes that map cleanly onto Unit 1's stream/verbose split.
- Chose `GenericFakeChatModel` (can emit tool calls, streams chunks) over the Strands `FakeModel`
  (framework-specific, doesn't port) for the boundary fake in tests.

## Risks & rollback

- **Prebuilt-agent API drift** (rename/import churn, `prompt` vs `system_prompt`). Mitigation: Task 0
  pins the exact entry point + kwarg against the resolved version before any agent code is written;
  `model.py`/`basic.py`/`research.py` all import it from one place.
- **`streaming=True` or node-name filtering behaves differently** than documented on the resolved
  Gemini integration (e.g. no token events, unexpected `langgraph_node` value). Mitigation: verify in
  Task 0 with a tiny live probe; the streaming code is isolated in `_run_and_stream`, one place to
  adjust.
- **`langchain-google-genai` co-installs a second Google SDK** (`google-generativeai` vs the
  `google-genai` already pulled by strands). Mitigation: inspect `uv.lock` after `uv add`; it's
  additive and harmless, but note it so a future reader isn't surprised.
- **Rollback:** the entire unit is new files under `langgraph_app/` + `tests/` plus a deps bump;
  reverting is deleting them and restoring `pyproject.toml`/`uv.lock`. `common/` and `strands_app/`
  are untouched.

## Open questions

- **`create_agent` vs `create_react_agent`** — resolved in Task 0 against the installed version, not
  now (we haven't installed anything). Whichever resolves cleanly with Gemini + a system prompt +
  streaming wins.
- **Does a `COORDINATOR_PROMPT` tweak help LangGraph?** The prompt says "Call the research_topic tool
  once for each subtopic," which is framework-neutral. If LangGraph's coordinator needs a nudge, add
  a `COORDINATOR_PROMPT` variant in `common/prompts.py` rather than inlining — decide during
  implementation, only if a live run shows it's needed.
