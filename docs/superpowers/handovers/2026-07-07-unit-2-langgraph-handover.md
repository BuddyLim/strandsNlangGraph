# Handover — Unit 2: LangGraph research assistant

**For:** the session that will build Unit 2.
**Status of Unit 1:** complete and merged to `master` (see log from `f3a5fc8` … `e42d855`).
**How to use this doc:** read it, then run your own `brainstorming → writing-plans →
subagent-driven-development` cycle. This is context, not a plan — don't skip the design step.

---

## Mission

This repo is a **comparative learning repo**: the same research-assistant task built in
**Strands** (Unit 1, done) and **LangGraph** (Unit 2, this handover), so the *only* thing that
differs when reading the two implementations is the framework itself. Unit 3 later deploys to
AWS Bedrock AgentCore.

**Unit 2 goal:** build `langgraph_app/` that mirrors `strands_app/` file-for-file, reusing all of
`common/` verbatim, producing the *same* `ResearchReport` and the *same* CLI behaviour — so a
reader can diff Strands vs LangGraph idioms side by side.

---

## What Unit 1 delivered (mirror this)

```
common/                     ← FRAMEWORK-NEUTRAL. Reuse verbatim. Do NOT import a framework here.
├─ config.py                # settings (google_api_key, model_id="gemini-2.5-flash", n_subtopics), require_api_key()
├─ types.py                 # ResearchRequest, SubFinding, ResearchReport  (Pydantic)
├─ tools.py                 # mock_search(subtopic) -> str  (plain deterministic fn; each framework wraps it)
└─ prompts.py               # SINGLE_AGENT_PROMPT, SUB_AGENT_PROMPT, COORDINATOR_PROMPT
strands_app/
├─ model.py                 # build_gemini_model(grounded=False) -> GeminiModel
├─ basic.py                 # Stage A: single agent + one tool
├─ research.py              # Stage B: agents-as-tools fan-out + streaming
└─ run.py                   # CLI: python -m strands_app.run "..." [--subtopics N] [--grounded] [--verbose]
tests/                      # flat layout, one test file per module; conftest.py has a FakeModel fixture
```

**Stage A** — smallest complete agent: model + the `mock_search` tool + `SINGLE_AGENT_PROMPT`.

**Stage B — the important one.** Strands' idiomatic **"agents as tools"**: the researcher sub-agent
is registered as a `@tool` (`research_topic(subtopic)`), and the **coordinator LLM decides the
subtopics and calls that tool once per subtopic**, then synthesizes. Each tool call spawns a fresh
sub-agent. Findings are collected out-of-band into a `list[SubFinding]` closed over by the tool;
the coordinator's returned text is the summary. Graceful degradation: a failing sub-topic records
`SubFinding(ok=False)` and never aborts the run.

**CLI output behaviour (match this exactly in Unit 2):**
- **Default = streaming**: prints `▸ researching: <subtopic>` per sub-agent, streams the summary
  **token by token**, then prints a labelled `## Sub-agent findings (N spawned)` recap.
- **`--verbose`**: raw framework trace + full `format_report`.
- `--subtopics N` is a **soft hint** to the coordinator (LLM still decides the count); `--grounded`
  swaps the mock tool for Gemini's native Google Search.

---

## The honest cross-framework mapping

| Strands (Unit 1) | LangGraph (Unit 2) |
|---|---|
| `GeminiModel(client_args={"api_key":…}, model_id=…)` | `ChatGoogleGenerativeAI(model=…, api_key=…, streaming=True)` |
| `Agent(model, tools, system_prompt)` | `create_react_agent(model, tools, prompt=…)` (prebuilt tool-calling agent) |
| researcher wrapped as `@tool`, coordinator LLM calls it per subtopic | a `research_topic` tool (LangChain `@tool`) whose body invokes a researcher `create_react_agent`; coordinator is a `create_react_agent` given that tool |
| streaming via a custom `callback_handler` printing `data` chunks | streaming via `graph.stream(..., stream_mode="messages")` or `astream_events()` — token chunks come as `(message_chunk, metadata)` |
| grounded: `GeminiModel(gemini_tools=[types.Tool(google_search=types.GoogleSearch())])` | grounded: `model.bind_tools([{"google_search": {}}])` |

**Key point:** the Strands agents-as-tools pattern maps onto a **LangGraph tool-calling (ReAct)
agent**, NOT `Send`. `Send` is a *different* LangGraph feature (map-reduce over a fixed list) and
would be code-orchestration, which is the pattern we deliberately moved away from. Keep the
comparison honest: coordinator LLM decides + delegates via a tool in both.

---

## Concrete LangGraph pointers (verify against installed versions during your plan)

- **Deps to add:** `uv add langgraph langchain-google-genai langchain-core`. Check resolved
  versions and pin the LangGraph API you target (the prebuilt-agent and streaming APIs move fast).
- **Model:** `ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=…, streaming=True)`.
  `streaming=True` is REQUIRED for token streaming — without it you get no token events.
- **Tool-calling agent:** `from langgraph.prebuilt import create_react_agent` (confirm the import
  path for the installed version). Supply model + tools + a system prompt.
- **Token streaming:** `stream_mode="messages"` streams tokens from anywhere in the graph
  (including tools/subgraphs) as `(chunk, metadata)`; `metadata["langgraph_node"]` tells you which
  node/subgraph produced it — useful to stream ONLY the coordinator's synthesis and keep sub-agents
  quiet, mirroring Unit 1. `astream_events()` is the fine-grained async alternative.
- **Subtopic progress:** simplest reliable source is the `research_topic` tool body itself (it has
  the subtopic as its argument) — print `▸ researching: <subtopic>` there, exactly like Unit 1's
  `_research_impl`. Don't try to parse it out of streamed tool-input deltas.
- **Findings collection:** the same out-of-band closure trick from Unit 1 works, OR use LangGraph
  state (a `findings` channel with a reducer). Decide in your brainstorm; keep the public
  `run_research(request, grounded=False, verbose=False) -> ResearchReport` signature identical.

---

## Constraints carried from Unit 1 (non-negotiable)

- **Gemini everywhere.** One provider, one key (`GOOGLE_API_KEY`).
- **`common/` stays framework-neutral** — no `langgraph`/`langchain` imports under `common/`, ever.
  Wrap `common.tools.mock_search` in LangChain's `@tool` inside `langgraph_app/`, not in `common/`.
- **Same public surface:** `run_research(request, grounded=False, verbose=False) -> ResearchReport`,
  same CLI flags, same streaming-vs-verbose behaviour, same `ResearchReport(question, summary,
  findings)` output. A user should be able to swap `strands_app` ↔ `langgraph_app` and see the same
  UX.
- **Tests never make live calls** in the default run; one opt-in `@pytest.mark.live` smoke test
  skipped without `GOOGLE_API_KEY`. Run tests with `uv run pytest`.
- **Scope:** target ~15 files (AGENTS.md). This mirrors ~7 Strands files + tests, so it fits one
  unit. TDD, frequent commits.

---

## Environment gotchas discovered in Unit 1 (will save you time)

- **Toolchain:** uv-managed, `[tool.uv] package = false`, run from repo root. Tests: `uv run pytest`.
  Lint: `uv run ruff check .`. Python 3.11+ (venv is 3.11; local python is 3.12).
- **`git commit` hangs when run directly** via this session's shell (an `rtk` hook + heredoc issue).
  Workarounds that WORK: `command git commit -F <msgfile>` (write the message to a file first with
  the Write tool). `command git add` works fine. Avoid `-m "$(cat <<EOF…)"` (heredoc-in-substitution
  hangs) and avoid redirecting `git diff > file` (also hangs — stream to a pipe instead).
- **Strands-specific facts** (won't bite LangGraph, but explain Unit 1 code): `Agent(model=object())`
  fails (Agent reads `model.stateful`), so tests use a `FakeModel` (see `tests/conftest.py`);
  calling a `@tool`-decorated function directly can hang (tests never do); `GeminiModel` rejects
  `gemini_tools=None` so the ungrounded path omits the kwarg. LangGraph will have its own such
  quirks — verify, don't assume.

---

## Reuse map (import these; build nothing new here)

- `from common.config import settings, require_api_key`
- `from common.types import ResearchRequest, SubFinding, ResearchReport`
- `from common.tools import mock_search`  ← wrap in LangChain `@tool` inside `langgraph_app/`
- `from common.prompts import SINGLE_AGENT_PROMPT, SUB_AGENT_PROMPT, COORDINATOR_PROMPT`
  - Prompts are framework-neutral English; reuse as-is. If LangGraph needs a tweaked coordinator
    instruction, add a `COORDINATOR_PROMPT` variant in `common/prompts.py` rather than inlining.

---

## Suggested workflow for Unit 2

1. **brainstorming** — settle: create_react_agent vs a hand-built `StateGraph`; findings via closure
   vs graph state; streaming via `stream_mode="messages"` vs `astream_events`; how to keep sub-agents
   quiet while streaming only the coordinator. Write a spec to `docs/superpowers/specs/`.
2. **writing-plans** — TDD tasks mirroring Unit 1: deps + model factory → Stage A → Stage B fan-out →
   CLI/streaming. Verify LangGraph API facts live (like Unit 1 did for Strands) and bake them into
   the plan so implementers don't guess.
3. **subagent-driven-development** — fresh implementer + independent reviewer per task; final
   whole-branch review; then `finishing-a-development-branch`.
4. Branch off `master` first (don't implement on `master`).

## Open questions to resolve in the Unit 2 brainstorm

- Does the installed LangGraph expose `create_react_agent` at `langgraph.prebuilt`, and does it accept
  a system prompt + streaming model cleanly with Gemini? (There was a historical issue with
  pre-bound tools on Gemini — check.)
- Best seam to keep sub-agents quiet while streaming only the coordinator's synthesis — filter by
  `metadata["langgraph_node"]` in `stream_mode="messages"`?
- Findings collection: closure side-channel (like Unit 1) or a state reducer? Pick the one that
  keeps `run_research` testable without live calls.

## Pointers

- Unit 1 spec: `docs/superpowers/specs/2026-07-07-strands-research-assistant-design.md`
- Unit 1 plan: `docs/superpowers/plans/2026-07-07-strands-research-assistant.md`
- Reference code to mirror: `strands_app/research.py`, `strands_app/run.py`, `tests/conftest.py`
- External: [LangGraph streaming docs](https://docs.langchain.com/oss/python/langgraph/streaming),
  [ReAct agent with Gemini + LangGraph](https://ai.google.dev/gemini-api/docs/langgraph-example),
  [ChatGoogleGenerativeAI grounding](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)
