# Unit 1 — Shared Core + Strands Research Assistant

**Status:** Approved design, pre-implementation
**Date:** 2026-07-07
**Repo purpose:** A comparative learning repo. Same task (a research assistant) built in
both Strands and LangGraph to learn (1) each framework's basic API and (2) each
framework's subagent-spawning model, then deployed to AWS Bedrock AgentCore as the
runtime. This spec covers **Unit 1 only** — the shared core plus the Strands half.
Units 2 (LangGraph) and 3 (AgentCore deploy) are sketched at the end and get their own
spec → plan → implementation cycles.

---

## Blast radius

New repo, no existing code to change. Files created in Unit 1:

| File | Rough LOC | Purpose |
|---|---|---|
| `pyproject.toml` | ~40 | deps + extras (`strands-agents[gemini]`) |
| `.env.example` | ~3 | `GOOGLE_API_KEY` template |
| `common/config.py` | ~30 | env loading, fail-fast on missing key |
| `common/prompts.py` | ~25 | coordinator + sub-agent system prompts |
| `common/tools.py` | ~40 | `mock_search` deterministic `@tool` |
| `common/types.py` | ~30 | Pydantic: `ResearchRequest`, `ResearchReport`, `SubFinding` |
| `strands_app/model.py` | ~15 | `build_gemini_model()` factory |
| `strands_app/basic.py` | ~30 | Stage A: single agent + one tool |
| `strands_app/research.py` | ~70 | Stage B: coordinator + agents-as-tools fan-out |
| `strands_app/run.py` | ~40 | CLI entrypoint, `--grounded` flag |
| `tests/test_common_tools.py` | ~25 | |
| `tests/test_strands_basic.py` | ~40 | |
| `tests/test_strands_research.py` | ~60 | |

**~13 files, well under the 15-file ceiling.** `common/` is written once here and reused
verbatim by Unit 2.

## Problem

Empty repo. We want to understand two agent-orchestration frameworks by building the same
non-trivial task in each, side by side, so that the only thing that differs when reading
the two implementations is the framework itself. Unit 1 delivers the shared foundation and
one complete, runnable, comparable half (Strands).

## Non-goals

- **Not** building LangGraph yet (Unit 2) or deploying to AgentCore yet (Unit 3).
- **Not** real web search as the primary tool — the mock keeps runs deterministic and free.
- **Not** a provider abstraction layer — Gemini is the single provider (see Decision log).
- **Not** production hardening (retries/backoff policies, observability wiring) beyond the
  minimal graceful-degradation described here.

## Approach

Approach A: a shared `common/` package consumed verbatim by each framework app, with the
Strands implementation in `strands_app/`. Two learning stages live side by side so the
basic-API lesson and the subagent lesson are each isolated.

### Layout

```
strandsNlangGraph/
├─ pyproject.toml
├─ .env.example
├─ common/
│  ├─ config.py      # GOOGLE_API_KEY, model_id, n_subtopics; fail-fast
│  ├─ prompts.py     # coordinator + sub-agent prompts (shared verbatim across frameworks)
│  ├─ tools.py       # mock_search(subtopic) -> canned, deterministic results
│  └─ types.py       # ResearchRequest, ResearchReport, SubFinding (Pydantic)
├─ strands_app/
│  ├─ model.py       # build_gemini_model() -> GeminiModel
│  ├─ basic.py       # Stage A: single agent + one tool
│  ├─ research.py    # Stage B: coordinator spawns N sub-agents (agents-as-tools)
│  └─ run.py         # CLI: python -m strands_app.run "question" [--grounded]
└─ tests/
```

### Stage A — basic API (`strands_app/basic.py`)

Smallest complete Strands agent, to learn the core surface (Agent construction, tool
registration, the tool-call loop):

```python
from strands import Agent
from strands_app.model import build_gemini_model
from common.tools import mock_search        # @tool-decorated function
from common.prompts import SINGLE_AGENT_PROMPT

agent = Agent(model=build_gemini_model(), tools=[mock_search],
              system_prompt=SINGLE_AGENT_PROMPT)
answer = agent("Give me a short brief on X")
```

### Stage B — subagent spawning (`strands_app/research.py`)

Strands' idiomatic "agents as tools" fan-out. A `researcher(subtopic)` sub-agent is wrapped
as a `@tool`; the coordinator calls it once per sub-topic and synthesizes:

```
coordinator agent
  ├─ decides sub-topics from the question
  ├─ calls researcher(subtopic)  ─┐  each call spins up a fresh sub-agent
  ├─ calls researcher(subtopic)  ─┤  (its own model + mock_search tool)
  └─ synthesizes → ResearchReport ┘
```

The `researcher` tool catches its own exceptions and returns a structured
"sub-topic failed: <reason>" string so one bad sub-topic degrades gracefully.

The **coordinator LLM decides how many subtopics** to research and drives the fan-out
itself (it is not a Python loop). `ResearchRequest.n_subtopics` / the `--subtopics` flag
is passed to the coordinator as a *soft target* ("aim for about N subtopics"), not a hard
count — the model remains in charge. This is the distinctive Strands "the model
orchestrates" lesson, and it maps in Unit 2 onto a LangGraph tool-calling agent.

### Model factory (`strands_app/model.py`)

```python
from strands.models.gemini import GeminiModel
from common.config import settings

def build_gemini_model() -> GeminiModel:
    return GeminiModel(client_args={"api_key": settings.google_api_key},
                       model_id=settings.model_id)   # gemini-2.5-flash
```

One place constructs the model; both stages import it. Unit 2 writes the LangGraph
equivalent (`ChatGoogleGenerativeAI`) — the only provider-wiring difference between apps.

### Data flow

```
CLI (run.py)
  → ResearchRequest(question, n_subtopics)        [common/types]
  → coordinator Agent (strands_app/research.py)   [framework layer]
      → researcher sub-agent @tool  ×N            [spawned per sub-topic]
          → mock_search(subtopic)                 [common/tools — deterministic]
      → synthesize
  → ResearchReport(summary, findings[])           [common/types]
  → printed to stdout
```

A `--grounded` flag on `run.py` swaps `mock_search` for Gemini native grounding
(`google_search` tool on the model) to contrast "framework routes the tool" vs "model
grounds internally."

### Error handling

- **Config**: `common/config.py` fails fast with a clear message if `GOOGLE_API_KEY` is
  missing — no half-initialized runs.
- **Sub-agent failure**: the `researcher` tool catches its own exceptions and returns a
  structured failure string; one bad sub-topic degrades gracefully instead of killing the
  report.
- **Model throttling**: Gemini raises `ModelThrottledException`; caught at the `run.py`
  boundary and surfaced as a clean CLI message, not a stack trace.

### Testing (behaviour, not internals)

- `test_common_tools.py` — `mock_search` returns deterministic results for a known
  sub-topic (pure, no model).
- `test_strands_basic.py` — Stage A agent, Gemini model faked at the boundary, reaches
  `mock_search` and returns a non-empty answer. Asserts the tool was reached, not how.
- `test_strands_research.py` — `_research_impl` spawns a researcher sub-agent and records a
  `SubFinding` (with a forced-error test proving graceful degradation); `run_research`
  collects findings via the tool closure and uses the coordinator's output as the summary;
  the coordinator registers `research_topic` as a tool. Agents/model faked at the boundary →
  fast, free, deterministic. The LLM-driven fan-out itself is exercised by the opt-in `live`
  test, since faking the coordinator's tool-calling loop would assert nothing real.

## Alternatives considered

- **Approach B — two independent mini-projects** (near-zero sharing). Rejected: duplicates
  tool/prompt/config code and makes framework differences indistinguishable from our own
  drift.
- **Approach C — one app, pluggable orchestrator behind an interface.** Rejected: forces
  both frameworks behind our abstraction, hiding the very idioms we want to learn.
- **Gemini native grounding as the only research tool.** Rejected as primary: the search
  happens inside the model call, so the framework tool-call loop stays hidden and goal #1
  (basic API) is under-served. Kept as a `--grounded` contrast variant instead.

## Decision log

- Chose Gemini-only over a provider abstraction because it's YAGNI for a learning repo and
  a real abstraction would hide framework idioms; a one-function model factory is enough.
- Chose a mock function tool over real web search as the primary tool because determinism +
  no extra credential keeps the focus on framework mechanics.
- Chose the idiomatic "agents as tools" pattern (coordinator LLM decides subtopics and calls
  a `research_topic` @tool per subtopic) over a code-orchestrated Python fan-out for Stage B,
  because learning-goal #2 is to see each framework's *distinctive* subagent mechanism, and
  LLM-driven delegation is Strands' signature. In Unit 2 this maps onto a LangGraph
  tool-calling (ReAct) agent, not `Send` — that is the honest cross-framework comparison.
  (An earlier plan draft used a code-orchestrated loop; reworked to this after review.)
- Chose to mock the model at the boundary in tests over live calls because tests must be
  fast, free, and deterministic.

## Risks & rollback

- **Strands Gemini provider API drifts** (class/args change). Mitigation: the model factory
  is the single point of change; verified against current docs
  (`GeminiModel(client_args={"api_key": ...}, model_id=...)`).
- **`common/` shape doesn't fit LangGraph cleanly** in Unit 2. Mitigation: `common/` holds
  only framework-neutral things (Pydantic types, prompts, a plain function tool, config);
  nothing Strands-specific leaks in. If a mismatch appears, adjust `common/` in Unit 2 — it
  has no external consumers yet.
- **Rollback**: the entire unit is new files in a fresh repo; reverting is deleting them.

## Open questions

- **Git**: repo is not yet initialized. Init and commit the spec, or hold off? (Default
  intent: `git init` and commit the design doc.)
- **Test runner / lint**: assume `pytest` + `ruff`? (Default: yes, added in `pyproject.toml`
  during implementation.)

---

## Units 2 & 3 (sketch only — separate specs later)

- **Unit 2 — LangGraph**: `langgraph_app/` mirrors `strands_app/` file-for-file, reusing all
  of `common/`. Stage A = a `StateGraph` with a tool node; Stage B = subgraphs / `Send`
  fan-out for sub-agents. Same CLI, same `ResearchReport` out. No changes to Unit 1.
- **Unit 3 — AgentCore deploy**: `deploy/` wraps each app's entrypoint in
  `BedrockAgentCoreApp` (`@app.entrypoint`), containerizes, injects `GOOGLE_API_KEY` as a
  secret, deploys via the AgentCore starter toolkit. Only meaningful once ≥1 app runs
  locally.
