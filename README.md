# strands-n-langgraph

A comparative learning repo: the **same** research assistant built twice — once in
[Strands](https://github.com/strands-agents) (`strands_app/`) and once in
[LangGraph](https://github.com/langchain-ai/langgraph) (`langgraph_app/`) — behind one shared
seam, then deployed to **AWS Bedrock AgentCore** (`deploy/` + `agentcore/`).

Because both implementations expose an identical `run_research` function and CLI, the *only*
thing that differs between the two trees is the framework itself. That makes the frameworks
directly diffable — and it makes the deploy layer nearly framework-agnostic.

The work is organised in three units:

| Unit | What | Where |
|------|------|-------|
| 1 | Strands research assistant | `strands_app/` |
| 2 | LangGraph research assistant | `langgraph_app/` |
| 3 | Deploy to AWS Bedrock AgentCore | `deploy/`, `agentcore/`, root `Dockerfile` |

## What this teaches

This repo is a hands-on exercise with three learning goals:

1. **Two frameworks, one task** — build the *identical* research assistant in
   [Strands](https://github.com/strands-agents) and
   [LangGraph](https://github.com/langchain-ai/langgraph) and diff the APIs side by side
   (`strands_app/` vs `langgraph_app/`), with `common/` held constant so only the framework differs.
2. **Spawning and orchestrating sub-agents** — how a coordinator agent fans out into subtopics and
   delegates each to a sub-agent (the agents-as-tools pattern), and how that same fan-out is
   expressed in each framework (compare `research.py` against LangGraph's hand-built
   `research_graph.py`).
3. **Deploying to AgentCore** — taking the same local agent to a managed AWS Bedrock AgentCore
   runtime: containerising it, wrapping the seam in an HTTP entrypoint, and wiring config, env, and
   secrets (`deploy/` + `agentcore/`).

## What the assistant does

Given a question, a **coordinator** agent fans out into *N* subtopics, delegates each to a
sub-agent, and synthesises the findings into a report. The public seam (identical in both apps):

```python
run_research(request, grounded=False, model=None, verbose=False) -> ResearchReport
```

- `ResearchRequest(question: str, n_subtopics: int = 3)`
- `ResearchReport(question, summary, findings: list[SubFinding])` — Pydantic; `.model_dump()` gives clean JSON
- `grounded=True` swaps the mock search tool for Gemini's native Google Search grounding

## Codebase shape

```
common/            Framework-neutral shared seam — reused verbatim by both apps
  types.py           ResearchRequest, ResearchReport, SubFinding
  config.py          Settings (GOOGLE_API_KEY, MODEL_ID, N_SUBTOPICS), require_api_key()
  prompts.py         Coordinator / sub-agent system prompts
  tools.py           mock_search tool
strands_app/       Unit 1 — Strands implementation
  model.py           build_gemini_model()  (Gemini via strands.models.gemini)
  basic.py           Stage A: single agent + one tool
  research.py        Stage B: coordinator + agents-as-tools fan-out  -> run_research()
  run.py             CLI: main(argv); `python -m strands_app.run`
langgraph_app/     Unit 2 — LangGraph implementation (same seam)
  model.py           build_gemini_model()  (Gemini via langchain-google-genai)
  basic.py           Stage A
  research.py        Stage B coordinator  -> run_research()
  research_graph.py  Pedagogical hand-built StateGraph (NOT wired to the CLI) — what the
                     prebuilt coordinator desugars to; exists to be diffed against research.py
  run.py             CLI: main(argv); `python -m langgraph_app.run`
deploy/            Unit 3 — AgentCore HTTP wrapper
  server.py          BedrockAgentCoreApp entrypoint wrapping run_research
agentcore/         AgentCore CLI project (aws/agentcore-cli)
  agentcore.json     Runtimes, credentials, build config
  aws-targets.json   Deployment target (account + region)
  cdk/               CDK infrastructure the CLI deploys
  .llm-context/      TypeScript type definitions for the JSON config (read-only)
  .env.local         Local secrets + APP for `agentcore dev` (gitignored)
tests/             Flat pytest layout, one file per module
docs/superpowers/  Design specs, plans, handovers, and the deploy runbook
```

The repo is `[tool.uv] package = false` — a **path-import workspace**, not an installable
package. Code imports across packages by path (`from common.types import ...`), so anything that
runs it must have `common/` and the app packages physically present on the path.

## How AgentCore is used

**One thin wrapper for both frameworks.** `deploy/server.py` wraps the shared `run_research` in a
`BedrockAgentCoreApp` HTTP entrypoint and returns `ResearchReport.model_dump()` (sync JSON — no
streaming):

```python
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict, context) -> dict:
    return _handle(payload)   # payload -> ResearchRequest -> run_research -> report.model_dump()
```

**Framework selection via an `APP` env var (Pattern A).** A `pydantic-settings` `DeploySettings`
reads `APP` (`strands` | `langgraph`) — process environment first, then a local `.env` — and the
matching `run_research` is imported once at startup. A misconfigured value fails fast on cold
start. One wrapper, one image recipe; the two frameworks differ only by which env value they run
under.

**Repo-root Docker build context.** Because of `package = false`, the image must contain
`common/` *and* the app packages, so the build context is the repo root (`codeLocation: "."` in
`agentcore.json`). The root `Dockerfile` is uv-based (ARM64) and runs:

```
opentelemetry-instrument python -m deploy.server
```

**`runtimes[]` — one entry per deployed runtime.** Each becomes an independent AgentCore Runtime
(its own image, ARN, and endpoint). Only `strands` is wired now; LangGraph slots in later by
adding a second entry with `envVars: [{ "name": "APP", "value": "langgraph" }]` — no code change.

```jsonc
// agentcore/agentcore.json (current)
"runtimes": [
  {
    "name": "strands",
    "build": "Container",
    "entrypoint": "deploy/server.py",
    "codeLocation": ".",
    "dockerfile": "Dockerfile",
    "protocol": "HTTP",
    "envVars": [{ "name": "APP", "value": "strands" }]
  }
]
```

**Invocation payload.** AgentCore hands the entrypoint the **raw JSON body**. `_handle` accepts
the question under any of:

- `{"question": "..."}` — explicit contract (wins if present)
- `{"prompt": "..."}` — the AgentCore console/CLI default shape
- `{"messages": [...]}` — Bedrock messages; the last user message's text is used

plus optional `n_subtopics` (default 3) and `grounded`.

**Environment & secrets.**

- **`APP`** selects the framework. It is applied from `agentcore.json` `envVars` at **cloud
  deploy** — but local `agentcore dev` does **not** read those `envVars` (nor the repo-root
  `.env`, which is docker-ignored from the container). So for local `agentcore dev`, set it in
  `agentcore/.env.local` (gitignored) or pass `--env APP=strands`:

  ```dotenv
  # agentcore/.env.local  (gitignored — read by `agentcore dev`)
  APP=strands
  ```

  (The plain `python -m deploy.server` path instead reads `APP` from the repo-root `.env`.)
- **`GOOGLE_API_KEY`** is the only runtime credential. In the cloud it is supplied by the
  `strandsNlanggraphGemini` **ApiKeyCredentialProvider** (AgentCore Identity), **not** baked into
  the image — `.env` is git- and docker-ignored.
- **Model:** Google **Gemini** (`gemini-2.5-flash` default, override with `MODEL_ID`). AgentCore
  only *hosts the container*; inference goes to Google's API, not to a Bedrock model.

**Deploy flow:** `agentcore dev` builds and runs the container locally; `agentcore deploy` pushes
it to AWS; invoke with `agentcore invoke '{"prompt": "..."}'`. The full, spend-flagged runbook is
in [`docs/superpowers/runbooks/2026-07-08-unit-3-agentcore-deploy-runbook.md`](docs/superpowers/runbooks/2026-07-08-unit-3-agentcore-deploy-runbook.md).

## Quickstart

```bash
# Install (needs uv; Python >= 3.11)
uv sync

# Set your Gemini key
cp .env.example .env    # then edit GOOGLE_API_KEY

# Run either CLI (positional question; --subtopics N, --grounded, --verbose)
uv run python -m strands_app.run "What is photosynthesis?"
uv run python -m langgraph_app.run "What is photosynthesis?" --subtopics 2

# Tests
uv run pytest                          # default suite
uv run --group deploy pytest           # includes deploy/ tests (needs bedrock-agentcore)

# Local AgentCore HTTP server (faithful stand-in for the deployed runtime)
APP=strands GOOGLE_API_KEY=<key> uv run --group deploy python -m deploy.server
# then, in another shell:
curl -s -X POST localhost:8080/invocations \
  -H 'content-type: application/json' \
  -d '{"prompt": "What is photosynthesis?"}' | jq .
```
