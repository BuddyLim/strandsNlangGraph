# Handover — Unit 3: Deploy to AWS Bedrock AgentCore

**For:** the session that will build Unit 3.
**Status of Units 1 & 2:** both complete and merged to `main`. Strands (`strands_app/`) and LangGraph
(`langgraph_app/`) each expose the *same* public seam and CLI. Repo is now on GitHub
(`origin` → github.com/BuddyLim/strandsNlangGraph, public).
**How to use this doc:** read it, then run your own `brainstorming → writing-plans →
subagent-driven-development` cycle. This is **context, not a plan** — don't skip the design step.
Like Unit 2, **verify the AgentCore SDK/toolkit APIs live before planning** — they move fast and
this doc's specifics may be stale.

---

## Mission

Comparative learning repo: the same research assistant built in Strands (Unit 1) and LangGraph
(Unit 2). **Unit 3 deploys it to AWS Bedrock AgentCore** — the runtime/hosting layer — so the same
agent that runs locally as a CLI runs as a managed cloud service. This is the "deploy" lesson; the
framework-comparison lesson is already done.

---

## What Units 1 & 2 give you to wrap (do NOT change these)

Both apps expose an **identical public seam** — this is the whole point, and it makes the deploy
layer nearly framework-agnostic:

```python
# strands_app/research.py  AND  langgraph_app/research.py
def run_research(request: ResearchRequest, grounded=False, model=None, verbose=False) -> ResearchReport
```

- `ResearchRequest(question: str, n_subtopics: int = 3)` — `common/types.py`
- `ResearchReport(question, summary, findings: list[SubFinding])` — `common/types.py` (Pydantic;
  `.model_dump()` gives clean JSON)
- CLI entrypoint `main(argv) -> int` in each app's `run.py` (streams by default, `--verbose`, `--grounded`).

**Honest observation for this unit:** because both apps expose the same `run_research`, the AgentCore
wrapper is essentially the *same code* for both. Unit 3 is where Strands and LangGraph look **most**
alike — the deployment surface has already absorbed the framework difference. Keep the comparison
honest by wrapping both through one small adapter rather than writing two bespoke handlers.

---

## The core design question

AgentCore Runtime wants an **HTTP service with a JSON in / JSON out entrypoint**; our apps produce a
`ResearchReport` and stream to stdout. The deploy layer's job is to bridge those:

```
AgentCore Runtime (HTTP)
   → payload {question, n_subtopics?, grounded?}   [JSON in]
   → ResearchRequest(**payload)                     [common/types]
   → run_research(request, grounded=…)              [strands_app OR langgraph_app — pick via config]
   → ResearchReport                                 [common/types]
   → report.model_dump()                            [JSON out]
```

Sketch of the entrypoint (verify the SDK names live):

```python
from bedrock_agentcore import BedrockAgentCoreApp   # VERIFY import path + API
app = BedrockAgentCoreApp()

@app.entrypoint
def handler(payload: dict) -> dict:
    request = ResearchRequest(question=payload["question"],
                              n_subtopics=payload.get("n_subtopics", 3))
    report = run_research(request, grounded=payload.get("grounded", False), verbose=True)
    return report.model_dump()

if __name__ == "__main__":
    app.run()   # runs a LOCAL HTTP server — your offline test path before deploying
```

**Decide in your brainstorm:** one entrypoint module parametrized by an env var (`APP=strands|langgraph`),
or two thin entrypoint modules sharing one adapter. Sync full-report response first (matches the sync
seam); streaming responses are a **stretch** — verify whether AgentCore's response streaming is worth
wiring to the token stream, or leave it for later.

---

## Concrete AgentCore pointers (VERIFY against current AWS docs during your plan)

- **SDK:** `bedrock-agentcore` (Python) — `BedrockAgentCoreApp`, `@app.entrypoint`, `app.run()`.
  Confirm the import path, the payload/response contract, and whether the handler may return a
  generator for streaming.
- **Deploy toolkit:** `bedrock-agentcore-starter-toolkit` — CLI `agentcore configure` → `agentcore
  launch` → `agentcore invoke`. It builds a container, pushes to ECR, and deploys to AgentCore Runtime.
- **Container:** AgentCore Runtime requires **ARM64** images. Reconcile with our **uv** toolchain —
  the toolkit historically builds from a `requirements.txt`/`Dockerfile`; decide how to produce that
  from `pyproject.toml`/`uv.lock` (e.g. `uv export --format requirements-txt`, or a uv-based Dockerfile).
- **Secrets:** inject `GOOGLE_API_KEY` at launch — env var via the toolkit, or AWS Secrets Manager.
  NEVER bake the key into the image or commit it. `.env` stays gitignored.
- **AWS prerequisites:** an AWS account + credentials, a region where AgentCore is available, an
  execution IAM role (Bedrock AgentCore + ECR + CloudWatch), and Docker/Finch for the local build.

---

## Constraints carried from Units 1 & 2 (non-negotiable)

- **Gemini everywhere.** AgentCore Runtime is **model-agnostic hosting** — it runs your container, so a
  Gemini-backed agent is fine; we are NOT switching to Bedrock models. VERIFY nothing in the runtime
  forces a Bedrock model. `GOOGLE_API_KEY` is the only credential the app needs at runtime.
- **`common/` stays framework-neutral**; **don't edit `strands_app/` or `langgraph_app/` internals** —
  the deploy code lives in a new `deploy/` dir and only *imports* their `run_research`.
- **Tests never make live calls / no real AWS calls in the default suite.** Unit-test the payload↔
  `ResearchRequest`/`ResearchReport` adapter with `run_research` stubbed (mirror how Units 1–2 tested
  `run.py`). An actual `agentcore launch` is a manual, opt-in step — not part of `uv run pytest`.
- **Scope:** small — a `deploy/` wrapper, a container recipe, packaging glue, adapter tests. Target
  well under 15 files. TDD the adapter; frequent commits.

---

## Environment gotchas

- **This unit costs real money and can't be fully tested offline.** Deploying hits AWS (ECR storage,
  AgentCore Runtime, CloudWatch). Do the local `app.run()` HTTP path first; only `agentcore launch`
  when the local server behaves. Flag spend before deploying.
- **`git commit` hangs when run directly** in this session — write the message to a file and use
  `command git commit -F <file>`. `command git add` is fine; don't redirect `git diff > file`. See
  [[env-git-commit-hang]].
- **uv ↔ container packaging** is the most likely friction point — resolve it early in the plan.
- Repo now has a remote: after committing, `git push` (branch tracks `origin/main`).

---

## Reuse map (import these; build nothing new here)

- `from common.types import ResearchRequest, ResearchReport, SubFinding`  ← payload/response shapes
- `from strands_app.research import run_research`  **or**  `from langgraph_app.research import run_research`
- `from common.config import require_api_key`  ← same fail-fast key check at startup

---

## Suggested workflow for Unit 3

1. **brainstorming** — settle: which app(s) to deploy and how to select between them; sync-report vs
   streaming response; the uv→container packaging approach; secrets mechanism; local-`app.run()` test
   story. Verify the AgentCore SDK/toolkit APIs live first. Write a spec to `docs/superpowers/specs/`.
2. **writing-plans** — TDD tasks: adapter (payload↔types, `run_research` stubbed) → entrypoint module →
   container recipe + packaging → deploy runbook (documented manual `configure/launch/invoke` steps,
   not automated in CI).
3. **subagent-driven-development** — fresh implementer + reviewer per task; final whole-branch review;
   then `finishing-a-development-branch`.
4. Branch off `main` first.

## Open questions to resolve in the Unit 3 brainstorm

- Deploy **both** apps (two runtimes, honest parity) or **one first**? Both share one adapter either way.
- **Sync full-report response** (simple, matches the seam) or wire AgentCore **response streaming** to
  the token stream (verify it exists and is worth it)?
- **Packaging:** `uv export` → `requirements.txt` for the toolkit's build, or a uv-native Dockerfile?
- **Local testing:** does `app.run()` give a faithful local stand-in for the deployed runtime, so most
  of the unit is testable without spending on AWS?
- **Secrets:** toolkit env-var injection vs AWS Secrets Manager for `GOOGLE_API_KEY`?

## Pointers

- Unit 2 handover (the model for this doc): `docs/superpowers/handovers/2026-07-07-unit-2-langgraph-handover.md`
- Reference seams to wrap: `strands_app/research.py`, `langgraph_app/research.py`, `common/types.py`
- Unit 1 & 2 specs/plans under `docs/superpowers/specs/` and `docs/superpowers/plans/`
- External (verify — may be stale): AWS Bedrock AgentCore developer guide; `bedrock-agentcore` SDK
  reference; `bedrock-agentcore-starter-toolkit` README.
