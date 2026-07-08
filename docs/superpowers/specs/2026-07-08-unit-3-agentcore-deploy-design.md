# Unit 3 — Deploy to AWS Bedrock AgentCore

**Status:** Approved design, pre-implementation
**Date:** 2026-07-08
**Repo purpose:** A comparative learning repo. The same research assistant built in Strands
(Unit 1) and LangGraph (Unit 2), both exposing an identical `run_research` seam and CLI. Unit 3
is the **deploy** lesson: the same agent that runs locally as a CLI runs as a managed cloud
service on AWS Bedrock AgentCore Runtime.

This spec covers **Unit 3 only**. It supersedes the specifics in the Unit 3 handover
(`docs/superpowers/handovers/2026-07-08-unit-3-agentcore-handover.md`) where they drifted: the
installed CLI is the **CDK-based `agentcore-cli`** (schema `agentcore.aws.dev/v1`), not the
Python `bedrock-agentcore-starter-toolkit` the handover assumed. The SDK entrypoint
(`BedrockAgentCoreApp`, `@app.entrypoint`, `app.run()`) is unchanged.

---

## Blast radius

`common/`, `strands_app/`, and `langgraph_app/` are reused verbatim — **no changes** (handover
rule: deploy code only *imports* `run_research`). New files sit in a new `deploy/` dir + repo
root + `tests/`, plus one deps bump and one config fix.

| File | Change | ~LOC | Purpose |
|---|---|---|---|
| `deploy/__init__.py` | new | 0 | package marker |
| `deploy/server.py` | **new** | ~40 | `BedrockAgentCoreApp` HTTP wrapper; `APP` env-var dispatch; fail-fast key check |
| `Dockerfile` (repo root) | **new** | ~30 | uv-based, ARM64, `CMD … python -m deploy.server` |
| `.dockerignore` (repo root) | new | ~15 | exclude `.venv/`, `.git/`, `app/`, caches |
| `pyproject.toml` | edit | ~+4 | add `[dependency-groups] deploy = [bedrock-agentcore, aws-opentelemetry-distro]` |
| `agentcore/agentcore.json` | edit | ~10 | fix `strands` runtime → `codeLocation:"."`, `entrypoint:"deploy/server.py"`, add `APP` env; `langgraph` entry added when its deploy lands |
| `tests/test_deploy_server.py` | **new** | ~50 | adapter unit tests, `run_research` stubbed |
| `app/MyAgent/` | **delete** | — | throwaway CLI demo scaffold (ships a committed `.venv`) |

**~5 new files + 2 edits + 1 deletion, well under the 15-file ceiling.**

## Problem

Units 1 & 2 produce a `ResearchReport` and stream to stdout via a CLI. AgentCore Runtime wants
an **HTTP service with a JSON-in / JSON-out entrypoint** on port 8080 (`/invocations` + `/ping`).
The deploy layer bridges those. The CLI-scaffolded `app/MyAgent/` is a self-contained demo agent
whose Docker build context (`app/MyAgent/`) **cannot see** `common/` or `strands_app/` at the
repo root — and this repo is `[tool.uv] package = false` (a path-import workspace, no installable
wheel), so the app code must be *physically present* in the image. That mismatch is why a bare
`strands_app/Dockerfile` can never work.

## Non-goals

- **Not** deploying LangGraph in this unit. The wrapper is written to support it (Pattern A,
  below), but only the `strands` runtime is wired now. LangGraph slots in later by adding one
  `runtimes[]` entry — no code change.
- **Not** streaming responses. Sync full-report first; response streaming is a later stretch.
- **Not** editing `strands_app/` / `langgraph_app/` / `common/`.
- **Not** automating `agentcore launch` in CI. Deploy is a manual, opt-in, spend-flagged step.
- **Not** switching to Bedrock models. Gemini everywhere; `GOOGLE_API_KEY` is the only runtime credential.

## Approach — Pattern A (one wrapper, `APP` env var)

The build context is the **repo root** (shared by every framework). Each deployment is just a
different entrypoint *selection*, carried by a per-runtime env var. `agentcore.json`'s
`runtimes[]` array is the mechanism for N independent runtimes (each its own image + ARN +
endpoint), all pointing at `codeLocation: "."`.

```
                        repo root  (ONE Docker build context)
 route/runtime layer    ├── common/          ← shared, in-context, untouched
                        ├── strands_app/     ← untouched (imports common)
                        ├── langgraph_app/   ← untouched (imports common)
                        └── deploy/server.py ← ONE thin HTTP wrapper (new)
                                  │  reads APP env → imports the matching run_research
        ┌─────────────────────────┴─────────────────────────┐
   runtime "strands"                                   runtime "langgraph"  (added later)
   envVars APP=strands                                 envVars APP=langgraph
   → own image + ARN + /invocations                    → own image + ARN + /invocations
```

Request flow (sync):

```
AgentCore HTTP runtime → POST /invocations {question, n_subtopics?, grounded?}
  → invoke(payload, context)              [deploy/server.py]
  → ResearchRequest(question=…, n_subtopics=…)   [common.types]
  → run_research(request, grounded=…)     [strands_app OR langgraph_app, per APP env]
  → ResearchReport
  → report.model_dump()                   → JSON out
```

`deploy/server.py` sketch (verify SDK import path at implementation time):

```python
import os
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from common.config import require_api_key
from common.types import ResearchRequest

APP = os.environ["APP"]  # "strands" | "langgraph" — fixed per runtime at launch
if APP == "strands":
    from strands_app.research import run_research
elif APP == "langgraph":
    from langgraph_app.research import run_research
else:
    raise RuntimeError(f"APP must be 'strands' or 'langgraph', got {APP!r}")

require_api_key()  # fail fast at cold start if GOOGLE_API_KEY is missing

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict, context) -> dict:
    request = ResearchRequest(
        question=payload["question"],
        n_subtopics=payload.get("n_subtopics", 3),
    )
    report = run_research(request, grounded=payload.get("grounded", False))
    return report.model_dump()

if __name__ == "__main__":
    app.run()  # local HTTP server — the offline test path
```

Dockerfile shape (repo-root context; adapted from the CLI scaffold, which was already
idiomatic uv — only the context and `CMD` target change):

```dockerfile
FROM public.ecr.aws/docker/library/python:3.12-slim-trixie
RUN pip install --no-cache-dir uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_NO_PROGRESS=1 PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
RUN useradd -m -u 1000 bedrock_agentcore
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --group deploy
COPY --chown=bedrock_agentcore:bedrock_agentcore . .
RUN uv sync --frozen --group deploy
USER bedrock_agentcore
EXPOSE 8080
CMD ["opentelemetry-instrument", "python", "-m", "deploy.server"]
```

Because `package = false`, `python -m deploy.server` runs from `/app`, putting the repo root on
`sys.path` so `import common`, `import strands_app`, `import langgraph_app` resolve. The
`--group deploy` flag pulls in `bedrock-agentcore` + `aws-opentelemetry-distro` on top of the
app deps already in `uv.lock`.

## Alternatives considered

- **Pattern B — two wrapper modules + two Dockerfiles** (`deploy/strands_server.py` +
  `Dockerfile`, `deploy/langgraph_server.py` + `Dockerfile.langgraph`, framework hardcoded in
  each `CMD`). Both patterns run both runtimes concurrently (concurrency comes from two
  `runtimes[]` entries, not from two images), so B buys nothing here while doubling the wrappers
  *and* Dockerfiles. Rejected. The only reason to prefer B is wanting the built image to be
  self-describing/immutable per framework — not a goal for a learning repo.
- **Self-contained `app/MyAgent/` (vendored)** — keep the CLI's per-agent subfolder as the build
  context and copy `common/` + `strands_app/` into it. Duplicates source (DRY + "don't edit
  strands_app" violation) and forces two copies to stay in sync. Rejected.
- **`CodeZip` build instead of Container** — let AWS build the image from zipped code. Loses
  control of the ARM64 base + uv, and the `package = false` path-import layout fights the managed
  builder. Rejected; Container was chosen.

## Decision log

- Chose **repo-root build context** over per-agent subfolder because `package = false` +
  `common/`-as-sibling means the app code must be copied into the image and cannot be `pip
  install`ed.
- Chose **`APP` env-var dispatch (Pattern A)** over per-framework Dockerfiles because both give
  identical concurrency; the env var is fixed per runtime at launch, so each deployed service is
  still permanently one framework (honest side-by-side).
- Chose **sync `.model_dump()` response** over streaming because it matches the existing
  `run_research` seam; streaming is deferred.
- Chose the **existing `strandsNlanggraphGemini` ApiKeyCredentialProvider** (already in
  `agentcore.json`) for the Gemini key over raw env injection or Secrets Manager — it's already
  scaffolded; the app reads `GOOGLE_API_KEY` from env unchanged *if* the provider injects under
  that name (implementation-time check).
- Chose to **pin Python 3.12 in the Dockerfile** and ignore `runtimeVersion: PYTHON_3_14` in the
  config, which is moot for a Container build (we own the base image); 3.12 matches `uv.lock`.

## Error handling

- **Missing/invalid `APP`** → `RuntimeError` at import time; the container fails fast on cold
  start rather than serving a half-configured agent (define the error out of the request path).
- **Missing `GOOGLE_API_KEY`** → `require_api_key()` raises at startup — same fail-fast contract
  as the CLIs, surfaced before the first request.
- **Bad payload** (`question` absent) → `KeyError`/Pydantic `ValidationError` from
  `ResearchRequest`; surfaced as a runtime error to the caller. No bespoke HTTP shaping in this
  unit — the SDK owns the HTTP envelope.
- **Model throttling / transient provider errors** propagate from `run_research`; the SDK returns
  a 5xx. Retry/backoff is out of scope.

## Testing

- **No live AWS. No live Gemini in `uv run pytest`.** `agentcore launch` is manual/opt-in.
- `tests/test_deploy_server.py` unit-tests `invoke()` with `run_research` **monkeypatched** and
  `APP=strands` set (mirrors how Units 1–2 tested `run.py`):
  - payload → `ResearchRequest` mapping (question + `n_subtopics` default and override)
  - `grounded` passthrough (default `False`, explicit `True`)
  - return value is `ResearchReport.model_dump()` (clean JSON dict)
  - `APP` unset / invalid → import raises `RuntimeError`
- **Local integration (manual, documented):** `APP=strands python -m deploy.server`, then
  `curl -X POST localhost:8080/invocations -d '{"question":"…"}'` — faithful stand-in for the
  deployed runtime, exercised before any AWS spend.

## Risks & rollback

- **Rollback:** deploy code is isolated. Delete `deploy/`, root `Dockerfile`/`.dockerignore`,
  the `deploy` dep group, and revert the `agentcore.json` edit — apps are untouched.
- **Top risk — CLI acceptance of `codeLocation: "."`:** unverified from the sandbox shell. The
  CLI scaffolds per-agent subfolders; it may resist a repo-root context. **First implementation
  step verifies this live** (`agentcore configure` / a dry `agentcore launch --local` or build)
  before any downstream work. If the CLI insists on a per-agent subfolder, do **not** fall back
  to symlinking/copying the app code into it (breaks DRY) — instead escalate: reconsider a
  `CodeZip` build or an alternate context layout, and flag the tradeoff to the user.
- **Real spend:** `agentcore launch` hits ECR + AgentCore Runtime + CloudWatch. Flag cost before
  the first launch; do the local `app.run()` path first.
- **Secret-injection var name:** if the credential provider injects under a name other than
  `GOOGLE_API_KEY`, add a one-line env alias in the Dockerfile/config — not an app change.

## Open questions

- Does the installed `agentcore-cli` accept `codeLocation: "."` with a root Dockerfile, and
  honor per-runtime `envVars` sharing one Dockerfile? (Resolved in implementation step 1.)
- Exact env var name the `strandsNlanggraphGemini` ApiKeyCredentialProvider injects the key as.
- ARM64 build: does the CLI build ARM64 by default on this (Apple Silicon) host, or is an
  explicit `--platform`/buildx flag needed?
