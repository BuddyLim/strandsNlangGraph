# Unit 3 — AgentCore Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the existing `strands_app` research assistant in an AWS Bedrock AgentCore HTTP runtime so the same agent that runs as a CLI runs as a managed cloud service — via a single `deploy/server.py`, a repo-root Dockerfile, and one `agentcore.json` runtime, with the LangGraph half deployable later by config alone.

**Architecture:** Pattern A. One thin `BedrockAgentCoreApp` wrapper (`deploy/server.py`) selects `strands` vs `langgraph` from an `APP` env var, maps `{question, n_subtopics?, grounded?}` → `ResearchRequest` → `run_research` → `ResearchReport.model_dump()` (sync, no streaming). The Docker build context is the **repo root** so `common/` + `strands_app/` are in-context (the repo is `[tool.uv] package = false` — a path-import workspace with no installable wheel). Each entry in `agentcore.json`'s `runtimes[]` becomes an independent runtime; they share one Dockerfile and differ only by `envVars.APP`.

**Tech Stack:** Python 3.12 (container base; repo floor ≥3.11), `bedrock-agentcore` + `aws-opentelemetry-distro` (new `deploy` dependency group), `uv`, `pytest`, the CDK-based `agentcore-cli`, Docker/Finch (ARM64).

## Global Constraints

- **Python** ≥ 3.11; run everything via `uv run`. The deploy unit tests need the new group: `uv run --group deploy pytest …`.
- **`common/`, `strands_app/`, `langgraph_app/` are FROZEN** — deploy code only *imports* `run_research`, `common.types`, `common.config.require_api_key`. Never edit those trees.
- **Public seam (unchanged, sync):** `run_research(request, grounded=False, model=None, verbose=False) -> ResearchReport`.
- **Response is sync full-report:** `ResearchReport.model_dump()`. No streaming in this unit.
- **Single provider:** Gemini only; `GOOGLE_API_KEY` is the only runtime credential; never bake it into the image.
- **`APP` env var** selects the framework, fixed per runtime: `"strands"` | `"langgraph"`. Anything else (including unset) fails fast at import with a `RuntimeError`.
- **Tests make no live AWS calls and no live Gemini calls** in `uv run pytest`. `run_research` is monkeypatched. `agentcore launch` is manual/opt-in/spend-flagged, never in CI.
- **Repo-root Docker build context** (`codeLocation: "."`); the Dockerfile lives at the repo root (schema requires `dockerfile` to be a filename at the context root).
- **Commit workflow (repo-specific):** `git commit` hangs when run directly. Write the message to a file under `/private/tmp/claude-501/-Users-limkuangtar-Code-strandsNlangGraph/5f5243fb-3930-4268-a4d0-d107024e4a32/scratchpad/` and use `command git commit -F <file>`. `command git add` is fine. Never redirect `git diff > file`.
- **Always use absolute paths in bash** — the shell cwd may have drifted.
- **Commit trailer:** end every commit message body with `Claude-Session: https://claude.ai/code/session_012JU7Vuotmr8mcFH8jt6gNa`.
- **Branch:** work on `unit-3-agentcore-deploy` (already created off `main`; the spec is already committed there).

---

### Task 1: CLI verification gate — repo-root context (HUMAN-DRIVEN, hard gate)

The whole of Pattern A rests on the installed `agentcore-cli` accepting a repo-root build context. The CLI scaffolds per-agent subfolders (`app/MyAgent/`), so this is unverified and is the top project risk. **The implementing agent cannot run `agentcore` (not on PATH in the sandbox shell) — present these commands to the human/orchestrator and wait for the result.**

**Files:** none (verification only).

**Gate rule:** Tasks 4 and 5 (Dockerfile, `agentcore.json`) MUST NOT start until this passes. Tasks 2 and 3 (deps + `deploy/server.py` + unit tests) are pure Python with no AWS surface and MAY proceed concurrently while awaiting the human's result.

- [ ] **Step 1: Hand the human the verification commands**

Ask the human to run, from the repo root:
```bash
agentcore --version
# In agentcore/agentcore.json, temporarily set the strands runtime's
#   codeLocation to "."  and  entrypoint to "deploy/server.py"  (Task 5 makes this permanent),
# with a repo-root Dockerfile present (Task 4). Then a build-only dry run:
agentcore launch --local        # or the CLI's equivalent local/build-only flag
```

- [ ] **Step 2: Record the pass/fail signal**

- **PASS** if the CLI accepts `codeLocation: "."`, finds the repo-root `Dockerfile`, and honors per-runtime `envVars` (APP) without demanding a per-agent subfolder. Record the exact working flag names for the runbook (Task 6).
- **FAIL** if the CLI rejects a repo-root context or forces the code into `app/<agent>/`. **STOP. Do not** symlink or copy `common/`/`strands_app/` into a subfolder (breaks DRY and the frozen-tree rule). Escalate to the user with the exact error and propose reconsidering a `CodeZip` build or an alternate context layout. Do not proceed to Tasks 4–5.

- [ ] **Step 3: Also record the two smaller unknowns for Task 6**

- Whether the CLI builds ARM64 by default on this Apple-Silicon host or needs an explicit `--platform linux/arm64` / buildx flag.
- The exact env var name the `strandsNlanggraphGemini` ApiKeyCredentialProvider injects the Gemini key as (must land as `GOOGLE_API_KEY`; if not, note the alias needed in Task 5's config).

---

### Task 2: Add the `deploy` dependency group and pin SDK facts

Nothing AgentCore-related is installed. Add the deps as a dedicated group and verify the two SDK facts every later task relies on: the `BedrockAgentCoreApp` import path and that `@app.entrypoint` leaves the function callable.

**Files:**
- Modify: `pyproject.toml` (`[dependency-groups] deploy`) + `uv.lock` (regenerated by `uv add`)

**Interfaces:**
- Produces (consumed by Tasks 3–4): confirmed `from bedrock_agentcore.runtime import BedrockAgentCoreApp`; confirmation that `@app.entrypoint` returns a normally-callable function; the local server port (expected `8080`).

- [ ] **Step 1: Add the dependencies to a `deploy` group**

Run (absolute path):
```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && uv add --group deploy bedrock-agentcore aws-opentelemetry-distro
```
Expected: writes `[dependency-groups] deploy = ["bedrock-agentcore", "aws-opentelemetry-distro"]` into `pyproject.toml` and updates `uv.lock`.

- [ ] **Step 2: Install and record the resolved version**

Run:
```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && uv sync --group deploy
uv run --group deploy python -c "import importlib.metadata as m; print('bedrock-agentcore', m.version('bedrock-agentcore'))"
```
Expected: prints a version (spec floor `>= 1.9.1` from the CLI scaffold — accept whatever resolves).

- [ ] **Step 3: Pin the SDK import path + entrypoint callability**

Run:
```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && uv run --group deploy python -c "
from bedrock_agentcore.runtime import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload, context):
    return {'ok': True}

print('CALLABLE', callable(invoke), invoke({'x': 1}, None))
print('HAS_RUN', hasattr(app, 'run'))
"
```
Expected: `CALLABLE True {'ok': True}` and `HAS_RUN True`.
- If `invoke` is **not** directly callable after decoration (the SDK replaces it), that's fine — Task 3 tests target the undecorated `_handle`, not `invoke`, precisely to avoid this coupling. Record the observed behavior in the commit message.

- [ ] **Step 4: Commit**

```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && command git add pyproject.toml uv.lock
printf '%s\n' 'add deploy dependency group (bedrock-agentcore, aws-opentelemetry-distro)' '' 'Claude-Session: https://claude.ai/code/session_012JU7Vuotmr8mcFH8jt6gNa' > /private/tmp/claude-501/-Users-limkuangtar-Code-strandsNlangGraph/5f5243fb-3930-4268-a4d0-d107024e4a32/scratchpad/msg.txt
command git commit -F /private/tmp/claude-501/-Users-limkuangtar-Code-strandsNlangGraph/5f5243fb-3930-4268-a4d0-d107024e4a32/scratchpad/msg.txt
```

---

### Task 3: `deploy/server.py` adapter + unit tests (TDD)

The core deliverable: the HTTP wrapper. Test-drive the pure `_handle` adapter and the `APP`-dispatch guard.

**Files:**
- Create: `deploy/__init__.py` (empty package marker)
- Create: `deploy/server.py`
- Test: `tests/test_deploy_server.py`

**Interfaces:**
- Consumes: `run_research(request, grounded=False, model=None, verbose=False) -> ResearchReport` (frozen seam); `ResearchRequest(question: str, n_subtopics: int = 3)`, `ResearchReport(question, summary, findings)` from `common.types`; `require_api_key()` from `common.config`; `BedrockAgentCoreApp` from Task 2.
- Produces: module `deploy.server` with `_handle(payload: dict) -> dict` (pure adapter), `invoke(payload, context)` (decorated entrypoint delegating to `_handle`), module-level `app`, and an import-time `RuntimeError` when `APP` ∉ {`strands`, `langgraph`}.

- [ ] **Step 1: Create the package marker**

```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && mkdir -p deploy && : > deploy/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_deploy_server.py`:
```python
import importlib
import sys

import pytest

# deploy.server imports bedrock_agentcore (deploy group). In the default env this
# skips cleanly; run the deploy tests with: uv run --group deploy pytest
pytest.importorskip("bedrock_agentcore")

from common.types import ResearchReport, SubFinding


def _load_server(monkeypatch, app="strands"):
    """Import deploy.server fresh with APP set and the key check stubbed out."""
    monkeypatch.setenv("APP", app)
    monkeypatch.setattr("common.config.require_api_key", lambda: "test-key")
    sys.modules.pop("deploy.server", None)
    return importlib.import_module("deploy.server")


def _fake_report(**kw):
    def fake(request, grounded=False, model=None, verbose=False):
        fake.calls.append({"request": request, "grounded": grounded})
        return ResearchReport(question=request.question, summary="ok", findings=[], **kw)
    fake.calls = []
    return fake


def test_handle_maps_question_and_defaults_subtopics_to_3(monkeypatch):
    server = _load_server(monkeypatch)
    fake = _fake_report()
    monkeypatch.setattr(server, "run_research", fake)

    server._handle({"question": "What is X?"})

    assert fake.calls[0]["request"].question == "What is X?"
    assert fake.calls[0]["request"].n_subtopics == 3


def test_handle_honours_explicit_subtopics(monkeypatch):
    server = _load_server(monkeypatch)
    fake = _fake_report()
    monkeypatch.setattr(server, "run_research", fake)

    server._handle({"question": "Q", "n_subtopics": 5})

    assert fake.calls[0]["request"].n_subtopics == 5


def test_handle_defaults_grounded_false_and_passes_true_through(monkeypatch):
    server = _load_server(monkeypatch)
    fake = _fake_report()
    monkeypatch.setattr(server, "run_research", fake)

    server._handle({"question": "Q"})
    server._handle({"question": "Q", "grounded": True})

    assert fake.calls[0]["grounded"] is False
    assert fake.calls[1]["grounded"] is True


def test_handle_returns_report_model_dump_dict(monkeypatch):
    server = _load_server(monkeypatch)

    def fake(request, grounded=False, model=None, verbose=False):
        return ResearchReport(
            question=request.question,
            summary="the summary",
            findings=[SubFinding(subtopic="alpha", findings="fa")],
        )

    monkeypatch.setattr(server, "run_research", fake)

    result = server._handle({"question": "Q"})

    assert isinstance(result, dict)
    assert result == {
        "question": "Q",
        "summary": "the summary",
        "findings": [{"subtopic": "alpha", "findings": "fa", "ok": True}],
    }


@pytest.mark.parametrize("bad", ["", "STRANDS", "both", "gpt"])
def test_invalid_app_raises_at_import(monkeypatch, bad):
    monkeypatch.setenv("APP", bad)
    monkeypatch.setattr("common.config.require_api_key", lambda: "test-key")
    sys.modules.pop("deploy.server", None)
    with pytest.raises(RuntimeError, match="APP must be"):
        importlib.import_module("deploy.server")


def test_unset_app_raises_at_import(monkeypatch):
    monkeypatch.delenv("APP", raising=False)
    monkeypatch.setattr("common.config.require_api_key", lambda: "test-key")
    sys.modules.pop("deploy.server", None)
    with pytest.raises(RuntimeError, match="APP must be"):
        importlib.import_module("deploy.server")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:
```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && uv run --group deploy pytest tests/test_deploy_server.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'deploy.server'`.

- [ ] **Step 4: Write the implementation**

Create `deploy/server.py`:
```python
"""AgentCore HTTP entrypoint wrapping the research assistant (Pattern A).

One wrapper serves both frameworks; the `APP` env var (fixed per runtime via
agentcore.json envVars) selects which `run_research` to import. Resolved once at
import so a misconfigured runtime fails fast on cold start, not per request.
"""

import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from common.config import require_api_key
from common.types import ResearchRequest

APP = os.environ.get("APP", "")
if APP == "strands":
    from strands_app.research import run_research
elif APP == "langgraph":
    from langgraph_app.research import run_research
else:
    raise RuntimeError(f"APP must be 'strands' or 'langgraph', got {APP!r}")

# Fail fast on a missing Gemini key at cold start, matching the CLI contract.
require_api_key()

app = BedrockAgentCoreApp()


def _handle(payload: dict) -> dict:
    """Map a JSON payload to a report dict: {question, n_subtopics?, grounded?} -> dict.

    Framework-free (no BedrockAgentCore types) so it is unit-testable without the
    runtime; `invoke` is the thin decorated entrypoint that delegates here.
    """
    request = ResearchRequest(
        question=payload["question"],
        n_subtopics=payload.get("n_subtopics", 3),
    )
    report = run_research(request, grounded=payload.get("grounded", False))
    return report.model_dump()


@app.entrypoint
def invoke(payload: dict, context) -> dict:
    return _handle(payload)


if __name__ == "__main__":
    app.run()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && uv run --group deploy pytest tests/test_deploy_server.py -v
```
Expected: PASS (7 tests: 4 `_handle`, 4 parametrized invalid + 1 unset — all green).

- [ ] **Step 6: Confirm the default suite still passes (deploy test skips without the group)**

Run:
```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && uv run pytest -q && uv run ruff check deploy tests/test_deploy_server.py
```
Expected: full suite green; `tests/test_deploy_server.py` reported skipped (bedrock_agentcore absent from the default env); ruff clean.

- [ ] **Step 7: Commit**

```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && command git add deploy/__init__.py deploy/server.py tests/test_deploy_server.py
printf '%s\n' 'add AgentCore HTTP wrapper with APP-dispatch adapter and tests' '' 'Claude-Session: https://claude.ai/code/session_012JU7Vuotmr8mcFH8jt6gNa' > /private/tmp/claude-501/-Users-limkuangtar-Code-strandsNlangGraph/5f5243fb-3930-4268-a4d0-d107024e4a32/scratchpad/msg.txt
command git commit -F /private/tmp/claude-501/-Users-limkuangtar-Code-strandsNlangGraph/5f5243fb-3930-4268-a4d0-d107024e4a32/scratchpad/msg.txt
```

---

### Task 4: Repo-root Dockerfile + `.dockerignore` (GATED on Task 1 PASS)

Produce the container recipe from the repo root so `common/` + `strands_app/` are in-context. Do not start until Task 1 has PASSED.

**Files:**
- Create: `Dockerfile` (repo root)
- Create: `.dockerignore` (repo root)

**Interfaces:**
- Consumes: the `deploy` group (Task 2), `deploy/server.py` (Task 3).
- Produces: an image whose `CMD` runs `opentelemetry-instrument python -m deploy.server` on port 8080.

- [ ] **Step 1: Write `.dockerignore`**

Create `/Users/limkuangtar/Code/strandsNlangGraph/.dockerignore`:
```
.venv/
app/
.git/
.pytest_cache/
.ruff_cache/
__pycache__/
*.py[cod]
docs/
tests/
.env
.env.*
```

- [ ] **Step 2: Write the `Dockerfile`**

Create `/Users/limkuangtar/Code/strandsNlangGraph/Dockerfile`:
```dockerfile
FROM public.ecr.aws/docker/library/python:3.12-slim-trixie

RUN pip install --no-cache-dir uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN useradd -m -u 1000 bedrock_agentcore

# Install deps first (cached layer) from the lockfile; --no-install-project
# because this repo is package = false (path-import workspace, no wheel).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --group deploy

# Then the source: common/ + strands_app/ + langgraph_app/ + deploy/ land at /app,
# so `python -m deploy.server` resolves the path-imports from the workdir.
COPY --chown=bedrock_agentcore:bedrock_agentcore . .
RUN uv sync --frozen --no-install-project --group deploy

USER bedrock_agentcore

# AgentCore Runtime HTTP service contract: /invocations + /ping on 8080.
EXPOSE 8080

CMD ["opentelemetry-instrument", "python", "-m", "deploy.server"]
```

- [ ] **Step 3: Build the image (test)**

If Docker/Finch is available to the agent, run:
```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && docker build --platform linux/arm64 -t strandsnlanggraph-strands .
```
Expected: build succeeds through both `uv sync` layers. **If Docker is not available in the sandbox, this is a human-driven step** (like Task 1) — hand the command to the human and record the result.

- [ ] **Step 4: Smoke-test the image imports cleanly**

Run (Docker available):
```bash
docker run --rm -e APP=strands -e GOOGLE_API_KEY=dummy-not-used strandsnlanggraph-strands \
  python -c "import deploy.server; print('IMPORT_OK', hasattr(deploy.server, 'app'))"
```
Expected: `IMPORT_OK True` (module body runs — `APP` valid, key present — but `app.run()` stays behind `__main__`, so nothing serves). This proves the build context resolves the path-imports.

- [ ] **Step 5: Commit**

```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && command git add Dockerfile .dockerignore
printf '%s\n' 'add repo-root ARM64 Dockerfile and .dockerignore for the strands runtime' '' 'Claude-Session: https://claude.ai/code/session_012JU7Vuotmr8mcFH8jt6gNa' > /private/tmp/claude-501/-Users-limkuangtar-Code-strandsNlangGraph/5f5243fb-3930-4268-a4d0-d107024e4a32/scratchpad/msg.txt
command git commit -F /private/tmp/claude-501/-Users-limkuangtar-Code-strandsNlangGraph/5f5243fb-3930-4268-a4d0-d107024e4a32/scratchpad/msg.txt
```

---

### Task 5: Fix `agentcore.json` + delete the demo scaffold (GATED on Task 1 PASS)

Point the `strands` runtime at the repo-root context, wire `APP=strands`, and remove the throwaway CLI demo (which ships a committed `.venv`).

**Files:**
- Modify: `agentcore/agentcore.json` (the single `strands` runtime entry)
- Delete: `app/MyAgent/` (untracked scaffold)

**Interfaces:**
- Consumes: `deploy/server.py` (entrypoint target), the repo-root `Dockerfile` (Task 4).
- Produces: a `runtimes[0]` with `build: "Container"`, `codeLocation: "."`, `entrypoint: "deploy/server.py"`, `dockerfile: "Dockerfile"`, `protocol: "HTTP"`, `envVars: [{name: "APP", value: "strands"}]`.

- [ ] **Step 1: Edit the strands runtime entry**

In `/Users/limkuangtar/Code/strandsNlangGraph/agentcore/agentcore.json`, replace the `runtimes[0]` object with:
```json
{
  "name": "strands",
  "build": "Container",
  "entrypoint": "deploy/server.py",
  "codeLocation": ".",
  "dockerfile": "Dockerfile",
  "networkMode": "PUBLIC",
  "protocol": "HTTP",
  "envVars": [{ "name": "APP", "value": "strands" }]
}
```
Leave `runtimeVersion` out (moot for a Container build — the Dockerfile pins Python 3.12). Keep the `credentials` block (`strandsNlanggraphGemini`) as-is. If Task 1 Step 3 found the provider injects the key under a name other than `GOOGLE_API_KEY`, add a matching `envVars` alias here and note it.

- [ ] **Step 2: Verify the JSON is valid and correctly shaped (test)**

Run:
```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && uv run python -c "
import json
cfg = json.load(open('agentcore/agentcore.json'))
rt = cfg['runtimes'][0]
assert rt['codeLocation'] == '.', rt['codeLocation']
assert rt['entrypoint'] == 'deploy/server.py', rt['entrypoint']
assert rt['dockerfile'] == 'Dockerfile', rt.get('dockerfile')
assert rt['build'] == 'Container', rt['build']
assert {'name': 'APP', 'value': 'strands'} in rt['envVars'], rt.get('envVars')
print('AGENTCORE_JSON_OK')
"
```
Expected: `AGENTCORE_JSON_OK`.

- [ ] **Step 3: Delete the throwaway scaffold**

Run:
```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && rm -rf app/MyAgent && rmdir app 2>/dev/null; ls app 2>&1 | head -1
```
Expected: `app` no longer exists (or is empty and removed). `app/MyAgent/` is untracked, so this needs no `git rm`.

- [ ] **Step 4: Commit**

```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && command git add agentcore/agentcore.json
printf '%s\n' 'point strands runtime at repo-root context and wire APP=strands' '' 'Removes the throwaway app/MyAgent CLI demo scaffold (was untracked).' '' 'Claude-Session: https://claude.ai/code/session_012JU7Vuotmr8mcFH8jt6gNa' > /private/tmp/claude-501/-Users-limkuangtar-Code-strandsNlangGraph/5f5243fb-3930-4268-a4d0-d107024e4a32/scratchpad/msg.txt
command git commit -F /private/tmp/claude-501/-Users-limkuangtar-Code-strandsNlangGraph/5f5243fb-3930-4268-a4d0-d107024e4a32/scratchpad/msg.txt
```

---

### Task 6: Deploy runbook (local test → manual launch)

Document the manual, spend-flagged path so a reader can go from local HTTP test to a live runtime — and knows exactly where money starts. Nothing here runs in CI.

**Files:**
- Create: `docs/superpowers/runbooks/2026-07-08-unit-3-agentcore-deploy-runbook.md`

- [ ] **Step 1: Write the runbook**

Create the file with these sections (fill flag names with the values Task 1 recorded):

```markdown
# Unit 3 — AgentCore Deploy Runbook (manual, spend-flagged)

## 0. Prerequisites
- AWS account + credentials; a region where AgentCore Runtime is available.
- Docker or Finch running (ARM64 builds).
- `GOOGLE_API_KEY` available to inject at launch (never baked into the image).

## 1. Local HTTP test FIRST (no AWS spend)
    cd /Users/limkuangtar/Code/strandsNlangGraph
    APP=strands GOOGLE_API_KEY=<key> uv run --group deploy python -m deploy.server
    # in another shell:
    curl -s -X POST localhost:8080/invocations \
      -H 'content-type: application/json' \
      -d '{"question":"What is photosynthesis?","n_subtopics":2}' | jq .
Expect a JSON ResearchReport ({question, summary, findings[]}). This is the
faithful stand-in for the deployed runtime — get it green before spending.

## 2. Configure (writes agentcore.json; no spend)
    agentcore configure        # confirm it accepts codeLocation "." (Task 1)

## 3. Launch — ⚠️ THIS SPENDS MONEY (ECR storage + AgentCore Runtime + CloudWatch)
    agentcore launch <recorded flags, e.g. --platform linux/arm64 if needed>
    # inject the Gemini key via the strandsNlanggraphGemini ApiKeyCredentialProvider
    # (or --env GOOGLE_API_KEY=... per the CLI). NEVER commit the key.

## 4. Invoke the live runtime
    agentcore invoke '{"question":"What is photosynthesis?","n_subtopics":2}'

## 5. Add LangGraph later (no code change)
Append a second runtimes[] entry: name "langgraph", same codeLocation ".",
entrypoint "deploy/server.py", dockerfile "Dockerfile", envVars APP=langgraph.
Re-run configure/launch. deploy/server.py already dispatches on APP.

## 6. Teardown (stop spend)
Delete the runtime(s) and the ECR image via the CLI/console when done.
```

- [ ] **Step 2: Commit**

```bash
cd /Users/limkuangtar/Code/strandsNlangGraph && command git add docs/superpowers/runbooks/2026-07-08-unit-3-agentcore-deploy-runbook.md
printf '%s\n' 'add Unit 3 AgentCore deploy runbook (local test then manual launch)' '' 'Claude-Session: https://claude.ai/code/session_012JU7Vuotmr8mcFH8jt6gNa' > /private/tmp/claude-501/-Users-limkuangtar-Code-strandsNlangGraph/5f5243fb-3930-4268-a4d0-d107024e4a32/scratchpad/msg.txt
command git commit -F /private/tmp/claude-501/-Users-limkuangtar-Code-strandsNlangGraph/5f5243fb-3930-4268-a4d0-d107024e4a32/scratchpad/msg.txt
```

---

## Self-Review

**Spec coverage:**
- Pattern A repo-root context → Tasks 3–5. ✓
- One `deploy/server.py` + `APP` dispatch → Task 3. ✓
- Sync `model_dump()` response → Task 3 (`_handle` + `test_handle_returns_report_model_dump_dict`). ✓
- Frozen `common/`/`strands_app/`/`langgraph_app/` → Global Constraints; only imports. ✓
- `deploy` dep group (bedrock-agentcore + aws-opentelemetry-distro) → Task 2. ✓
- `agentcore.json` fix (codeLocation ".", entrypoint, APP env) + delete `app/MyAgent/` → Task 5. ✓
- Tests: monkeypatched `run_research`, no live AWS/Gemini, payload→request mapping, grounded passthrough, model_dump shape, APP unset/invalid → RuntimeError → Task 3. ✓
- CLI `codeLocation "."` risk as first gate → Task 1. ✓
- Secrets via existing credential provider / GOOGLE_API_KEY env → Task 1 Step 3 + Task 5 Step 1 + runbook. ✓
- Local `app.run()` curl path + manual configure/launch/invoke, spend-flagged, not in CI → Task 6. ✓
- Python 3.12 pinned in Dockerfile, runtimeVersion dropped → Task 4 + Task 5 Step 1. ✓

**Placeholder scan:** none — all code and commands are concrete. Runbook flag names are the one deliberately-deferred value, resolved by Task 1 (the gate whose purpose is to discover them).

**Type consistency:** `_handle(payload: dict) -> dict` and `invoke(payload, context)` names match across Task 3 tests, implementation, and the Task 4 smoke test. `run_research(request, grounded=…)` matches the frozen seam. `ResearchReport.model_dump()` keys (`question`, `summary`, `findings[].{subtopic,findings,ok}`) match `common/types.py`.

**Ordering / gating:** Task 1 (human/CLI) gates Tasks 4–5 only; Tasks 2–3 (pure Python) may proceed concurrently. Task 3's deploy tests require the Task 2 group (`uv run --group deploy pytest`); the default suite stays green via `pytest.importorskip`.
