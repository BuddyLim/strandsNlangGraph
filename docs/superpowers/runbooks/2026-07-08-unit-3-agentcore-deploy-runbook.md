# Unit 3 — AgentCore Deploy Runbook (manual, spend-flagged)

> Flag names marked `<recorded flags>` are resolved by the Task 1 CLI
> verification gate (repo-root `codeLocation "."`, ARM64 default, and the Gemini
> key env-var name). Fill them in once that gate has been run.

## 0. Prerequisites
- AWS account + credentials; a region where AgentCore Runtime is available.
- Docker or Finch running (ARM64 builds).
- `GOOGLE_API_KEY` available to inject at launch (never baked into the image).

## 1. Local HTTP test FIRST (no AWS spend)
Two local paths:

    # a) plain module run — APP comes from repo-root .env (or inline)
    cd /Users/limkuangtar/Code/strandsNlangGraph
    APP=strands GOOGLE_API_KEY=<key> uv run --group deploy python -m deploy.server

    # b) via the CLI (builds + runs the container, hot-reload)
    agentcore dev --env APP=strands

    # then, in another shell:
    curl -s -X POST localhost:8080/invocations \
      -H 'content-type: application/json' \
      -d '{"prompt":"What is photosynthesis?"}' | jq .
Expect a JSON ResearchReport ({question, summary, findings[]}). This is the
faithful stand-in for the deployed runtime — get it green before spending.

> **Local env gotcha:** `agentcore.json` `envVars` (e.g. `APP=strands`) are
> applied at **cloud deploy only**, NOT by `agentcore dev`. Locally, provide
> `APP` via `agentcore dev --env APP=strands` or `agentcore/.env.local`. The
> repo-root `.env` fallback in `deploy/server.py` does not reach the dev
> container (it is stripped by `.dockerignore`).

## 2. Configure / deploy (this CLI uses `agentcore dev` + `agentcore deploy`)
    agentcore deploy           # confirm it accepts codeLocation "." (Task 1)

## 3. Launch — ⚠️ THIS SPENDS MONEY (ECR storage + AgentCore Runtime + CloudWatch)
    agentcore launch <recorded flags, e.g. --platform linux/arm64 if needed>
    # inject the Gemini key via the strandsNlanggraphGemini ApiKeyCredentialProvider
    # (or --env GOOGLE_API_KEY=... per the CLI). NEVER commit the key.

## 4. Invoke the live runtime
The entrypoint accepts three payload shapes for the question (the AgentCore
console/CLI send `prompt` or `messages`, never a custom key):

    agentcore invoke '{"prompt":"What is photosynthesis?"}'
    agentcore invoke '{"question":"What is photosynthesis?","n_subtopics":2}'
    # messages[] (Bedrock shape) is also accepted; last user text becomes the question.

Optional keys: `n_subtopics` (default 3), `grounded` (default false).

## 5. Add LangGraph later (no code change)
Append a second runtimes[] entry: name "langgraph", same codeLocation ".",
entrypoint "deploy/server.py", dockerfile "Dockerfile", envVars APP=langgraph.
Re-run configure/launch. deploy/server.py already dispatches on APP.

## 6. Teardown (stop spend)
Delete the runtime(s) and the ECR image via the CLI/console when done.
