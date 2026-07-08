# Unit 3 — AgentCore Deploy Runbook (manual, spend-flagged)

> Flag names marked `<recorded flags>` are resolved by the Task 1 CLI
> verification gate (repo-root `codeLocation "."`, ARM64 default, and the Gemini
> key env-var name). Fill them in once that gate has been run.

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
