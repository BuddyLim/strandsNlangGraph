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
