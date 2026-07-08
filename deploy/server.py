"""AgentCore HTTP entrypoint wrapping the research assistant (Pattern A).

One wrapper serves both frameworks; the `APP` setting selects which
`run_research` to import. It is read from the process environment first — the
AgentCore runtime injects it via agentcore.json `envVars` — and falls back to a
local `.env` for development, the same source order `common.config` uses for the
API key. Resolved once at import so a misconfigured runtime fails fast on cold
start, not per request.
"""

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.config import require_api_key
from common.types import ResearchRequest


class DeploySettings(BaseSettings):
    """Deploy-layer config. `app` picks the framework runtime to serve.

    A raw ``os.environ`` read would miss ``.env`` (pydantic-settings loads it
    into a settings object, not into the process environment), so local runs
    could not configure ``APP`` the way they configure ``GOOGLE_API_KEY``.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app: str = ""


APP = DeploySettings().app
if APP == "strands":
    from strands_app.research import run_research
elif APP == "langgraph":
    from langgraph_app.research import run_research
else:
    raise RuntimeError(f"APP must be 'strands' or 'langgraph', got {APP!r}")

# Fail fast on a missing Gemini key at cold start, matching the CLI contract.
require_api_key()

app = BedrockAgentCoreApp()


def _last_user_text(messages: list) -> str:
    """Best-effort text of the most recent user message in a Bedrock messages list.

    Content may be a plain string or a list of blocks (``[{"text": ...}]``); both
    shapes appear depending on the caller. Returns "" if no user text is found.
    """
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            ).strip()
            if text:
                return text
    return ""


def _extract_question(payload: dict) -> str:
    """Pull the research question out of an AgentCore invocation payload.

    The runtime hands the entrypoint the raw JSON body. Direct callers use our
    explicit ``{"question": ...}`` contract, but the AgentCore console and CLI
    send the framework-native ``{"prompt": ...}`` or ``{"messages": [...]}``
    shapes — accept all three rather than 500 on the common case.
    """
    question = payload.get("question") or payload.get("prompt")
    if not question:
        question = _last_user_text(payload.get("messages") or [])
    if not question:
        raise ValueError(
            "payload must include a non-empty 'question', 'prompt', or 'messages' entry"
        )
    return question


def _handle(payload: dict) -> dict:
    """Map a JSON payload to a report dict.

    Accepts ``question`` / ``prompt`` / ``messages`` for the research question,
    plus optional ``n_subtopics`` and ``grounded``. Framework-free (no
    BedrockAgentCore types) so it is unit-testable without the runtime; ``invoke``
    is the thin decorated entrypoint that delegates here.
    """
    request = ResearchRequest(
        question=_extract_question(payload),
        n_subtopics=payload.get("n_subtopics", 3),
    )
    report = run_research(request, grounded=payload.get("grounded", False))
    return report.model_dump()


@app.entrypoint
def invoke(payload: dict, context) -> dict:
    return _handle(payload)


if __name__ == "__main__":
    app.run()
