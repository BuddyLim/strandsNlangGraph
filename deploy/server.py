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
