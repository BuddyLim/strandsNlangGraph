from google.genai import types
from strands.models.gemini import GeminiModel

from common.config import require_api_key, settings


def build_gemini_model(grounded: bool = False) -> GeminiModel:
    """Construct the Gemini model both stages share.

    grounded=True attaches Gemini's native Google Search tool, so the model
    grounds its answers internally instead of calling our framework-level tool.
    Constructing the model does not call the API; the key is validated here so
    failures are early and clear.
    """
    api_key = require_api_key()
    # GeminiModel raises TypeError on gemini_tools=None (verified against
    # strands-agents 1.45.0), so only pass the kwarg in grounded mode.
    kwargs = {"client_args": {"api_key": api_key}, "model_id": settings.model_id}
    if grounded:
        kwargs["gemini_tools"] = [types.Tool(google_search=types.GoogleSearch())]
    return GeminiModel(**kwargs)
