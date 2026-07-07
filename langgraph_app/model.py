from langchain_google_genai import ChatGoogleGenerativeAI

from common.config import require_api_key, settings


def build_gemini_model(grounded: bool = False):
    """Construct the Gemini model both stages share.

    grounded=True binds Gemini's native Google Search so the model grounds its
    answers internally instead of calling our framework-level tool. Construction
    does not call the API; the key is validated here so failures are early and clear.
    streaming=True is required for token streaming (stream_mode="messages" emits
    no token events without it).
    """
    api_key = require_api_key()
    model = ChatGoogleGenerativeAI(
        model=settings.model_id,
        google_api_key=api_key,
        streaming=True,
    )
    # bind_tools returns a bound runnable; only the ungrounded path returns the
    # bare chat model. Mirrors Unit 1's grounded/plain split.
    return model.bind_tools([{"google_search": {}}]) if grounded else model
