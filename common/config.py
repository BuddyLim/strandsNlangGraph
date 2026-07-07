from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment and a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = ""
    model_id: str = "gemini-2.5-flash"
    n_subtopics: int = 3


settings = Settings()


def require_api_key() -> str:
    """Return the Gemini API key, or fail fast with an actionable message.

    Called at model-construction time (not import time) so tests can import
    modules without a key present.
    """
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return settings.google_api_key
