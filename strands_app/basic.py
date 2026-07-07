from strands import Agent, tool

from common.prompts import SINGLE_AGENT_PROMPT
from common.tools import mock_search as _mock_search
from strands_app.model import build_gemini_model


@tool
def mock_search(subtopic: str) -> str:
    """Search for information about a subtopic and return findings."""
    return _mock_search(subtopic)


def build_basic_agent(model=None) -> Agent:
    """The smallest complete Strands agent: one model, one tool, one prompt."""
    return Agent(
        model=model or build_gemini_model(),
        tools=[mock_search],
        system_prompt=SINGLE_AGENT_PROMPT,
    )


def answer_question(question: str, model=None) -> str:
    """Answer a single question with the basic agent."""
    agent = build_basic_agent(model)
    return str(agent(question))
