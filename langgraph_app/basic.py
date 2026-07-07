from langchain.agents import create_agent
from langchain_core.tools import tool

from common.prompts import SINGLE_AGENT_PROMPT
from common.tools import mock_search as _mock_search
from langgraph_app.model import build_gemini_model


@tool
def mock_search(subtopic: str) -> str:
    """Search for information about a subtopic and return findings."""
    return _mock_search(subtopic)


def build_basic_agent(model=None):
    """The smallest complete LangGraph agent: one model, one tool, one prompt."""
    return create_agent(
        model or build_gemini_model(),
        tools=[mock_search],
        system_prompt=SINGLE_AGENT_PROMPT,
    )


def answer_question(question: str, model=None) -> str:
    """Answer a single question with the basic agent."""
    agent = build_basic_agent(model)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
