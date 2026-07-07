from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """A request to research a question by fanning out into subtopics."""

    question: str
    n_subtopics: int = 3


class SubFinding(BaseModel):
    """One sub-agent's result for a single subtopic. `ok=False` marks a
    subtopic that failed and was degraded gracefully rather than aborting."""

    subtopic: str
    findings: str
    ok: bool = True


class ResearchReport(BaseModel):
    """The synthesized answer plus the per-subtopic findings it was built from."""

    question: str
    summary: str
    findings: list[SubFinding] = Field(default_factory=list)
