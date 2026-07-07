# Strands Research Assistant (Unit 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared `common/` package and a Strands research assistant (single-agent basic stage + subagent-spawning fan-out stage) that runs locally against Gemini.

**Architecture:** A framework-neutral `common/` package (config, Pydantic types, a deterministic mock search tool, shared prompts) is consumed by `strands_app/`. Stage A is a single Strands `Agent` with one tool. Stage B is a code-orchestrated fan-out: a planner splits the question into N subtopics, one fresh sub-agent is spawned per subtopic, and a synthesis agent merges the findings into a `ResearchReport`. A `--grounded` flag swaps the mock tool for Gemini's native Google Search grounding.

**Tech Stack:** Python 3.11+, `strands-agents[gemini]`, `pydantic`, `pydantic-settings`, `google-genai` (transitively, for grounding types), `pytest`, `ruff`.

## Global Constraints

- Python 3.11+ (`requires-python = ">=3.11"`).
- Single model provider: Gemini via `strands.models.gemini.GeminiModel`, model id `gemini-2.5-flash`.
- `common/` MUST stay framework-neutral — no `strands` imports anywhere under `common/` (it is reused verbatim by the future LangGraph app).
- Tests never make live model calls. The model is injected/stubbed at the boundary. Live checks are separate and skipped without `GOOGLE_API_KEY`.
- Config fails fast with a clear message when `GOOGLE_API_KEY` is missing — but only at model-construction time, never at import time (so tests can import freely).
- Commit after every task.

---

### Task 1: Project scaffold + config

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `common/__init__.py` (empty)
- Create: `common/config.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `common.config.settings` (a `Settings` instance with `google_api_key: str`, `model_id: str`, `n_subtopics: int`); `common.config.require_api_key() -> str` (returns the key or raises `RuntimeError`).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "strands-n-langgraph"
version = "0.1.0"
description = "Comparative learning repo: Strands vs LangGraph research assistant"
requires-python = ">=3.11"
dependencies = [
    "strands-agents[gemini]>=0.1.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]

[tool.pytest.ini_options]
markers = ["live: makes real Gemini API calls; skipped without GOOGLE_API_KEY"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Write `.env.example`**

```
GOOGLE_API_KEY=your-gemini-api-key-here
```

- [ ] **Step 3: Create empty package markers**

Create `common/__init__.py` and `tests/__init__.py` as empty files.

- [ ] **Step 4: Write the failing test** — `tests/test_config.py`

```python
import pytest
from common.config import Settings, require_api_key


def test_settings_have_gemini_defaults():
    s = Settings(google_api_key="x")
    assert s.model_id == "gemini-2.5-flash"
    assert s.n_subtopics == 3


def test_require_api_key_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.setattr("common.config.settings", Settings(google_api_key=""))
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        require_api_key()


def test_require_api_key_returns_key_when_present(monkeypatch):
    monkeypatch.setattr("common.config.settings", Settings(google_api_key="secret"))
    assert require_api_key() == "secret"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'common.config'`

- [ ] **Step 6: Write `common/config.py`**

```python
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
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .env.example common/ tests/
git commit -m "add project scaffold and fail-fast config"
```

---

### Task 2: Shared Pydantic types

**Files:**
- Create: `common/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ResearchRequest(question: str, n_subtopics: int = 3)`
  - `SubFinding(subtopic: str, findings: str, ok: bool = True)`
  - `ResearchReport(question: str, summary: str, findings: list[SubFinding])`
  - `SubtopicPlan(subtopics: list[str])`

- [ ] **Step 1: Write the failing test** — `tests/test_types.py`

```python
from common.types import ResearchRequest, SubFinding, ResearchReport, SubtopicPlan


def test_research_request_defaults_to_three_subtopics():
    req = ResearchRequest(question="What is X?")
    assert req.n_subtopics == 3


def test_sub_finding_defaults_to_ok():
    f = SubFinding(subtopic="a", findings="text")
    assert f.ok is True


def test_research_report_holds_findings():
    report = ResearchReport(
        question="Q", summary="S",
        findings=[SubFinding(subtopic="a", findings="t")],
    )
    assert len(report.findings) == 1
    assert report.summary == "S"


def test_subtopic_plan_holds_list():
    plan = SubtopicPlan(subtopics=["a", "b"])
    assert plan.subtopics == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'common.types'`

- [ ] **Step 3: Write `common/types.py`**

```python
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


class SubtopicPlan(BaseModel):
    """Structured output of the planner: the subtopics to research."""

    subtopics: list[str]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_types.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add common/types.py tests/test_types.py
git commit -m "add shared research Pydantic types"
```

---

### Task 3: Shared mock search tool + prompts

**Files:**
- Create: `common/tools.py`
- Create: `common/prompts.py`
- Test: `tests/test_common_tools.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `common.tools.mock_search(subtopic: str) -> str` — deterministic canned results. NOTE: this is a plain function (no `strands` import — `common/` stays framework-neutral). Each framework wraps it in its own tool decorator.
  - `common.prompts.SINGLE_AGENT_PROMPT`, `PLANNER_PROMPT`, `SUB_AGENT_PROMPT`, `SYNTHESIS_PROMPT` (str constants).

- [ ] **Step 1: Write the failing test** — `tests/test_common_tools.py`

```python
from common.tools import mock_search


def test_mock_search_is_deterministic():
    assert mock_search("photosynthesis") == mock_search("photosynthesis")


def test_mock_search_mentions_the_subtopic():
    result = mock_search("quantum tunneling")
    assert "quantum tunneling" in result


def test_mock_search_returns_nonempty_for_any_subtopic():
    assert mock_search("anything at all").strip() != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_common_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'common.tools'`

- [ ] **Step 3: Write `common/tools.py`**

```python
def mock_search(subtopic: str) -> str:
    """Return deterministic, canned 'search results' for a subtopic.

    Framework-neutral on purpose: this is a plain function so both the Strands
    and LangGraph apps can wrap it in their own tool abstraction. Deterministic
    output keeps test runs free and reproducible. Swap for a real search later
    behind this same signature.
    """
    return (
        f"Search results for '{subtopic}':\n"
        f"1. Overview of {subtopic}: a concise, factual summary.\n"
        f"2. Key considerations regarding {subtopic}.\n"
        f"3. A commonly cited example involving {subtopic}."
    )
```

- [ ] **Step 4: Write `common/prompts.py`**

```python
SINGLE_AGENT_PROMPT = (
    "You are a concise research assistant. Use the search tool when you need "
    "facts, then answer in a short, well-structured brief."
)

PLANNER_PROMPT = (
    "You break a research question into distinct, non-overlapping subtopics. "
    "Return exactly the requested number of subtopics, each a short phrase."
)

SUB_AGENT_PROMPT = (
    "You are a focused researcher investigating ONE subtopic. Use the search "
    "tool, then report only factual findings for that subtopic in 2-4 sentences."
)

SYNTHESIS_PROMPT = (
    "You synthesize per-subtopic findings into one coherent answer to the "
    "original question. Be concise and do not invent facts beyond the findings."
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_common_tools.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add common/tools.py common/prompts.py tests/test_common_tools.py
git commit -m "add shared mock search tool and prompts"
```

---

### Task 4: Strands model factory (mock + grounded)

**Files:**
- Create: `strands_app/__init__.py` (empty)
- Create: `strands_app/model.py`
- Test: `tests/test_strands_model.py`

**Interfaces:**
- Consumes: `common.config.settings`, `common.config.require_api_key`.
- Produces: `strands_app.model.build_gemini_model(grounded: bool = False) -> GeminiModel`.
  When `grounded=True`, the model carries Gemini's native Google Search tool.

- [ ] **Step 1: Create `strands_app/__init__.py`** (empty file).

- [ ] **Step 2: Write the failing test** — `tests/test_strands_model.py`

```python
import pytest
from common.config import Settings


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setattr("common.config.settings", Settings(google_api_key="test-key"))
    monkeypatch.setattr("strands_app.model.settings", Settings(google_api_key="test-key"))


def test_build_model_uses_configured_model_id():
    from strands_app.model import build_gemini_model
    model = build_gemini_model()
    # GeminiModel exposes its config; model_id must match settings.
    assert model.get_config()["model_id"] == "gemini-2.5-flash"


def test_build_model_grounded_attaches_search_tool():
    from strands_app.model import build_gemini_model
    grounded = build_gemini_model(grounded=True)
    plain = build_gemini_model(grounded=False)
    # Grounded model carries at least one native gemini tool; plain carries none.
    assert grounded.get_config().get("gemini_tools")
    assert not plain.get_config().get("gemini_tools")


def test_build_model_raises_without_key(monkeypatch):
    monkeypatch.setattr("common.config.settings", Settings(google_api_key=""))
    monkeypatch.setattr("strands_app.model.settings", Settings(google_api_key=""))
    from strands_app.model import build_gemini_model
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        build_gemini_model()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_strands_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strands_app.model'`

- [ ] **Step 4: Write `strands_app/model.py`**

```python
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
    gemini_tools = (
        [types.Tool(google_search=types.GoogleSearch())] if grounded else None
    )
    return GeminiModel(
        client_args={"api_key": api_key},
        model_id=settings.model_id,
        gemini_tools=gemini_tools,
    )
```

> Note: if `GeminiModel` rejects `gemini_tools=None`, pass the kwarg only when
> `grounded` is true by building a `kwargs` dict. Verify in Step 5.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_strands_model.py -v`
Expected: PASS (3 passed). If `test_build_model_grounded_attaches_search_tool`
fails on the config key name, print `build_gemini_model(grounded=True).get_config()`
and adjust the assertion/key to the actual field, then re-run.

- [ ] **Step 6: Commit**

```bash
git add strands_app/__init__.py strands_app/model.py tests/test_strands_model.py
git commit -m "add Strands Gemini model factory with grounded mode"
```

---

### Task 5: Stage A — basic single-agent

**Files:**
- Create: `strands_app/basic.py`
- Test: `tests/test_strands_basic.py`

**Interfaces:**
- Consumes: `strands_app.model.build_gemini_model`, `common.tools.mock_search`, `common.prompts.SINGLE_AGENT_PROMPT`.
- Produces:
  - `strands_app.basic.build_basic_agent(model=None) -> Agent`
  - `strands_app.basic.answer_question(question: str, model=None) -> str`

- [ ] **Step 1: Write the failing test** — `tests/test_strands_basic.py`

```python
from strands_app import basic


class _StubAgent:
    def __init__(self):
        self.called_with = None

    def __call__(self, prompt):
        self.called_with = prompt
        return "canned answer"


def test_answer_question_delegates_to_agent_and_returns_text(monkeypatch):
    stub = _StubAgent()
    monkeypatch.setattr(basic, "build_basic_agent", lambda model=None: stub)
    result = basic.answer_question("What is X?")
    assert result == "canned answer"
    assert stub.called_with == "What is X?"


def test_basic_agent_registers_mock_search_as_a_tool():
    # Construct with an injected sentinel model so no API key/model is needed.
    agent = basic.build_basic_agent(model=object())
    assert "mock_search" in agent.tool_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_strands_basic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strands_app.basic'`

- [ ] **Step 3: Write `strands_app/basic.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strands_basic.py -v`
Expected: PASS (2 passed). If `agent.tool_names` is not the correct attribute
for the installed Strands version, inspect `dir(agent)` for the tool registry
accessor and update the assertion + any downstream use.

- [ ] **Step 5: Commit**

```bash
git add strands_app/basic.py tests/test_strands_basic.py
git commit -m "add Strands Stage A basic single-agent"
```

---

### Task 6: Stage B — subagent-spawning research fan-out

**Files:**
- Create: `strands_app/research.py`
- Test: `tests/test_strands_research.py`

**Interfaces:**
- Consumes: `strands_app.model.build_gemini_model`, `strands_app.basic.mock_search` (the `@tool`-wrapped version), `common.prompts` (`PLANNER_PROMPT`, `SUB_AGENT_PROMPT`, `SYNTHESIS_PROMPT`), `common.types` (`ResearchRequest`, `ResearchReport`, `SubFinding`, `SubtopicPlan`).
- Produces:
  - `plan_subtopics(question: str, n: int, model=None) -> list[str]`
  - `research_subtopic(subtopic: str, grounded: bool = False, model=None) -> SubFinding`
  - `synthesize(question: str, findings: list[SubFinding], model=None) -> str`
  - `run_research(request: ResearchRequest, grounded: bool = False, model=None) -> ResearchReport`

- [ ] **Step 1: Write the failing test** — `tests/test_strands_research.py`

```python
import pytest
from common.types import ResearchRequest, SubFinding
from strands_app import research


def test_run_research_fans_out_one_finding_per_subtopic(monkeypatch):
    monkeypatch.setattr(research, "plan_subtopics",
                        lambda q, n, model=None: ["a", "b", "c"])
    monkeypatch.setattr(research, "research_subtopic",
                        lambda st, grounded=False, model=None:
                        SubFinding(subtopic=st, findings=f"found {st}"))
    monkeypatch.setattr(research, "synthesize",
                        lambda q, findings, model=None: "final summary")

    report = research.run_research(ResearchRequest(question="Q", n_subtopics=3))

    assert report.summary == "final summary"
    assert [f.subtopic for f in report.findings] == ["a", "b", "c"]


def test_research_subtopic_degrades_gracefully_on_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("model exploded")
    # Agent construction inside research_subtopic raises.
    monkeypatch.setattr(research, "Agent", _boom)

    finding = research.research_subtopic("doomed subtopic", model=object())

    assert finding.ok is False
    assert "doomed subtopic" == finding.subtopic
    assert "model exploded" in finding.findings


def test_synthesize_includes_each_subtopic_in_the_prompt(monkeypatch):
    seen = {}

    class _StubAgent:
        def __init__(self, **kwargs): ...
        def __call__(self, prompt):
            seen["prompt"] = prompt
            return "synthesized"

    monkeypatch.setattr(research, "Agent", _StubAgent)
    findings = [SubFinding(subtopic="alpha", findings="x"),
                SubFinding(subtopic="beta", findings="y")]

    out = research.synthesize("Q", findings, model=object())

    assert out == "synthesized"
    assert "alpha" in seen["prompt"] and "beta" in seen["prompt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_strands_research.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strands_app.research'`

- [ ] **Step 3: Write `strands_app/research.py`**

```python
from strands import Agent, tool

from common.prompts import PLANNER_PROMPT, SUB_AGENT_PROMPT, SYNTHESIS_PROMPT
from common.tools import mock_search as _mock_search
from common.types import ResearchReport, ResearchRequest, SubFinding, SubtopicPlan
from strands_app.model import build_gemini_model


@tool
def mock_search(subtopic: str) -> str:
    """Search for information about a subtopic and return findings."""
    return _mock_search(subtopic)


def plan_subtopics(question: str, n: int, model=None) -> list[str]:
    """Split the question into n distinct subtopics using structured output."""
    planner = Agent(model=model or build_gemini_model(),
                    system_prompt=PLANNER_PROMPT)
    plan: SubtopicPlan = planner.structured_output(
        SubtopicPlan,
        f"Question: {question}\nProduce exactly {n} subtopics.",
    )
    return plan.subtopics[:n]


def research_subtopic(subtopic: str, grounded: bool = False, model=None) -> SubFinding:
    """Spawn a fresh sub-agent to research one subtopic.

    Each call constructs its own Agent — that construction IS the subagent
    spawn. Failures are caught and returned as a degraded SubFinding so one bad
    subtopic never aborts the whole report.
    """
    try:
        sub_model = model or build_gemini_model(grounded=grounded)
        tools = None if grounded else [mock_search]
        researcher = Agent(model=sub_model, tools=tools,
                           system_prompt=SUB_AGENT_PROMPT)
        answer = str(researcher(f"Research this subtopic: {subtopic}"))
        return SubFinding(subtopic=subtopic, findings=answer, ok=True)
    except Exception as exc:  # noqa: BLE001 — deliberate graceful degradation boundary
        return SubFinding(subtopic=subtopic, findings=f"failed: {exc}", ok=False)


def synthesize(question: str, findings: list[SubFinding], model=None) -> str:
    """Merge per-subtopic findings into one answer to the original question."""
    synthesizer = Agent(model=model or build_gemini_model(),
                        system_prompt=SYNTHESIS_PROMPT)
    joined = "\n\n".join(f"## {f.subtopic}\n{f.findings}" for f in findings)
    return str(synthesizer(
        f"Question: {question}\n\nFindings:\n{joined}\n\nWrite a synthesis."
    ))


def run_research(request: ResearchRequest, grounded: bool = False,
                 model=None) -> ResearchReport:
    """Plan subtopics, spawn one sub-agent per subtopic, then synthesize."""
    subtopics = plan_subtopics(request.question, request.n_subtopics, model=model)
    findings = [research_subtopic(st, grounded=grounded, model=model)
                for st in subtopics]
    summary = synthesize(request.question, findings, model=model)
    return ResearchReport(question=request.question, summary=summary,
                          findings=findings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strands_research.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add strands_app/research.py tests/test_strands_research.py
git commit -m "add Strands Stage B subagent-spawning research fan-out"
```

---

### Task 7: CLI entrypoint + live smoke test

**Files:**
- Create: `strands_app/run.py`
- Test: `tests/test_strands_run.py`

**Interfaces:**
- Consumes: `common.config.settings`, `common.types.ResearchRequest`, `strands_app.research.run_research`.
- Produces:
  - `strands_app.run.format_report(report: ResearchReport) -> str`
  - `strands_app.run.main(argv: list[str] | None = None) -> int`
  - CLI: `python -m strands_app.run "question" [--subtopics N] [--grounded]`

- [ ] **Step 1: Write the failing test** — `tests/test_strands_run.py`

```python
import os
import pytest
from common.types import ResearchReport, SubFinding, ResearchRequest
from strands_app import run


def test_format_report_shows_summary_and_each_subtopic():
    report = ResearchReport(
        question="Q", summary="the summary",
        findings=[SubFinding(subtopic="alpha", findings="x"),
                  SubFinding(subtopic="beta", findings="y")],
    )
    text = run.format_report(report)
    assert "the summary" in text
    assert "alpha" in text and "beta" in text


def test_main_wires_args_into_run_research(monkeypatch, capsys):
    captured = {}

    def _fake_run_research(request, grounded=False, model=None):
        captured["request"] = request
        captured["grounded"] = grounded
        return ResearchReport(question=request.question, summary="ok", findings=[])

    monkeypatch.setattr(run, "run_research", _fake_run_research)
    code = run.main(["What is X?", "--subtopics", "2", "--grounded"])

    assert code == 0
    assert captured["request"].question == "What is X?"
    assert captured["request"].n_subtopics == 2
    assert captured["grounded"] is True
    assert "ok" in capsys.readouterr().out


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="needs GOOGLE_API_KEY")
def test_live_end_to_end_smoke():
    report = run_research_live()
    assert report.summary.strip() != ""
    assert len(report.findings) == 2


def run_research_live():
    from strands_app.research import run_research
    return run_research(ResearchRequest(question="What is photosynthesis?",
                                        n_subtopics=2))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_strands_run.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strands_app.run'`
(The `live` test is skipped — no key in CI.)

- [ ] **Step 3: Write `strands_app/run.py`**

```python
import argparse
import sys

from common.config import settings
from common.types import ResearchReport, ResearchRequest
from strands_app.research import run_research


def format_report(report: ResearchReport) -> str:
    """Render a report as readable CLI text."""
    lines = [f"# Research: {report.question}", "", "## Summary", report.summary, ""]
    for f in report.findings:
        status = "" if f.ok else " (failed)"
        lines += [f"## {f.subtopic}{status}", f.findings, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strands research assistant")
    parser.add_argument("question", help="the research question")
    parser.add_argument("--subtopics", type=int, default=settings.n_subtopics,
                        help="number of subtopics to fan out into")
    parser.add_argument("--grounded", action="store_true",
                        help="use Gemini native Google Search instead of the mock tool")
    args = parser.parse_args(argv)

    request = ResearchRequest(question=args.question, n_subtopics=args.subtopics)
    try:
        report = run_research(request, grounded=args.grounded)
    except RuntimeError as exc:
        # Config failures (e.g. missing key) — clean message, no stack trace.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: never dump a stack trace
        # Covers model throttling / transient provider errors surfaced by Strands.
        print(f"research failed: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strands_run.py -v`
Expected: PASS (2 passed, 1 skipped)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all pass, the single `live` test skipped.

- [ ] **Step 6: Live verification (manual, requires a real key)**

```bash
cp .env.example .env   # then edit .env to add a real GOOGLE_API_KEY
python -m strands_app.run "What is photosynthesis?" --subtopics 2
```
Expected: a printed report with a non-empty summary and two subtopic sections.
Optionally run the grounded path: `python -m strands_app.run "Latest Mars mission news" --grounded`.

- [ ] **Step 7: Commit**

```bash
git add strands_app/run.py tests/test_strands_run.py
git commit -m "add Strands CLI entrypoint and live smoke test"
```

---

## Self-Review

**Spec coverage:**
- Repo layout (Approach A) → Tasks 1–7 create exactly the spec's file tree (`deploy/` intentionally deferred to Unit 3).
- Stage A basic API → Task 5.
- Stage B subagent spawning → Task 6 (one `Agent` spawned per subtopic).
- Model factory → Task 4.
- Mock tool + grounding variant → Task 3 (mock) + Task 4 (`grounded=True`) + Task 7 (`--grounded` flag).
- Config fail-fast → Task 1.
- Sub-agent graceful degradation → Task 6 (`research_subtopic` try/except + test).
- Model-throttling clean CLI message → Task 7 `main` catches `RuntimeError` (config) and any other exception at the CLI boundary, printing a one-line message and returning exit code 1. This covers `ModelThrottledException` without hard-coding its import path (caught by the broad `except`), so no stack trace ever reaches the user.
- Tests at boundaries, deterministic → every task mocks at the boundary; one opt-in `live` test.

**Placeholder scan:** one `>` note in Task 4 flags an SDK-detail verification (the grounded-config field name via `get_config()`) — a verification instruction with a concrete fallback, not an unfilled placeholder.

**Type consistency:** `SubFinding(subtopic, findings, ok)`, `ResearchReport(question, summary, findings)`, `SubtopicPlan(subtopics)`, and the four `research.py` function signatures are used identically across Tasks 2, 6, and 7. `mock_search` is a plain function in `common/tools.py` (Task 3) and re-wrapped as a `@tool` in both `basic.py` and `research.py` (Tasks 5, 6) — consistent.
