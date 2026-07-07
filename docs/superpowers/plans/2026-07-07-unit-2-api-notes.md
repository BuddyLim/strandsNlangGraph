# Unit 2 (LangGraph) — API facts pinned by live probes

Task 0 verification. All facts below come from running the commands shown
against the actually-installed package versions on 2026-07-07, on branch
`unit-2-langgraph`. `GOOGLE_API_KEY` was **not** set in this environment, so
the live-model coordinator-node probe could not run against a real model;
everything else (including the fake-model probe, which needs no key) ran live.

## API PINS (summary for later tasks)

| Fact | Value |
|---|---|
| prebuilt-agent import | `from langgraph.prebuilt import create_react_agent` (primary `from langchain.agents import create_agent` is **unavailable** — `langchain` package not installed) |
| system-prompt kwarg | `prompt` (keyword-only) |
| `langgraph` version | `1.2.8` |
| `langchain-core` version | `1.4.8` |
| `langchain-google-genai` version | `4.2.7` |
| second Google SDK co-installed? | No — only pre-existing `google-genai==1.75.0` (+ `google-auth==2.55.1`); `google-generativeai` was NOT installed |
| `streaming=True` on `ChatGoogleGenerativeAI` | Accepted — inherited from `BaseChatModel.__init__`, not an explicit param on `ChatGoogleGenerativeAI.__init__` |
| `_COORDINATOR_NODE` | `"agent"` (**default, not live-confirmed** — no `GOOGLE_API_KEY` available; must be confirmed against a real model at first live run) |
| `GenericFakeChatModel` through `create_react_agent`? | **Does NOT survive** — raises `NotImplementedError` from `BaseChatModel.bind_tools` (fake model has no tool-calling support). **Decision: Task 2's routing test must use the stub-agent wiring test only; tool routing is verified only in the live smoke test, not via a fake-model unit test.** |

## Step 1 — Add the dependencies

```
uv add langgraph langchain-google-genai langchain-core
```

Resolved and installed cleanly (82 packages resolved, 18 installed, 1 replaced:
`websockets` downgraded 16.0 → 15.0.1 to satisfy a shared constraint). Key new
packages: `langgraph==1.2.8`, `langgraph-prebuilt==1.1.0`,
`langgraph-checkpoint==4.1.1`, `langgraph-sdk==0.4.2`,
`langchain-core==1.4.8`, `langchain-google-genai==4.2.7`, `langsmith==0.9.8`.

`pyproject.toml` `[project].dependencies` now includes (alphabetized by uv):

```
"langchain-core>=1.4.8",
"langchain-google-genai>=4.2.7",
"langgraph>=1.2.8",
```

`uv.lock` was regenerated accordingly.

## Step 2 — Resolved versions

```
uv run python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['langgraph','langchain-core','langchain-google-genai']})"
```
Output:
```
{'langgraph': '1.2.8', 'langchain-core': '1.4.8', 'langchain-google-genai': '4.2.7'}
```

```
uv pip list | grep -i google
```
Output:
```
google-auth                             2.55.1
google-genai                            1.75.0
langchain-google-genai                  4.2.7
```

No `google-generativeai` package was co-installed. Only the pre-existing
`google-genai` SDK (used by the Strands app) and `google-auth` (a transitive
dependency) are present alongside `langchain-google-genai`.

## Step 3 — Prebuilt-agent entry point + system-prompt kwarg

Primary probe:
```
uv run python -c "from langchain.agents import create_agent; import inspect; print(inspect.signature(create_agent))"
```
Result: **fails** — `ModuleNotFoundError: No module named 'langchain'`. The
umbrella `langchain` package (which hosts the new `langchain.agents.create_agent`
entry point) is not part of this dependency set (only `langchain-core` was
installed, per Step 1's exact package list) and was not separately added,
consistent with the brief's task scope.

Fallback probe:
```
uv run python -c "from langgraph.prebuilt import create_react_agent; import inspect; print(inspect.signature(create_react_agent))"
```
Result: **succeeds**. Abbreviated signature:
```
create_react_agent(
    model: ...,
    tools: Sequence[BaseTool | Callable | dict] | ToolNode,
    *,
    prompt: SystemMessage | str | Callable | Runnable | None = None,
    response_format=None,
    pre_model_hook=None,
    post_model_hook=None,
    state_schema=None,
    context_schema=None,
    checkpointer=None,
    store=None,
    interrupt_before=None,
    interrupt_after=None,
    debug=False,
    version: Literal['v1','v2'] = 'v2',
    name=None,
    **deprecated_kwargs,
) -> CompiledStateGraph
```

**Decision: use `from langgraph.prebuilt import create_react_agent` as the
prebuilt-agent entry point for Unit 2. The system-prompt kwarg is `prompt`**
(keyword-only), not `system_prompt`. Later tasks should write
`create_react_agent(model, tools=[...], prompt=...)`.

Note (informational, not actioned): at call time in Step 5, LangGraph emitted
`LangGraphDeprecatedSinceV10: create_react_agent has been moved to
langchain.agents. Please update your import to
from langchain.agents import create_agent. Deprecated in LangGraph V1.0 to be
removed in V2.0.` This confirms `create_react_agent` still works in
`langgraph==1.2.8` (deprecated, not removed) but signals it may need to move
to `langchain.agents.create_agent` in a future package bump once `langchain`
is added as a dependency. Flagged for awareness; out of scope for Task 0.

## Step 4 — Model + fake-model + tool imports

```
uv run python -c "
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
import inspect
print('streaming' in inspect.signature(ChatGoogleGenerativeAI.__init__).parameters or 'streaming param inherited')
print('ok')
"
```
Output:
```
streaming param inherited
ok
```

All four imports succeed. `streaming` is not an explicit parameter of
`ChatGoogleGenerativeAI.__init__` but is inherited from `BaseChatModel`, so
`ChatGoogleGenerativeAI(..., streaming=True)` is accepted.

## Step 5 — Coordinator stream node name + fake-model viability

```
uv run python -c "
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_core.messages.tool import ToolCall
try:
    from langchain.agents import create_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent as create_agent

@tool
def echo(x: str) -> str:
    'echo a string'
    return x

fake = GenericFakeChatModel(messages=iter([
    AIMessage(content='', tool_calls=[ToolCall(name='echo', args={'x':'hi'}, id='c1')]),
    AIMessage(content='done'),
]))
agent = create_agent(fake, tools=[echo])
seen = set()
for chunk, meta in agent.stream({'messages': [{'role':'user','content':'go'}]}, stream_mode='messages'):
    seen.add(meta.get('langgraph_node'))
print('NODES:', seen)
res = agent.invoke({'messages': [{'role':'user','content':'go'}]})
print('FINAL:', res['messages'][-1].content)
"
```

Result: **raises before printing `NODES:`**:
```
NotImplementedError
  ... langgraph/prebuilt/chat_agent_executor.py:586, in create_react_agent
    model = cast(BaseChatModel, model).bind_tools(...)
  ... langchain_core/language_models/chat_models.py:2355, in bind_tools
    raise NotImplementedError
```

`GenericFakeChatModel` does not implement `bind_tools`, so `create_react_agent`
cannot bind the `echo` tool to it. Per the brief's documented fallback branch:

**Fake model does NOT survive `create_react_agent`.** Consequence for Task 2:
the routing test must be a **stub-agent wiring test** (asserting the graph is
built and wired correctly, e.g. checking node names / edges on the compiled
graph) rather than a fake-model end-to-end tool-routing test. Actual tool
routing through a real tool call can only be verified by the live smoke test
against a real Gemini model.

`GOOGLE_API_KEY` was not set (`echo "${GOOGLE_API_KEY:+set}"` printed empty),
so the live-model half of this probe (streaming a real `ChatGoogleGenerativeAI`
run and inspecting `meta['langgraph_node']`) was not run.

**`_COORDINATOR_NODE` is set to the default `"agent"`** (the conventional node
name `create_react_agent` uses for its model-calling node) but this is
**unconfirmed by a live probe** in this environment. **Flag: confirm
`_COORDINATOR_NODE = "agent"` against a real streamed run the first time a
`GOOGLE_API_KEY` is available** (e.g. during Task 2's or a later task's live
smoke test) before relying on it for stream-filtering logic.

## Open follow-ups for later tasks

- Confirm `_COORDINATOR_NODE = "agent"` live once an API key is available.
- If a future task wants the newer `langchain.agents.create_agent` API (to
  pre-empt the V2.0 removal of `create_react_agent` from `langgraph.prebuilt`),
  it will need to add `langchain` as an explicit dependency — not done here,
  out of scope for Task 0.
- Task 2 routing test: use a stub-agent wiring test (assert graph structure),
  not a `GenericFakeChatModel` tool-routing test.
