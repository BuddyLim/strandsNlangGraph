Guidelines for how Claude should approach development in this codebase.

---

## Scope & PR Discipline

- Target ~15 files changed per session. If the natural scope exceeds this, flag it and suggest splitting the work before writing any code.
- Before writing code, list the files you expect to touch. If the list feels large, say so.
- Features are product concepts — break them into independently mergeable engineering units. Each session should target one unit.
- Refactors and feature work should not be mixed. If you spot something worth refactoring while implementing a feature, note it in a comment (`# TODO: refactor candidate — <reason>`) and move on.
- Aim for the smallest change that moves the codebase to a valid, non-broken state.

---

## During Plan Mode

When producing a plan — whether in Plan Mode or as a written design doc — follow the structure below. Optimize for information-per-word: bullets, diagrams, and the decision log should do most of the work, with prose only where it earns its place. The goal is a plan that's reviewable in one pass.

## During Plan Mode

Follow this structure exactly. Mark inapplicable sections `N/A — <reason>`.

**Blast radius** — files expected to change with rough LOC delta. If this exceeds 15 files, stop and propose a split before continuing.
**Problem** — what's broken or missing, one short paragraph
**Non-goals** — what this plan deliberately won't do
**Approach** — chosen design. Include an ASCII diagram when the change 
crosses layers (route ↔ service ↔ repository ↔ frontend) or introduces 
async flow (SSE, background jobs, websocket). Label each box with its 
layer. Schema-only changes don't need a diagram — call out the migration 
in prose.
**Alternatives considered** — at least one rejected alternative and why
**Decision log** — non-obvious choices only. Format: `Chose X over Y because Z`
**Risks & rollback** — what could go wrong, how to undo
**Open questions** — anything needed from you before coding begins

**Lead with `Blast radius`, not prose.** List files and rough LOC delta before writing the approach. This forces the ~15-file scope check to happen up front. If the list exceeds 15, stop and propose a split before continuing.

**Include at least one diagram when the plan crosses layer boundaries, changes schema, or introduces async flow.** ASCII is fine. Label each box with its layer (route / service / repository / external) so layer violations show up at planning time, not review time. If a reviewer would have to reconstruct the topology from prose, draw it.

**Decision log: non-obvious choices only, one line each.** Format: `Chose X over Y because Z`. If the alternative wouldn't survive five seconds of scrutiny, skip it. The log catches the small forks that prose hides — `service method vs route helper`, `JSON column vs join table`, `sync vs async`. The big architectural shape belongs in `Alternatives considered`.

---

## Module Design (after Ousterhout)

- **Prefer deep modules over shallow ones.** A module should hide significant complexity behind a simple interface. Avoid splitting logic into many small functions or classes just for the sake of it — each abstraction should earn its existence.
- **Different layer, different abstraction.** Each layer (e.g. repository, service, route) should have its own vocabulary. Don't pass raw DB models up to the API layer, and don't leak HTTP concepts down into business logic.
- **Define errors out of existence where possible.** Prefer designs that make invalid states unrepresentable over scattered defensive checks. When errors must exist, handle them at the right layer — not everywhere.
- **General-purpose over special-purpose.** When writing a module, ask whether a slightly more general version would serve multiple use cases without added complexity. Avoid one-off abstractions.

---

## Comments & Documentation

- Comments should explain _why_, not _what_. The code says what; the comment explains the reasoning, tradeoffs, or constraints that aren't obvious from reading it.
- Don't comment things that are self-evident from the code.
- Document interfaces, not implementations. A function's docstring should describe its contract — inputs, outputs, side effects — not narrate the body.

---

## FastAPI (Backend)

- Follow a controller / service / repository folder structure. Keep route handlers thin — business logic belongs in the service layer.
- Services should not import from routes. Repositories should not import from services.
- Use Pydantic models for all request/response shapes. Don't pass raw dicts across layer boundaries.
- Prefer explicit dependency injection via `Depends()` over module-level globals.

---

## React + Vite (Frontend)

- Co-locate components with the routes that use them unless they're genuinely reusable.
- Server components by default; opt into client components only when you need interactivity or browser APIs.
- Data fetching belongs close to where it's rendered — avoid prop-drilling fetched data through multiple layers.
- Keep API calls in a dedicated layer (e.g. `lib/api/`), not scattered across components.

---

## Testing

- Tests are part of the implementation, not an afterthought. Write them in the same session unless explicitly told otherwise.
- **Test behaviour, not implementation.** Tests should assert on inputs and outputs, not on how the internals work. If refactoring breaks a test without changing observable behaviour, the test was wrong.
- **Test at the right layer.** Unit test services and pure logic. Integration test repositories and API routes. Don't unit test things that are better covered by integration tests, and don't write integration tests for things that are trivially covered by unit tests.
- One test file per module. Keep test names parallel to source modules (e.g. `tests/test_user_service.py` mirrors `app/services/user_service.py`). The backend uses a flat test layout — don't introduce subdirectories.
- Prefer real objects over mocks where it's not expensive. Mock at boundaries — external APIs, third-party services, the filesystem — not between internal layers.
- Each test should have one reason to fail. Avoid asserting on multiple unrelated behaviours in a single test.
- Tests should be readable as documentation. A test name should describe the scenario and expected outcome: `test_create_user_returns_409_when_email_already_exists`, not `test_create_user_error`.

---

## Error Handling

- **Handle errors at the right layer, not everywhere.** Repositories surface DB exceptions. Services translate them into domain errors. Routes translate domain errors into HTTP responses. Don't catch and re-raise through every layer.
- Prefer domain-specific exceptions over generic ones. `UserNotFoundError` is more useful than `ValueError("user not found")`.
- In FastAPI, use exception handlers registered at the app level for consistent HTTP error shaping. Don't write `try/except` blocks in route handlers unless the handling is genuinely route-specific.
- Never expose internal error details (stack traces, DB messages) in API responses. Log them server-side; return a clean, structured error shape to the client.
- Distinguish between expected errors (user input, not found, conflict) and unexpected errors (infra failure, unhandled exception). Handle them differently — expected errors are part of the domain, unexpected errors should alert.

---

## Naming

- Names should be precise enough that a reader doesn't need to look at the implementation to understand the intent. If you find yourself writing a comment to explain what a variable holds, the name is probably wrong.
- Avoid generic names: `data`, `result`, `info`, `handler`, `manager`, `util`. Name things by what they specifically represent.
- Functions should be named for what they return or what they do — not how they do it. `get_active_users()` over `query_users_with_active_flag()`.
- Boolean variables and functions should read as assertions: `is_expired`, `has_permission`, `can_publish` — not `check_expiry` or `expiry_status`.
- Be consistent with the codebase's existing vocabulary. If the codebase calls it a `policy`, don't introduce `rule` or `regulation` for the same concept.
- When a name feels hard to choose, that's often a signal the abstraction itself isn't clear yet. Pause and reconsider the design before forcing a name.

---

## Git Commit Granularity

- Commit messages should complete the sentence: _"This commit will..."_ — e.g. `add user authentication middleware`, not `auth stuff` or `wip`.
- Don't bundle a refactor and a feature in the same commit. If a refactor was necessary to enable a feature, commit the refactor first, then the feature.
- Avoid committing commented-out code, debug logs, or `print` statements.
- If a session produces work that spans multiple logical changes, flag it and suggest how to split the commits before finishing.

---

## General

- When in doubt between two approaches, briefly note the tradeoff and your reasoning before implementing. Don't just pick one silently.
- Consistency with existing patterns in the codebase takes precedence over personal preference.
- Don't introduce a new dependency without flagging it. Prefer solving problems with what's already in the stack.

