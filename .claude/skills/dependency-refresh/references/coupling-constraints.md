# Coupling constraints & gotchas

Project-specific constraints hit during real refreshes — check here before
fighting a resolver conflict or a CI surprise, and add new ones in the same PR
that hits them. Key points only; verify current specifics against the resolver
and release notes.

## uv & the cooldown mechanism

- **Relative `exclude-newer` needs a recent uv.** Older uv warns
  (`Failed to parse pyproject.toml during settings discovery`), ignores the
  whole `[tool.uv]` table, and a re-lock with it silently strips the cooldown
  metadata from `uv.lock`. Always lock with the repo's pinned uv version.
- **A cooldown conflict looks like an unsatisfiable pin** — `uv lock` fails
  with an `exclude-newer` hint. That's enforcement, not breakage: hold the
  bump, or (security fixes only) add a commented temporary
  `[tool.uv.exclude-newer-package]` override per SKILL.md.
- **Re-locks refetch index metadata**, so unrelated wheel entries can drift out
  of `uv.lock` with no version change. Harmless for supported platforms.

## GitHub Actions

- **Node runtime deprecations show up as CI warnings, not lockfile conflicts**,
  and pinned `@vN` refs don't auto-upgrade. Check each action's `runs.using`
  — including the nested `uses:` lines inside composite actions — and bump to
  a major that ships the current runtime. Watch for behavior changes riding
  along (e.g. `docker/build-push-action` turning on provenance attestations
  by default).
- **Some actions stop publishing floating major tags** for supply-chain
  hardening (`astral-sh/setup-uv` as of v8) and must be pinned to an exact
  release.

## LangChain / LangGraph ecosystem

- **`langchain` ⇄ `langgraph` move in lockstep** — the meta-package pins a
  narrow `langgraph` range, so neither can be bumped past the other.
- **`langgraph` ⇄ `langgraph-checkpoint` ⇄ the checkpointer packages** upgrade
  as one coupled unit, and checkpointer majors can change stored-checkpoint
  compatibility or drop APIs (the mongodb saver dropped its async class;
  `src/memory/mongodb.py` bridges it). The test suite mocks the real backends,
  so verify old checkpoints still read under new savers manually.
- **LLM SDK bumps chain**: `langchain-<provider>` pulls a newer provider SDK
  which drags its own floors (e.g. `openai` → `jiter`). Expect to move the
  whole chain together.
- **Provider adapters can cap shared libs** (e.g. `langchain-google-vertexai`
  capping `pyarrow`) — a floor that won't move is usually one of these.
- **Adapter majors can change real-API behavior** (transports, structured
  output defaults) that the fake-model e2e can't exercise — check those paths
  with a real provider key before trusting the bump in production.
- **`langfuse` SDK usage here is tiny** (`CallbackHandler` +
  `auth_check()`), so SDK majors are usually version-bumps; the heavyweight
  self-host infra requirements belong to the Langfuse *server*, which this
  repo doesn't bundle.

## Tooling & misc

- **`ruff` formatter output can change on minor bumps** — expect one-time
  `ruff format` diffs, not just new lint rules.
- **Uncapped dev-tool majors flood in via `uv lock --upgrade`** (this bit with
  mypy). Cap temporarily to keep a disruptive major out of a routine round,
  and lift the cap in a focused pass.
- **`numpy` leads Python-version drops** — its floor is usually the first hard
  constraint when deciding when to drop an old Python minor.
- **Don't write resolver picks down as requirements.** Pin-reconciliation once
  produced a `>=` floor on a transitive-only package that nothing required,
  which later blocked an unrelated upgrade. Transitive-only packages don't
  belong in `[project] dependencies` at all.

## Python interpreter

- **Validate new Python minors on a stable patch release, not an rc** — rc
  builds have hit already-fixed interpreter/pydantic bugs that don't reproduce
  on stable.
