# Coupling constraints & gotchas (learned the hard way)

Project-specific constraints hit during real refreshes. Check the relevant
entry before fighting a resolver conflict or a CI surprise; add new entries in
the same PR that hits them. Entries note the refresh that learned them so the
provenance is one `git log` away — but this file is reusable knowledge, not a
journal: what each round bumped lives in that round's PR description.

## uv & the cooldown mechanism

- **Relative `exclude-newer` needs a recent uv.** The repo's
  `[tool.uv] exclude-newer = "7 days"` (and per-package
  `[tool.uv.exclude-newer-package]`) parse only on newer uv (the repo's pinned
  uv 0.11.x handles them; uv 0.8.x does not). An older uv prints
  `warning: Failed to parse pyproject.toml during settings discovery` and
  ignores the whole `[tool.uv]` table: `uv sync --frozen` still works, but a
  re-lock with an old uv **silently strips the cooldown metadata from
  `uv.lock`**. Always lock with the repo's pinned uv version (see README /
  `.github/workflows/test.yml`).
- **The lock records the cooldown** as `exclude-newer-span = "P7D"` (plus a
  placeholder `exclude-newer` timestamp kept for backwards compatibility).
  Adding or changing the setting forces a full re-resolve of the lockfile even
  with no dependency changes — expect a small header diff.
- **A cooldown conflict looks like an unsatisfiable pin.** If a pin's floor is
  a release younger than the cooldown, `uv lock` fails with a hint naming
  `exclude-newer` and the earliest allowed date. That's enforcement, not
  breakage: hold the bump, or (security fixes only) add a commented temporary
  `[tool.uv.exclude-newer-package]` override per the SKILL.md policy.
- **Re-locks can shed index-drifted artifacts.** A fresh resolve refetches file
  lists, so unrelated wheel entries can drop out of `uv.lock` (seen 2026-07-20:
  greenlet 3.5.3's s390x/riscv64 wheels vanished on re-lock with no version
  change, cooldown or not). Version-less churn like this is index drift —
  harmless for the platforms this repo supports.

## GitHub Actions

- **Node runtime deprecations show up as CI warnings, not lockfile conflicts.**
  GitHub periodically deprecates the Node.js runtime an Action ships with
  (16 → 20 → 24), and pinned `uses: owner/action@vN` refs don't auto-upgrade.
  Check each action's `action.yml` for its `runs.using` value (or the warning
  banner on a run) and bump to the earliest major shipping the current runtime.
  `docker/build-push-action` v4 also turned on SLSA provenance attestations by
  default — set `provenance: false` explicitly if you don't want multi-manifest
  images as a side effect. Composite actions (`runs.using: composite`, e.g.
  `codecov/codecov-action`) can hide a stale nested action (it called
  `actions/github-script@v7`, Node 20) — check their `action.yml` for their own
  `uses:` lines, not just the top-level `runs.using`.
- **`astral-sh/setup-uv` stopped publishing floating major tags as of v8.0.0**
  (deliberate supply-chain hardening, same motivation as the tj-actions
  incident). `@v7` still floats; `@v8` requires pinning the exact release,
  e.g. `astral-sh/setup-uv@v8.3.2`.

## LangChain / LangGraph ecosystem

- **`langchain` ⇄ `langgraph` move in lockstep.** The `langchain` meta-package
  pins a narrow `langgraph` range. e.g. `langchain 1.3.x` requires
  `langgraph >=1.2.5,<1.3`, while `langchain 1.2.18` requires
  `langgraph >=1.1.10,<1.2`. You cannot bump one past the other.
- **`langgraph` ⇄ `langgraph-checkpoint` base ⇄ the checkpointer packages.**
  `langgraph` **1.0–1.1.x** wants `langgraph-checkpoint <5,>=2.1` (works with
  `langgraph-checkpoint-{postgres,sqlite}` **2.x** / `-mongodb` **0.1–0.3**).
  `langgraph` **1.2.0+** requires `langgraph-checkpoint >=4.1`, which forces
  the checkpointers to **3.x** (postgres/sqlite) and **0.4.x** (mongodb).
  Landed as one coupled unit. Verified: 2.x-written SQLite and Postgres
  checkpoints read and continue correctly under the 3.x savers (round-tripped
  manually — old writer venv → new reader venv — since the test suite mocks
  `initialize_database` and never exercises real backends). **MongoDB was
  untested at the time** (no `mongod` in that sandbox; import/wiring only).
  `langgraph-checkpoint-mongodb` **0.4.0** also dropped
  `AsyncMongoDBSaver`/the `.aio` submodule in favor of a sync `MongoDBSaver`;
  `src/memory/mongodb.py` wraps it in a small async context manager
  (`_AsyncMongoDBSaver`) running connect/close via `asyncio.to_thread`.
- **`aiosqlite <0.22` was required by `langgraph-checkpoint-sqlite` 2.x**
  (no longer applies). The 2.x SQLite saver called `Connection.is_alive()`,
  removed in aiosqlite 0.22; the 3.x saver doesn't, so the pin was dropped
  with the checkpointer bump.
- **`langchain-mcp-adapters` 0.3.0 tightened its `connections` typing.**
  `MultiServerMCPClient` takes an invariant `dict[str, Connection]`, so
  `github_mcp_agent.py`'s connections dict needs the explicit annotation to
  keep `mypy src` clean (PR #312).
- **`langchain-google-genai` 4.x** dropped the gRPC transport and changed the
  `with_structured_output` default method — real behavior changes on the
  actual Gemini API path that the fake-model smoke can't exercise. Check that
  path with a real `GOOGLE_API_KEY` before relying on it in production.
- **`langchain-google-vertexai` caps `pyarrow`.** `3.2.4` (latest checked)
  requires `pyarrow>=19.0.1,<24.0.0`, so our `pyarrow` floor can't move to
  `24.x` until vertexai allows it — re-check whenever vertexai bumps.
- **`langchain-openai` → `openai` → `jiter` floor.** Bumping `langchain-openai`
  pulled a newer `openai` that required `jiter >=0.10`, so the `jiter` pin had
  to move too. Expect chains like this when bumping the LLM SDKs.
- **`langfuse` v4 SDK is a client-only change for this repo.** The repo's
  entire surface is `from langfuse.langchain import CallbackHandler` (no-arg)
  plus `Langfuse().auth_check()` in `/health` — none of the v4 tracing-API
  reshapes touch it. The ClickHouse/Redis/S3 infra requirement people
  associate with "newer Langfuse" belongs to the self-hosted *server* (v3,
  2024), not the SDK; this repo bundles no server infra. One real risk for
  self-hosters: an old server paired with a much newer SDK has broken
  `auth_check()`/tracing — keep the server current.

## Formatting, typing, and misc Python deps

- **`ruff` formatter output can change on minor bumps.** 0.15 reformatted a
  conditional lambda in `streamlit_app.py` (added parens) — expect one-time
  `ruff format` diffs on formatter-behavior changes, not just lint rules
  (PR #312).
- **`mypy` majors flood in via `uv lock --upgrade` if uncapped.** It was
  capped `<2.0` to keep the 2.x major out of a routine round, then bumped
  deliberately in a focused pass (no new errors surfaced, cap lifted). Re-add
  a cap if a disruptive major appears mid-cycle.
- **`numpy` leads Python-version drops.** `numpy 2.5` dropped Python 3.11 —
  numpy's floor is usually the first hard constraint when deciding when to
  drop an old Python minor.
- **Don't write resolver picks down as requirements.** Our own `grpcio >=1.81.1`
  floor came from pin-reconciliation (step 6) writing down whatever the
  resolver picked. `grpcio` is transitive-only (nothing in `src/` imports
  `grpc`), no dependent required that floor, and it later blocked the
  `langgraph-cli[inmem]` chain needed for Python 3.14 (`langgraph-api >=0.8`
  pins `grpcio<1.81.0`). Fixed by deleting the line — the package is still
  pulled transitively at the same version. When reconciling a `>=` floor,
  grep the package's block in `uv.lock` for its reverse-deps' `specifier`s
  first.

## Python interpreter

- **Validate new Python minors on a stable patch release, not an rc.** A
  3.14.0rc2-only sandbox hit a real but already-fixed CPython/pydantic
  interaction (`TypeError: _eval_type() got an unexpected keyword argument
  'prefer_fwd_module'`, pydantic#12544) that does not reproduce on 3.14.0
  stable. If a validation run hits this, `uv python install 3.14` (a current
  patch), don't treat it as a regression.
