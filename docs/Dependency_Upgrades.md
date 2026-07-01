# Dependency & Version Upgrade Management

A maintainer guide for keeping `agent-service-toolkit`'s dependencies current without
breaking the build. It captures the workflow plus the project-specific coupling
constraints that have bitten past upgrades, so future refresh PRs are faster and safer.

## Purpose

Do a broad dependency refresh periodically (e.g. quarterly, or when you want a specific
new feature/fix). The goal each time is to take the **safe, low-effort bumps** in one PR
and to **triage the majors** into their own follow-ups with a clear ROI rationale, rather
than upgrading everything blindly.

## Where versions live

| What | File |
|---|---|
| Runtime deps + version pins | `pyproject.toml` → `[project] dependencies` |
| Dev tooling (ruff, mypy, pytest, …) | `pyproject.toml` → `[dependency-groups] dev` |
| Minimal client/Streamlit deps (a subset, **kept in sync** with the main list) | `pyproject.toml` → `[dependency-groups] client` |
| Fully resolved versions (the source of truth for what actually installs) | `uv.lock` |
| Lint/format target Python | `pyproject.toml` → `[tool.ruff] target-version` |
| CI test matrix | `.github/workflows/test.yml` (`python-version`) |
| Container base image | `docker/Dockerfile.app`, `docker/Dockerfile.service` |
| Supported Python range + classifiers | `pyproject.toml` → `requires-python`, `classifiers` |

## Upgrade workflow (the recipe)

1. **Survey.** Compare resolved versions in `uv.lock` against the latest on PyPI. The
   PyPI JSON API is handy for scripting: `https://pypi.org/pypi/<package>/json` →
   `info.version` (latest) and `info.requires_dist` / `info.requires_python` (constraints).
2. **Triage** each package into a tier (see below).
3. **Apply the safe bumps** by editing the pins in `pyproject.toml`. Preserve the existing
   `~=` (compatible-release) style for app deps and `>=` floors for loosely-pinned libs.
4. **Re-resolve:** `uv lock --upgrade`. Read the conflict messages carefully — they tell you
   exactly which transitive constraint blocks a bump (this is how the coupling issues below
   surface).
5. **Reconcile pins to the lock (transparency).** Where the resolver picked a version higher
   than what's written in `pyproject.toml` (common for `>=` floors), raise the written
   pin to match the locked version. This is purely cosmetic — re-running `uv lock` after
   should report **no changes** — but it keeps `pyproject.toml` honest about what installs.
6. **Sync + verify:** `uv sync --frozen`, then:
   - `uv run ruff check .` and `uv run ruff format --check .`
   - `uv run mypy src`
   - `uv run pytest`
7. **Live-test the running service** end-to-end (see [Live end-to-end test](#live-end-to-end-test-no-api-key-needed)
   below). Unit tests mock the LLM and transport, so a running-service check is the only way to
   catch integration regressions from bumps to FastAPI/uvicorn/langgraph/the checkpointers.
8. Commit, push, open the PR. CI runs the matrix in `.github/workflows/test.yml` (including a
   `test-docker` job that builds the images and exercises `docker compose`).

## Live end-to-end test (no API key needed)

The service ships a **fake model** (`USE_FAKE_MODEL=true`) that satisfies the "at least one LLM
key" startup check and returns a canned reply — so you can drive the real HTTP API on the
upgraded stack without any provider credentials. Leaving `AUTH_SECRET` unset makes the
endpoints unauthenticated, so no bearer token is needed either.

Run it natively (fastest — exercises the same `run_service.py`/uvicorn entrypoint and the
full dependency stack, just not containerized):

```sh
PYTHONPATH=src USE_FAKE_MODEL=true HOST=127.0.0.1 PORT=8080 uv run python src/run_service.py &
# then, once /health is up:
curl -s localhost:8080/health                       # {"status":"ok"}  -> app + lifespan booted
curl -s localhost:8080/info                          # agents/models listed
curl -s -XPOST localhost:8080/invoke   -H 'content-type: application/json' \
     -d '{"message":"hi","agent_id":"chatbot","model":"fake"}'         # -> completion + run_id
curl -s -N -XPOST localhost:8080/stream -H 'content-type: application/json' \
     -d '{"message":"hi","agent_id":"chatbot","model":"fake","stream_tokens":true}'  # SSE tokens
# persistence: invoke twice with the same "thread_id", then POST /history for that thread
# and confirm the checkpointer returns the prior turns (validates the checkpointer packages).
```

What each check validates: `/health` + `/info` = FastAPI/uvicorn/pydantic boot and agent
wiring; `/invoke` = a full graph run (langgraph + langchain + langsmith `run_id`); `/stream` =
the SSE `StreamingResponse` path; `/history` on a reused `thread_id` = the checkpointer
(langgraph-checkpoint-sqlite/aiosqlite by default). Also start the Streamlit app
(`streamlit run src/streamlit_app.py`) to confirm the client renders.

**Containerized test.** To validate the image itself (base image + in-image `uv sync`), run
`docker compose up --build` and hit the same endpoints on the mapped ports — this mirrors the
`test-docker` CI job. Note it requires pulling the `python:3.12.3-slim` base image, so it needs
outbound Docker Hub access (not available in every sandbox; CI handles it via docker-in-docker).

## Triage principles

- **Minor / patch bumps** within a major are generally safe — batch them.
- **Major bumps** (and pre-1.0 `0.x` minor bumps, which don't guarantee SemVer stability):
  assess the **actual code/behavior change required** against how the repo *uses* the
  package, weigh **ROI**, and if it's non-trivial, **hold it for its own PR**. Grep for the
  import to see the real surface area before assuming a major is scary — often the repo only
  touches a tiny, stable part of the API.
- **Watch version-coupled packages.** Some packages move in lockstep and a bump in one drags
  others (see below). The resolver will tell you, but knowing the couplings up front saves a
  round-trip.
- **Transitive-only deps** (nothing in the repo imports them) are lower risk for *our* code —
  their risk is whatever *consumes* them (e.g. Streamlit ↔ pandas/pyarrow).

## Coupling constraints & gotchas (learned the hard way)

These are real issues hit during the June 2026 refresh — check them first next time:

- **`langchain` ⇄ `langgraph` move in lockstep.** The `langchain` meta-package pins a narrow
  `langgraph` range. e.g. `langchain 1.3.x` requires `langgraph >=1.2.5,<1.3`, while
  `langchain 1.2.18` requires `langgraph >=1.1.10,<1.2`. You cannot bump one past the other.
- **`langgraph` ⇄ `langgraph-checkpoint` base ⇄ the checkpointer packages.**
  `langgraph` **1.0–1.1.x** wants `langgraph-checkpoint <5,>=2.1` (works with
  `langgraph-checkpoint-{postgres,sqlite}` **2.x** / `-mongodb` **0.1–0.3**). `langgraph`
  **1.2.0+** requires `langgraph-checkpoint >=4.1`, which **forces** the checkpointers to
  **3.x** (postgres/sqlite) and **0.4.x** (mongodb). This was landed as one coupled unit
  (langgraph 1.1→1.2, checkpointers 2.x→3.x/0.4.x). Verified: 2.x-written SQLite and Postgres
  checkpoints read and continue correctly under the 3.x savers (round-tripped manually — old
  writer venv → new reader venv — since the test suite mocks `initialize_database` and never
  exercises real backends). **MongoDB is untested** — no `mongod` available in the sandbox
  this was developed in; only import/wiring was smoke-tested against a refused connection.
  `langgraph-checkpoint-mongodb` **0.4.0** also dropped `AsyncMongoDBSaver`/the `.aio`
  submodule entirely in favor of a sync `MongoDBSaver` (async methods bridge via a thread
  executor internally); `src/memory/mongodb.py` now wraps it in a small async context manager
  (`_AsyncMongoDBSaver`) that runs connect/close via `asyncio.to_thread`.
- **`aiosqlite <0.22` was required by `langgraph-checkpoint-sqlite` 2.x** (removed — no
  longer applies). The 2.x SQLite saver called `Connection.is_alive()`, which `aiosqlite`
  removed in 0.22. The 3.x saver doesn't call it, so the pin was dropped along with the
  checkpointer bump above.
- **`numpy 2.5` dropped Python 3.11.** The repo has since dropped 3.11 (now `>=3.12`), so the
  `numpy ~=2.4.6` pin is no longer forced by the Python floor and can be revisited in a future
  safe-bumps round.
- **`langchain-openai` → `openai` → `jiter` floor.** Bumping `langchain-openai` pulled a newer
  `openai` that required `jiter >=0.10`, so the `jiter` pin had to move too. Expect chains like
  this when bumping the LLM SDKs.
- **`mypy` is unpinned in the `dev` group.** A plain `uv lock --upgrade` will happily jump it
  to the next major (2.x) and flood you with new type errors. If you want to hold it back,
  add an explicit cap (e.g. `mypy <2.0`).
- **Our own `grpcio` floor was over-constrained, and it blocked Python 3.14.** `grpcio` is
  transitive-only (nothing in `src/` imports `grpc`; it's pulled in by chromadb,
  google-api-core, opentelemetry-exporter-otlp-proto-grpc, etc.) — checking `uv.lock`, *none*
  of those actual consumers record a version-specific requirement on it. The `>=1.81.1` floor
  in `pyproject.toml` was purely a leftover from the "reconcile pins to lock" step (§5) of a
  past refresh — i.e. whatever the resolver happened to pick, written down as if it were a
  requirement. That incidentally blocked bumping the dev-only `langgraph-cli[inmem]` chain
  past `langgraph-api 0.7.27`, since `langgraph-api >=0.8.0` pins `grpcio<1.81.0` (needed to
  get a `jsonschema-rs` version with Python 3.14 wheels — see the Python version policy
  section). **Lesson: when reconciling a `>=` floor to match the lock, sanity-check the
  written pin against what the lockfile's actual dependents require** (`grep` the package
  block in `uv.lock` for its own `dependencies`/reverse-deps and their `specifier`s) before
  writing it down as gospel — otherwise a future coupling can be blocked by a floor nothing
  actually needs. Fixed by dropping the floor entirely (bare `"grpcio"`) and letting the
  resolver pick freely; it now lands on `1.80.0`, satisfying `langgraph-api`.

## Currently deferred upgrades (backlog)

Majors intentionally held out of the safe round, each needing its own PR:

| Upgrade | From → To | Why deferred / ROI |
|---|---|---|
| **langchain-google-genai** | 3.x → 4.x | Migrates to the unified `google-genai` SDK: drops gRPC transport (REST only), changes `with_structured_output` default to `method="json_schema"`. Repo surface is light (`ChatGoogleGenerativeAI` in `core/llm.py`); mostly a Gemini-path regression test. |
| **langfuse** | 3.x → 4.x | Deliberately pinned to v3 (`~=3.10`, PR #309 / issue #250). v4 is an observation-centric rewrite (`start_observation`, decomposed trace updates, changed default OTel span export). Revisit deliberately. |
| **pandas** | 2.x → 3.0 | Transitive-only (nothing in the repo imports pandas; only Streamlit consumes it, and it allows `<4`). 3.0 is a real major (Copy-on-Write default, PyArrow-backed strings). Bump in isolation and smoke-test Streamlit's dataframe/chat rendering — or drop the explicit `pandas` dep entirely and let Streamlit pull it. |
| **mypy** | 1.x → 2.0 | Dev-only; will surface new type errors. Do it in a focused tooling PR so type-fix churn doesn't mix with a dependency refresh. |

## Python version policy

The project follows CPython's support cycle: a new minor each October, ~5 years of support
(roughly 18 months of bugfixes, then security-only). As of mid-2026, **3.14.0 is the latest
stable release** (2025-10-07).

| Version | Released | Security EOL |
|---|---|---|
| 3.10 | Oct 2021 | Oct 2026 |
| 3.11 | Oct 2022 | Oct 2027 |
| 3.12 | Oct 2023 | Oct 2028 |
| 3.13 | Oct 2024 | Oct 2029 |
| 3.14 | Oct 2025 | Oct 2030 |

Current declarations: `requires-python = ">=3.12,<3.15"`, classifiers + CI matrix cover
3.12/3.13/3.14, ruff targets `py312`, and the Docker images use `python:3.12.3-slim` (still
fine — `<3.15` is just the upper bound, the base image doesn't need to move to 3.14). Python
3.11 support was dropped earlier; 3.14 support landed after clearing two sequential
`langgraph-cli[inmem]`-chain blockers (see the coupling-constraints section): first the
`langgraph-checkpoint` floor capping `jsonschema-rs`, then an over-tight `grpcio` floor of our
own capping `langgraph-api`. Once both cleared, `uv sync` on a real Python 3.14.6 interpreter
worked cleanly and the full suite (ruff/mypy/pytest/live e2e) passed.

**One thing to watch, not a repo issue:** validating locally against a **3.14.0 pre-release
build** (`rc2`, the only one cached in the dev sandbox this was built in) hit a real but
already-fixed CPython/pydantic interaction
(`TypeError: _eval_type() got an unexpected keyword argument 'prefer_fwd_module'`,
[pydantic#12544](https://github.com/pydantic/pydantic/issues/12544)) that does **not**
reproduce on the final 3.14.0+ stable release. If a local validation run hits this on Python
3.14, install a current patch release (`uv python install 3.14`, not an `rc`) rather than
treating it as a regression.
