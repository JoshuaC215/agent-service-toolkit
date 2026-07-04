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
| GitHub Actions versions (`actions/checkout`, `setup-python`, `setup-uv`, `docker/*`, `codecov-action`, …) | `.github/workflows/*.yml` (`uses:`) |
| `uv` CLI version — CI, quickstart docs, and Docker images (**keep all three in sync**) | `.github/workflows/test.yml` (`astral-sh/setup-uv` → `version:`), `README.md` install snippet, `docker/Dockerfile.app`/`docker/Dockerfile.service` (`pip install uv==`) |

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

These are real issues hit during the June/July 2026 refresh — check them first next time:

- **GitHub Actions Node runtime deprecations show up as CI warnings, not lockfile conflicts.**
  GitHub periodically deprecates the Node.js runtime an Action ships with (16 → 20 → 24), and
  pinned `uses: owner/action@vN` refs don't auto-upgrade. Check each action's `action.yml` for its
  `runs.using` value (or just check for the warning banner on a run) and bump to the earliest
  major that ships the current runtime. `docker/build-push-action` v4 also turned on SLSA
  provenance attestations by default — set `provenance: false` explicitly if you don't want
  multi-manifest images as a side effect of an otherwise-unrelated runtime bump. Composite actions
  (`runs.using: composite`, e.g. `codecov/codecov-action`) can still be hiding a stale nested
  action (it called `actions/github-script@v7`, Node 20) — check their `action.yml` for their own
  `uses:` lines, not just the top-level `runs.using`.
- **`astral-sh/setup-uv` stopped publishing floating major tags as of v8.0.0** (a deliberate
  supply-chain-hardening move, same motivation as the tj-actions incident). `@v7` still floats;
  `@v8` requires pinning the exact release, e.g. `astral-sh/setup-uv@v8.2.0`.
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
- **`langchain-mcp-adapters` 0.3.0 tightened its `connections` typing.** `MultiServerMCPClient`
  now takes an invariant `dict[str, Connection]` instead of a looser mapping type, so
  `github_mcp_agent.py`'s connections dict needed an explicit `dict[str, Connection]`
  annotation to keep `mypy src` clean (landed in PR #312).
- **`ruff` 0.15's formatter reformats conditional lambdas.** Bumping past 0.15 reformatted a
  conditional lambda in `streamlit_app.py` (added parens) — expect a one-time `ruff format`
  diff like this on any ruff minor bump that changes formatter behavior, not just lint rules
  (landed in PR #312).
- **`langchain-google-vertexai` caps `pyarrow`.** `langchain-google-vertexai==3.2.4` (the latest
  release) depends on `pyarrow>=19.0.1,<24.0.0`, so our `pyarrow >=23.0.1` floor can't move to
  `24.x` until `langchain-google-vertexai` releases a version that allows it — checked during
  the July 2026 safe-bumps round, re-check next time `langchain-google-vertexai` bumps.
- **`numpy 2.5` dropped Python 3.11.** The repo has since dropped 3.11 (now `>=3.12`), so the
  `numpy ~=2.4.6` pin was no longer forced by the Python floor — bumped to `~=2.5.0` in the July
  2026 safe-bumps round.
- **`langchain-openai` → `openai` → `jiter` floor.** Bumping `langchain-openai` pulled a newer
  `openai` that required `jiter >=0.10`, so the `jiter` pin had to move too. Expect chains like
  this when bumping the LLM SDKs.
- **`mypy` was capped at `<2.0` in the `dev` group** to keep the 2.x major out of a routine
  safe-bumps round (a plain `uv lock --upgrade` would otherwise jump it and flood you with new
  type errors). Bumped deliberately to `~=2.1.0` in a focused pass — no new errors surfaced
  against this repo's code, so the cap was lifted rather than re-added.
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
  actually needs. Fixed by removing `grpcio` from `[project] dependencies` entirely —
  confirmed (by diffing `uv.lock` with/without the line) that nothing about what installs
  changes; it's still pulled in transitively at the same version (`1.80.0`) via
  `google-api-core[grpc]` / `googleapis-common-protos[grpc]` / `langgraph-api`. Since it's
  transitive-only, there's nothing to gain from listing it explicitly, and doing so is what
  created the artificial floor in the first place.

## Currently deferred upgrades (backlog)

Majors intentionally held out of the safe round, each needing its own PR:

*None currently.* Add a table row here (`| Upgrade | From → To | Why deferred / ROI |`) as new
majors get triaged and held out of a safe-bumps round.

**Landed since the table above was written:**
- **langfuse** 3.15.0 → 4.12.0: previously deliberately pinned to v3 (`~=3.10`, PR #309 / issue #250)
  pending a validated look at v4. Investigated and landed: this repo's entire Langfuse surface is
  `from langfuse.langchain import CallbackHandler` (no-arg `CallbackHandler()`) plus
  `Langfuse().auth_check()` in the `/health` check — it never touches the low-level tracing API
  (`start_span`/`start_generation` → `start_observation`, `update_current_trace()` →
  `propagate_attributes()`, the new default OTel span-filtering) that actually changed shape in
  v4. Confirmed both call sites are unchanged in v4 (same constructor signatures), so this was a
  version-bump, not a rewrite: `uv lock --upgrade-package langfuse` picked v4.12.0 with **zero**
  new/changed transitive deps (a 2-line lockfile diff). `uv run mypy src`, the full test suite,
  and a live fake-model e2e all pass, including a check that `/invoke` and `/stream` complete
  normally with `LANGFUSE_TRACING=true` and the `CallbackHandler` attached even when no real
  Langfuse server is reachable (`auth_check()` correctly surfaces a 403 into the existing
  try/except in `/health` rather than crashing anything).
  **On the self-hosting/infra question that motivated deferring this**: the ClickHouse + Redis +
  S3 requirement people associate with "later Langfuse versions" belongs to the **Langfuse
  platform** (the self-hosted server) going to v3 back in 2024 — it is *not* new to the Python
  SDK v4 bump, and a maintainer has confirmed SDK v4 made "no changes to underlying
  infrastructure" for self-hosters. This repo doesn't bundle any Langfuse server infra itself
  (nothing in `compose.yaml`) — it's purely a client pointed at `LANGFUSE_HOST`/keys, so this was
  never actually a repo concern, just a byproduct of how self-hosted Langfuse evolved
  independently of this SDK version. One real, SDK-version-agnostic risk to flag for
  self-hosters: there's a documented case of `auth_check()`/tracing breaking when an old
  self-hosted server version is paired with a much newer SDK — self-hosters should keep their
  server reasonably current, independent of this bump. Not validated against a live self-hosted
  or Langfuse Cloud server with real credentials in this pass (sandbox has no Docker daemon to
  spin up the self-host stack) — do that check with real credentials before relying on it in
  production, same caveat pattern as the Gemini bump below.
- **mypy** 1.x → 2.1.0 (dropped the `<2.0` cap): no new type errors surfaced against this repo's code.
- **langchain-google-genai** 3.0.3 → 4.2.6: repo surface is light (`ChatGoogleGenerativeAI` in `core/llm.py`, plus generic `with_structured_output` in `agents/interrupt_agent.py`); `uv run mypy src`, the full test suite, and a live fake-model e2e pass (`test_llm.py` covers `GoogleModelName` construction). The gRPC-transport drop and `with_structured_output` default-method change are real behavior changes on the actual Gemini API path — not exercised by the fake-model smoke test — so give that path a manual check with a real `GOOGLE_API_KEY` before relying on it in production.
- **Docker base images** bumped `python:3.12.3-slim` → `python:3.13.14-slim` in `docker/Dockerfile.app` and `docker/Dockerfile.service` (already within the `>=3.12,<3.15` floor and CI's 3.12/3.13/3.14 matrix).
- **pandas** 2.2.3 → 3.0.3 (transitive-only; nothing in `src/` imports pandas, only Streamlit consumes it): `uv lock --upgrade-package pandas` dropped the now-unneeded `pytz`/`tzdata` transitive deps. `uv run mypy src` and the full test suite pass; live-tested the Streamlit app end-to-end against the fake-model service (chat send/receive, streaming, feedback widget all render correctly under 3.0's Copy-on-Write default).
- **July 2026 safe-bumps round** (`uv lock --upgrade` plus a few tilde-pin bumps, all within existing majors or transitive-only): `numpy` 2.4.6→2.5.0, `pyowm` 3.3.0→3.5.0, `python-dotenv` 1.0.1→1.2.2 (both the main and `client` dependency groups), `aiosqlite` 0.21.0→0.22.1 (floor-only pin, no `pyproject.toml` change needed), and dev-only `pytest-env` 1.2.0→1.6.0. Plus transitive-only movement `uv lock --upgrade` picked up on its own: `cryptography` 44.0.3→49.0.0, `wrapt` 1.17.3→2.2.2, `sse-starlette` 2.1.3→3.3.4, `pymongo` 4.12.1→4.16.0, `anthropic` 0.113.0→0.115.0, `boto3`/`botocore`, `google-cloud-aiplatform`, `narwhals`, `packaging`, `pillow`, `pyopenssl` (all patch/minor). None of these are imported directly in `src/` except `numpy`/`pyowm`/`python-dotenv`, all with unchanged call sites. `uv run ruff check`/`format --check`, `mypy src`, and the full test suite (126 passed, 2 skipped) all clean; live-tested the FastAPI service (`/health`, `/invoke`, `/history` checkpointer round-trip) and the Streamlit app end-to-end (sent a message through a real browser against the fake-model service — response, streaming, and the feedback widget all rendered correctly).
- **GitHub Actions + `uv` CLI runtime refresh** (PR #318, landed the same day as the round above): see the coupling-constraints entries above for the Node-runtime-deprecation and `setup-uv` immutable-tag details.

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
3.12/3.13/3.14, ruff targets `py312`, and the Docker images use `python:3.13.14-slim` (bumped
from `3.12.3-slim`; `<3.15` is just the upper bound, so the base image doesn't need to move to
3.14). Python 3.11 support was dropped earlier; 3.14 support landed after clearing two sequential
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
