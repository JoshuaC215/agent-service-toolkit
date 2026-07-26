---
name: smoke-test
description: >-
  Run and interpret scripts/smoke_test.sh, the on-demand checks for optional
  dependencies that CI and the default docker compose don't cover: the Postgres
  and MongoDB checkpointers, the AG-UI endpoint, and LangFuse tracing. Use when
  you've touched the memory/checkpointer layer, service startup/health, the
  AG-UI adapter, LangFuse tracing, or bumped one of those dependencies — and want
  to confirm the real integration still works without waiting for a full CI run.
  Also use when deciding WHETHER such a check is even warranted for a change.
---

# Optional-dependency smoke tests

`scripts/smoke_test.sh` exercises the parts of this project that aren't in the fast
unit suite or the default CI path because they need real infrastructure: the
**Postgres** and **MongoDB** checkpointers, the **AG-UI** endpoint, and **LangFuse**
tracing. It spins up the real dependency in Docker, runs the service against it, and
verifies the integration end-to-end, then tears everything down.

"Smoke test" here means a quick *is-this-obviously-broken* confidence check, not
exhaustive integration coverage. It's deliberately cheap to run one target and skip the
rest.

## First: should you run this at all?

Most changes do **not** need it. It pulls images, starts containers, and takes minutes —
that time and token cost isn't worth it for work that doesn't touch these paths. Skip it
for: adding or editing an agent graph, prompts, or tools; docs; client/Streamlit-only
changes; anything already covered by `uv run pytest` and normal CI.

**Run the relevant target when your change touches its path:**

| You changed… | Run |
| --- | --- |
| `src/memory/*`, `initialize_database`, a saver/store class, checkpointer wiring | `postgres` and/or `mongo` |
| `langgraph-checkpoint-postgres` / `-mongodb`, `pymongo`, `psycopg` versions | `postgres` / `mongo` |
| `src/service/agui.py`, the AG-UI adapter, `ag-ui-*` deps | `agui` |
| LangFuse tracing (the `CallbackHandler` wiring, `/health` auth check), `langfuse` dep | `langfuse` |
| Service startup / lifespan / `/health` broadly | the most relevant one or two |
| Nothing specific — periodic pre-release confidence pass | `all` |

Prefer the **narrowest** target that covers your change. Run `all` (which includes the
heavy LangFuse stack) only for a deliberate full pass — not as a reflex.

## Running it

```sh
./scripts/smoke_test.sh                 # default: postgres, mongo, agui
./scripts/smoke_test.sh mongo           # a single target
./scripts/smoke_test.sh postgres agui   # a subset
./scripts/smoke_test.sh langfuse        # heavy: ~5GB stack, run on its own
./scripts/smoke_test.sh all             # everything, incl. langfuse
```

`langfuse` is excluded from the default run because it starts LangFuse's full 6-service
self-host stack (~5GB of images). A green run ends with `--- All smoke tests passed ---`;
exit code is non-zero on any failure or an unknown target.

## Interpreting the result — and why a green ≠ "it works" trap is real

The whole design point is that **a passing API test does not prove the intended
dependency was used.** The persistence test just checks that history survives across two
invokes, which is true for *any* working checkpointer — including SQLite. So if
`DATABASE_TYPE=mongo` silently didn't take effect (dropped env, a fallback), the pytest
step would still pass while nothing touched Mongo.

That's why each target does a **second, independent check** that the specific dependency
was really exercised, and that's the line you should actually trust:

- `✓ verified: N postgres checkpoint rows for this run's thread`
- `✓ verified: N mongo checkpoint documents for this run's thread`
- `✓ verified: N LangFuse traces recorded for this run`
- `✓ verified: AG-UI streamed a complete run with the expected response`

Each uses a **unique per-run thread id** (`SMOKE_THREAD_ID`), so the count reflects *this*
run's data even against a non-empty volume. A `✗ FAIL: … was NOT exercised as intended`
means the dependency wasn't actually hit — treat that as a real failure, not flakiness.

Other guards worth knowing when you read a failure:
- **Port already in use:** `start_service` refuses to run if something is already on
  `:8080`, so the health check can't accidentally pass against an unrelated process.
- **Service died on startup:** it fails fast and dumps the service log instead of waiting
  out the full timeout.
- **LangFuse ingestion is async:** traces flow through a worker → ClickHouse pipeline, so
  the script polls the traces API for a bit (usually a few seconds) before deciding.

## Cloud-environment quirks (Claude Code on the web / sandboxes)

This script is meant to be runnable by an agent in a sandboxed cloud environment. A few
things bite there specifically:

- **Docker daemon may not be running** (and can stop between steps). If `docker` errors
  with "Cannot connect to the daemon," start it: `(dockerd > /tmp/dockerd.log 2>&1 &)`
  then wait a few seconds.
- **Don't build the service image in-sandbox.** Container builds can't reach package
  registries through the egress proxy (TLS/CA), so `docker compose build` fails. This is
  exactly why the script runs the service on the **host** via `uv` and only puts the
  *dependencies* in Docker. Don't "fix" it by baking the proxy CA into the committed
  Dockerfile — that's sandbox-specific.
- **Registry allowlist for LangFuse.** The LangFuse stack's `minio` image comes from
  `cgr.dev` (Chainguard). In a restricted-egress environment add **`cgr.dev`** to the
  network allowlist, or the pull 403s. (If a pull still 403s afterward on a *different*
  host, add whatever host it names — some registries serve blobs from a separate CDN.)
- **`pkill -f run_service.py` will kill your own shell.** `-f` matches full command lines,
  and your command text contains that string, so it self-matches. To free port 8080 use
  `fuser -k 8080/tcp` instead.
- **Long foreground `sleep` loops can be killed** by the sandbox. When running the script
  (or waiting on it), launch it in the background (`nohup … &`) and poll its log file
  rather than blocking in the foreground.
- **`docker compose exec` needs `.env`; `docker exec` doesn't.** The host-based flow
  deliberately omits `.env`, and `docker compose exec` re-parses the full config (which
  references `env_file: .env`) and fails without it. The script queries containers via
  `docker exec <id>` (id from `docker compose ps -q`) to sidestep this.

## Extending it

The script is built around one principle: **every target proves the real dependency was
used, not just that the API returned something.** Keep that when adding a target.

To add a new optional-dependency target:
1. Add a `smoke_<name>()` function following the existing shape: start the dependency
   (Docker), `start_service` on the host with the right env, run an API-level check, then
   a **dependency-identity check** (query the dependency directly for this run's data via
   `assert_positive_count`, or assert on captured output like the AG-UI target).
2. Register it in the `case` in the target loop and, if it should be part of a full pass,
   in the `all` expansion. Decide whether it's light enough for the default set (DB-style)
   or heavy enough to be opt-in (LangFuse-style).
3. If it needs an add-on compose file, follow the `docker/compose.<dep>.yaml` pattern; if
   it's a large external stack, prefer fetching upstream's compose pinned to a tag (as the
   `langfuse` target does via `LANGFUSE_REF`) rather than vendoring it.

Keep these out of the CI docker job: they live in `tests/smoke/` (not `tests/integration/`,
which CI's `test-docker` job scopes to) and are marked `@pytest.mark.docker` so they're
skipped unless `--run-docker` is passed.
