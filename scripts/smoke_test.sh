#!/usr/bin/env bash
# On-demand smoke tests for docker-backed integration paths that aren't in the
# fast unit-test suite or the default CI run: the Postgres and MongoDB
# checkpointers, the AG-UI endpoint, and LangFuse tracing. Lets a maintainer (or
# agent) verify these still work without waiting for a full CI cycle.
#
# Usage:
#   ./scripts/smoke_test.sh                 # default targets: postgres, mongo, agui
#   ./scripts/smoke_test.sh mongo           # run a single target
#   ./scripts/smoke_test.sh postgres agui   # run a subset
#   ./scripts/smoke_test.sh langfuse        # run the heavy langfuse target on its own
#   ./scripts/smoke_test.sh all             # everything, including langfuse
#
# Targets: postgres, mongo, agui, langfuse
#   langfuse is excluded from the default run: it spins up LangFuse's full
#   self-host stack (6 services, ~5GB of images) and takes noticeably longer, so
#   run it explicitly or via `all`. It also needs the cgr.dev container registry
#   reachable (for the minio image) — in a restricted-egress cloud environment,
#   add cgr.dev to the network allowlist first.
#
# Only databases run in Docker; the service itself runs on the host via uv,
# pointed at localhost. That's deliberate — building the service image requires
# reaching package registries from inside the build container, which is blocked
# in some sandboxed agent environments. Running on the host sidesteps that while
# still exercising real database containers.
#
# A green run is meant to actually mean something. Beyond the pytest/API check,
# each target independently verifies the intended dependency was really used:
# the DB targets query the container for this run's thread, and langfuse queries
# its API for the trace. This is what catches a silent fallback (e.g. to SQLite)
# that would otherwise pass the API-level test against any working checkpointer.
#
# Requires: docker, docker compose, uv, node (AG-UI client), python3, curl
set -euo pipefail
cd "$(dirname "$0")/.."

# Unique per run so the backend verification below reflects THIS run's data even
# if the database volume isn't empty. Exported so the pytest test uses the same
# thread id (see tests/smoke/test_persistence.py).
SMOKE_THREAD_ID="smoke-test-$(date +%s)-$$"
export SMOKE_THREAD_ID

# LangFuse's self-host compose is fetched from upstream at this pinned tag rather
# than vendored into the repo. Bump this to move to a newer LangFuse.
LANGFUSE_REF="v3.205.1"

SERVICE_PID=""
SERVICE_LOG=""
LANGFUSE_COMPOSE=""  # temp compose file, set while the langfuse target runs

start_service() {
  # Start the agent service on the host with the given backend env, then wait
  # until it reports healthy. Args: KEY=VALUE ... connection settings.
  if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
    echo "  ✗ refusing to start: something is already listening on :8080"
    return 1
  fi
  SERVICE_LOG="$(mktemp)"
  env USE_FAKE_MODEL=true "$@" uv run python src/run_service.py > "$SERVICE_LOG" 2>&1 &
  SERVICE_PID=$!
  for _ in $(seq 1 30); do
    if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
      echo "  ✗ service exited during startup; log:"
      cat "$SERVICE_LOG"
      return 1
    fi
    sleep 2
  done
  echo "  ✗ service did not become healthy within 60s; log:"
  cat "$SERVICE_LOG"
  return 1
}

stop_service() {
  if [[ -n "$SERVICE_PID" ]]; then
    kill "$SERVICE_PID" 2>/dev/null || true
    wait "$SERVICE_PID" 2>/dev/null || true
    SERVICE_PID=""
  fi
  if [[ -n "$SERVICE_LOG" ]]; then
    rm -f "$SERVICE_LOG"
    SERVICE_LOG=""
  fi
}

wait_healthy() {
  # Wait until a container reports healthy. Args: container id.
  local cid="$1" status=""
  for _ in $(seq 1 20); do
    status="$(docker inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo missing)"
    [[ "$status" == "healthy" ]] && return 0
    echo "  waiting for database... ($status)"
    sleep 2
  done
  echo "  ✗ database never became healthy"
  return 1
}

assert_positive_count() {
  # Args: count-string, label. Fails loudly unless count is an integer > 0.
  local n="$1" label="$2"
  if [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 )); then
    echo "  ✓ verified: $n $label"
  else
    echo "  ✗ FAIL: expected >0 $label, got '$n' — backend was NOT exercised as intended"
    return 1
  fi
}

cleanup() {
  echo "--- Tearing down ---"
  stop_service
  # down removes every service in the merged project (postgres + mongo), so this
  # one call cleans up regardless of which target was running.
  docker compose -f compose.yaml -f docker/compose.mongo.yaml down -v >/dev/null 2>&1 || true
  if [[ -n "$LANGFUSE_COMPOSE" && -f "$LANGFUSE_COMPOSE" ]]; then
    docker compose -f "$LANGFUSE_COMPOSE" down -v >/dev/null 2>&1 || true
    rm -f "$LANGFUSE_COMPOSE"
  fi
}
trap cleanup EXIT

smoke_postgres() {
  echo "=== Postgres checkpointer (DATABASE_TYPE=postgres) ==="
  docker compose -f compose.yaml up -d postgres
  local cid
  cid="$(docker compose -f compose.yaml ps -q postgres)"
  wait_healthy "$cid"
  start_service DATABASE_TYPE=postgres POSTGRES_HOST=localhost POSTGRES_PORT=5432 \
    POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres POSTGRES_DB=agent_service
  uv run pytest tests/smoke/test_persistence.py -v --run-docker
  local n
  n="$(docker exec -e PGPASSWORD=postgres "$cid" psql -U postgres -d agent_service -tAc \
    "select count(*) from checkpoints where thread_id='$SMOKE_THREAD_ID'" 2>/dev/null | tr -d '[:space:]')" || true
  assert_positive_count "$n" "postgres checkpoint rows for this run's thread"
  stop_service
  docker compose -f compose.yaml down -v
}

smoke_mongo() {
  echo "=== MongoDB checkpointer (DATABASE_TYPE=mongo) ==="
  local files=(-f compose.yaml -f docker/compose.mongo.yaml)
  docker compose "${files[@]}" up -d mongo
  local cid
  cid="$(docker compose "${files[@]}" ps -q mongo)"
  wait_healthy "$cid"
  start_service DATABASE_TYPE=mongo MONGO_HOST=localhost MONGO_PORT=27017 MONGO_DB=agent_service
  uv run pytest tests/smoke/test_persistence.py -v --run-docker
  local n
  n="$(docker exec "$cid" mongosh agent_service --quiet --eval \
    "db.checkpoints.countDocuments({thread_id:'$SMOKE_THREAD_ID'})" 2>/dev/null | tr -d '[:space:]')" || true
  assert_positive_count "$n" "mongo checkpoint documents for this run's thread"
  stop_service
  docker compose "${files[@]}" down -v
}

smoke_agui() {
  echo "=== AG-UI endpoint ==="
  # AG-UI is backend-agnostic, so the default SQLite checkpointer is fine here
  # and no database container is needed.
  start_service
  local out
  out="$(cd scripts/agui-client && npm install --silent && \
    AGENT_URL=http://localhost:8080 node client.mjs "Tell me a joke!" chatbot)" || true
  echo "$out"
  # A green exit isn't enough: confirm the stream actually completed and returned
  # the fake model's response, not an empty or partial run.
  if ! grep -q "RUN_FINISHED" <<<"$out"; then
    echo "  ✗ FAIL: AG-UI stream did not reach RUN_FINISHED"
    return 1
  fi
  if ! grep -q "This is a test response from the fake model." <<<"$out"; then
    echo "  ✗ FAIL: AG-UI did not return the expected assistant response"
    return 1
  fi
  echo "  ✓ verified: AG-UI streamed a complete run with the expected response"
  stop_service
}

smoke_langfuse() {
  echo "=== LangFuse tracing (self-hosted) ==="
  local pk="pk-lf-smoke-public" sk="sk-lf-smoke-secret"

  # Fetch LangFuse's official self-host compose (pinned) rather than vendoring it.
  # Bare mktemp (no --suffix) for macOS/BSD portability; `docker compose -f`
  # doesn't care about the file extension.
  LANGFUSE_COMPOSE="$(mktemp)"
  echo "  fetching LangFuse compose @ $LANGFUSE_REF"
  if ! curl -sSL "https://raw.githubusercontent.com/langfuse/langfuse/$LANGFUSE_REF/docker-compose.yml" \
    -o "$LANGFUSE_COMPOSE"; then
    echo "  ✗ FAIL: could not fetch LangFuse compose"
    return 1
  fi

  # LANGFUSE_INIT_* seeds an org/project/user and known API keys on first boot, so
  # no manual signup is needed and the keys below are deterministic.
  echo "  starting LangFuse stack (this pulls ~5GB the first time)..."
  LANGFUSE_INIT_ORG_ID=smoke-org LANGFUSE_INIT_ORG_NAME=smoke \
  LANGFUSE_INIT_PROJECT_ID=smoke-project LANGFUSE_INIT_PROJECT_NAME=smoke \
  LANGFUSE_INIT_PROJECT_PUBLIC_KEY="$pk" LANGFUSE_INIT_PROJECT_SECRET_KEY="$sk" \
  LANGFUSE_INIT_USER_EMAIL=smoke@example.com LANGFUSE_INIT_USER_NAME=smoke \
  LANGFUSE_INIT_USER_PASSWORD=smokepassword123 \
    docker compose -f "$LANGFUSE_COMPOSE" up -d

  echo "  waiting for langfuse-web..."
  for _ in $(seq 1 40); do
    curl -sf http://localhost:3000/api/public/health >/dev/null 2>&1 && break
    sleep 3
  done
  if ! curl -sf http://localhost:3000/api/public/health >/dev/null 2>&1; then
    echo "  ✗ FAIL: langfuse-web did not become ready"
    return 1
  fi

  start_service LANGFUSE_TRACING=true LANGFUSE_HOST=http://localhost:3000 \
    LANGFUSE_PUBLIC_KEY="$pk" LANGFUSE_SECRET_KEY="$sk"

  # (1) service-level: /health runs langfuse.auth_check() against the instance.
  if curl -s http://localhost:8080/health | grep -q '"langfuse":"connected"'; then
    echo "  ✓ /health reports langfuse connected"
  else
    echo "  ✗ FAIL: /health did not report langfuse connected"
    return 1
  fi

  # (2) decisive: a traced invoke must actually produce a trace in LangFuse.
  uv run python -c "
import sys; sys.path.insert(0, 'src')
from client import AgentClient
c = AgentClient('http://localhost:8080')
r = c.invoke('Trace me please', thread_id='$SMOKE_THREAD_ID', model='fake')
assert r.type == 'ai', r
print('  traced invoke ok')
"
  echo "  waiting for the trace to land in LangFuse (ingestion is async)..."
  local n=0
  for _ in $(seq 1 20); do
    n="$(curl -s -u "$pk:$sk" "http://localhost:3000/api/public/traces?limit=5" \
      | python3 -c 'import sys, json; print(len(json.load(sys.stdin).get("data", [])))' 2>/dev/null || echo 0)"
    [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 )) && break
    sleep 3
  done
  assert_positive_count "$n" "LangFuse traces recorded for this run"

  stop_service
  docker compose -f "$LANGFUSE_COMPOSE" down -v
  rm -f "$LANGFUSE_COMPOSE"
  LANGFUSE_COMPOSE=""
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(postgres mongo agui)

# Expand "all" to every target, including the heavy langfuse one.
expanded=()
for t in "${targets[@]}"; do
  if [[ "$t" == "all" ]]; then
    expanded+=(postgres mongo agui langfuse)
  else
    expanded+=("$t")
  fi
done
targets=("${expanded[@]}")

for t in "${targets[@]}"; do
  case "$t" in
    postgres) smoke_postgres ;;
    mongo)    smoke_mongo ;;
    agui)     smoke_agui ;;
    langfuse) smoke_langfuse ;;
    *) echo "unknown target: $t (valid: postgres, mongo, agui, langfuse, all)"; exit 2 ;;
  esac
done

echo "--- All smoke tests passed ---"
