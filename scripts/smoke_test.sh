#!/usr/bin/env bash
# On-demand smoke tests for docker-backed integration paths that aren't in the
# fast unit-test suite or the default CI run: the Postgres and MongoDB
# checkpointers and the AG-UI endpoint. Lets a maintainer (or agent) verify
# these still work without waiting for a full CI cycle.
#
# Usage:
#   ./scripts/smoke_test.sh                 # run all targets
#   ./scripts/smoke_test.sh mongo           # run a single target
#   ./scripts/smoke_test.sh postgres agui   # run a subset
#
# Targets: postgres, mongo, agui
#
# Only databases run in Docker; the service itself runs on the host via uv,
# pointed at localhost. That's deliberate — building the service image requires
# reaching package registries from inside the build container, which is blocked
# in some sandboxed agent environments. Running on the host sidesteps that while
# still exercising real database containers.
#
# Each database target does two independent checks so a green run actually means
# something: (1) the pytest persistence test over the HTTP API, and (2) a direct
# query of the database container confirming this run's thread was written to
# that specific backend. (2) is what catches a silent fallback to SQLite — the
# pytest test alone would pass against any working checkpointer.
#
# Requires: docker, docker compose, uv, node (for the AG-UI client)
set -euo pipefail
cd "$(dirname "$0")/.."

# Unique per run so the backend verification below reflects THIS run's data even
# if the database volume isn't empty. Exported so the pytest test uses the same
# thread id (see tests/smoke/test_persistence.py).
export SMOKE_THREAD_ID="smoke-test-$(date +%s)-$$"

SERVICE_PID=""
SERVICE_LOG=""

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
  local i
  for i in $(seq 1 30); do
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

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(postgres mongo agui)

for t in "${targets[@]}"; do
  case "$t" in
    postgres) smoke_postgres ;;
    mongo)    smoke_mongo ;;
    agui)     smoke_agui ;;
    *) echo "unknown target: $t (valid: postgres, mongo, agui)"; exit 2 ;;
  esac
done

echo "--- All smoke tests passed ---"
