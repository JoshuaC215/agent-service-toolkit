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
# Requires: docker, docker compose, uv, node (for the AG-UI client)
set -euo pipefail
cd "$(dirname "$0")/.."

SERVICE_PID=""
SERVICE_LOG=""

start_service() {
  # Start the agent service on the host with the given backend env, then wait
  # until it reports healthy. Args: KEY=VALUE ... connection settings.
  SERVICE_LOG="$(mktemp)"
  env USE_FAKE_MODEL=true "$@" uv run python src/run_service.py > "$SERVICE_LOG" 2>&1 &
  SERVICE_PID=$!
  if ! timeout 60 bash -c 'until curl -sf http://localhost:8080/health >/dev/null; do sleep 2; done'; then
    echo "  service failed to become healthy; log:"
    cat "$SERVICE_LOG"
    return 1
  fi
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
  echo "  database never became healthy"
  return 1
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
  echo "=== Postgres checkpointer ==="
  docker compose -f compose.yaml up -d postgres
  wait_healthy "$(docker compose -f compose.yaml ps -q postgres)"
  start_service DATABASE_TYPE=postgres POSTGRES_HOST=localhost POSTGRES_PORT=5432 \
    POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres POSTGRES_DB=agent_service
  uv run pytest tests/smoke/test_persistence.py -v --run-docker
  stop_service
  docker compose -f compose.yaml down -v
}

smoke_mongo() {
  echo "=== MongoDB checkpointer ==="
  local files=(-f compose.yaml -f docker/compose.mongo.yaml)
  docker compose "${files[@]}" up -d mongo
  wait_healthy "$(docker compose "${files[@]}" ps -q mongo)"
  start_service DATABASE_TYPE=mongo MONGO_HOST=localhost MONGO_PORT=27017 MONGO_DB=agent_service
  uv run pytest tests/smoke/test_persistence.py -v --run-docker
  stop_service
  docker compose "${files[@]}" down -v
}

smoke_agui() {
  echo "=== AG-UI endpoint ==="
  # AG-UI is backend-agnostic, so the default SQLite checkpointer is fine here
  # and no database container is needed.
  start_service
  ( cd scripts/agui-client && npm install --silent && \
    AGENT_URL=http://localhost:8080 node client.mjs "Tell me a joke!" chatbot )
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
