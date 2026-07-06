#!/usr/bin/env bash
# Smoke test for optional dependencies (MongoDB checkpointer, AG-UI) that aren't
# exercised by the default docker compose stack or CI. Run this periodically as
# a maintainer (or agent) to catch regressions without adding weight to either.
#
# This only uses Docker for the stateful dependency (MongoDB) and runs the
# service itself directly on the host via uv. That's deliberate: rebuilding the
# service image is slow and, in some sandboxed agent environments, outright
# blocked (containerized builds can't reach package registries there). Running
# the service on the host sidesteps that while still exercising a real MongoDB.
#
# If you want the fully containerized version instead (service running in
# Docker too), see the comment in docker/compose.optional.yaml.
#
# Usage: ./scripts/smoke_test_optional.sh
#
# Requires: docker, docker compose, uv, node (for the AG-UI client)
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE="docker compose -f compose.yaml -f docker/compose.optional.yaml"
SERVICE_LOG="$(mktemp)"
SERVICE_PID=""

cleanup() {
  echo "--- Tearing down ---"
  if [[ -n "$SERVICE_PID" ]]; then
    kill "$SERVICE_PID" 2>/dev/null || true
    wait "$SERVICE_PID" 2>/dev/null || true
  fi
  $COMPOSE down -v
  rm -f "$SERVICE_LOG"
}
trap cleanup EXIT

echo "--- Starting MongoDB ---"
$COMPOSE up -d mongo

echo "--- Waiting for MongoDB to be healthy ---"
mongo_container="$($COMPOSE ps -q mongo)"
for _ in $(seq 1 20); do
  status="$(docker inspect -f '{{.State.Health.Status}}' "$mongo_container")"
  [[ "$status" == "healthy" ]] && break
  echo "waiting for mongo... ($status)"
  sleep 2
done
[[ "$status" == "healthy" ]] || { echo "mongo never became healthy"; exit 1; }

echo "--- Starting agent service on the host (DATABASE_TYPE=mongo) ---"
USE_FAKE_MODEL=true \
DATABASE_TYPE=mongo \
MONGO_HOST=localhost \
MONGO_PORT=27017 \
MONGO_DB=agent_service \
  uv run python src/run_service.py > "$SERVICE_LOG" 2>&1 &
SERVICE_PID=$!

echo "--- Waiting for agent service to be healthy ---"
timeout 60 bash -c '
  until curl -sf http://localhost:8080/health > /dev/null; do
    echo "waiting for service..."
    sleep 2
  done
' || { echo "--- service log ---"; cat "$SERVICE_LOG"; exit 1; }

echo "--- Running Mongo checkpointer persistence test ---"
uv run pytest tests/optional/test_optional_deps.py -v --run-docker

echo "--- Running AG-UI smoke test ---"
( cd scripts/agui-client && npm install --silent && AGENT_URL=http://localhost:8080 node client.mjs "Tell me a joke!" chatbot )

echo "--- All optional-dependency smoke tests passed ---"
