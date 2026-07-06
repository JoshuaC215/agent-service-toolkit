#!/usr/bin/env bash
# Smoke test for optional dependencies (MongoDB checkpointer, AG-UI) that aren't
# exercised by the default docker compose stack or CI. Run this periodically as
# a maintainer to catch regressions without adding weight to either.
#
# Usage: ./scripts/smoke_test_optional.sh
#
# Requires: docker, docker compose, uv, node (for the AG-UI client)
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE="docker compose -f compose.yaml -f docker/compose.optional.yaml"

cleanup() {
  echo "--- Tearing down containers ---"
  $COMPOSE down -v
}
trap cleanup EXIT

export USE_FAKE_MODEL=true

echo "--- Building and starting stack (postgres, mongo, agent_service) ---"
$COMPOSE up -d --build agent_service mongo postgres

echo "--- Waiting for agent_service to be healthy ---"
timeout 60 bash -c '
  until curl -sf http://localhost:8080/health > /dev/null; do
    echo "waiting for service..."
    sleep 2
  done
'

echo "--- Running Mongo checkpointer persistence test ---"
uv run pytest tests/integration/test_optional_deps.py -v --run-docker

echo "--- Running AG-UI smoke test ---"
( cd scripts/agui-client && npm install --silent && node client.mjs "Tell me a joke!" chatbot )

echo "--- All optional-dependency smoke tests passed ---"
