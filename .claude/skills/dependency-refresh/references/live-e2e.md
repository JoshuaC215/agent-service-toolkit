# Live end-to-end verification (no API key needed)

Unit tests mock the LLM and transport, so a running-service check is the only
way to catch integration regressions from bumps to FastAPI/uvicorn/langgraph/
the checkpointers, and a real browser pass is the only thing that catches
Streamlit-stack UI breakage.

## Fake-model HTTP ladder

The service ships a fake model (`USE_FAKE_MODEL=true`) that satisfies the "at
least one LLM key" startup check and returns a canned reply — so you can drive
the real HTTP API on the upgraded stack without provider credentials. Leaving
`AUTH_SECRET` unset keeps the endpoints unauthenticated (no bearer token).

Run it natively (fastest — exercises the same `run_service.py`/uvicorn
entrypoint and the full dependency stack, just not containerized):

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

What each check validates: `/health` + `/info` = FastAPI/uvicorn/pydantic boot
and agent wiring; `/invoke` = a full graph run (langgraph + langchain +
langsmith `run_id`); `/stream` = the SSE `StreamingResponse` path; `/history`
on a reused `thread_id` = the checkpointer
(langgraph-checkpoint-sqlite/aiosqlite by default).

## Streamlit UI e2e (browser)

CI never drives the actual Streamlit interface — pytest mocks the transport and
the docker CI job only checks health endpoints — so a real browser pass is the
only thing that catches Streamlit/pandas/pyarrow-level UI breakage from a bump.
With the fake-model service from above still running:

```sh
uv run streamlit run src/streamlit_app.py --server.headless true --server.port 8501 &
# quick single-message smoke:
uv run --with playwright python scripts/smoke_live_app.py http://localhost:8501
# wider coverage (multi-turn resume, settings selectors, feedback widget, streaming toggle, ...):
uv run --with playwright python scripts/e2e_ui_tests.py http://localhost:8501
```

`smoke_live_app.py` sends one chat message and verifies a streamed response
renders and settles. `e2e_ui_tests.py` is a small suite of key user journeys
built on the same idea — run `--list` to see them, or pass names to run a
subset. Both exit 0 on pass and save a diagnostic screenshot on failure, and
both are URL-parameterized so the same commands also run against the deployed
app (omit the URL, or set `LIVE_APP_URL`). Run at least the smoke before
opening any PR that bumps `streamlit`, its rendering stack (pandas, pyarrow,
pillow, altair), or the client/schema code the UI consumes; run the fuller
suite when a bump could plausibly touch chat history, settings, or the
feedback/streaming paths.

## Containerized test

To validate the image itself (base image + in-image `uv sync`), run
`docker compose up --build` and hit the same endpoints on the mapped ports —
this mirrors the `test-docker` CI job. It requires pulling the Python slim
base image, so it needs outbound Docker Hub access (not available in every
sandbox; CI handles it via docker-in-docker).
