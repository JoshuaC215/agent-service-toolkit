# AG-UI Protocol Support

The service exposes every agent over the [AG-UI protocol](https://docs.ag-ui.com) — an open,
event-based standard for connecting agents to user-facing applications, used by
[CopilotKit](https://docs.copilotkit.ai) and a growing set of frameworks. This lets you build a
production React/Next.js frontend for your agents while keeping the Streamlit app for development.

The heavy lifting (translating LangGraph execution into AG-UI events) is done by the official
[`ag-ui-langgraph`](https://pypi.org/project/ag-ui-langgraph/) package. The service adds a thin
adapter (`src/service/agui.py`) that wires it into the agent registry, bearer auth, and Langfuse
tracing.

## Endpoints

| Endpoint | Description |
| --- | --- |
| `POST /agui/{agent_id}/run` | Run an agent, streaming AG-UI events over SSE |
| `POST /agui/run` | Same, using the default agent |

The request body is the standard AG-UI `RunAgentInput`. The same `AUTH_SECRET` bearer auth as the
rest of the API applies.

## Connecting a frontend

The expected production setup is the standard CopilotKit architecture: your frontend talks to a
[CopilotKit runtime](https://docs.copilotkit.ai) (e.g. a Next.js API route), which connects to
this service server-side using the AG-UI `HttpAgent`:

```ts
import { HttpAgent } from "@ag-ui/client";

const agent = new HttpAgent({
  url: "http://your-service:8080/agui/research-assistant/run",
  headers: { Authorization: `Bearer ${process.env.AUTH_SECRET}` },
});
```

The runtime holds the bearer token and acts as the trusted layer between browsers and the agent
service, the same role the Streamlit app plays for the vanilla API. Direct browser access to the
endpoint is possible for experiments, but you'd need to add CORS middleware to the service
yourself and your `AUTH_SECRET` would be visible to browsers — prefer the runtime pattern.

## Trying it out

A minimal reference client using the official SDK is included:

```sh
# start the service in one shell (or docker compose watch)
python src/run_service.py

# in another shell
cd scripts/agui-client
npm install
node client.mjs "Tell me a joke!" chatbot
```

Use `THREAD_ID` to continue a conversation, and `AUTH_SECRET` / `AGENT_URL` as needed:

```sh
THREAD_ID=my-thread node client.mjs "And another one" chatbot
```

## Behavior notes

- **Threads are shared with the vanilla API.** Both protocols use the same checkpointer keyed by
  thread ID, so a conversation started on `/stream` can be continued over AG-UI and vice versa.
  One caveat: messages are deduplicated by ID, so don't replay past messages with newly generated
  IDs (well-behaved AG-UI clients preserve IDs and are fine).
- **Per-request configuration** goes in `forwardedProps.configurable` — the AG-UI equivalent of
  the vanilla API's `model` / `user_id` / `agent_config` fields, e.g.
  `{"forwardedProps": {"configurable": {"model": "gpt-5.2"}}}`. Keys managed by the protocol
  (`thread_id`, `checkpoint_id`, `checkpoint_ns`) are rejected. `model` is checked against
  `AVAILABLE_MODELS` (400 if not allowed), same as the vanilla API.
- **Interrupts** (human-in-the-loop) surface as a `CUSTOM` event named `on_interrupt`. Resume by
  running the same thread again with `{"forwardedProps": {"command": {"resume": <answer>}}}`.
- **Graph state is client-visible.** AG-UI's shared-state feature sends `STATE_SNAPSHOT` events
  containing the full graph state, so don't put secrets or internal-only data in agent state.
- **`RAW` passthrough events are filtered out** by the adapter. Standard AG-UI clients ignore
  them, and they would expose server-side internals (including fully rendered prompts) to
  callers. If you need the full event firehose for debugging (e.g. the AG-UI Event Inspector)
  behind a trusted layer, remove the filter in `src/service/agui.py`.
- **`/feedback` and `/history` are not bridged.** The AG-UI `runId` is client-generated and isn't
  used as the LangSmith run ID, so star feedback doesn't apply to AG-UI runs. AG-UI clients
  manage their own message history from the event stream.
- The `ag-ui-langgraph` package is pre-1.0 and pinned accordingly. If a future LangGraph major
  release ever conflicts with it, the integration should be dropped or lag behind rather than
  hold back core upgrades.
