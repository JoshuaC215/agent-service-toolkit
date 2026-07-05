// Minimal AG-UI client for manually validating the service's /agui endpoints,
// built on the official @ag-ui/client SDK. See docs/AGUI.md for details.
//
// Usage:
//   cd scripts/agui-client
//   npm install
//   node client.mjs [message] [agent]
//
// Environment variables:
//   AGENT_URL   - base URL of the agent service (default: http://localhost:8080)
//   AUTH_SECRET - bearer token, if the service has one configured
//   THREAD_ID   - reuse a thread to continue a conversation (default: random)

import { randomUUID } from "crypto";
import { HttpAgent } from "@ag-ui/client";

const message = process.argv[2] ?? "Tell me a joke!";
const agentId = process.argv[3] ?? "chatbot";
const baseUrl = process.env.AGENT_URL ?? "http://localhost:8080";
const threadId = process.env.THREAD_ID ?? randomUUID();

const agent = new HttpAgent({
  url: `${baseUrl}/agui/${agentId}/run`,
  threadId,
  headers: process.env.AUTH_SECRET
    ? { Authorization: `Bearer ${process.env.AUTH_SECRET}` }
    : {},
});

console.log(`agent: ${agentId}  thread: ${threadId}`);
console.log(`user: ${message}\n`);

agent.messages = [{ id: randomUUID(), role: "user", content: message }];

const eventTypes = new Set();
let printedPrefix = false;
try {
  await agent.runAgent(
    { runId: randomUUID() },
    {
      onEvent({ event }) {
        eventTypes.add(event.type);
      },
      onTextMessageContentEvent({ event }) {
        // Print each delta as it arrives rather than re-rendering the full buffer -
        // an in-place overwrite via \r depends on terminal support that isn't
        // consistent (e.g. piping through `docker compose logs`).
        if (!printedPrefix) {
          process.stdout.write("assistant: ");
          printedPrefix = true;
        }
        process.stdout.write(event.delta);
      },
      onCustomEvent({ event }) {
        if (event.name === "on_interrupt") {
          console.log(`[interrupted] ${JSON.stringify(event.value)}`);
          console.log(
            "resume by running the same thread with forwardedProps {command: {resume: <answer>}} - see docs/AGUI.md"
          );
        }
      },
    }
  );
} catch (err) {
  console.error(`\nrun failed: ${err.message ?? err}`);
  if (String(err.message).includes("401")) {
    console.error("hint: set AUTH_SECRET to the service's bearer token");
  }
  process.exit(1);
}

console.log(`\n\nevent types received: ${[...eventTypes].join(", ")}`);
console.log(`continue this thread with: THREAD_ID=${threadId} node client.mjs '<message>' ${agentId}`);
