# Human refund approval API

## Customer identity and order ownership

This release uses server-issued static access keys, not password accounts or
expiring login sessions. Configure `CUSTOMER_ACCESS_KEYS` as a JSON map from
customer IDs to distinct random keys of at least 32 characters. Customer IDs and
keys must not overlap staff identities in `APPROVAL_REVIEWERS`. Keep real keys
only in backend configuration; do not commit them. Recreate containers after
changing `.env`. Removing a key revokes its subsequent API requests.

Customer requests send `X-User-Key`; staff requests send `X-Approval-Key`.
`GET /auth/me` resolves identity on the server. The frontend's customer login
screen forwards the assigned key, and logout clears local session state.
The optional shared `AUTH_SECRET` remains an additional service-level control,
not a substitute for user identity.

After applying migration `d2e681c4a907`, historical orders have no owner and are
visible only to staff. In the human workspace's **订单归属** tab, staff may assign
an order to a configured customer. The corresponding API is
`PUT /orders/{order_id}/owner` with `{"owner_id":"customer-1"}`. It allows initial
assignment or the same owner, but refuses transferring an already assigned order.
All linked tickets and approvals inherit visibility from the order.

Order/ticket reads and customer approval tools filter by authenticated ownership.
Creating/changing orders, changing ticket status, human review, and simulated
execution remain staff-only. Customer tools use request identity rather than
model-supplied IDs. Conversation ownership is registered on first use and checked
before reading or resuming; historical unregistered chats are not claimed by a
new customer. Request `user_id` cannot override identity. Customer chat currently
uses only `order-agent`; alternate agents, AG-UI, and feedback are restricted until
their access models are completed.

Validation:

```powershell
uv run --extra local-rag pytest tests/integration/test_customer_access.py -q
```

## My orders

Customer chat now supports "查询我的订单" through `list_my_orders`. Its only
arguments are `limit` (1–100, default 20) and `offset` (nonnegative, default 0).
The authenticated request context determines the owner. Extra parameters such as
`user_id` or `owner_id` are rejected, including in the tool's input schema.

`GET /orders/me?limit=20&offset=0` returns `orders`, `limit`, `offset`, and
`has_more`. Results sort by creation time descending with order ID descending
as a deterministic tie-breaker. An empty page returns an empty list and
`has_more: false`. This endpoint requires a customer identity; it is not a
staff-wide order listing. Unassigned orders and other customers' orders are
excluded. No new database migration is needed for this query feature.

The order agent must use the tool rather than asking for an order ID just to
list orders. It must distinguish a failed query from an empty page and must not
describe a partial page as all orders. Listing does not create a ticket or
submit a refund. A new chat avoids older context asserting the tool is absent.

## Simulated refund execution

Apply the new migration once from the host project directory:

```powershell
uv run --extra local-rag alembic upgrade head
```

Expected migration head: `b7d20a8f914c`. Compose Watch syncs execution code but
does not apply database migrations. The new table is `refund_executions`.

In **人工审批 → 模拟退款**, enter an approved ticket, verify its order amount,
check the simulation confirmation, and click **执行模拟退款**. Use the same tab
to query the saved result. No payment provider is called, no funds move, and
order/ticket status remains unchanged. Amount and currency come from the order;
clients cannot supply them. This first version models only full-order refunds
for positive-value `paid` orders; partial refunds and real payments are excluded.

The existing staff authentication also protects these routes:

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/refund-approvals/by-ticket/{ticket_id}/simulate-refund` | Execute or return an existing simulation |
| GET | `/refund-approvals/by-ticket/{ticket_id}/execution` | Read the saved execution |

POST body: `{"confirm_simulation":true}`. Actor identity is derived from the
staff credential. The response includes `created`, `simulated: true`,
`funds_transferred: false`, and the execution record. Its status is explicitly
`simulated_succeeded`, with an execution ID, approval ID, order ID, amount,
currency, original executor, result message, and timestamp.

Only approved applications on active tickets can first execute. Repeating the
same approved application returns the original record and `created: false`,
without changing its actor/time. A different approval for the same order returns
409. PostgreSQL row locks plus unique approval/order constraints arbitrate
concurrent requests. SQLite tests cover the unique-insert arbitration; they do
not substitute for PostgreSQL lock validation. Since the simulation is only a
database transaction, a rollback leaves no execution. This is not an exactly-once
design for an external payment provider; real payments need provider idempotency
and reconciliation. Business rejections return errors without successful records;
there is no injected payment failure or retry state in this deterministic simulator.

Run execution and page tests:

```powershell
uv run --extra local-rag pytest tests/integration/test_refund_execution_service.py tests/app/test_approval_page.py tests/service/test_refund_approvals.py -q
```

PostgreSQL concurrency validation (requires the local Docker database and `.env`):

```powershell
uv run --extra local-rag pytest tests/integration/test_refund_execution_postgres.py --run-docker -q
```

This creates and removes a unique `test_refund_*` schema; it does not read or
write existing business tables. The configured database user needs schema-create
permission. It verifies concurrent requests return one execution and a second
ticket for the same order cannot create another execution.

## Docker development

Stop the local backend and Streamlit processes with Ctrl+C before switching;
the containers publish the same 8080 and 8501 ports. Keep Docker Desktop running.
From the project directory, start development mode with:

```powershell
docker compose up --build --watch
```

Keep that terminal open. Source changes sync and restart the corresponding
service; dependency files and Dockerfiles trigger image rebuilds. Initial sync
also includes edits made before Watch starts. PostgreSQL and Chroma retain their
named volumes, and the downloaded local embedding model is mounted read-only.
The first image build requires package downloads and may take several minutes.

After changing `.env`, recreate the services to reload container environment:

```powershell
docker compose up -d --force-recreate agent_service streamlit_app
```

Do not run `down -v`: it deletes named data volumes. Model/schema changes sync
as code, but database schema changes still require an explicit migration.
Voice is disabled in the frontend container by default. Existing approval
credentials come from backend `.env`; enter the assigned key in the UI.

The approval service records decisions only. It never executes a refund or
changes order or ticket status. Historical tickets are not automatically submitted.

## Configuration

Set `APPROVAL_REVIEWERS` in the backend environment to a JSON object mapping
reviewer IDs to independent random credentials. Each credential must be unique
and at least 32 characters; IDs must contain non-whitespace text and fit 64
characters. Generate credentials with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
Store the mapping privately; do not commit it or send credentials in chat.
An empty or invalid mapping disables approval endpoints with HTTP 503.
Restart the backend after configuration changes.

Send the assigned credential in `X-Approval-Key`. If the service's `AUTH_SECRET`
is enabled, also send its usual `Authorization: Bearer ...` header.
These static credentials are a first version of staff authentication, not a
user-account or role-management system. The reviewer ID is resolved server-side.

## Endpoints

All endpoints require approval credentials, including submission and reads.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/refund-approvals/by-ticket/{ticket_id}` | Submit an existing active ticket for approval |
| GET | `/refund-approvals/by-ticket/{ticket_id}` | Read the approval record |
| GET | `/refund-approvals/pending` | List pending approvals, oldest first |
| POST | `/refund-approvals/by-ticket/{ticket_id}/review` | Approve or reject |

Review body: `{"decision":"approved","reason":"Evidence reviewed"}`.
The alternative decision is `rejected`. Reasons must contain 1–4000 characters
after trimming. Client-supplied reviewer IDs and extra review fields are rejected.
Submission accepts no decision fields and always creates a pending record.
Only tickets in `open` or `in_progress` may be submitted or reviewed. Staff must
confirm that the ticket actually requests a refund: the current free-text
`issue_type` is not a reliable structured refund classification.

Missing records return 404; duplicate submissions, completed approvals, and
inactive tickets return 409. Invalid review bodies return 422. Missing or wrong
credentials return 401. There is no reopen or re-review operation in this version.

## Validation

Run `uv run --extra local-rag pytest tests/service/test_refund_approvals.py -vv`.
These tests use an isolated FastAPI app and mocked database calls to check auth,
identity derivation, request validation, and error responses. The database service
tests are separate; SQLite tests do not validate PostgreSQL row-lock behavior.

## Streamlit workspace

Start the existing Streamlit app and select **人工审批** in the sidebar's
**工作区** control. The page uses the existing native Streamlit theme. Enter the
assigned key in the password field. The backend `APPROVAL_REVIEWERS` mapping is
not needed by the frontend; only the human operator's key is entered at runtime.
When service bearer authentication is enabled, configure frontend `AUTH_SECRET`
as usual. Requests use the configured `AGENT_URL` and bypass environment proxies.

The page provides pending reviews, submission of existing refund tickets, and
result lookup. Review details display order facts and the complete ticket
description. An explicit checkbox and nonempty reason are required to send a
decision. The backend rechecks status even if the screen is stale. Network errors
do not trigger automatic write retries; query the result before trying again.
Exit the approval identity using the sidebar button to clear its session key.
No approval credentials or decisions are sent to the chat model.

Run the new UI/client tests with:

```powershell
uv run --extra local-rag pytest tests/client/test_approval_client.py tests/app/test_approval_page.py -q
```

## Agent submission and lookup

The order agent now has `submit_refund_approval(ticket_id)` and
`get_refund_approval(ticket_id)` tools. These call the shared business/repository
layer directly using the existing chat service access model; they do not possess
staff credentials or expose human review. Ticket ownership authentication is not
implemented by these tools, consistent with the existing ticket query tools.

Explicitly request submission for a confirmed refund ticket. Creating a ticket
or consulting policies alone should not submit it. Duplicate submission returns
an error and the agent should query the existing approval. If a write's result
cannot be confirmed, the agent should query before retrying. A missing approval
is not a rejection, and an approved record does not establish refund execution.
These user-intent rules are model instructions, not an authorization mechanism.

Example: ask the order agent to submit an existing test refund ticket for human
review, open the human workspace to review it, then ask the agent for that
ticket's approval status. This creates a real approval record and human review
is final in this version; no refund is executed.

Tool adapter tests:

```powershell
uv run --extra local-rag pytest tests/agents/test_approval_tools.py -q
```
