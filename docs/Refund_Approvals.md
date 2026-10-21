# Human refund approval API

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
