# Third-party adaptations

The project's framework is based on JoshuaC215/agent-service-toolkit, retaining
the existing root MIT license and history.

## HelpPilot

Source: https://github.com/poysa213/HelpPilot

Files consulted on 2026-10-22:

- `helppilot/rag.py` on `main`: query/passage CrossEncoder scoring and sorting
  scores together with passage metadata. Adapted in
  `src/agents/policy_reranker.py` and integrated with `PolicyRAG`.
- The repository README: triage, bounded grounding review and final response.
  Implemented anew in `support_review.py` and `order_agent.py`, not copied from
  `helppilot/graph.py` (that source could not be fetched in this session).
- `LICENSE`: MIT, with upstream's literal copyright placeholder preserved in
  `third_party/helppilot/LICENSE`.

The retrieved `main` source was not resolved to a commit SHA; no pinned upstream
commit is claimed. Local adaptations include Chinese BGE reranking, explicit
model download and offline loading, retained cosine filtering, no tool calls
during review rewrites, hidden draft output, and our existing PostgreSQL staff
approval/identity workflow. We did not import HelpPilot's Groq configuration,
SQLite schema, example customers, approval resume authorization, or eval scores.
