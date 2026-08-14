"""Thread enumeration for the /threads endpoints.

Threads are derived from the checkpointer rather than a table of their own, so listing
them means reading checkpoint metadata the way LangGraph writes it. The invariants that
makes possible are documented at the constants below.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from schema import ThreadSummary
from service.utils import convert_message_content_to_string, messages_from_checkpoint

logger = logging.getLogger(__name__)

# LangGraph writes the input checkpoint at step -1 once per thread; later turns continue
# from the last step. Don't swap in another step - single-turn threads never reach step 1.
THREAD_HEAD_STEP = -1

# Heads are ordered by thread creation, so over-fetch and re-sort by tip to approximate
# "most recently updated". Threads created before the oldest head fetched fall off.
MAX_THREAD_HEADS = 200
HEAD_PAGE_SIZE = 200

# A head row isn't always a distinct thread: agents with subgraphs write one per subgraph
# call, inheriting the parent's metadata. Bounds the paging that compensates for it.
MAX_HEAD_ROWS = 1000

TITLE_MAX_LENGTH = 60


async def _list_thread_heads(checkpointer: Any, user_id: str, agent_id: str) -> list[Any]:
    """Return one head checkpoint per thread, newest thread first."""
    heads: list[Any] = []
    seen: set[str] = set()
    rows_scanned = 0
    before = None
    while len(seen) < MAX_THREAD_HEADS and rows_scanned < MAX_HEAD_ROWS:
        page = [
            c
            async for c in checkpointer.alist(
                None,
                filter={"user_id": user_id, "agent_id": agent_id, "step": THREAD_HEAD_STEP},
                before=before,
                limit=HEAD_PAGE_SIZE,
            )
        ]
        if not page:
            break
        rows_scanned += len(page)
        short_page = len(page) < HEAD_PAGE_SIZE
        for row in page:
            thread_id = row.config["configurable"]["thread_id"]
            if thread_id in seen:
                continue
            seen.add(thread_id)
            heads.append(row)
        if short_page:
            break
        before = RunnableConfig(
            configurable={"checkpoint_id": page[-1].config["configurable"]["checkpoint_id"]}
        )
    return heads


async def list_user_threads(
    checkpointer: Any, user_id: str, agent_id: str, limit: int
) -> list[ThreadSummary]:
    """List a user's threads for an agent, most recently updated first."""
    summaries: list[tuple[str, ThreadSummary]] = []
    for head in await _list_thread_heads(checkpointer, user_id, agent_id):
        thread_id = head.config["configurable"]["thread_id"]
        stored_user_id = head.metadata.get("user_id")
        stored_agent_id = head.metadata.get("agent_id")
        if stored_user_id != user_id or stored_agent_id != agent_id:
            logger.warning(
                f"Checkpointer returned thread {thread_id} with user_id "
                f"{stored_user_id!r}/agent_id {stored_agent_id!r}, expected "
                f"{user_id!r}/{agent_id!r} — skipping to avoid a "
                "cross-user or cross-agent leak."
            )
            continue

        # The head has no messages yet, so the title and updated_at come from the tip.
        tip = await checkpointer.aget_tuple(RunnableConfig(configurable={"thread_id": thread_id}))
        if tip is None:
            continue
        messages = messages_from_checkpoint(tip.checkpoint)
        first_human = next((m for m in messages if isinstance(m, HumanMessage)), None)
        summaries.append(
            (
                tip.config["configurable"]["checkpoint_id"],
                ThreadSummary(
                    thread_id=thread_id,
                    agent_id=agent_id,
                    updated_at=tip.checkpoint.get("ts"),
                    title=convert_message_content_to_string(first_human.content)[:TITLE_MAX_LENGTH]
                    if first_human
                    else None,
                ),
            )
        )

    # Checkpoint IDs are time-ordered UUIDs, so the tip's ID sorts by last update.
    summaries.sort(key=lambda item: item[0], reverse=True)
    return [summary for _, summary in summaries[:limit]]
