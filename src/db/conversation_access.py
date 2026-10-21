from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from db.access import AccessDenied, principal
from db.models import ConversationOwner
from db.session import AsyncSessionLocal


async def check_conversation(thread_id: str, *, create: bool = False):
    identity = principal()
    if not thread_id or len(thread_id) > 64:
        raise AccessDenied("INVALID_THREAD_ID")
    async with AsyncSessionLocal() as session:
        if create:
            dialect = session.get_bind().dialect.name
            insert = pg_insert if dialect == "postgresql" else sqlite_insert
            await session.execute(
                insert(ConversationOwner)
                .values(thread_id=thread_id, user_id=identity.user_id)
                .on_conflict_do_nothing()
            )
            await session.commit()
        owner = await session.get(ConversationOwner, thread_id)
        if owner is None or owner.user_id != identity.user_id:
            raise AccessDenied("CONVERSATION_NOT_FOUND")
