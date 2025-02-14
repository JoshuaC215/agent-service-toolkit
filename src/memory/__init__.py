from langgraph.checkpoint.base import BaseCheckpointSaver

from core.settings import DatabaseType, settings
from memory.postgres import get_postgres_saver
from memory.sqlite import get_sqlite_saver


def initialize_database() -> BaseCheckpointSaver:
    """
    Initialize the appropriate database checkpointer based on configuration.
    Returns an initialized AsyncCheckpointer instance.
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_saver()
    else:  # Default to SQLite
        return get_sqlite_saver()


__all__ = ["initialize_database"]
