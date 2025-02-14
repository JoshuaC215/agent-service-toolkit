from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.settings import settings


def get_sqlite_saver() -> BaseCheckpointSaver:
    """Initialize and return a SQLite saver instance."""
    return AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)
