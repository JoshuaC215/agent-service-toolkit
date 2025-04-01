from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from core.settings import settings


def get_sqlite_saver() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """Initialize and return a SQLite saver instance."""
    return AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)


def get_sqlite_store() -> BaseStore:
    """Initialize and return a store instance for long-term memory.
    
    Note: SQLite-specific store isn't available in LangGraph,
    so we use InMemoryStore as a fallback.
    """
    return InMemoryStore()