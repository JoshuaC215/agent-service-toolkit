import asyncio
import logging
import urllib.parse
from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from core.settings import settings

logger = logging.getLogger(__name__)


def _has_auth_credentials() -> bool:
    required_auth = ["MONGO_USER", "MONGO_PASSWORD", "MONGO_AUTH_SOURCE"]
    set_auth = [var for var in required_auth if getattr(settings, var, None)]
    if len(set_auth) > 0 and len(set_auth) != len(required_auth):
        raise ValueError(
            f"If any of the following environment variables are set, all must be set: {', '.join(required_auth)}."
        )
    return len(set_auth) == len(required_auth)


def validate_mongo_config() -> None:
    """
    Validate that all required MongoDB configuration is present.
    Raises ValueError if any required configuration is missing.
    """
    required_always = ["MONGO_HOST", "MONGO_PORT", "MONGO_DB"]
    missing_always = [var for var in required_always if not getattr(settings, var, None)]
    if missing_always:
        raise ValueError(
            f"Missing required MongoDB configuration: {', '.join(missing_always)}. "
            "These environment variables must be set to use MongoDB persistence."
        )

    _has_auth_credentials()


def get_mongo_connection_string() -> str:
    """Build and return the MongoDB connection string from settings."""

    if _has_auth_credentials():
        if settings.MONGO_PASSWORD is None:  # for type checking
            raise ValueError("MONGO_PASSWORD is not set")
        password = settings.MONGO_PASSWORD.get_secret_value().strip()
        password_escaped = urllib.parse.quote_plus(password)
        return (
            f"mongodb://{settings.MONGO_USER}:{password_escaped}@"
            f"{settings.MONGO_HOST}:{settings.MONGO_PORT}/"
            f"?authSource={settings.MONGO_AUTH_SOURCE}"
        )
    else:
        return f"mongodb://{settings.MONGO_HOST}:{settings.MONGO_PORT}/"


class _AsyncMongoDBSaver(AbstractAsyncContextManager[MongoDBSaver]):
    """Async context manager wrapping MongoDBSaver, which is sync-only as of
    langgraph-checkpoint-mongodb 0.4 (it bridges to async internally via a thread executor).
    Connecting and building the saver's indexes does blocking I/O, so both are run off the
    event loop thread.
    """

    def __init__(self, conn_string: str, db_name: str):
        self._conn_string = conn_string
        self._db_name = db_name
        self._saver: MongoDBSaver | None = None

    async def __aenter__(self) -> MongoDBSaver:
        def _connect() -> MongoDBSaver:
            client: MongoClient = MongoClient(self._conn_string)
            return MongoDBSaver(client, db_name=self._db_name)

        self._saver = await asyncio.to_thread(_connect)
        return self._saver

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._saver is not None:
            await asyncio.to_thread(self._saver.close)


def get_mongo_saver() -> AbstractAsyncContextManager[MongoDBSaver]:
    """Initialize and return a MongoDB saver instance."""
    validate_mongo_config()
    if settings.MONGO_DB is None:  # for type checking
        raise ValueError("MONGO_DB is not set")
    return _AsyncMongoDBSaver(get_mongo_connection_string(), settings.MONGO_DB)
