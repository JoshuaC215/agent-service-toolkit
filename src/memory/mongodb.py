import logging
import urllib.parse
from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

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


def get_mongo_saver() -> AbstractAsyncContextManager[AsyncMongoDBSaver]:
    """Initialize and return a MongoDB saver instance."""
    validate_mongo_config()
    if settings.MONGO_DB is None:  # for type checking
        raise ValueError("MONGO_DB is not set")
    return AsyncMongoDBSaver.from_conn_string(
        get_mongo_connection_string(), db_name=settings.MONGO_DB
    )
