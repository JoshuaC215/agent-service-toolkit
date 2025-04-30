import logging
import urllib.parse
from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from core.settings import settings

logger = logging.getLogger(__name__)


def validate_mongo_config() -> None:
    """
    Validate that all required MongoDB configuration is present.
    Raises ValueError if any required configuration is missing.
    """
    required_always = ["MONGO_HOST", "MONGO_PORT", "MONGO_DB"]
    missing_always = [var for var in required_always if not getattr(settings, var, None)]
    if missing_always:
        raise ValueError(
            f"Missing required MongoDB configuration: {
                ', '.join(missing_always)}. "
            'These environment variables must be set to use MongoDB persistence.'
        )

    user = getattr(settings, "MONGO_USER", None)
    password = getattr(settings, "MONGO_PASS", None)

    has_user = user is not None and user.strip() != ""
    has_password = False
    if password is not None:
        try:
            password_str = password.get_secret_value()
            has_password = password_str.strip() != ""
        except AttributeError:
            has_password = False

    if has_user or has_password:
        if not (has_user and has_password):
            missing = []
            if not has_user:
                missing.append("MONGO_USER")
            if not has_password:
                missing.append("MONGO_PASS")
            raise ValueError(
                f"Both MONGO_USER and MONGO_PASS must be set if one is provided. "
                f"Missing: {', '.join(missing)}"
            )
        auth_source = getattr(settings, "MONGO_AUTH_SOURCE", None)
        if not auth_source or auth_source.strip() == "":
            raise ValueError(
                "MONGO_AUTH_SOURCE is required when MONGO_USER and MONGO_PASS are set."
            )


def get_mongo_connection_string() -> str:
    """Build and return the MongoDB connection string from settings."""
    user = getattr(settings, "MONGO_USER", None)
    password = getattr(settings, "MONGO_PASS", None)

    has_user = user is not None and user.strip() != "" if user else False
    has_password = False
    if password is not None:
        try:
            password_str = password.get_secret_value()
            has_password = password_str.strip() != ""
        except AttributeError:
            has_password = False

    if has_user and has_password:
        password_escaped = urllib.parse.quote_plus(password.get_secret_value())
        auth_source = settings.MONGO_AUTH_SOURCE
        return (
            f"mongodb://{user}:{password_escaped}@"
            f"{settings.MONGO_HOST}:{settings.MONGO_PORT}/"
            f"?authSource={auth_source}"
        )
    else:
        return f"mongodb://{settings.MONGO_HOST}:{settings.MONGO_PORT}/"


def get_mongo_saver() -> AbstractAsyncContextManager[AsyncMongoDBSaver]:
    """Initialize and return a MongoDB saver instance."""
    validate_mongo_config()
    return AsyncMongoDBSaver.from_conn_string(
        get_mongo_connection_string(), db_name=settings.MONGO_DB
    )
