import logging
from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from core.settings import settings

logger = logging.getLogger(__name__)


def validate_postgres_config() -> None:
    """
    Validate that all required PostgreSQL configuration is present.
    Raises ValueError if any required configuration is missing.
    """
    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]

    missing = [var for var in required_vars if not getattr(settings, var, None)]
    if missing:
        raise ValueError(
            f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
            "These environment variables must be set to use PostgreSQL persistence."
        )


def get_postgres_connection_string() -> str:
    """Build and return the PostgreSQL connection string from settings."""
    return (
        f"postgresql://{settings.POSTGRES_USER}:"
        f"{settings.POSTGRES_PASSWORD.get_secret_value()}@"
        f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
        f"{settings.POSTGRES_DB}"
    )


def create_connection_pool() -> AsyncConnectionPool:
    """Create and return a PostgreSQL connection pool with configured settings."""
    conn_string = get_postgres_connection_string()

    # Create connection pool with settings from config
    pool = AsyncConnectionPool(
        conn_string,
        min_size=settings.POSTGRES_MIN_SIZE,
        max_size=settings.POSTGRES_POOL_SIZE,
        max_idle=settings.POSTGRES_MAX_IDLE,
    )

    logger.info(
        f"Created PostgreSQL connection pool: min_size={settings.POSTGRES_MIN_SIZE}, "
        f"max_size={settings.POSTGRES_POOL_SIZE}, max_idle={settings.POSTGRES_MAX_IDLE}"
    )

    return pool


def get_postgres_saver() -> AbstractAsyncContextManager[AsyncPostgresSaver]:
    """Initialize and return a PostgreSQL saver instance with connection pool."""
    validate_postgres_config()

    # Create connection pool with custom settings
    pool = create_connection_pool()

    # Initialize saver with the pool
    saver = AsyncPostgresSaver(conn=pool)
    return saver
