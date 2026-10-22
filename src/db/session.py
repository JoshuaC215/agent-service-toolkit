
from collections.abc import AsyncIterator

from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.settings import settings


def build_database_url() -> URL:
    """Build the SQLAlchemy PostgreSQL connection URL."""

    required = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]

    missing = [
        name for name in required
        if not getattr(settings, name)
    ]

    if missing:
        raise ValueError(
            f"Missing PostgreSQL settings: {', '.join(missing)}"
        )

    password = settings.POSTGRES_PASSWORD
    assert password is not None

    return URL.create(
        drivername="postgresql+psycopg",
        username=settings.POSTGRES_USER,
        password=password.get_secret_value(),
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
    )


async_engine = create_async_engine(
    build_database_url(),
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Provide a database session."""

    async with AsyncSessionLocal() as session:
        yield session