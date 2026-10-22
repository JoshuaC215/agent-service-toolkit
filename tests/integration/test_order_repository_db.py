
import asyncio
import selectors
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from db.session import build_database_url
from db.repositories.order_repository import OrderRepository


async def check_real_orders():
    # Windows connects to Docker PostgreSQL via localhost.
    database_url = build_database_url().set(
        host="127.0.0.1"
    )

    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
    )

    try:
        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

        async with session_factory() as session:
            # 1. Verify database connection
            result = await session.execute(text("SELECT 1"))
            assert result.scalar_one() == 1

            repo = OrderRepository(session)

            # 2. Query an existing order
            order1 = await repo.get_by_id("ORD001")

            assert order1 is not None
            assert order1.status == "paid"
            assert order1.amount == Decimal("299.00")

            # 3. Query another existing order
            order2 = await repo.get_by_id("ORD002")

            assert order2 is not None
            assert order2.status == "shipped"
            assert order2.amount == Decimal("599.00")

            # 4. Query a nonexistent order
            missing = await repo.get_by_id("ORD999")

            assert missing is None

    finally:
        await engine.dispose()


def test_order_repository_real_postgres():
    # psycopg async requires SelectorEventLoop on Windows.
    with asyncio.Runner(
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    ) as runner:
        runner.run(check_real_orders())