
import asyncio
import selectors
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from db.models import Base, Order, Ticket
from db.repositories.ticket_repository import TicketRepository
from db.session import build_database_url


async def exercise_ticket_repository():
    # Always use the isolated test database.
    database_url = build_database_url().set(
        host="127.0.0.1",
        database="agentops_test",
    )

    if database_url.database != "agentops_test":
        raise RuntimeError("Unsafe test database configuration")

    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
    )

    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )

    suffix = uuid4().hex[:10].upper()
    order_id = f"TESTORD{suffix}"
    ticket_id = f"TKT{suffix}"
    rollback_ticket_id = f"RBK{suffix}"

    schema_ready = False

    try:
        # Create missing tables only in agentops_test.
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        schema_ready = True

        # 1. Create a test order for the ticket foreign key.
        async with session_factory() as session:
            session.add(
                Order(
                    order_id=order_id,
                    status="paid",
                    amount=Decimal("25.00"),
                    currency="CNY",
                )
            )
            await session.commit()

        # 2. Create a ticket through the Repository.
        async with session_factory() as session:
            repo = TicketRepository(session)

            ticket = Ticket(
                ticket_id=ticket_id,
                order_id=order_id,
                issue_type="refund",
                description="Integration test ticket",
                status="open",
            )

            created = await repo.create(ticket)

            assert created.ticket_id == ticket_id
            await session.commit()

        # 3. Query the ticket and list tickets by order.
        async with session_factory() as session:
            repo = TicketRepository(session)

            found = await repo.get_by_id(ticket_id)

            assert found is not None
            assert found.order_id == order_id
            assert found.issue_type == "refund"
            assert found.status == "open"

            tickets = await repo.list_by_order_id(order_id)

            assert [item.ticket_id for item in tickets] == [
                ticket_id
            ]

            # 4. Update status and commit the transaction.
            await repo.update_status(found, "in_progress")
            await session.commit()

        # 5. Verify the update persisted and missing IDs return None.
        async with session_factory() as session:
            repo = TicketRepository(session)

            updated = await repo.get_by_id(ticket_id)

            assert updated is not None
            assert updated.status == "in_progress"

            missing = await repo.get_by_id(f"MISSING{suffix}")
            assert missing is None

            # Create a second ticket, but deliberately roll back.
            rollback_ticket = Ticket(
                ticket_id=rollback_ticket_id,
                order_id=order_id,
                issue_type="shipping",
                description="Rollback test ticket",
                status="open",
            )

            await repo.create(rollback_ticket)
            await session.rollback()

        # A new session must not see the rolled-back ticket.
        async with session_factory() as session:
            repo = TicketRepository(session)

            rolled_back = await repo.get_by_id(
                rollback_ticket_id
            )

            assert rolled_back is None

        print("TicketRepository PostgreSQL checks passed")

    finally:
        try:
            # Remove only the records created by this test.
            if schema_ready:
                async with session_factory() as session:
                    await session.execute(
                        delete(Ticket).where(
                            Ticket.ticket_id.in_(
                                [ticket_id, rollback_ticket_id]
                            )
                        )
                    )

                    await session.execute(
                        delete(Order).where(
                            Order.order_id == order_id
                        )
                    )

                    await session.commit()
        finally:
            await engine.dispose()


def test_ticket_repository_real_postgres():
    # Use SelectorEventLoop for psycopg async on Windows.
    with asyncio.Runner(
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    ) as runner:
        runner.run(exercise_ticket_repository())