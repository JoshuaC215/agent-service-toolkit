
import asyncio
import selectors
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

import agents.ticket_tools as ticket_tools
from db.models import Base, Order, Ticket
from db.session import build_database_url


async def exercise_ticket_tools(session_factory):
    suffix = uuid4().hex[:10].upper()
    order_id = f"TORD{suffix}"
    cleanup_needed = False

    try:
        # 1. Create a temporary test order.
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

        # The order was committed successfully.
        # Temporary data must be cleaned up in finally.
        cleanup_needed = True

        # 2. Create a ticket through the LangChain tool.
        created = await ticket_tools.create_ticket.ainvoke(
            {
                "order_id": order_id,
                "issue_type": "refund",
                "description": "Integration test ticket",
            }
        )

        assert created["success"] is True, created

        ticket_id = created["ticket"]["ticket_id"]

        assert ticket_id.startswith("TKT")
        assert created["ticket"]["order_id"] == order_id
        assert created["ticket"]["status"] == "open"

        # 3. Retrieve the ticket by its actual ID.
        found = await ticket_tools.get_ticket.ainvoke(
            {"ticket_id": ticket_id}
        )

        assert found["found"] is True, found
        assert found["ticket"]["ticket_id"] == ticket_id

        # 4. List tickets associated with the order.
        listed = await ticket_tools.list_tickets_by_order.ainvoke(
            {"order_id": order_id}
        )

        assert listed["found"] is True, listed
        assert listed["count"] == 1
        assert listed["tickets"][0]["ticket_id"] == ticket_id

        # 5. Reject a ticket request for a nonexistent order.
        missing = await ticket_tools.create_ticket.ainvoke(
            {
                "order_id": f"MISSING{suffix}",
                "issue_type": "refund",
                "description": "This ticket must not be created",
            }
        )

        assert missing["success"] is False, missing
        assert missing["error"] == "ORDER_NOT_FOUND"

        # 6. A nonexistent ticket must not be fabricated.
        missing_ticket = await ticket_tools.get_ticket.ainvoke(
            {"ticket_id": f"MISSING{suffix}"}
        )

        assert missing_ticket["found"] is False, missing_ticket
        assert missing_ticket["error"] == "TICKET_NOT_FOUND"

        print("Ticket tools PostgreSQL checks passed")

    finally:
        # Delete temporary tickets before deleting their order.
        if cleanup_needed:
            async with session_factory() as session:
                await session.execute(
                    delete(Ticket).where(
                        Ticket.order_id == order_id
                    )
                )

                await session.execute(
                    delete(Order).where(
                        Order.order_id == order_id
                    )
                )

                await session.commit()


def test_ticket_tools_real_postgres(monkeypatch):
    # Always use the isolated test database.
    database_url = build_database_url().set(
        host="127.0.0.1",
        database="agentops_test",
    )

    if database_url.database != "agentops_test":
        raise RuntimeError(
            "Unsafe test database configuration"
        )

    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
    )

    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )

    async def run():
        try:
            # Ensure the ORM tables exist in the test database.
            async with engine.begin() as connection:
                await connection.run_sync(
                    Base.metadata.create_all
                )

            # Make all ticket tools use the isolated test database.
            monkeypatch.setattr(
                ticket_tools,
                "AsyncSessionLocal",
                session_factory,
            )

            await exercise_ticket_tools(session_factory)

        finally:
            # Release database connections even if assertions fail.
            await engine.dispose()

    # Use a selector-based event loop for Windows compatibility.
    with asyncio.Runner(
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    ) as runner:
        runner.run(run())