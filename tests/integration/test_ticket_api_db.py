import asyncio
import selectors
from decimal import Decimal
from uuid import uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from db.models import Base, Order, Ticket
from db.session import build_database_url, get_db_session
from service.tickets import router as tickets_router


async def exercise_ticket_api():
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
    order_id = f"TORD{suffix}"
    ticket_id = f"TKT{suffix}"

    schema_ready = False

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        schema_ready = True

        # Seed an order for the ticket foreign key.
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

        async def override_db_session():
            async with session_factory() as session:
                yield session

        app = FastAPI()
        app.include_router(tickets_router)
        app.dependency_overrides[
            get_db_session
        ] = override_db_session

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:

            payload = {
                "ticket_id": ticket_id,
                "order_id": order_id,
                "issue_type": "refund",
                "description": "Integration test ticket",
            }

            # 1. Create a ticket.
            response = await client.post(
                "/tickets",
                json=payload,
            )
            assert response.status_code == 201, response.text
            assert response.json()["status"] == "open"

            # 2. Query the ticket by ID.
            response = await client.get(
                f"/tickets/{ticket_id}"
            )
            assert response.status_code == 200, response.text
            assert response.json()["order_id"] == order_id

            # 3. List tickets belonging to the order.
            response = await client.get(
                f"/tickets/by-order/{order_id}"
            )
            assert response.status_code == 200, response.text
            assert len(response.json()) == 1

            # 4. Update ticket status.
            response = await client.patch(
                f"/tickets/{ticket_id}",
                json={"status": "in_progress"},
            )
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "in_progress"

            # 5. Duplicate ticket ID should return 409.
            response = await client.post(
                "/tickets",
                json=payload,
            )
            assert response.status_code == 409, response.text

            # 6. A nonexistent order should return 404.
            missing_order_payload = {
                **payload,
                "ticket_id": f"BAD{suffix}",
                "order_id": f"MISSING{suffix}",
            }
            response = await client.post(
                "/tickets",
                json=missing_order_payload,
            )
            assert response.status_code == 404, response.text

            # 7. A nonexistent ticket should return 404.
            response = await client.get(
                f"/tickets/NOTFOUND{suffix}"
            )
            assert response.status_code == 404, response.text

        print("Ticket API PostgreSQL checks passed")

    finally:
        try:
            if schema_ready:
                async with session_factory() as session:
                    await session.execute(
                        delete(Ticket).where(
                            Ticket.ticket_id == ticket_id
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


def test_ticket_api_real_postgres():
    with asyncio.Runner(
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    ) as runner:
        runner.run(exercise_ticket_api())