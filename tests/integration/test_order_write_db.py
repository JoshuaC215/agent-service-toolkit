
import asyncio
import selectors
from decimal import Decimal
from uuid import uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from db.models import Base, Order
from db.session import build_database_url, get_db_session
from service.orders import router as orders_router


async def run_checks():
    # Explicitly use the isolated test database.
    url = build_database_url().set(
        host="127.0.0.1",
        database="agentops_test",
    )

    if url.database != "agentops_test":
        raise RuntimeError("Unsafe test database")

    engine = create_async_engine(url, pool_pre_ping=True)
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False
    )

    order_id = f"TEST{uuid4().hex[:12]}"
    rollback_id = f"ROLL{uuid4().hex[:12]}"
    db_ready = False

    try:
        # Create the business table only in agentops_test.
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        db_ready = True

        async def test_session():
            async with session_factory() as session:
                yield session

        app = FastAPI()
        app.include_router(orders_router)
        app.dependency_overrides[get_db_session] = test_session

        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:

            payload = {
                "order_id": order_id,
                "status": "pending",
                "amount": "129.90",
                "currency": "CNY",
            }

            # 1. Create an order.
            response = await client.post("/orders", json=payload)
            assert response.status_code == 201, response.text

            # 2. Duplicate order should return 409.
            response = await client.post("/orders", json=payload)
            assert response.status_code == 409, response.text

            # 3. Update the order status.
            response = await client.patch(
                f"/orders/{order_id}",
                json={"status": "shipped"},
            )
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "shipped"

            # 4. Query the updated order.
            response = await client.get(f"/orders/{order_id}")
            assert response.status_code == 200
            assert response.json()["amount"] == "129.90"

            # 5. Updating a missing order should return 404.
            response = await client.patch(
                "/orders/ORDER_NOT_EXIST",
                json={"status": "paid"},
            )
            assert response.status_code == 404

            # 6. Invalid amount should return 422.
            response = await client.post(
                "/orders",
                json={**payload, "amount": "-1.00"},
            )
            assert response.status_code == 422

        # 7. Verify the update was committed to PostgreSQL.
        async with session_factory() as session:
            order = await session.get(Order, order_id)
            assert order is not None
            assert order.status == "shipped"
            assert order.amount == Decimal("129.90")

        # 8. Verify uncommitted changes are rolled back.
        async with session_factory() as session:
            session.add(
                Order(
                    order_id=rollback_id,
                    status="pending",
                    amount=Decimal("10.00"),
                    currency="CNY",
                )
            )
            await session.flush()
            # No commit: closing the session rolls back.

        async with session_factory() as session:
            assert await session.get(Order, rollback_id) is None

    finally:
        try:
            if db_ready:
                # Remove the test order, preserving other data.
                async with session_factory() as session:
                    order = await session.get(Order, order_id)
                    if order is not None:
                        await session.delete(order)
                        await session.commit()
        finally:
            await engine.dispose()


def test_order_write_real_postgres():
    # Use SelectorEventLoop for Windows psycopg compatibility.
    with asyncio.Runner(
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    ) as runner:
        runner.run(run_checks())