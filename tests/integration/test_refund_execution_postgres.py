"""Real PostgreSQL concurrency checks in a temporary, isolated schema."""

import asyncio
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from dotenv import dotenv_values
from sqlalchemy import func, select
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema

from db.access import Principal, current_principal

from db.models import Base, Order, RefundApproval, RefundExecution, Ticket
from db.refund_approval_service import ApprovalError
from db.refund_execution_service import RefundExecutionService


async def exercise_postgres():
    config = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
    url = URL.create(
        "postgresql+psycopg",
        username=config["POSTGRES_USER"],
        password=config["POSTGRES_PASSWORD"],
        host="127.0.0.1",
        port=int(config.get("POSTGRES_PORT") or 5432),
        database=config["POSTGRES_DB"],
    )
    # No existing table is used: all ORM operations translate to this unique schema.
    schema = f"test_refund_{uuid4().hex}"
    engine = create_async_engine(url, execution_options={"schema_translate_map": {None: schema}})
    factory = async_sessionmaker(engine, expire_on_commit=False)
    created = False
    try:
        async with engine.begin() as connection:
            await connection.execute(CreateSchema(schema))
            created = True
            await connection.run_sync(Base.metadata.create_all)
        async with factory() as session:
            session.add(Order(order_id="O1", status="paid", amount=Decimal("299"), currency="CNY"))
            await session.flush()
            for number in (1, 2):
                session.add(
                    Ticket(
                        ticket_id=f"T{number}",
                        order_id="O1",
                        issue_type="refund",
                        description="isolated test",
                        status="open",
                    )
                )
            await session.flush()
            for number in (1, 2):
                session.add(
                    RefundApproval(
                        approval_id=f"A{number}",
                        ticket_id=f"T{number}",
                        status="approved",
                        reviewer_id="test-staff",
                        reason="test",
                        reviewed_at=datetime.now(UTC),
                    )
                )
            await session.commit()

        async def execute(ticket_id):
            async with factory() as session:
                execution, fresh = await RefundExecutionService(session).execute(
                    ticket_id, "test-staff"
                )
                return execution.execution_id, fresh

        results = await asyncio.gather(execute("T1"), execute("T1"))
        assert results[0][0] == results[1][0]
        assert sum(fresh for _, fresh in results) == 1
        async with factory() as session:
            with pytest.raises(ApprovalError, match="ORDER_ALREADY_SIMULATED"):
                await RefundExecutionService(session).execute("T2", "test-staff")
            assert await session.scalar(select(func.count()).select_from(RefundExecution)) == 1
    finally:
        if created:
            async with engine.begin() as connection:
                await connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        await engine.dispose()


@pytest.mark.docker
def test_postgres_concurrent_execution():
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else asyncio.new_event_loop
    with asyncio.Runner(loop_factory=factory) as runner:
        runner.run(exercise_postgres())


@pytest.fixture(autouse=True)
def trusted_staff_identity():
    token = current_principal.set(Principal('test-staff', 'staff'))
    try:
        yield
    finally:
        current_principal.reset(token)
