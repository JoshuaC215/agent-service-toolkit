
import asyncio
import selectors
import sys

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import agents.order_tools as order_tools
from agents.order_tools import get_order
from db.session import build_database_url


def make_test_loop():
    """Use a SelectorEventLoop on Windows for async Psycopg."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )

    return asyncio.new_event_loop()


@pytest.fixture(scope="module")
def selector_runner():
    # Both tests share one event loop. This also avoids
    # switching loops while reusing the database engine.
    with asyncio.Runner(
        loop_factory=make_test_loop
    ) as runner:
        yield runner


@pytest.fixture(autouse=True)
def local_order_sessions(monkeypatch, selector_runner):
    url = build_database_url()
    # Compose service names resolve inside containers, not on the Windows host.
    if url.host == "postgres":
        url = url.set(host="127.0.0.1")
    engine = create_async_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
    monkeypatch.setattr(
        order_tools,
        "AsyncSessionLocal",
        async_sessionmaker(engine, expire_on_commit=False),
    )
    try:
        yield
    finally:
        selector_runner.run(engine.dispose())


def test_order_exists(selector_runner):
    result = selector_runner.run(
        get_order.ainvoke({"order_id": "ORD001"})
    )

    assert result["found"] is True
    assert result["order"]["status"] == "paid"
    assert result["order"]["amount"] == "299.00"


def test_order_not_found(selector_runner):
    result = selector_runner.run(
        get_order.ainvoke({"order_id": "ORD999"})
    )

    assert result["found"] is False
    assert result["error"] == "ORDER_NOT_FOUND"
