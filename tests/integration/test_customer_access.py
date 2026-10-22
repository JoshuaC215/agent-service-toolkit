from decimal import Decimal

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.responses import JSONResponse, StreamingResponse

from core.settings import settings
from db.access import AccessDenied, Principal, current_principal
from db.models import Base, Order, RefundApproval, Ticket
from db.refund_approval_service import ApprovalError, RefundApprovalService
from db.repositories.order_repository import OrderRepository
from db.repositories.refund_approval_repository import RefundApprovalRepository
from db.repositories.ticket_repository import TicketRepository
from db.session import get_db_session
from service.identity import IdentityMiddleware
from service.identity import router as identity_router
from service.orders import router as orders_router
from service.tickets import router as tickets_router


@pytest_asyncio.fixture
async def access_app(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'access.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        for number, owner in ((1, "alice"), (2, "bob"), (3, None)):
            session.add(
                Order(
                    order_id=f"O{number}",
                    owner_id=owner,
                    status="paid",
                    amount=Decimal("10"),
                    currency="CNY",
                )
            )
        await session.flush()
        for number in (1, 2):
            session.add(
                Ticket(
                    ticket_id=f"T{number}",
                    order_id=f"O{number}",
                    issue_type="refund",
                    description="test",
                    status="open",
                )
            )
        await session.flush()
        for number in (1, 2):
            session.add(
                RefundApproval(approval_id=f"A{number}", ticket_id=f"T{number}", status="pending")
            )
        await session.commit()
    monkeypatch.setattr(
        settings,
        "CUSTOMER_ACCESS_KEYS",
        {
            "alice": SecretStr("a" * 40),
            "bob": SecretStr("b" * 40),
        },
    )
    monkeypatch.setattr(settings, "APPROVAL_REVIEWERS", {"staff": SecretStr("s" * 40)})
    app = FastAPI()
    app.add_middleware(IdentityMiddleware)
    for router in (identity_router, orders_router, tickets_router):
        app.include_router(router)

    @app.exception_handler(AccessDenied)
    async def forbidden(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=403)

    async def session_override():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = session_override
    yield app, factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_api_owner_filter_and_anonymous_denial(access_app):
    app, _ = access_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/orders/O1")).status_code == 401
        alice = {"X-User-Key": "a" * 40, "X-User-Id": "bob"}
        assert (await client.get("/auth/me", headers=alice)).json()["user_id"] == "alice"
        assert (await client.get("/orders/O1", headers=alice)).status_code == 200
        for order in ("O2", "O3", "missing"):
            assert (await client.get(f"/orders/{order}", headers=alice)).status_code == 404
        assert (await client.get("/tickets/T2", headers=alice)).status_code == 404
        assert (await client.get("/tickets/by-order/O2", headers=alice)).status_code == 404
        assert (await client.get("/tickets/T1", headers=alice)).status_code == 200
        staff = {"X-Approval-Key": "s" * 40}
        assert (await client.get("/orders/O3", headers=staff)).status_code == 200


@pytest.mark.asyncio
async def test_customer_cannot_write_status_or_claim_order(access_app):
    app, _ = access_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"X-User-Key": "a" * 40},
    ) as client:
        assert (await client.patch("/orders/O1", json={"status": "refunded"})).status_code == 403
        assert (await client.patch("/tickets/T1", json={"status": "closed"})).status_code == 403
        assert (await client.put("/orders/O3/owner", json={"owner_id": "alice"})).status_code == 403
        response = await client.post(
            "/tickets",
            json={
                "ticket_id": "EVIL",
                "order_id": "O2",
                "issue_type": "refund",
                "description": "test",
            },
        )
        assert response.status_code == 404
        assert (await client.post("/agui/run", json={})).status_code == 403
        assert (await client.post("/chatbot/invoke", json={})).status_code == 403


@pytest.mark.asyncio
async def test_staff_initial_assignment_only(access_app):
    app, _ = access_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"X-Approval-Key": "s" * 40},
    ) as client:
        assert (await client.put("/orders/O3/owner", json={"owner_id": "alice"})).status_code == 200
        assert (await client.put("/orders/O3/owner", json={"owner_id": "bob"})).status_code == 409


@pytest.mark.asyncio
async def test_repository_and_approval_tool_paths_are_scoped(access_app):
    _, factory = access_app
    token = current_principal.set(Principal("alice", "customer"))
    try:
        async with factory() as session:
            assert await OrderRepository(session).get_by_id("O2") is None
            assert await TicketRepository(session).get_by_id("T2") is None
            assert await RefundApprovalRepository(session).get_by_ticket_id("T2") is None
            with pytest.raises(ApprovalError, match="TICKET_NOT_FOUND"):
                await RefundApprovalService(session).submit("T2")
            with pytest.raises(AccessDenied, match="STAFF_REQUIRED"):
                await RefundApprovalService(session).review(
                    "T1", decision="approved", reviewer_id="forged", reason="ok"
                )
    finally:
        current_principal.reset(token)


@pytest.mark.asyncio
async def test_stream_keeps_identity_and_request_context_does_not_leak(access_app):
    app, factory = access_app

    @app.get("/stream-test")
    async def stream():
        async def body():
            async with factory() as session:
                own = await OrderRepository(session).get_by_id("O1")
                other = await OrderRepository(session).get_by_id("O2")
                yield f"{own is not None},{other is not None}"

        return StreamingResponse(body())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/stream-test", headers={"X-User-Key": "a" * 40})
        assert response.text == "True,False"
        assert (await client.get("/orders/O1")).status_code == 401
    assert current_principal.get() is None


@pytest.mark.asyncio
async def test_conversation_cannot_be_claimed_or_read_by_another_user(access_app, monkeypatch):
    _, factory = access_app
    from db import conversation_access

    monkeypatch.setattr(conversation_access, "AsyncSessionLocal", factory)
    token = current_principal.set(Principal("alice", "customer"))
    try:
        await conversation_access.check_conversation("thread1", create=True)
        await conversation_access.check_conversation("thread1")
        current_principal.set(Principal("bob", "customer"))
        for create in (True, False):
            with pytest.raises(AccessDenied, match="CONVERSATION_NOT_FOUND"):
                await conversation_access.check_conversation("thread1", create=create)
    finally:
        current_principal.reset(token)


@pytest.mark.asyncio
async def test_my_orders_api_only_lists_authenticated_owner(access_app):
    app, _ = access_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/orders/me")).status_code == 401
        for key, expected in (("a", "O1"), ("b", "O2")):
            response = await client.get("/orders/me", headers={"X-User-Key": key * 40})
            assert response.status_code == 200
            body = response.json()
            assert [order["order_id"] for order in body["orders"]] == [expected]
            assert body["has_more"] is False
            assert body["limit"] == 20
        assert (
            await client.get("/orders/me", headers={"X-Approval-Key": "s" * 40})
        ).status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["user_id=bob", "owner_id=bob", "limit=101", "offset=-1"])
async def test_my_orders_rejects_owner_override_and_invalid_pagination(access_app, query):
    app, _ = access_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"X-User-Key": "a" * 40},
    ) as client:
        assert (await client.get(f"/orders/me?{query}")).status_code == 422


@pytest.mark.asyncio
async def test_my_orders_stable_pagination_and_empty_page(access_app):
    from datetime import datetime

    app, factory = access_app
    async with factory() as session:
        for order_id in ("O4", "O5"):
            session.add(
                Order(
                    order_id=order_id,
                    owner_id="alice",
                    status="paid",
                    amount=Decimal("20"),
                    currency="CNY",
                    created_at=datetime(2030, 1, 1),
                )
            )
        await session.commit()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"X-User-Key": "a" * 40},
    ) as client:
        pages = [(await client.get(f"/orders/me?limit=1&offset={i}")).json() for i in range(4)]
        assert [page["orders"][0]["order_id"] for page in pages[:3]] == ["O5", "O4", "O1"]
        assert [page["has_more"] for page in pages] == [True, True, False, False]
        assert pages[3]["orders"] == []


@pytest.mark.asyncio
async def test_my_orders_tool_uses_request_identity_and_forbids_user_parameter(
    access_app, monkeypatch
):
    from pydantic import ValidationError

    from agents import order_tools

    _, factory = access_app
    monkeypatch.setattr(order_tools, "AsyncSessionLocal", factory)
    with pytest.raises(ValidationError):
        order_tools.list_my_orders.args_schema.model_validate({"user_id": "bob"})
    assert set(order_tools.list_my_orders.args_schema.model_fields) == {"limit", "offset"}
    token = current_principal.set(Principal("alice", "customer"))
    try:
        result = await order_tools.list_my_orders.coroutine()
        assert result["success"] is True
        assert [order["order_id"] for order in result["orders"]] == ["O1"]
        current_principal.set(Principal("bob", "customer"))
        result = await order_tools.list_my_orders.coroutine()
        assert [order["order_id"] for order in result["orders"]] == ["O2"]
        current_principal.set(None)
        result = await order_tools.list_my_orders.coroutine()
        assert result == {"success": False, "error": "LOGIN_REQUIRED"}
    finally:
        current_principal.reset(token)
