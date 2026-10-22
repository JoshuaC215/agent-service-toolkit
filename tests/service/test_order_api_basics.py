
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

import service.orders as orders_api
from schema.orders import OrderCreate, OrderResponse, OrderStatusUpdate


def make_order():
    """Create fake order data without accessing PostgreSQL."""
    now = datetime.now(timezone.utc)

    return SimpleNamespace(
        order_id="ORD001",
        status="paid",
        amount=Decimal("299.00"),
        currency="CNY",
        created_at=now,
        updated_at=now,
    )


def test_order_create_valid():
    order = OrderCreate(
        order_id="ORD004",
        status="pending",
        amount="129.90",
        currency="CNY",
    )

    assert order.amount == Decimal("129.90")


@pytest.mark.parametrize("amount", ["-1.00", "12.345"])
def test_order_create_invalid_amount(amount):
    with pytest.raises(ValidationError):
        OrderCreate(
            order_id="ORD004",
            status="pending",
            amount=amount,
            currency="CNY",
        )


def test_order_status_cannot_be_empty():
    with pytest.raises(ValidationError):
        OrderStatusUpdate(status="")


def test_order_response_from_object():
    response = OrderResponse.model_validate(make_order())

    assert response.order_id == "ORD001"
    assert response.amount == Decimal("299.00")
    assert '"299.00"' in response.model_dump_json()


@pytest.fixture
def client(monkeypatch):
    class FakeRepository:
        def __init__(self, session):
            self.session = session

        async def get_by_id(self, order_id):
            if order_id == "ORD001":
                return make_order()
            return None

    async def fake_db_session():
        yield object()

    monkeypatch.setattr(
        orders_api,
        "OrderRepository",
        FakeRepository,
    )

    app = FastAPI()
    app.include_router(orders_api.router)
    app.dependency_overrides[
        orders_api.get_db_session
    ] = fake_db_session

    with TestClient(app) as test_client:
        yield test_client


def test_get_order_success(client):
    response = client.get("/orders/ORD001")

    assert response.status_code == 200
    assert response.json()["order_id"] == "ORD001"
    assert response.json()["amount"] == "299.00"


def test_get_order_not_found(client):
    response = client.get("/orders/ORD999")

    assert response.status_code == 404
    assert response.json()["detail"] == "ORDER_NOT_FOUND"