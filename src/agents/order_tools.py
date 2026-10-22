
from langchain_core.tools import tool

from db.repositories.order_repository import OrderRepository
from db.session import AsyncSessionLocal


@tool
async def get_order(order_id: str) -> dict:
    """Query an order by its exact order ID from PostgreSQL."""

    async with AsyncSessionLocal() as session:
        repository = OrderRepository(session)

        order = await repository.get_by_id(order_id)

        if order is None:
            return {
                "found": False,
                "error": "ORDER_NOT_FOUND",
            }

        return {
            "found": True,
            "order": {
                "order_id": order.order_id,
                "status": order.status,
                "amount": str(order.amount),
                "currency": order.currency,
            },
        }