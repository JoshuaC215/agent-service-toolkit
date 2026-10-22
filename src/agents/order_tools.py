import logging

from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from db.access import AccessDenied
from db.repositories.order_repository import OrderRepository
from db.session import AsyncSessionLocal
from schema.orders import MyOrdersQuery

logger = logging.getLogger(__name__)


@tool(args_schema=MyOrdersQuery)
async def list_my_orders(limit: int = 20, offset: int = 0) -> dict:
    """List the logged-in customer's orders, newest first. No order ID is needed.

    Identity is determined by the backend, never supplied by the model or user.
    Use limit/offset for pagination and has_more to know whether more exist.
    This read-only tool never creates tickets or submits refund applications.
    """
    try:
        async with AsyncSessionLocal() as session:
            orders, has_more = await OrderRepository(session).list_mine(limit=limit, offset=offset)
            return {
                "success": True,
                "orders": [
                    {
                        "order_id": order.order_id,
                        "status": order.status,
                        "amount": str(order.amount),
                        "currency": order.currency,
                    }
                    for order in orders
                ],
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
            }
    except AccessDenied as exc:
        return {"success": False, "error": str(exc)}
    except SQLAlchemyError as exc:
        logger.warning("Order listing unavailable (%s)", type(exc).__name__)
        return {"success": False, "error": "ORDER_LIST_UNAVAILABLE"}


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
