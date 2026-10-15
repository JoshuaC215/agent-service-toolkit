from langchain_core.tools import tool


MOCK_ORDERS = {
    "ORD001": {
        "order_id": "ORD001",
        "status": "paid",
        "amount": 299.0,
        "currency": "CNY",
    },
    "ORD002": {
        "order_id": "ORD002",
        "status": "shipped",
        "amount": 599.0,
        "currency": "CNY",
    },
}


@tool
def get_order(order_id: str) -> dict:
    """Query an order by its exact order ID."""

    order = MOCK_ORDERS.get(order_id)

    if order is None:
        return {
            "found": False,
            "error": "ORDER_NOT_FOUND",
        }

    return {
        "found": True,
        "order": order,
    }