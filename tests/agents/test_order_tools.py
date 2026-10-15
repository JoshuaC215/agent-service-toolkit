from agents.order_tools import get_order


def test_order_exists():
    result = get_order.invoke({"order_id": "ORD001"})

    assert result["found"] is True
    assert result["order"]["status"] == "paid"
    assert result["order"]["amount"] == 299.0


def test_order_not_found():
    result = get_order.invoke({"order_id": "ORD999"})

    assert result["found"] is False
    assert result["error"] == "ORDER_NOT_FOUND"