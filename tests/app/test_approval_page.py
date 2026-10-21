from unittest.mock import Mock

from streamlit.testing.v1 import AppTest


def test_login_empty_queue_and_logout(monkeypatch):
    client = Mock()
    client.pending.return_value = []
    monkeypatch.setattr("client.approval_page.ApprovalClient", Mock(return_value=client))
    app = AppTest.from_string(
        "from client.approval_page import draw_approval_page\ndraw_approval_page('http://test')"
    ).run(timeout=15)
    assert not app.exception
    client.pending.assert_not_called()
    app.sidebar.text_input[0].set_value("test-key").run(timeout=15)
    assert not app.exception
    assert len(app.tabs) == 3
    client.pending.assert_called_once()
    app.sidebar.button[0].click().run(timeout=15)
    assert not app.exception
    assert app.sidebar.text_input[0].value == ""


def test_pending_review_requires_reason_and_confirmation(monkeypatch):
    client = Mock()
    client.pending.return_value = [{"ticket_id": "T1", "created_at": "2026-10-21T09:00:00Z"}]
    client.ticket.return_value = {
        "ticket_id": "T1",
        "order_id": "O1",
        "status": "open",
        "issue_type": "refund",
        "description": "Evidence missing",
    }
    client.order.return_value = {"status": "paid", "amount": "10.00", "currency": "CNY"}
    monkeypatch.setattr("client.approval_page.ApprovalClient", Mock(return_value=client))
    app = AppTest.from_string(
        "from client.approval_page import draw_approval_page\ndraw_approval_page('http://test')"
    ).run(timeout=15)
    app.sidebar.text_input[0].set_value("test-key").run(timeout=15)
    assert not app.exception
    next(button for button in app.button if button.label == "提交审批决定").click().run(timeout=15)
    client.review.assert_not_called()
    assert app.warning
    client.review.return_value = {"status": "approved", "reviewer_id": "staff-1"}
    client.pending.return_value = []
    app.text_area[0].set_value("Evidence checked")
    app.checkbox[0].check()
    # Keep the record available for the submission rerun, then refresh the queue.
    client.pending.side_effect = [
        [{"ticket_id": "T1", "created_at": "2026-10-21T09:00:00Z"}], []
    ]
    next(button for button in app.button if button.label == "提交审批决定").click().run(timeout=15)
    assert not app.exception
    client.review.assert_called_once_with("T1", "approved", "Evidence checked")
    assert any("staff-1" in message.value for message in app.success)
