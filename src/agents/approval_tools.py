"""Customer-facing submission and status tools; no review capability."""

import logging

from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from db.refund_approval_service import ApprovalError, RefundApprovalService
from db.repositories.refund_approval_repository import RefundApprovalRepository
from db.session import AsyncSessionLocal
from schema.refund_approvals import ApprovalResponse

logger = logging.getLogger(__name__)


def _valid_ticket_id(ticket_id: str) -> bool:
    return 1 <= len(ticket_id) <= 32


@tool
async def submit_refund_approval(ticket_id: str) -> dict:
    """Submit an existing refund ticket for human review only on explicit user request.

    Confirm the exact ticket and refund intent first. Policy consultation and
    ticket creation alone do not authorize submission. Never approve or execute
    a refund. Duplicate submission must use the existing approval, not a new ticket.
    """
    ticket_id = ticket_id.strip()
    if not _valid_ticket_id(ticket_id):
        return {"success": False, "error": "INVALID_TICKET_ID"}
    try:
        async with AsyncSessionLocal() as session:
            approval = await RefundApprovalService(session).submit(ticket_id)
            return {
                "success": True,
                "approval": ApprovalResponse.model_validate(approval).model_dump(mode="json"),
                "refund_executed": False,
            }
    except ApprovalError as exc:
        return {"success": False, "error": str(exc)}
    except SQLAlchemyError as exc:
        logger.warning("Approval submission failed (%s)", type(exc).__name__)
        return {
            "success": False,
            "error": "APPROVAL_SUBMISSION_UNCONFIRMED",
            "message": "Query approval status before retrying; submission may have committed.",
        }


@tool
async def get_refund_approval(ticket_id: str) -> dict:
    """Read the human approval status for an exact ticket ID without modifying data.

    Pending means awaiting a decision; approved does not mean a refund was
    executed. Missing approval does not mean rejected. Never ask for staff keys.
    """
    ticket_id = ticket_id.strip()
    if not _valid_ticket_id(ticket_id):
        return {"found": False, "error": "INVALID_TICKET_ID"}
    try:
        async with AsyncSessionLocal() as session:
            approval = await RefundApprovalRepository(session).get_by_ticket_id(ticket_id)
            if approval is None:
                return {"found": False, "error": "APPROVAL_NOT_FOUND"}
            return {
                "found": True,
                "approval": ApprovalResponse.model_validate(approval).model_dump(mode="json"),
                "refund_execution_status": "not_checked",
            }
    except SQLAlchemyError as exc:
        logger.warning("Approval lookup failed (%s)", type(exc).__name__)
        return {"found": False, "error": "APPROVAL_LOOKUP_UNAVAILABLE"}
