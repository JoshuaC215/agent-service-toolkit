import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.settings import settings
from db.refund_approval_service import ApprovalError, RefundApprovalService
from db.refund_execution_service import RefundExecutionService
from db.repositories.refund_approval_repository import RefundApprovalRepository
from db.session import get_db_session
from schema.refund_approvals import ApprovalDecision, ApprovalResponse
from schema.refund_executions import ExecutionResponse, SimulationRequest, SimulationResponse


async def require_reviewer(
    key: Annotated[str | None, Header(alias="X-Approval-Key")] = None,
) -> str:
    """Resolve a reviewer from server-owned credentials, never request identity."""
    credentials = settings.APPROVAL_REVIEWERS
    values = [secret.get_secret_value() for secret in credentials.values()]
    if (
        not credentials
        or any(not name.strip() or len(name) > 64 for name in credentials)
        or any(len(value) < 32 for value in values)
        or len(set(values)) != len(values)
    ):
        raise HTTPException(503, "APPROVAL_AUTH_NOT_CONFIGURED")
    if key is not None:
        for name, secret in credentials.items():
            if secrets.compare_digest(key.encode(), secret.get_secret_value().encode()):
                return name
    raise HTTPException(401, "INVALID_APPROVAL_CREDENTIALS")


Reviewer = Annotated[str, Depends(require_reviewer)]
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
router = APIRouter(
    prefix="/refund-approvals", tags=["refund-approvals"], dependencies=[Depends(require_reviewer)]
)


def approval_http_error(exc: ApprovalError) -> HTTPException:
    code = str(exc)
    if code in ("TICKET_NOT_FOUND", "APPROVAL_NOT_FOUND", "ORDER_NOT_FOUND", "EXECUTION_NOT_FOUND"):
        return HTTPException(404, code)
    if code in (
        "TICKET_NOT_ACTIVE",
        "APPROVAL_ALREADY_EXISTS",
        "APPROVAL_ALREADY_REVIEWED",
        "APPROVAL_NOT_APPROVED",
        "ORDER_NOT_REFUNDABLE",
        "ORDER_ALREADY_SIMULATED",
    ):
        return HTTPException(409, code)
    return HTTPException(422, code)


@router.get("/pending", response_model=list[ApprovalResponse])
async def list_pending_approvals(session: DbSession):
    return await RefundApprovalRepository(session).list_pending()


@router.post("/by-ticket/{ticket_id}", response_model=ApprovalResponse, status_code=201)
async def submit_approval(ticket_id: str, session: DbSession):
    try:
        return await RefundApprovalService(session).submit(ticket_id)
    except ApprovalError as exc:
        raise approval_http_error(exc) from exc


@router.get("/by-ticket/{ticket_id}", response_model=ApprovalResponse)
async def get_approval(ticket_id: str, session: DbSession):
    approval = await RefundApprovalRepository(session).get_by_ticket_id(ticket_id)
    if approval is None:
        raise HTTPException(404, "APPROVAL_NOT_FOUND")
    return approval


@router.post("/by-ticket/{ticket_id}/review", response_model=ApprovalResponse)
async def review_approval(
    ticket_id: str, payload: ApprovalDecision, session: DbSession, reviewer: Reviewer
):
    try:
        return await RefundApprovalService(session).review(
            ticket_id, decision=payload.decision, reviewer_id=reviewer, reason=payload.reason
        )
    except ApprovalError as exc:
        raise approval_http_error(exc) from exc


@router.post("/by-ticket/{ticket_id}/simulate-refund", response_model=SimulationResponse)
async def simulate_refund(
    ticket_id: str, payload: SimulationRequest, session: DbSession, reviewer: Reviewer
):
    """Record an atomic full-order simulation. Never calls a payment provider."""
    try:
        execution, created = await RefundExecutionService(session).execute(ticket_id, reviewer)
        return SimulationResponse(
            execution=ExecutionResponse.model_validate(execution), created=created
        )
    except ApprovalError as exc:
        raise approval_http_error(exc) from exc


@router.get("/by-ticket/{ticket_id}/execution", response_model=ExecutionResponse)
async def get_refund_execution(ticket_id: str, session: DbSession):
    try:
        execution = await RefundExecutionService(session).get(ticket_id)
        if execution is None:
            raise HTTPException(404, "EXECUTION_NOT_FOUND")
        return execution
    except ApprovalError as exc:
        raise approval_http_error(exc) from exc
