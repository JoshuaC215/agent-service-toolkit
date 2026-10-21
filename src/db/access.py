"""Request identity, set only by trusted authentication middleware."""

from contextvars import ContextVar
from dataclasses import dataclass

from sqlalchemy import false, select

from db.models import Order, Ticket


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str


current_principal: ContextVar[Principal | None] = ContextVar("principal", default=None)


class AccessDenied(PermissionError):
    pass


def principal() -> Principal:
    identity = current_principal.get()
    if identity is None:
        raise AccessDenied("LOGIN_REQUIRED")
    return identity


def require_staff():
    if principal().role != "staff":
        raise AccessDenied("STAFF_REQUIRED")


def order_scope():
    identity = current_principal.get()
    if identity is None:
        return false()
    if identity.role == "staff":
        from sqlalchemy import true

        return true()
    return Order.owner_id == identity.user_id


def ticket_scope():
    return Ticket.order_id.in_(select(Order.order_id).where(order_scope()))
