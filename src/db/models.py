
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

class Base(DeclarativeBase):
    pass

class Order(Base):
    __tablename__ = "orders"

    __table_args__ = (
        CheckConstraint(
            "amount >= 0",
            name="ck_orders_amount_non_negative",
        ),
    )

    order_id: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    tickets: Mapped[list["Ticket"]] = relationship(
        back_populates="order",
    )

class Ticket(Base):
    __tablename__ = "tickets"

    ticket_id: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
    )

    order_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey(
            "orders.order_id",
            name="fk_tickets_order_id_orders",
        ),
        nullable=False,
        index=True,
    )

    issue_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="open",
        server_default=text("'open'"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    order: Mapped["Order"] = relationship(
        back_populates="tickets",
    )

    refund_approval: Mapped["RefundApproval | None"] = relationship(
        back_populates="ticket",
        uselist=False,
    )

class RefundApproval(Base):
    __tablename__ = "refund_approvals"

    __table_args__ = (
        UniqueConstraint(
            "ticket_id",
            name="uq_refund_approvals_ticket_id",
        ),
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected')",
            name="ck_refund_approvals_status",
        ),
        CheckConstraint(
            """
            (
                status = 'pending'
                AND reviewer_id IS NULL
                AND reviewed_at IS NULL
                AND reason IS NULL
            )
            OR
            (
                status IN ('approved', 'rejected')
                AND reviewer_id IS NOT NULL
                AND length(trim(reviewer_id)) > 0
                AND reviewed_at IS NOT NULL
                AND reason IS NOT NULL
                AND length(trim(reason)) > 0
            )
            """,
            name="ck_refund_approvals_review_fields",
        ),
    )

    approval_id: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
    )

    ticket_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey(
            "tickets.ticket_id",
            name="fk_refund_approvals_ticket_id_tickets",
        ),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="pending",
        server_default=text("'pending'"),
    )

    reviewer_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    ticket: Mapped["Ticket"] = relationship(
        back_populates="refund_approval",
    )