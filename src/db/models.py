
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
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