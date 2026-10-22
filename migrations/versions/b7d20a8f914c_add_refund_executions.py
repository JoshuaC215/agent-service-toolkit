"""Add full-order simulated refund execution records.

Revision ID: b7d20a8f914c
Revises: 98a319d076ae
"""

import sqlalchemy as sa
from alembic import op

revision = "b7d20a8f914c"
down_revision = "98a319d076ae"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "refund_executions",
        sa.Column("execution_id", sa.String(32), primary_key=True),
        sa.Column(
            "approval_id",
            sa.String(32),
            sa.ForeignKey("refund_approvals.approval_id"),
            nullable=False,
        ),
        sa.Column("order_id", sa.String(32), sa.ForeignKey("orders.order_id"), nullable=False),
        sa.Column("amount", sa.Numeric(10, 2), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("executor_id", sa.String(64), nullable=False),
        sa.Column("result_message", sa.Text(), nullable=False),
        sa.Column(
            "executed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("approval_id", name="uq_refund_executions_approval"),
        sa.UniqueConstraint("order_id", name="uq_refund_executions_order"),
        sa.CheckConstraint("amount > 0", name="ck_refund_executions_amount"),
        sa.CheckConstraint("status = 'simulated_succeeded'", name="ck_refund_executions_status"),
        sa.CheckConstraint("length(trim(executor_id)) > 0", name="ck_refund_executions_executor"),
    )


def downgrade():
    op.drop_table("refund_executions")
