"""Add order and conversation ownership without assigning historical data."""

import sqlalchemy as sa
from alembic import op

revision = "d2e681c4a907"
down_revision = "b7d20a8f914c"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("orders", sa.Column("owner_id", sa.String(64), nullable=True))
    op.create_index("ix_orders_owner_id", "orders", ["owner_id"])
    op.create_table(
        "conversation_owners",
        sa.Column("thread_id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
    )
    op.create_index("ix_conversation_owners_user_id", "conversation_owners", ["user_id"])


def downgrade():
    op.drop_index("ix_conversation_owners_user_id", table_name="conversation_owners")
    op.drop_table("conversation_owners")
    op.drop_index("ix_orders_owner_id", table_name="orders")
    op.drop_column("orders", "owner_id")
