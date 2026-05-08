"""Add price, entry_price, stop_loss, take_profit to signals

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("signals", sa.Column("price", sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("entry_price", sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("stop_loss", sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("take_profit", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("signals") as batch_op:
        batch_op.drop_column("take_profit")
        batch_op.drop_column("stop_loss")
        batch_op.drop_column("entry_price")
        batch_op.drop_column("price")
