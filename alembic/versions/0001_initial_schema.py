"""Initial schema — assets, signals, agent_predictions, agent_performance

Revision ID: 0001
Revises:
Create Date: 2026-05-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── market_assets ──
    op.create_table(
        "market_assets",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("symbol", sa.String(20), unique=True, nullable=False, index=True),
        sa.Column("asset_type", sa.String(10), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("exchange", sa.String(50), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── signals ──
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("asset_id", sa.String(32), sa.ForeignKey("market_assets.id"), nullable=False, index=True),
        sa.Column("direction", sa.String(10), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("meta_agent_version", sa.String(20), server_default="0.2.1"),
        sa.Column("agent_weights", sa.JSON(), nullable=True),
        sa.Column("consensus_dispersion", sa.Float(), nullable=True),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── agent_predictions ──
    op.create_table(
        "agent_predictions",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("signal_id", sa.Integer(), sa.ForeignKey("signals.id"), nullable=False, index=True),
        sa.Column("agent_name", sa.String(100), nullable=False, index=True),
        sa.Column("agent_id", sa.Integer(), nullable=False),
        sa.Column("tier", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("vote", sa.String(10), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── agent_performance ──
    op.create_table(
        "agent_performance",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("agent_name", sa.String(100), unique=True, nullable=False, index=True),
        sa.Column("agent_id", sa.Integer(), unique=True, nullable=False),
        sa.Column("tier", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("accuracy_ema", sa.Float(), server_default="0.5"),
        sa.Column("total_predictions", sa.Integer(), server_default="0"),
        sa.Column("correct_predictions", sa.Integer(), server_default="0"),
        sa.Column("weight", sa.Float(), server_default="0.0"),
        sa.Column("last_updated", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("agent_performance")
    op.drop_table("agent_predictions")
    op.drop_table("signals")
    op.drop_table("market_assets")
