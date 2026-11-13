"""Keep originating trace context with retained memories.

The columns move with memory invalidation/restoration and are removed when the
memory is deleted. They contain identifiers only and outlast request-history
retention, so delayed consolidation can link to each contributing retain.

Revision ID: a8c3e7f1b942
Revises: k7b1c2d3e4f5
Create Date: 2026-11-13
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a8c3e7f1b942"
down_revision: str | Sequence[str] | None = "k7b1c2d3e4f5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    for table in ("memory_units", "invalidated_memory_units"):
        op.execute(
            f"ALTER TABLE {_pg_schema_prefix()}{table} "
            "ADD COLUMN origin_traceparent TEXT, ADD COLUMN origin_tracestate TEXT"
        )


def _pg_downgrade() -> None:
    for table in ("memory_units", "invalidated_memory_units"):
        op.execute(
            f"ALTER TABLE {_pg_schema_prefix()}{table} DROP COLUMN origin_traceparent, DROP COLUMN origin_tracestate"
        )


def _oracle_upgrade() -> None:
    for table in ("memory_units", "invalidated_memory_units"):
        op.execute(f"ALTER TABLE {table} ADD (origin_traceparent VARCHAR2(55), origin_tracestate CLOB)")


def _oracle_downgrade() -> None:
    for table in ("memory_units", "invalidated_memory_units"):
        op.execute(f"ALTER TABLE {table} DROP (origin_traceparent, origin_tracestate)")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
