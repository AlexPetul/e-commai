"""product price float

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-09

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "product",
        "price",
        server_default=None,
    )

    op.alter_column(
        "product",
        "price",
        existing_type=sa.String(length=100),
        type_=sa.Numeric(10, 2),
        existing_nullable=False,
        postgresql_using="""
            CASE
                WHEN price = '' THEN 0
                ELSE price::numeric
            END
        """,
    )


def downgrade() -> None:
    op.alter_column(
        "product",
        "price",
        existing_type=sa.Float(),
        type_=sa.String(length=100),
        existing_nullable=False,
        server_default="",
        postgresql_using="price::text",
    )
