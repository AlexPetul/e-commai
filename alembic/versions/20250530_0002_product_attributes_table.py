"""product_categories table

Revision ID: 0002
Revises: 0001
Create Date: 2025-05-30

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "product_categories",
        sa.Column("name", sa.Text, primary_key=True),
        sa.Column("attributes", JSONB, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("product_categories")
