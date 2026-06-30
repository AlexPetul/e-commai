"""product table

Revision ID: 0003
Revises: 0002
Create Date: 2025-05-31

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "product",
        sa.Column("url", sa.String(length=500), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("tech_specs", sa.Text(), nullable=False, server_default=""),
        sa.Column("price", sa.String(length=100), nullable=False, server_default=""),
        sa.Column("currency", sa.String(length=20), nullable=False, server_default=""),
        sa.Column("category_name", sa.Text(), nullable=False),
        sa.Column("image", sa.Text(), nullable=True),
        sa.Column("attributes", sa.Text(), nullable=False, server_default=""),
        sa.ForeignKeyConstraint(
            ["category_name"],
            ["product_categories.name"],
        ),
    )
    op.create_index(op.f("ix_product_category_name"), "product", ["category_name"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_product_category_name"), table_name="product")
    op.drop_table("product")
