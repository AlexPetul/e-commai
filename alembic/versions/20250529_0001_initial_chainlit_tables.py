"""initial chainlit tables

Revision ID: 0001
Revises:
Create Date: 2025-05-29

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("identifier", sa.Text, nullable=False, unique=True),
        sa.Column("createdAt", sa.Text, nullable=True),
        sa.Column("metadata", sa.Text, nullable=True),
    )

    op.create_table(
        "threads",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("createdAt", sa.Text, nullable=True),
        sa.Column("name", sa.Text, nullable=True),
        sa.Column("userId", sa.Text, nullable=True),
        sa.Column("userIdentifier", sa.Text, nullable=True),
        sa.Column("tags", sa.Text, nullable=True),
        sa.Column("metadata", sa.Text, nullable=True),
    )

    op.create_table(
        "steps",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=True),
        sa.Column("type", sa.Text, nullable=True),
        sa.Column("threadId", sa.Text, nullable=False),
        sa.Column("parentId", sa.Text, nullable=True),
        sa.Column("streaming", sa.Boolean, nullable=True),
        sa.Column("waitForAnswer", sa.Boolean, nullable=True),
        sa.Column("isError", sa.Boolean, nullable=True),
        sa.Column("metadata", sa.Text, nullable=True),
        sa.Column("tags", sa.Text, nullable=True),
        sa.Column("input", sa.Text, nullable=True),
        sa.Column("output", sa.Text, nullable=True),
        sa.Column("createdAt", sa.Text, nullable=True),
        sa.Column("start", sa.Text, nullable=True),
        sa.Column("end", sa.Text, nullable=True),
        sa.Column("generation", sa.Text, nullable=True),
        sa.Column("showInput", sa.Text, nullable=True),
        sa.Column("language", sa.Text, nullable=True),
        sa.Column("defaultOpen", sa.Boolean, nullable=True),
        sa.Column("autoCollapse", sa.Boolean, nullable=True),
        sa.Column("command", sa.Text, nullable=True),
        sa.Column("modes", sa.Text, nullable=True),
        sa.Column("icon", sa.Text, nullable=True),
    )

    op.create_table(
        "feedbacks",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("forId", sa.Text, nullable=False),
        sa.Column("value", sa.SmallInteger, nullable=False),
        sa.Column("comment", sa.Text, nullable=True),
    )

    op.create_table(
        "elements",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("threadId", sa.Text, nullable=True),
        sa.Column("type", sa.Text, nullable=True),
        sa.Column("chainlitKey", sa.Text, nullable=True),
        sa.Column("url", sa.Text, nullable=True),
        sa.Column("objectKey", sa.Text, nullable=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("display", sa.Text, nullable=True),
        sa.Column("size", sa.Text, nullable=True),
        sa.Column("language", sa.Text, nullable=True),
        sa.Column("page", sa.Integer, nullable=True),
        sa.Column("props", sa.Text, nullable=True),
        sa.Column("autoPlay", sa.Boolean, nullable=True),
        sa.Column("playerConfig", sa.Text, nullable=True),
        sa.Column("forId", sa.Text, nullable=True),
        sa.Column("mime", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("elements")
    op.drop_table("feedbacks")
    op.drop_table("steps")
    op.drop_table("threads")
    op.drop_table("users")
