from sqlalchemy import Boolean, Integer, SmallInteger, Text
from sqlalchemy.orm import Mapped, mapped_column

from ydachnik_chatbot.infrastructure.db.models.base import Base


class ChainlitUser(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    identifier: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    created_at: Mapped[str | None] = mapped_column("createdAt", Text)
    metadata_: Mapped[str | None] = mapped_column("metadata", Text)


class ChainlitThread(Base):
    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[str | None] = mapped_column("createdAt", Text)
    name: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[str | None] = mapped_column("userId", Text)
    user_identifier: Mapped[str | None] = mapped_column("userIdentifier", Text)
    tags: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[str | None] = mapped_column("metadata", Text)


class ChainlitStep(Base):
    __tablename__ = "steps"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    type: Mapped[str | None] = mapped_column(Text)
    thread_id: Mapped[str] = mapped_column("threadId", Text, nullable=False)
    parent_id: Mapped[str | None] = mapped_column("parentId", Text)
    streaming: Mapped[bool | None] = mapped_column(Boolean)
    wait_for_answer: Mapped[bool | None] = mapped_column("waitForAnswer", Boolean)
    is_error: Mapped[bool | None] = mapped_column("isError", Boolean)
    metadata_: Mapped[str | None] = mapped_column("metadata", Text)
    tags: Mapped[str | None] = mapped_column(Text)
    input: Mapped[str | None] = mapped_column(Text)
    output: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str | None] = mapped_column("createdAt", Text)
    start: Mapped[str | None] = mapped_column(Text)
    end: Mapped[str | None] = mapped_column(Text)
    generation: Mapped[str | None] = mapped_column(Text)
    show_input: Mapped[str | None] = mapped_column("showInput", Text)
    language: Mapped[str | None] = mapped_column(Text)
    default_open: Mapped[bool | None] = mapped_column("defaultOpen", Boolean)
    auto_collapse: Mapped[bool | None] = mapped_column("autoCollapse", Boolean)
    command: Mapped[str | None] = mapped_column(Text)
    modes: Mapped[str | None] = mapped_column(Text)
    icon: Mapped[str | None] = mapped_column(Text)


class ChainlitFeedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    for_id: Mapped[str] = mapped_column("forId", Text, nullable=False)
    value: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)


class ChainlitElement(Base):
    __tablename__ = "elements"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    thread_id: Mapped[str | None] = mapped_column("threadId", Text)
    type: Mapped[str | None] = mapped_column(Text)
    chainlit_key: Mapped[str | None] = mapped_column("chainlitKey", Text)
    url: Mapped[str | None] = mapped_column(Text)
    object_key: Mapped[str | None] = mapped_column("objectKey", Text)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    display: Mapped[str | None] = mapped_column(Text)
    size: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)
    props: Mapped[str | None] = mapped_column(Text)
    auto_play: Mapped[bool | None] = mapped_column("autoPlay", Boolean)
    player_config: Mapped[str | None] = mapped_column("playerConfig", Text)
    for_id: Mapped[str | None] = mapped_column("forId", Text)
    mime: Mapped[str | None] = mapped_column(Text)
