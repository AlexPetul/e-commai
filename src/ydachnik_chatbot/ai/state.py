from typing import Literal

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    route: Literal["product_consultant", "support_consultant", "off_topic"] | None
    off_topic_message: str | None
    product_category: str | None
    product_attribute_schema: list[dict] | None
    product_attributes: dict[str, str] | None
