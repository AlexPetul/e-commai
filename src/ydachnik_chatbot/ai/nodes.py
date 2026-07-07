from typing import Any

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from ydachnik_chatbot.ai.category_candidates import get_nearest_category_candidates
from ydachnik_chatbot.ai.llm import (
    category_classifier_llm,
    product_consultant_llm,
    support_consultant_llm,
)
from ydachnik_chatbot.ai.prompts import (
    SUPPORT_SYSTEM_PROMPT,
    get_product_category_prompt,
    get_product_system_prompt,
    load_customer_support_context,
)
from ydachnik_chatbot.ai.prompts_store.langsmith_store import langsmith_prompt_store
from ydachnik_chatbot.ai.state import AgentState
from ydachnik_chatbot.ai.tools import (
    fetch_page_main_block,
    load_category_attributes,
    save_product_attributes,
    search_products,
)
from ydachnik_chatbot.ai.utils import get_human_system_messages, get_latest_human_message_text

product_tools = [load_category_attributes, save_product_attributes, search_products]
support_tools = [fetch_page_main_block]


class ProductCategory(BaseModel):
    category: str | None = None


class ProductAttributes(BaseModel):
    attributes: dict[str, str] = Field(default_factory=dict)


async def product_category_classifier_node(state: AgentState) -> dict:
    messages = get_human_system_messages(state)
    latest_query = get_latest_human_message_text(state)
    candidate_categories = await get_nearest_category_candidates(latest_query)

    llm = category_classifier_llm.with_structured_output(ProductCategory)
    result: ProductCategory = await llm.ainvoke(
        [
            SystemMessage(
                content=get_product_category_prompt(
                    state.get("product_category"),
                )
            ),
            SystemMessage(
                content=(
                    "Candidate categories returned by the classifier helper tool:\n"
                    + "\n".join(f"- {category}" for category in candidate_categories)
                    if candidate_categories
                    else "Candidate categories returned by the classifier helper tool: none"
                )
            ),
            *messages,
        ]
    )
    selected_category = result.category
    updates: dict[str, Any] = {"product_category": selected_category}

    if selected_category != state.get("product_category"):
        updates["product_attribute_schema"] = None
        updates["product_attributes"] = None

    return updates


async def product_consultant_node(state: AgentState) -> dict[str, list[Any]]:
    schema: list[str] | None = state.get("product_attribute_schema")
    attributes: dict[str, str] | None = state.get("product_attributes")
    category = state.get("product_category")

    context_lines = [
        f"- product_category: {category or 'null'}",
        f"- product_attribute_schema: {schema if schema else 'NOT_LOADED'}",
    ]
    if attributes:
        context_lines.append(f"- product_attributes: {attributes}")

    system_prompt = await langsmith_prompt_store.retrieve("product_consultant")
    system_messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content="Current state:\n" + "\n".join(context_lines)),
    ]

    llm = product_consultant_llm.bind_tools(product_tools)
    response = await llm.ainvoke(system_messages + state["messages"])
    return {"messages": [response]}


async def support_consultant_node(state: AgentState) -> dict[str, list[Any]]:
    """LangGraph node responsible for support-agent reasoning (sub-agent).

    This node is called via routing rules.
    """
    llm = support_consultant_llm.bind_tools(support_tools)
    system_prompt = await langsmith_prompt_store.retrieve("support_consultant")
    response = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            SystemMessage(content=load_customer_support_context()),
        ]
        + state["messages"]
    )

    return {"messages": [response]}
