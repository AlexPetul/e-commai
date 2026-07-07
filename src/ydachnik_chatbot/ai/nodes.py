from typing import Any

from langchain_core.messages import SystemMessage

from ydachnik_chatbot.ai.llm import product_consultant_llm, support_consultant_llm
from ydachnik_chatbot.ai.prompts import (
    load_customer_support_context,
)
from ydachnik_chatbot.ai.prompts_store.langsmith_store import LangsmithPromptStore
from ydachnik_chatbot.ai.state import AgentState
from ydachnik_chatbot.ai.tools import (
    fetch_page,
    get_category_candidates_for_classification,
    get_price_range,
    load_category_attributes,
    save_product_attributes,
    search_products,
    set_product_category,
)

product_tools = [
    set_product_category,
    get_category_candidates_for_classification,
    load_category_attributes,
    save_product_attributes,
    search_products,
    get_price_range,
]
support_tools = [fetch_page]


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

    system_prompt = await LangsmithPromptStore.retrieve("product_consultant")
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
    system_prompt = await LangsmithPromptStore.retrieve("support_consultant")
    response = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            SystemMessage(content=load_customer_support_context()),
        ]
        + state["messages"]
    )

    return {"messages": [response]}
