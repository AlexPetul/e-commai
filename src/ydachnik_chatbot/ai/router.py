from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import END
from pydantic import BaseModel, Field

from ydachnik_chatbot.ai.llm import intent_router_llm
from ydachnik_chatbot.ai.prompts import INTENT_ROUTER_PROMPT
from ydachnik_chatbot.ai.state import AgentState


class RouteDecision(BaseModel):
    route: Literal["product_consultant", "support_consultant", "off_topic"] = Field(...)
    message: str | None = None


def _router_messages(state: AgentState) -> list:
    # Keep only plain human messages and AI text replies (no tool calls / tool results),
    # so the router never receives a dangling ToolMessage or an AIMessage with tool_calls.
    return [
        m
        for m in state["messages"]
        if isinstance(m, HumanMessage)
        or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))
    ][-4:]


async def intent_router(state: AgentState) -> dict[str, Any]:
    router_llm = intent_router_llm.with_structured_output(RouteDecision, method="function_calling")

    decision: RouteDecision = await router_llm.ainvoke(
        [SystemMessage(content=INTENT_ROUTER_PROMPT), *_router_messages(state)]
    )
    result: dict[str, Any] = {"route": decision.route}

    if decision.message:
        result["messages"] = [AIMessage(content=decision.message)]

    return result


def route_conversation(state: AgentState) -> str:
    route = state["route"]

    if route == "off_topic":
        return END

    return route
