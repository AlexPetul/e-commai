from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ydachnik_chatbot.ai.nodes import (
    product_consultant_node,
    product_tools,
    support_consultant_node,
    support_tools,
)
from ydachnik_chatbot.ai.router import intent_router, route_conversation
from ydachnik_chatbot.ai.state import AgentState


def _tool_error_handler(error: Exception) -> str:
    return f"Tool error ({type(error).__name__}): {error}"


def build_graph(checkpointer: BaseCheckpointSaver):
    builder = StateGraph(AgentState)

    builder.add_node("intent_router", intent_router, metadata={"disable_streaming": True})
    builder.add_node("product_consultant", product_consultant_node)
    builder.add_node(
        "product_tools", ToolNode(product_tools, handle_tool_errors=_tool_error_handler)
    )
    builder.add_node("support_consultant", support_consultant_node)
    builder.add_node(
        "support_tools", ToolNode(support_tools, handle_tool_errors=_tool_error_handler)
    )

    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        route_conversation,
        {
            "product_consultant": "product_consultant",
            "support_consultant": "support_consultant",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "product_consultant", tools_condition, {"tools": "product_tools", END: END}
    )
    builder.add_conditional_edges(
        "support_consultant", tools_condition, {"tools": "support_tools", END: END}
    )
    builder.add_edge("product_tools", "product_consultant")
    builder.add_edge("support_tools", "support_consultant")

    return builder.compile(checkpointer=checkpointer)
