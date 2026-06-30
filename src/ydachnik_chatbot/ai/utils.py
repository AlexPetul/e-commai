from langchain_core.messages import AIMessage, HumanMessage

from ydachnik_chatbot.ai.state import AgentState


def get_human_system_messages(state: AgentState) -> list:
    return [
        m
        for m in state["messages"]
        if isinstance(m, HumanMessage)
        or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))
    ]


def get_latest_human_message_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            text = message.content.strip()
            if text:
                return text
    return ""
