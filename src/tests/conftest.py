import pytest
from langgraph.checkpoint.memory import MemorySaver

from ydachnik_chatbot.ai.graph import build_graph


@pytest.fixture
def compiled_graph():
    return build_graph(checkpointer=MemorySaver())
