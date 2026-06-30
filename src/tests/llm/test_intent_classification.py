import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.ai
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message, route",
    [
        ("Hi! Write me a receipe for pancakes.", "off_topic"),
        ("Whats the point of life?", "off_topic"),
        ("Help me to choose a lawnmower", "product_consultant"),
        ("How the delivery works?", "support_consultant"),
    ],
)
async def test_intent_router(compiled_graph, message: str, route: str):
    compiled_graph.update_state(
        config={"configurable": {"thread_id": "1"}},
        values={"messages": [HumanMessage(content=message)]},
        as_node="__start__",
    )

    result = await compiled_graph.ainvoke(
        None,
        config={"configurable": {"thread_id": "1"}},
        interrupt_after=["intent_router"],
    )

    assert result["route"] == route
