from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from ydachnik_chatbot.ai.router import RouteDecision, intent_router


@pytest.mark.asyncio
async def test_intent_router_returns_route():
    state = {"messages": [HumanMessage(content="")]}

    decision = RouteDecision(route="product_consultant", message="Message")

    router = MagicMock()
    router.ainvoke = AsyncMock(return_value=decision)

    llm = MagicMock()
    llm.with_structured_output.return_value = router

    with patch("ydachnik_chatbot.ai.router.intent_router_llm", llm):
        result = await intent_router(state)

    router.ainvoke.assert_awaited_once()

    assert result["route"] == decision.route
    assert result["messages"][-1].content == decision.message
