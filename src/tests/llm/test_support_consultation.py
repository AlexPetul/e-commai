import uuid

import pytest
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import evaluate
from deepeval.metrics import GEval, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from ydachnik_chatbot.ai.graph import build_graph
from ydachnik_chatbot.ai.state import AgentState


@pytest.mark.ai
@pytest.mark.asyncio
async def test_graph():
    graph = build_graph(checkpointer=MemorySaver())
    goldens = [
        Golden(
            name="delivery_conditions",
            input="Hi, is the delivery free?",
            expected_output=(
                """
            Delivery is available across Minsk and Belarus, with free delivery in Minsk
            for orders over 150 BYN (and for selected brands regardless of order value);
            shipping fees for other orders depend on the destination and product weight,
            while delivery timelines are typically 1–3 days in Belarus.
            """
            ),
            expected_tools=[
                ToolCall(
                    name="fetch_page_main_block",
                    input_parameters={"url": "https://www.ydachnik.by/customers/delivery/"},
                ),
            ],
        ),
        Golden(
            name="warranty",
            input="What if i dont like the product? Can i return it?",
            expected_output=(
                """
            All purchases include a warranty and receipt,
            allowing eligible returns or exchanges within 14 days if
            the item is unused and meets return conditions, while defective products
            can be returned or replaced with proof of purchase and
            a service center confirmation of a manufacturing defect.
            """
            ),
            expected_tools=[
                ToolCall(
                    name="fetch_page_main_block",
                    input_parameters={"url": "https://www.ydachnik.by/customers/warranty/"},
                ),
            ],
        ),
        Golden(
            name="discount",
            input="Do you have any discount programs?",
            expected_output=(
                """
            The loyalty program offers 3–7% discounts based on cumulative purchase amounts,
            valid across all stores, with exclusions for promotional items, certain brands
            (e.g. STIHL and some Karcher products), installment purchases,
            and other active discounts.
            """
            ),
            expected_tools=[
                ToolCall(
                    name="fetch_page_main_block",
                    input_parameters={"url": "https://www.ydachnik.by/customers/discont/"},
                ),
            ],
        ),
    ]
    dataset = EvaluationDataset(goldens=goldens)

    for golden in dataset.goldens:
        state: AgentState = await graph.ainvoke(
            {"messages": [HumanMessage(content=golden.input)]},
            config={"configurable": {"thread_id": uuid.uuid4().hex}},
        )

        tools_called = []

        for message in state["messages"]:
            if hasattr(message, "tool_calls"):
                for tool_call in message.tool_calls:
                    tools_called.append(
                        ToolCall(
                            name=tool_call["name"],
                            input_parameters=tool_call.get("args", {}),
                        )
                    )

        dataset.add_test_case(
            LLMTestCase(
                input=golden.input,
                expected_output=golden.expected_output,
                actual_output=state["messages"][-1].content,
                tools_called=tools_called,
                expected_tools=golden.expected_tools,
            )
        )

    evaluate(
        test_cases=dataset.test_cases,
        metrics=[
            GEval(
                name="Support consulation",
                model="gpt-5.4-mini",
                threshold=0.6,
                criteria="Determine whether the assistant correctly answers the user's question.",
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                ],
            ),
            ToolCorrectnessMetric(),
        ],
    )
