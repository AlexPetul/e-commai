import asyncio
from pathlib import Path

import pytest
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import evaluate
from deepeval.metrics import TopicAdherenceMetric, TurnRelevancyMetric
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Turn
from langchain_core.messages import HumanMessage
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=1)

def run_async(coro):
    future = _executor.submit(asyncio.run, coro)
    return future.result()


@pytest.mark.ai
def test_graph(compiled_graph):
    def model_callback(input: str, thread_id: str) -> Turn:
        async def inner():
            result = await compiled_graph.ainvoke(
                {"messages": [HumanMessage(content=input)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            return Turn(role="assistant", content=result["messages"][-1].content)

        return run_async(inner())

    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(
        file_path=str(Path(__file__).parent / "goldens" / "rag_flow_dataset.json")
    )

    simulator = ConversationSimulator(model_callback=model_callback)
    conversational_test_cases = simulator.simulate(
        conversational_goldens=dataset.goldens,
        max_user_simulations=10,
    )

    evaluate(
        test_cases=conversational_test_cases,
        metrics=[
            TurnRelevancyMetric(model="gpt-4o-mini"),
            TopicAdherenceMetric(
                model="gpt-4o-mini",
                relevant_topics=[
                    "Garden equipment and outdoor maintenance",
                    "Power tools and machinery",
                    "Hand tools and workshop equipment",
                    "Construction and renovation supplies",
                    "Home improvement and household products",
                    "Cleaning equipment and pressure washers",
                    "Automotive tools and accessories",
                    "Product recommendations and comparisons",
                    "Store services (delivery, warranty, returns, discounts, payment)",
                    "Choosing products based on budget, brand, and intended use",
                ],
            ),
        ],
    )
