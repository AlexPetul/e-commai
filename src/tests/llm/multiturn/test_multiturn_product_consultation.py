import uuid

import pytest
from langchain_core.messages import HumanMessage
from langsmith import aevaluate
from langsmith.schemas import Run
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openevals.simulators import (
    create_llm_simulated_user,
    run_multiturn_simulation_async,
)

DATASET_NAME = "Product Consultation"


trajectory_evaluator = create_llm_as_judge(
    model="openai:gpt-4o-mini",
    feedback_key="conversation_quality",
    prompt="""
You are evaluating an AI shopping assistant.

Evaluate the COMPLETE conversation.

Score from 0.0 to 1.0.

Criteria:

1. The assistant stayed on topic.
2. Recommendations matched the user's needs.
3. The assistant asked useful clarifying questions.
4. There was no hallucinated product information.
5. The assistant remembered earlier turns.
6. The overall customer experience was good.

Conversation:

{outputs}
""",
)

correctness = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:gpt-4o-mini",
)


def extract_final_answer(trajectory: list) -> str:
    for message in reversed(trajectory):
        if isinstance(message, dict):
            if message.get("role") == "assistant":
                content = message.get("content")
                if content:
                    return content
        else:
            if getattr(message, "role", None) == "assistant":
                content = getattr(message, "content", "")
                if content:
                    return content
    return ""


async def stopping_condition(trajectory: list, turn_counter) -> bool:
    tool_calls = trajectory[-1].get("tool_calls", [])
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        if function.get("name") == "search_products":
            return True
    return False


@pytest.mark.ai
@pytest.mark.asyncio
async def test_multiturn_product_consultation(compiled_graph):
    async def target(inputs: dict, langsmith_extra=None):
        thread_id = uuid.uuid4().hex

        async def app(next_message, *, thread_id: str, **_):
            state = await compiled_graph.ainvoke(
                {"messages": [HumanMessage(content=next_message["content"])]},
                config={"configurable": {"thread_id": thread_id}},
            )

            return {
                "role": "assistant",
                "content": state["messages"][-1].content,
                "tool_calls": state["messages"][-1].tool_calls,
            }

        message = f"""
        Scenario: {inputs["scenario"]}
        You should generate an initial message to start a conversation based
        on the scenario, as you were a person who is vising an online shop.
        Don't include all the information in the first message. Let the assistant
        ask you clarifying questions.
        """

        simulated_user = create_llm_simulated_user(
            system=message,
            model="openai:gpt-4o-mini",
        )

        result = await run_multiturn_simulation_async(
            app=app,
            user=simulated_user,
            max_turns=10,
            thread_id=thread_id,
            stopping_condition=stopping_condition,
        )

        trajectory = result["trajectory"]
        return {
            "answer": extract_final_answer(trajectory),
            "trajectory": trajectory,
        }

    def llm_correctness(run: Run, example):
        return correctness(
            inputs=example.inputs["scenario"],
            outputs=run.outputs,
            reference_outputs=example.outputs["expected_output"],
        )

    def llm_trajectory_quality(run: Run, example):
        return trajectory_evaluator(
            inputs=example.inputs["scenario"],
            outputs=run.outputs,
        )

    await aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            llm_correctness,
            llm_trajectory_quality,
        ],
        experiment_prefix="product-consultation",
        upload_results=True,
        max_concurrency=4,
    )
