import uuid

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langsmith import aevaluate
from langsmith.schemas import Run
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from ydachnik_chatbot.ai.graph import build_graph

DATASET_NAME = "Support QA"


def extract_tool_calls(messages: list) -> list[dict]:
    tool_calls = []

    for message in messages:
        if isinstance(message, dict):
            tool_calls.extend(message.get("tool_calls", []))
        else:
            tool_calls.extend(getattr(message, "tool_calls", []) or [])

    return tool_calls


def tool_correctness(run: Run, example):
    messages = (run.outputs or {}).get("messages", [])
    expected_tool = example.outputs["expected_tool"]
    passed = any(tc.get("name") == expected_tool for tc in extract_tool_calls(messages))

    return {
        "key": "tool_correctness",
        "score": passed,
    }


correctness = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:gpt-4o-mini",
)


@pytest.mark.ai
@pytest.mark.asyncio
async def test_support_consultation():
    graph = build_graph(
        checkpointer=MemorySaver(),
    )

    async def target(inputs: dict, langsmith_extra=None):
        state = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=inputs["question"],
                    )
                ]
            },
            config={
                "configurable": {
                    "thread_id": uuid.uuid4().hex,
                }
            },
        )

        answer = state["messages"][-1].content

        return {
            "answer": answer,
            "messages": state["messages"],
        }

    def llm_correctness(run: Run, example):
        return correctness(
            inputs=example.inputs["question"],
            outputs=run.outputs["answer"],
            reference_outputs=example.outputs["reference_answer"],
        )

    await aevaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            tool_correctness,
            llm_correctness,
        ],
        experiment_prefix="support-consultation",
        upload_results=True,
        max_concurrency=4,
    )
