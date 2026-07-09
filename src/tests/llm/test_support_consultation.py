# import uuid
#
# import pytest
# from deepeval.dataset import EvaluationDataset, Golden
# from deepeval.evaluate import evaluate
# from deepeval.metrics import GEval, ToolCorrectnessMetric
# from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall
# from langchain_core.messages import HumanMessage
# from langgraph.checkpoint.memory import MemorySaver
#
# from ydachnik_chatbot.ai.graph import build_graph
# from ydachnik_chatbot.ai.state import AgentState
#
#
# @pytest.mark.ai
# @pytest.mark.asyncio
# async def test_support_consultation():
#     graph = build_graph(checkpointer=MemorySaver())
#     goldens = [
#         Golden(
#             name="delivery_conditions",
#             input="Hi, is the delivery free?",
#             expected_tools=[
#                 ToolCall(
#                     name="fetch_page",
#                     input_parameters={"url": "https://www.ydachnik.by/customers/delivery/"},
#                 ),
#             ],
#         ),
#         Golden(
#             name="warranty",
#             input="What if i dont like the product? Can i return it?",
#             expected_tools=[
#                 ToolCall(
#                     name="fetch_page",
#                     input_parameters={"url": "https://www.ydachnik.by/customers/warranty/"},
#                 ),
#             ],
#         ),
#         Golden(
#             name="discount",
#             input="Do you have any discount programs?",
#             expected_tools=[
#                 ToolCall(
#                     name="fetch_page",
#                     input_parameters={"url": "https://www.ydachnik.by/customers/discont/"},
#                 ),
#             ],
#         ),
#     ]
#     dataset = EvaluationDataset(goldens=goldens)
#
#     for golden in dataset.goldens:
#         state: AgentState = await graph.ainvoke(
#             {"messages": [HumanMessage(content=golden.input)]},
#             config={"configurable": {"thread_id": uuid.uuid4().hex}},
#         )
#
#         tools_called = []
#
#         for message in state["messages"]:
#             if hasattr(message, "tool_calls"):
#                 for tool_call in message.tool_calls:
#                     tools_called.append(
#                         ToolCall(
#                             name=tool_call["name"],
#                             input_parameters=tool_call.get("args", {}),
#                         )
#                     )
#
#         dataset.add_test_case(
#             LLMTestCase(
#                 input=golden.input,
#                 expected_output=golden.expected_output,
#                 actual_output=state["messages"][-1].content,
#                 tools_called=tools_called,
#                 expected_tools=golden.expected_tools,
#             )
#         )
#
#     evaluate(
#         test_cases=dataset.test_cases,
#         metrics=[
#             GEval(
#                 name="Support consulation",
#                 model="gpt-4o-mini",
#                 threshold=0.6,
#                 criteria="Determine whether the assistant correctly answers the user's question.",
#                 evaluation_params=[
#                     SingleTurnParams.INPUT,
#                     SingleTurnParams.ACTUAL_OUTPUT,
#                 ],
#             ),
#             ToolCorrectnessMetric(),
#         ],
#     )
