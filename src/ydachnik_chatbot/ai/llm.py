from langchain_openai import ChatOpenAI

from ydachnik_chatbot.settings import settings

product_consultant_llm = ChatOpenAI(
    model=settings.openai_model,
    temperature=0.4,
    streaming=True,
)

support_consultant_llm = ChatOpenAI(
    model=settings.openai_model,
    streaming=True,
)

intent_router_llm = ChatOpenAI(
    model=settings.openai_model,
    streaming=False,
)

query_transformation_llm = ChatOpenAI(
    model=settings.openai_model,
    disable_streaming=True,
)
