from langchain_openai import ChatOpenAI

from ydachnik_chatbot.settings import settings

_api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None

product_consultant_llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=_api_key,
    temperature=0.4,
    streaming=True,
)

support_consultant_llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=_api_key,
    streaming=True,
)

category_classifier_llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=_api_key,
    streaming=False,
)

intent_router_llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=_api_key,
    streaming=False,
)
