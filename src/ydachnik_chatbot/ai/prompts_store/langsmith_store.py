from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langsmith import AsyncClient

from ydachnik_chatbot.ai.prompts_store.base_store import PromptStore


@lru_cache(maxsize=1)
def get_langsmith_client() -> AsyncClient:
    return AsyncClient()


class LangsmithPromptStore(PromptStore):
    def __init__(self, client: AsyncClient) -> None:
        self.client = client

    async def retrieve(self, name: str) -> str:
        prompt_template: ChatPromptTemplate = await self.client.pull_prompt(name)
        return prompt_template.messages[0].prompt.template


langsmith_prompt_store = LangsmithPromptStore(get_langsmith_client())
