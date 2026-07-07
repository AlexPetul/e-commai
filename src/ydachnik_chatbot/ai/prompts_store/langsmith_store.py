from langchain_core.prompts import ChatPromptTemplate
from langsmith import AsyncClient


class LangsmithPromptStore:
    @classmethod
    async def retrieve(cls, name: str) -> str:
        async with AsyncClient() as client:
            prompt_template: ChatPromptTemplate = await client.pull_prompt(name)
            return prompt_template.messages[0].prompt.template
