from typing import Protocol


class PromptStore(Protocol):
    async def retrieve(self, name: str) -> str: ...
