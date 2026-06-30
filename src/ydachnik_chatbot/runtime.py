from typing import Any

from langchain_postgres import PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore

agent_graph: Any | None = None
engine: PGEngine | None = None
vectorstore: AsyncPGVectorStore | None = None
