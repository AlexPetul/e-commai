import logging
from typing import cast

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex

from ydachnik_chatbot.infrastructure.db import get_engine
from ydachnik_chatbot.settings import settings

logger = logging.getLogger(__name__)

_vectorstore: AsyncPGVectorStore | None = None


embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    dimensions=settings.embedding_dimensions,
)


async def init_vector_store(
    engine: PGEngine,
    *,
    overwrite_existing: bool = False,
) -> None:
    global _vectorstore
    _vectorstore = await AsyncPGVectorStore.create(
        engine=engine,
        table_name=settings.vector_table_name,
        embedding_service=embeddings,
    )

    if overwrite_existing:
        await _vectorstore.adrop_vector_index()
        await _vectorstore.aapply_vector_index(HNSWIndex())
    elif not await _vectorstore.is_valid_index():
        await _vectorstore.aapply_vector_index(HNSWIndex())

    logger.info("Vectorstore initialized.")


async def get_vectorstore() -> AsyncPGVectorStore:
    if _vectorstore is None:
        await init_vector_store(get_engine(settings))

    return cast(AsyncPGVectorStore, _vectorstore)
