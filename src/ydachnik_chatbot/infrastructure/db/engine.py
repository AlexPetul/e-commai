import logging

from langchain_postgres import Column, PGEngine
from sqlalchemy import text

from ydachnik_chatbot.settings import AppSettings

logger = logging.getLogger(__name__)

engine: PGEngine | None = None


def get_engine(settings: AppSettings) -> PGEngine:
    global engine
    if engine is None:
        engine = PGEngine.from_connection_string(url=settings.database_url)
    return engine


async def _table_exists(engine: PGEngine, table_name: str) -> bool:
    async with engine._pool.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT EXISTS (SELECT FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = :name)"
            ),
            {"name": table_name},
        )
        return bool(result.scalar())


async def init_vectorstore_table(
    engine: PGEngine,
    table_name: str,
    vector_size: int,
    *,
    overwrite_existing: bool = False,
) -> None:
    if not overwrite_existing and await _table_exists(engine, table_name):
        logger.info("Vector table %s already exists, skipped initialization.", table_name)
        return

    await engine.ainit_vectorstore_table(
        table_name=table_name,
        vector_size=vector_size,
        id_column=Column(name="langchain_id", data_type="VARCHAR"),
        overwrite_existing=overwrite_existing,
    )
    logger.info("Vector table %s initialized.", table_name)
