import logging
import uuid
from contextlib import asynccontextmanager

from chainlit.utils import mount_chainlit
from fastapi import FastAPI, Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from ydachnik_chatbot import runtime
from ydachnik_chatbot.ai import build_graph
from ydachnik_chatbot.ai.category_candidates import warmup_category_selector
from ydachnik_chatbot.infrastructure.db import get_engine, init_vectorstore_table
from ydachnik_chatbot.infrastructure.db.vectorstore import get_vectorstore, init_vector_store
from ydachnik_chatbot.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    engine = get_engine(settings)
    runtime.engine = engine
    app.state.engine = engine
    await init_vectorstore_table(
        engine,
        settings.vector_table_name,
        settings.embedding_dimensions,
    )
    await init_vector_store(engine)
    vectorstore = await get_vectorstore()
    runtime.vectorstore = vectorstore
    app.state.vectorstore = vectorstore
    await warmup_category_selector()

    async with AsyncPostgresSaver.from_conn_string(settings.checkpointer_dsn) as checkpointer:
        await checkpointer.setup()
        runtime.agent_graph = build_graph(checkpointer)
        app.state.agent_graph = runtime.agent_graph
        yield

    await app.state.engine.close()
    runtime.engine = None
    runtime.vectorstore = None
    runtime.agent_graph = None
    logger.info("Shutting down...")


app = FastAPI(title="Ydachnik chatbot", version="1.0.0", lifespan=lifespan)

mount_chainlit(app=app, target="src/ydachnik_chatbot/chat.py", path="")


@app.middleware("http")
async def ensure_user_id_cookie(request: Request, call_next):
    response = await call_next(request)

    is_secure = request.url.scheme == "https"

    if settings.user_id_cookie_name not in request.cookies:
        response.set_cookie(
            key=settings.user_id_cookie_name,
            value=str(uuid.uuid4()),
            max_age=settings.user_id_cookie_max_age,
            httponly=True,
            secure=is_secure,
            samesite="none" if is_secure else "lax",
        )

    return response
