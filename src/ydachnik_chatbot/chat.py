import logging
import uuid
from http.cookies import SimpleCookie

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from fastapi import Response
from langchain_core.messages import HumanMessage, ToolMessage
from starlette.datastructures import Headers

from ydachnik_chatbot import runtime
from ydachnik_chatbot.ai.stt import SpeechTranscriber
from ydachnik_chatbot.settings import settings

logger = logging.getLogger(__name__)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Хочу подобрать газонокосилку",
            message="Можешь мне помочь подобрать газонокосилку?",
            icon="/public/lawn_mower.jpg",
        ),
    ]


@cl.header_auth_callback
async def header_auth_callback(headers: Headers) -> cl.User | None:
    """Authenticate anonymous users via a persistent cookie set by the middleware.
    This is required to enable persistence of chat history.
    """
    cookie = SimpleCookie()
    cookie.load(headers.get("cookie", ""))
    user_id_cookie = cookie.get(settings.user_id_cookie_name)
    user_id = user_id_cookie.value if user_id_cookie else str(uuid.uuid4())
    return cl.User(identifier=user_id)


@cl.on_logout
def on_logout(_, response: Response):
    response.delete_cookie(settings.user_id_cookie_name)


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=settings.database_url)


def _thread_id() -> str:
    return cl.context.session.thread_id


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Enable users to continue a conversation. Function must be declared, even if it's empty."""
    cl.user_session.set("thread_id", thread["id"])


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", _thread_id())


async def process_user_message(content: str):
    agent_graph = runtime.agent_graph
    if agent_graph is None:
        await cl.Message(content="Сервис ещё запускается, попробуйте снова.").send()
        return

    thread_id = _thread_id()
    answer = cl.Message(content="")
    try:
        async for msg, metadata in agent_graph.astream(
            {"messages": [HumanMessage(content=content)]},
            stream_mode="messages",
            config={"configurable": {"thread_id": thread_id}},
        ):
            internal_node = metadata.get("disable_streaming")

            if metadata.get("langgraph_node") == "intent_router" and msg.content:
                await cl.Message(content=msg.content).send()

            if (
                not internal_node
                and not isinstance(msg, ToolMessage)
                and not msg.tool_calls
                and msg.content
            ):
                await answer.stream_token(msg.content)
    except Exception:
        logger.exception("Error while processing message")
        await cl.Message(content="Произошла ошибка. Попробуйте ещё раз.").send()
        return

    await answer.send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    await process_user_message(message.content)


@cl.on_audio_start
async def on_audio_start() -> bool:
    cl.user_session.set("pending_audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    chunks: list[bytes] = cl.user_session.get("pending_audio_chunks", [])
    chunks.append(chunk.data)
    cl.user_session.set("pending_audio_chunks", chunks)


@cl.on_audio_end
async def on_audio_end() -> None:
    chunks: list[bytes] = cl.user_session.get("pending_audio_chunks") or []
    audio = b"".join(chunks)
    await process_transcript_turn(audio)


@cl.on_chat_end
@cl.on_stop
async def on_end() -> None:
    cl.user_session.set("pending_audio_chunks", [])


async def process_transcript_turn(audio: bytes):
    if not audio:
        return

    transcript_message = cl.Message(content="", type="user_message")

    async def on_delta(delta: str):
        await transcript_message.stream_token(delta)

    transcriber = SpeechTranscriber(sample_rate_hz=24000)
    transcript = await transcriber.transcribe_streaming(audio, on_delta=on_delta)
    if not transcript.strip():
        logger.warning("Transcription finished with no text")
        return

    transcript_message.content = transcript
    await transcript_message.send()

    cl.user_session.set("pending_audio_chunks", [])

    await process_user_message(transcript)
