import io
import wave
from collections.abc import Callable, Coroutine
from typing import Any

from openai import AsyncOpenAI, AsyncStream
from openai.types.audio import (
    TranscriptionStreamEvent,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
)

from ydachnik_chatbot.settings import settings


def to_wav(audio_bytes: bytes, sample_rate_hz: int) -> bytes:
    """Wrap raw PCM16 samples into a WAV container for the transcription API."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(audio_bytes)
    return buffer.getvalue()


class SpeechTranscriber:
    """Stream speech transcription from buffered audio."""

    def __init__(self, sample_rate_hz: int):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._sample_rate_hz = sample_rate_hz

    async def close(self):
        await self._client.close()

    async def transcribe_streaming(
        self,
        audio_bytes: bytes,
        *,
        on_delta: Callable[[str], Coroutine[Any, Any, None]],
    ) -> str:
        """Transcribe audio and stream partial text to the provided callback."""
        wav = to_wav(audio_bytes, self._sample_rate_hz)

        wav_file = io.BytesIO(wav)
        wav_file.name = "speech.wav"
        wav_file.seek(0)

        transcript = ""

        async with self._client.audio.transcriptions.with_streaming_response.create(
            file=wav_file,
            model=settings.audio_transcription_model,
            response_format="text",
            stream=True,
        ) as response:
            stream: AsyncStream[TranscriptionStreamEvent] = await response.parse()  # ty: ignore[invalid-assignment]
            async for event in stream:
                if isinstance(event, TranscriptionTextDeltaEvent):
                    await on_delta(event.delta)
                elif isinstance(event, TranscriptionTextDoneEvent):
                    transcript = event.text

        return transcript
