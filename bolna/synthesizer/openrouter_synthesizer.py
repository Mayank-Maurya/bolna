"""
OpenRouter text-to-speech synthesizer for bolna.

OpenRouter exposes an OpenAI-compatible audio API at ``/api/v1/audio/speech`` that
fronts TTS models from several providers (OpenAI, Google, Mistral, Kokoro, ...) behind
one key and billing account. It is a one-shot HTTP endpoint — send ``{model, input,
voice}`` and receive a raw audio byte stream — so this is an HTTP ``BaseSynthesizer``
(like the OpenAI synthesizer), not a streaming WebSocket one.

We request ``mp3`` (self-describing and safely decodable) and convert/resample to the
target rate; OpenRouter's ``pcm`` output has no documented, model-stable sample rate, so
mp3 avoids guessing it. Telephony (``use_mulaw``) renders 8 kHz mu-law.
"""

import io
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import audio_to_mulaw8k, convert_audio_to_wav, resample
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache

logger = configure_logger(__name__)
load_dotenv()

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        voice,
        model="openai/gpt-4o-mini-tts",
        audio_format="mp3",
        sampling_rate=8000,
        stream=False,
        buffer_size=400,
        speed=1.0,
        caching=True,
        synthesizer_key=None,
        base_url=None,
        **kwargs,
    ):
        super().__init__(kwargs.get("task_manager_instance"), stream, buffer_size)
        self.voice = voice
        self.model = model
        self.speed = float(speed) if speed is not None else 1.0
        self.sample_rate = int(sampling_rate) if isinstance(sampling_rate, str) else sampling_rate
        # OpenRouter TTS is one-shot HTTP; bolna's streaming loop doesn't apply.
        self.stream = False
        self.use_mulaw = kwargs.get("use_mulaw", False)

        api_key = synthesizer_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key is required, either as synthesizer_key or OPENROUTER_API_KEY")

        base_url = base_url or os.getenv("OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

    def supports_websocket(self):
        return False

    # ------------------------------------------------------------------
    # BaseSynthesizer hooks
    # ------------------------------------------------------------------

    def _get_http_audio_format(self):
        return "mulaw" if self.use_mulaw else "wav"

    def _process_http_audio(self, audio):
        """Decode the mp3 response, then resample to the target rate (mu-law for telephony)."""
        if not audio:
            return audio
        wav = convert_audio_to_wav(audio, source_format="mp3")
        if self.use_mulaw:
            return audio_to_mulaw8k(wav, rate_hint=self.sample_rate, format_hint="wav")
        return resample(wav, self.sample_rate, format="wav")

    async def _generate_http(self, text):
        params = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": "mp3",
        }
        # Only send speed when it deviates from the default — some routed models reject it.
        if self.speed and self.speed != 1.0:
            params["speed"] = self.speed

        response = await self.async_client.audio.speech.create(**params)
        buffer = io.BytesIO()
        for chunk in response.iter_bytes(chunk_size=4096):
            buffer.write(chunk)
        return buffer.getvalue()

    async def synthesize(self, text):
        """One-shot render (used by prewarm/handoff). Return WAV so downstream mu-law
        conversion can decode it — a raw mp3 container would be skipped as undecodable."""
        audio = await self._generate_http(text)
        if not audio:
            return None
        return convert_audio_to_wav(audio, source_format="mp3")

    # ------------------------------------------------------------------
    # generate() — HTTP loop from BaseSynthesizer (handles caching)
    # ------------------------------------------------------------------

    async def generate(self):
        async for packet in self._generate_http_loop():
            yield packet
