"""OpenRouter TTS synthesizer: wiring, config, request params, and audio routing.

Audio decoding (mp3 -> wav) shells out to ffmpeg, so the helpers are monkeypatched;
these tests assert control flow and the request payload, not real transcoding.
"""

from unittest.mock import AsyncMock

import pytest

from bolna.enums import SynthesizerProvider
from bolna.models import OpenRouterConfig, Synthesizer
from bolna.providers import SUPPORTED_SYNTHESIZER_MODELS
from bolna.synthesizer import OpenRouterSynthesizer


def _synth(**overrides):
    cfg = dict(voice="alloy", model="openai/gpt-4o-mini-tts", synthesizer_key="k")
    cfg.update(overrides)
    return OpenRouterSynthesizer(**cfg)


class _FakeSpeechResponse:
    """Mimics the OpenAI SDK binary speech response (sync iter_bytes)."""

    def __init__(self, data):
        self._data = data

    def iter_bytes(self, chunk_size=4096):
        mid = len(self._data) // 2
        yield self._data[:mid]
        yield self._data[mid:]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_provider_is_registered():
    assert SynthesizerProvider.OPENROUTER.value == "openrouter"
    assert SUPPORTED_SYNTHESIZER_MODELS["openrouter"] is OpenRouterSynthesizer


def test_synthesizer_model_builds_openrouter_config():
    synth = Synthesizer(
        provider="openrouter",
        provider_config={"voice": "alloy", "model": "google/gemini-3.1-flash-tts"},
    )
    assert isinstance(synth.provider_config, OpenRouterConfig)
    assert synth.provider_config.model == "google/gemini-3.1-flash-tts"


def test_config_defaults_model_and_speed():
    cfg = OpenRouterConfig(voice="alloy")
    assert cfg.model == "openai/gpt-4o-mini-tts"
    assert cfg.speed == 1.0


# ---------------------------------------------------------------------------
# Construction / auth
# ---------------------------------------------------------------------------

def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError) as exc:
        OpenRouterSynthesizer(voice="alloy", synthesizer_key=None)
    assert "OpenRouter API key" in str(exc.value)


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    s = OpenRouterSynthesizer(voice="alloy")  # no explicit key -> falls back to env
    assert s.async_client is not None


def test_does_not_advertise_websocket():
    # OpenRouter TTS is one-shot HTTP; it must not be treated as a streaming WS synth.
    assert _synth().supports_websocket() is False
    assert _synth().stream is False


# ---------------------------------------------------------------------------
# Audio format + routing
# ---------------------------------------------------------------------------

def test_web_format_is_wav():
    s = _synth()
    assert s.use_mulaw is False
    assert s._get_http_audio_format() == "wav"


def test_telephony_format_is_mulaw():
    s = _synth(use_mulaw=True)
    assert s.use_mulaw is True
    assert s._get_http_audio_format() == "mulaw"


def _patch_audio_helpers(monkeypatch):
    mod = "bolna.synthesizer.openrouter_synthesizer."
    monkeypatch.setattr(mod + "convert_audio_to_wav", lambda audio, source_format=None: b"WAV(" + audio + b")")
    monkeypatch.setattr(mod + "resample", lambda wav, rate, format=None: b"RS(" + wav + b")")
    monkeypatch.setattr(mod + "audio_to_mulaw8k", lambda wav, rate_hint=0, format_hint="": b"UL(" + wav + b")")


def test_process_http_audio_web_path(monkeypatch):
    _patch_audio_helpers(monkeypatch)
    s = _synth()
    assert s._process_http_audio(b"mp3") == b"RS(WAV(mp3))"


def test_process_http_audio_telephony_path(monkeypatch):
    _patch_audio_helpers(monkeypatch)
    s = _synth(use_mulaw=True)
    assert s._process_http_audio(b"mp3") == b"UL(WAV(mp3))"


def test_process_http_audio_handles_empty():
    assert _synth()._process_http_audio(b"") == b""


# ---------------------------------------------------------------------------
# Request payload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_http_sends_expected_params():
    s = _synth()
    s.async_client.audio.speech.create = AsyncMock(return_value=_FakeSpeechResponse(b"ID3audio"))
    out = await s._generate_http("hello world")
    assert out == b"ID3audio"
    kwargs = s.async_client.audio.speech.create.call_args.kwargs
    assert kwargs == {
        "model": "openai/gpt-4o-mini-tts",
        "voice": "alloy",
        "input": "hello world",
        "response_format": "mp3",
    }  # speed omitted at default 1.0


@pytest.mark.asyncio
async def test_generate_http_includes_non_default_speed():
    s = _synth(speed=1.25)
    s.async_client.audio.speech.create = AsyncMock(return_value=_FakeSpeechResponse(b"x"))
    await s._generate_http("hi")
    assert s.async_client.audio.speech.create.call_args.kwargs["speed"] == 1.25


@pytest.mark.asyncio
async def test_synthesize_returns_wav(monkeypatch):
    _patch_audio_helpers(monkeypatch)
    s = _synth()
    s._generate_http = AsyncMock(return_value=b"mp3")
    assert await s.synthesize("hi") == b"WAV(mp3)"
