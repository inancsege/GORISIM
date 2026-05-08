"""FastAPI dependency providers — pipelines are singletons created in lifespan."""
from __future__ import annotations

from dataclasses import dataclass

from gorisim.sign_to_text.pipeline import SignToTextPipeline
from gorisim.speech_to_sign.pipeline import SpeechToSignPipeline


@dataclass
class PipelineRegistry:
    sign: SignToTextPipeline | None = None
    speech: SpeechToSignPipeline | None = None


_REGISTRY = PipelineRegistry()


def get_registry() -> PipelineRegistry:
    return _REGISTRY


def get_sign_pipeline() -> SignToTextPipeline:
    if _REGISTRY.sign is None:
        raise RuntimeError("Sign pipeline not loaded — application not started yet")
    return _REGISTRY.sign


def get_speech_pipeline() -> SpeechToSignPipeline:
    if _REGISTRY.speech is None:
        raise RuntimeError("Speech pipeline not loaded — application not started yet")
    return _REGISTRY.speech
