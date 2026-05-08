"""API request/response schemas — re-exports pipeline result types."""
from __future__ import annotations

from pydantic import BaseModel

from gorisim.sign_to_text.pipeline import Alternative, SignToTextResult
from gorisim.speech_to_sign.pipeline import MatchedSign, SpeechToSignResult


class HealthResponse(BaseModel):
    status: str
    device: str
    version: str
    models_loaded: bool


class EnrollResponse(BaseModel):
    profile: str
    num_samples: int
    ok: bool


class ProfileInfo(BaseModel):
    name: str


__all__ = [
    "Alternative",
    "EnrollResponse",
    "HealthResponse",
    "MatchedSign",
    "ProfileInfo",
    "SignToTextResult",
    "SpeechToSignResult",
]
