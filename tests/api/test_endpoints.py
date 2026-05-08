from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from gorisim.api import app as app_module
from gorisim.api.deps import get_sign_pipeline, get_speech_pipeline
from gorisim.sign_to_text.pipeline import Alternative, SignToTextResult
from gorisim.speech_to_sign.pipeline import SpeechToSignResult


@pytest.fixture
def client(tmp_path):
    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    app_module.app.router.lifespan_context = noop_lifespan

    fake_sign = MagicMock()
    fake_sign.run.return_value = SignToTextResult(
        text="merhaba",
        confidence=0.9,
        alternatives=[Alternative(text="evet", confidence=0.05)],
        duration_ms=10,
    )
    fake_speech = MagicMock()
    out_video = tmp_path / "out.mp4"
    out_video.write_bytes(b"x")
    fake_speech.run.return_value = SpeechToSignResult(
        transcript="merhaba",
        lemmas=["merhaba"],
        matched_signs=[],
        missing_words=[],
        output_video_path=out_video,
        duration_ms=10,
    )

    app_module.app.dependency_overrides[get_sign_pipeline] = lambda: fake_sign
    app_module.app.dependency_overrides[get_speech_pipeline] = lambda: fake_speech
    yield TestClient(app_module.app)
    app_module.app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_sign_to_text_endpoint(client, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 1000)
    with video.open("rb") as f:
        r = client.post("/sign-to-text", files={"video": ("v.mp4", f, "video/mp4")})
    assert r.status_code == 200
    assert r.json()["text"] == "merhaba"


def test_speech_to_sign_endpoint(client, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 1000)
    with audio.open("rb") as f:
        r = client.post("/speech-to-sign", files={"audio": ("a.wav", f, "audio/wav")})
    assert r.status_code == 200
    assert r.json()["transcript"] == "merhaba"


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower() or "<!doctype" in r.text.lower()
