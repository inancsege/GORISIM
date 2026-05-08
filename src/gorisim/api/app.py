"""FastAPI application — bidirectional translator endpoints."""

from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from gorisim import __version__
from gorisim.api.deps import (
    PipelineRegistry,
    get_registry,
    get_sign_pipeline,
    get_speech_pipeline,
)
from gorisim.api.schemas import (
    EnrollResponse,
    HealthResponse,
    ProfileInfo,
    SignToTextResult,
    SpeechToSignResult,
)
from gorisim.config import get_settings
from gorisim.devices import pick_device
from gorisim.sign_to_text.pipeline import SignToTextPipeline
from gorisim.speech_to_sign.enroll import (
    delete_profile,
    enroll_user,
    list_profiles,
)
from gorisim.speech_to_sign.pipeline import SpeechToSignPipeline
from gorisim.speech_to_sign.verify import SpeakerVerifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    reg = get_registry()
    reg.sign = SignToTextPipeline.load()
    reg.speech = SpeechToSignPipeline.load()
    yield
    reg.sign = None
    reg.speech = None


app = FastAPI(title="GORISIM Translator", version=__version__, lifespan=lifespan)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _save_upload(upload: UploadFile, suffix: str) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
    with tmp.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return tmp


def _check_size(path: Path, max_mb: int) -> None:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413, detail=f"File too large ({size_mb:.1f} MB > {max_mb} MB)"
        )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<html><body><h1>GORISIM</h1><p>UI not built yet (Task 28).</p></body></html>"
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
async def health(reg: PipelineRegistry = Depends(get_registry)) -> HealthResponse:  # noqa: B008
    s = get_settings()
    return HealthResponse(
        status="ok",
        device=str(pick_device(s.device)),
        version=__version__,
        models_loaded=reg.sign is not None and reg.speech is not None,
    )


@app.post("/sign-to-text", response_model=SignToTextResult)
async def sign_to_text(
    video: UploadFile = File(...),  # noqa: B008
    pipeline: SignToTextPipeline = Depends(get_sign_pipeline),  # noqa: B008
) -> SignToTextResult:
    s = get_settings()
    suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
    path = _save_upload(video, suffix)
    try:
        _check_size(path, s.max_video_mb)
        return pipeline.run(path)
    finally:
        path.unlink(missing_ok=True)


@app.post("/speech-to-sign", response_model=SpeechToSignResult)
async def speech_to_sign(
    audio: UploadFile = File(...),  # noqa: B008
    profile: str | None = Form(default=None),  # noqa: B008
    pipeline: SpeechToSignPipeline = Depends(get_speech_pipeline),  # noqa: B008
) -> SpeechToSignResult:
    s = get_settings()
    suffix = Path(audio.filename or "a.wav").suffix or ".wav"
    path = _save_upload(audio, suffix)
    try:
        _check_size(path, s.max_audio_mb)
        return pipeline.run(path, profile=profile or None)
    finally:
        path.unlink(missing_ok=True)


@app.get("/clips/{clip_id}.mp4")
async def serve_clip(clip_id: str) -> FileResponse:
    s = get_settings()
    if not clip_id.isalnum() or len(clip_id) > 64:
        raise HTTPException(404)
    target = s.data_dir / ".output" / f"{clip_id}.mp4"
    if not target.exists():
        raise HTTPException(404)
    return FileResponse(str(target), media_type="video/mp4")


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(
    name: str = Form(...),  # noqa: B008
    audio: list[UploadFile] = File(...),  # noqa: B008
) -> EnrollResponse:
    if not audio:
        raise HTTPException(400, detail="At least one audio file required")
    saved: list[Path] = []
    try:
        for uf in audio:
            saved.append(_save_upload(uf, Path(uf.filename or "x.wav").suffix or ".wav"))
        verifier = SpeakerVerifier()
        enroll_user(name=name, wav_paths=saved, verifier=verifier)
        return EnrollResponse(profile=name, num_samples=len(saved), ok=True)
    finally:
        for p in saved:
            p.unlink(missing_ok=True)


@app.get("/profiles", response_model=list[ProfileInfo])
async def get_profiles() -> list[ProfileInfo]:
    return [ProfileInfo(name=n) for n in list_profiles()]


@app.delete("/profiles/{name}")
async def remove_profile(name: str) -> JSONResponse:
    delete_profile(name)
    return JSONResponse({"ok": True})
