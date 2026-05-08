from pathlib import Path

from gorisim.config import Settings


def test_settings_defaults_when_env_empty(monkeypatch, tmp_path):
    for key in [
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "GORISIM_DEVICE",
        "GORISIM_DIARIZATION",
        "GORISIM_USE_LLM",
        "GORISIM_NO_PROFILE_MODE",
        "GORISIM_WHISPER_MODEL",
        "GORISIM_MAX_VIDEO_MB",
        "GORISIM_MAX_VIDEO_SECONDS",
        "GORISIM_MAX_AUDIO_MB",
        "GORISIM_MAX_AUDIO_SECONDS",
        "GORISIM_MODELS_DIR",
        "GORISIM_DATA_DIR",
        "GORISIM_CLIP_TTL_SECONDS",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    s = Settings(_env_file=None)  # type: ignore[call-arg]

    assert s.hf_token is None
    assert s.openai_api_key is None
    assert s.device == "auto"
    assert s.diarization is True
    assert s.use_llm is False
    assert s.no_profile_mode == "speaker_zero"
    assert s.whisper_model == "large-v3-turbo"
    assert s.max_video_mb == 100
    assert s.max_video_seconds == 30
    assert s.max_audio_mb == 50
    assert s.max_audio_seconds == 120
    assert s.models_dir == Path("./models")
    assert s.data_dir == Path("./data")
    assert s.clip_ttl_seconds == 3600


def test_settings_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_test_xyz")
    monkeypatch.setenv("GORISIM_DEVICE", "cpu")
    monkeypatch.setenv("GORISIM_DIARIZATION", "off")
    monkeypatch.setenv("GORISIM_USE_LLM", "true")
    monkeypatch.setenv("GORISIM_MAX_VIDEO_MB", "250")
    monkeypatch.chdir(tmp_path)

    s = Settings(_env_file=None)  # type: ignore[call-arg]

    assert s.hf_token == "hf_test_xyz"
    assert s.device == "cpu"
    assert s.diarization is False
    assert s.use_llm is True
    assert s.max_video_mb == 250


def test_settings_invalid_device_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("GORISIM_DEVICE", "tpu")
    monkeypatch.chdir(tmp_path)

    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]
