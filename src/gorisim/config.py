from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    device: Literal["auto", "cpu", "mps", "cuda"] = Field(
        default="auto", validation_alias="GORISIM_DEVICE"
    )
    diarization: bool = Field(default=True, validation_alias="GORISIM_DIARIZATION")
    use_llm: bool = Field(default=False, validation_alias="GORISIM_USE_LLM")
    no_profile_mode: Literal["speaker_zero", "all"] = Field(
        default="speaker_zero", validation_alias="GORISIM_NO_PROFILE_MODE"
    )
    whisper_model: str = Field(default="large-v3-turbo", validation_alias="GORISIM_WHISPER_MODEL")

    max_video_mb: int = Field(default=100, validation_alias="GORISIM_MAX_VIDEO_MB")
    max_video_seconds: int = Field(default=30, validation_alias="GORISIM_MAX_VIDEO_SECONDS")
    max_audio_mb: int = Field(default=50, validation_alias="GORISIM_MAX_AUDIO_MB")
    max_audio_seconds: int = Field(default=120, validation_alias="GORISIM_MAX_AUDIO_SECONDS")

    models_dir: Path = Field(default=Path("./models"), validation_alias="GORISIM_MODELS_DIR")
    data_dir: Path = Field(default=Path("./data"), validation_alias="GORISIM_DATA_DIR")
    clip_ttl_seconds: int = Field(default=3600, validation_alias="GORISIM_CLIP_TTL_SECONDS")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_for_test() -> None:
    global _settings
    _settings = None
