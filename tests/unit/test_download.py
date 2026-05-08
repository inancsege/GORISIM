from unittest.mock import patch

import pytest

from gorisim.config import reset_settings_for_test
from gorisim.download import (
    MANIFEST,
    Asset,
    ensure_dirs,
    fetch_hf_repo,
    fetch_http,
    is_present,
    require_hf_token,
)


def test_manifest_includes_required_assets():
    names = {a.name for a in MANIFEST}
    assert "hrnet_wholebody" in names
    assert "rgb_final_finetuned" in names
    assert "pyannote_diarization_3_1" in names
    assert "speechbrain_ecapa" in names
    assert "faster_whisper" in names
    assert "autsl_signlist_csv" in names


def test_ensure_dirs_creates_models_and_data(tmp_path):
    ensure_dirs(models_dir=tmp_path / "m", data_dir=tmp_path / "d")
    assert (tmp_path / "m").is_dir()
    assert (tmp_path / "d").is_dir()


def test_is_present_returns_false_for_missing_file(tmp_path):
    asset = Asset(name="x", target=tmp_path / "missing.bin", url="https://example/x", sha256=None)
    assert is_present(asset) is False


def test_is_present_returns_true_for_existing_file(tmp_path):
    f = tmp_path / "exists.bin"
    f.write_bytes(b"hello")
    asset = Asset(name="x", target=f, url="https://example/x", sha256=None)
    assert is_present(asset) is True


def test_fetch_http_writes_file(tmp_path):
    target = tmp_path / "out.bin"
    chunks = [b"a" * 1024, b"b" * 1024, b""]

    class FakeResp:
        status_code = 200
        headers = {"content-length": "2048"}
        def iter_content(self, chunk_size: int):
            yield from [c for c in chunks if c]
        def raise_for_status(self):
            pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    with patch("requests.get", return_value=FakeResp()) as mock_get:
        fetch_http(url="https://example/x", target=target)
        mock_get.assert_called_once()
    assert target.read_bytes() == b"a" * 1024 + b"b" * 1024


def test_require_hf_token_raises_when_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    reset_settings_for_test()
    with pytest.raises(SystemExit):
        require_hf_token()


def test_fetch_hf_repo_calls_snapshot_download(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    reset_settings_for_test()
    target = tmp_path / "repo"

    with patch("gorisim.download.snapshot_download") as mock_dl:
        mock_dl.return_value = str(target)
        fetch_hf_repo(repo_id="pyannote/speaker-diarization-3.1", target=target)
        mock_dl.assert_called_once()
        kwargs = mock_dl.call_args.kwargs
        assert kwargs["repo_id"] == "pyannote/speaker-diarization-3.1"
        assert kwargs["token"] == "hf_test"
