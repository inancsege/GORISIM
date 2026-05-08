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
from gorisim.download import main as download_main


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

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    with patch("requests.get", return_value=FakeResp()) as mock_get:
        fetch_http(url="https://example/x", target=target)
        mock_get.assert_called_once()
    assert target.read_bytes() == b"a" * 1024 + b"b" * 1024


def test_require_hf_token_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)  # ignore any .env in the real cwd
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


def test_main_skips_present_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("GORISIM_MODELS_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("GORISIM_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    from gorisim.config import reset_settings_for_test

    reset_settings_for_test()

    # Create every manifest file as a placeholder so all are "present"
    from gorisim.download import _build_manifest

    for asset in _build_manifest():
        if asset.target.suffix:
            asset.target.parent.mkdir(parents=True, exist_ok=True)
            asset.target.write_bytes(b"x")
        else:
            asset.target.mkdir(parents=True, exist_ok=True)

    with (
        patch("gorisim.download.fetch_http") as mock_http,
        patch("gorisim.download.fetch_hf_repo") as mock_hf,
        patch("gorisim.download.fetch_hf_whisper_ct2") as mock_ct2,
    ):
        rc = download_main()
        mock_http.assert_not_called()
        mock_hf.assert_not_called()
        mock_ct2.assert_not_called()
    assert rc == 0


def test_main_fetches_missing_http_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("GORISIM_MODELS_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("GORISIM_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    from gorisim.config import reset_settings_for_test

    reset_settings_for_test()

    # Pre-create all *non*-HTTP assets as present (HF dirs); leave HTTP ones missing.
    (tmp_path / "m" / "pyannote").mkdir(parents=True, exist_ok=True)
    (tmp_path / "m" / "speechbrain" / "spkrec-ecapa-voxceleb").mkdir(parents=True, exist_ok=True)
    (tmp_path / "m" / "faster-whisper").mkdir(parents=True, exist_ok=True)
    # Leave the 3 HTTP assets (hrnet_wholebody, rgb_final_finetuned, autsl_signlist_csv) MISSING.

    with (
        patch("gorisim.download.fetch_http") as mock_http,
        patch("gorisim.download.fetch_hf_repo") as mock_hf,
        patch("gorisim.download.fetch_hf_whisper_ct2") as mock_ct2,
    ):
        rc = download_main()
    assert rc == 0
    # Three HTTP assets should have triggered fetch_http
    assert mock_http.call_count == 3
    # HF assets are present, so no HF calls
    mock_hf.assert_not_called()
    mock_ct2.assert_not_called()
