from gorisim.download import MANIFEST, Asset, ensure_dirs, is_present


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
