from unittest.mock import MagicMock

import numpy as np

from gorisim.speech_to_sign.enroll import delete_profile, enroll_user, list_profiles


def test_enroll_writes_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("GORISIM_DATA_DIR", str(tmp_path))
    from gorisim.config import reset_settings_for_test
    reset_settings_for_test()

    fake_verifier = MagicMock()
    fake_verifier.embed.side_effect = [np.ones(192, dtype=np.float32), np.ones(192, dtype=np.float32) * 2]

    p = enroll_user(name="ege", wav_paths=[tmp_path / "a.wav", tmp_path / "b.wav"], verifier=fake_verifier)
    assert p.exists()
    arr = np.load(p)
    assert arr.shape == (192,)
    # mean of 1.0 and 2.0
    assert np.allclose(arr.mean(), 1.5)


def test_list_and_delete_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("GORISIM_DATA_DIR", str(tmp_path))
    from gorisim.config import reset_settings_for_test
    reset_settings_for_test()

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "ege.npy").write_bytes(b"x")
    (profiles_dir / "ali.npy").write_bytes(b"x")

    assert sorted(list_profiles()) == ["ali", "ege"]
    delete_profile("ege")
    assert list_profiles() == ["ali"]
