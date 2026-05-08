import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest
from gorisim.speech_to_sign.stitcher import stitch_clips


def _make_clip(path: Path, color: tuple[int, int, int], seconds: float = 0.5) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=#{color[0]:02x}{color[1]:02x}{color[2]:02x}:size=64x64:duration={seconds}:rate=24",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_stitch_concatenates_clips(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _make_clip(a, (255, 0, 0))
    _make_clip(b, (0, 255, 0))
    out = stitch_clips([a, b], output_path=tmp_path / "out.mp4")
    assert out.exists()
    assert out.stat().st_size > 0


def test_stitch_empty_list_raises(tmp_path):
    with pytest.raises(ValueError):
        stitch_clips([], output_path=tmp_path / "out.mp4")
