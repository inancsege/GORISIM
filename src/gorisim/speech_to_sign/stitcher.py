"""Concatenate sign-language clips into a single mp4 using imageio-ffmpeg."""
from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


def stitch_clips(clip_paths: list[Path], *, output_path: Path) -> Path:
    if not clip_paths:
        raise ValueError("stitch_clips: empty input list")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            # ffmpeg concat demuxer requires single-quoted, escaped paths
            f.write(f"file {shlex.quote(str(p.resolve()))}\n")
        list_path = Path(f.name)
    try:
        cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # Re-encode if -c copy fails (codec mismatch)
        cmd_reencode = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        try:
            subprocess.run(cmd_reencode, check=True, capture_output=True)
        except subprocess.CalledProcessError as e2:
            raise RuntimeError(f"ffmpeg failed: {e2.stderr.decode(errors='ignore')}") from e2
    finally:
        list_path.unlink(missing_ok=True)
    return output_path
