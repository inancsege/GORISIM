"""End-to-end Speech → Sign pipeline."""

from __future__ import annotations

import subprocess
import time
import uuid
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import imageio_ffmpeg
import pandas as pd
from pydantic import BaseModel

from gorisim.config import get_settings
from gorisim.speech_to_sign.lemmatize import extract_stems
from gorisim.speech_to_sign.stitcher import stitch_clips
from gorisim.speech_to_sign.vocabulary import Vocabulary

if TYPE_CHECKING:
    from gorisim.speech_to_sign.diarize import Diarizer, Turn
    from gorisim.speech_to_sign.stt import TurkishASR
    from gorisim.speech_to_sign.verify import SpeakerVerifier


class MatchedSign(BaseModel):
    lemma: str
    class_id: int
    turkish: str
    english: str


class SpeechToSignResult(BaseModel):
    transcript: str
    lemmas: list[str]
    matched_signs: list[MatchedSign]
    missing_words: list[str]
    output_video_path: Path
    duration_ms: int


def _slice_wav(audio_path: Path, turns: list[Turn], target: Path) -> Path:
    with wave.open(str(audio_path), "rb") as src:
        n_channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        with wave.open(str(target), "wb") as dst:
            dst.setnchannels(n_channels)
            dst.setsampwidth(sampwidth)
            dst.setframerate(framerate)
            for turn in turns:
                src.setpos(int(turn.start * framerate))
                n = int((turn.end - turn.start) * framerate)
                dst.writeframes(src.readframes(n))
    return target


class SpeechToSignPipeline:
    def __init__(
        self,
        asr: TurkishASR | None = None,
        diarizer: Diarizer | None = None,
        verifier: SpeakerVerifier | None = None,
        vocabulary: Vocabulary | None = None,
        signlist_csv: Path | None = None,
    ):
        s = get_settings()
        if asr is None:
            from gorisim.speech_to_sign.stt import TurkishASR as _TurkishASR

            self.asr: TurkishASR = _TurkishASR()
        else:
            self.asr = asr
        if diarizer is not None:
            self.diarizer: Diarizer | None = diarizer
        elif s.diarization:
            from gorisim.speech_to_sign.diarize import Diarizer as _Diarizer

            self.diarizer = _Diarizer()
        else:
            self.diarizer = None
        if verifier is not None:
            self.verifier: SpeakerVerifier | None = verifier
        elif s.diarization:
            from gorisim.speech_to_sign.verify import SpeakerVerifier as _SpeakerVerifier

            self.verifier = _SpeakerVerifier()
        else:
            self.verifier = None
        csv_path = signlist_csv or (s.data_dir / "SignList_ClassId_TR_EN.csv")
        self.vocab = vocabulary or Vocabulary.load(csv_path)
        self._english: dict[int, str] = {}
        df = pd.read_csv(csv_path, encoding="latin5", on_bad_lines="skip")
        for row in df.itertuples():
            self._english[int(row.ClassId)] = str(row.EN)

    @classmethod
    def load(cls) -> SpeechToSignPipeline:
        return cls()

    def _filter_audio(self, audio_path: Path, profile: str | None, work_dir: Path) -> Path:
        s = get_settings()
        if not s.diarization or self.diarizer is None:
            return audio_path
        turns = self.diarizer.diarize(audio_path)
        if profile is None:
            if s.no_profile_mode == "all":
                return audio_path
            kept_speaker = turns[0].speaker if turns else None
            kept = [t for t in turns if t.speaker == kept_speaker]
        else:
            from gorisim.speech_to_sign.enroll import load_profile

            assert self.verifier is not None
            embedding = load_profile(profile)
            kept: list[Turn] = []
            for turn in turns:
                slice_path = work_dir / f"slice_{turn.start:.2f}.wav"
                _slice_wav(audio_path, [turn], slice_path)
                if self.verifier.matches(slice_path, embedding):
                    kept.append(turn)
        if not kept:
            return audio_path  # fail-open: nothing matched, fall back to full audio
        merged = work_dir / "merged.wav"
        return _slice_wav(audio_path, kept, merged)

    def run(self, audio_path: Path, *, profile: str | None = None) -> SpeechToSignResult:
        s = get_settings()
        t0 = time.perf_counter()
        work_dir = s.data_dir / ".cache" / f"speech_{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=True)

        filtered = self._filter_audio(audio_path, profile, work_dir)
        transcript = self.asr.transcribe(filtered)
        lemmas = extract_stems(transcript)

        matched: list[MatchedSign] = []
        missing: list[str] = []
        clip_paths: list[Path] = []
        for lemma in lemmas:
            cid = self.vocab.lookup(lemma)
            if cid is None:
                missing.append(lemma)
                continue
            clip = s.data_dir / "sign_clips" / f"{cid}.mp4"
            if not clip.exists():
                missing.append(lemma)
                continue
            matched.append(
                MatchedSign(
                    lemma=lemma,
                    class_id=cid,
                    turkish=self.vocab.label(cid),
                    english=self._english.get(cid, ""),
                )
            )
            clip_paths.append(clip)

        out_dir = s.data_dir / ".output"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_video = out_dir / f"{uuid.uuid4().hex}.mp4"
        if clip_paths:
            stitch_clips(clip_paths, output_path=output_video)
        else:
            # Empty result: emit a 1-frame black mp4 so downstream callers always get a path
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:size=64x64:duration=1:rate=24",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(output_video),
                ],
                check=True,
                capture_output=True,
            )

        return SpeechToSignResult(
            transcript=transcript,
            lemmas=lemmas,
            matched_signs=matched,
            missing_words=missing,
            output_video_path=output_video,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
