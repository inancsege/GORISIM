from pathlib import Path

from gorisim.speech_to_sign.pipeline import (
    MatchedSign,
    SpeechToSignResult,
)


def test_matched_sign_schema():
    m = MatchedSign(lemma="merhaba", class_id=0, turkish="merhaba", english="hello")
    assert m.class_id == 0


def test_result_schema():
    r = SpeechToSignResult(
        transcript="merhaba",
        lemmas=["merhaba"],
        matched_signs=[
            MatchedSign(lemma="merhaba", class_id=0, turkish="merhaba", english="hello")
        ],
        missing_words=[],
        output_video_path=Path("/tmp/x.mp4"),
        duration_ms=10,
    )
    assert r.duration_ms == 10
    assert len(r.matched_signs) == 1
