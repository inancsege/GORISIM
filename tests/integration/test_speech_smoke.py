from pathlib import Path

import pytest

from gorisim.speech_to_sign.pipeline import SpeechToSignPipeline, SpeechToSignResult


@pytest.mark.slow
def test_speech_pipeline_runs_end_to_end():
    fixture = Path(__file__).parent.parent / "fixtures" / "speech_sample.wav"
    if not fixture.exists():
        pytest.skip(f"Fixture not found: {fixture}")

    pipeline = SpeechToSignPipeline.load()
    result = pipeline.run(fixture)

    assert isinstance(result, SpeechToSignResult)
    assert isinstance(result.transcript, str)
    assert isinstance(result.lemmas, list)
    assert result.output_video_path.exists()
    assert result.duration_ms >= 0
