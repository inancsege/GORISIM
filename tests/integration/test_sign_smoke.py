from pathlib import Path

import pytest
from gorisim.sign_to_text.pipeline import SignToTextPipeline, SignToTextResult


@pytest.mark.slow
def test_sign_pipeline_runs_end_to_end():
    fixture = Path(__file__).parent.parent / "fixtures" / "sign_sample.mp4"
    if not fixture.exists():
        pytest.skip(f"Fixture not found: {fixture}")

    pipeline = SignToTextPipeline.load()
    result = pipeline.run(fixture)

    assert isinstance(result, SignToTextResult)
    assert isinstance(result.text, str) and result.text
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.alternatives) == 4
    assert result.duration_ms >= 0
