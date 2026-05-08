from gorisim.sign_to_text.pipeline import Alternative, SignToTextResult


def test_alternative_schema():
    a = Alternative(text="merhaba", confidence=0.42)
    assert a.text == "merhaba"
    assert 0 <= a.confidence <= 1


def test_sign_to_text_result_schema():
    r = SignToTextResult(
        text="merhaba",
        confidence=0.9,
        alternatives=[Alternative(text="selam", confidence=0.05)],
        duration_ms=123,
    )
    assert r.duration_ms == 123
    assert len(r.alternatives) == 1
