from gorisim.speech_to_sign.lemmatize import extract_stems


def test_empty_string():
    assert extract_stems("") == []


def test_single_word():
    out = extract_stems("merhaba")
    assert out == ["merhaba"]


def test_garbage_input_does_not_raise():
    # zeyrek raises IndexError on some inputs in the original code
    out = extract_stems("xyz123 !@# ")
    assert isinstance(out, list)


def test_lowercases_output():
    out = extract_stems("MERHABA")
    assert out == ["merhaba"]
