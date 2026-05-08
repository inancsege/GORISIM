from pathlib import Path

import pytest
from gorisim.speech_to_sign.vocabulary import Vocabulary, normalize_turkish


@pytest.fixture
def vocab():
    return Vocabulary.load(Path(__file__).parent.parent / "fixtures" / "signlist_mini.csv")


def test_normalize_turkish_handles_dotless_i():
    assert normalize_turkish("İstanbul") == "istanbul"
    assert normalize_turkish("ISTANBUL") == "ıstanbul"


def test_normalize_turkish_preserves_dotted_dotless_distinction():
    assert normalize_turkish("ı") != normalize_turkish("i")


def test_lookup_exact(vocab):
    assert vocab.lookup("merhaba") == 0
    assert vocab.lookup("evet") == 1


def test_lookup_case_insensitive(vocab):
    assert vocab.lookup("MERHABA") == 0
    assert vocab.lookup("Hayır") == 2


def test_lookup_dotted_capital_i(vocab):
    assert vocab.lookup("istanbul") == 4
    assert vocab.lookup("İSTANBUL") == 4


def test_lookup_missing_returns_none(vocab):
    assert vocab.lookup("kayısı") is None


def test_label_lookup_by_id(vocab):
    assert vocab.label(0) == "merhaba"


def test_label_lookup_unknown_id_returns_class_n(vocab):
    assert vocab.label(999) == "class_999"
