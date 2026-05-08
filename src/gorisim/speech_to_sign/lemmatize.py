"""Turkish lemmatization wrapper — robust against zeyrek IndexError edge cases."""

from __future__ import annotations

import re

from zeyrek import MorphAnalyzer

from gorisim.speech_to_sign.vocabulary import normalize_turkish

_analyzer: MorphAnalyzer | None = None
_TOKEN_RE = re.compile(r"\b[\wçğıöşüÇĞİÖŞÜ]+\b", flags=re.UNICODE)


def _get_analyzer() -> MorphAnalyzer:
    global _analyzer
    if _analyzer is None:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        _analyzer = MorphAnalyzer()
    return _analyzer


def extract_stems(text: str) -> list[str]:
    """Return ordered stems for the words in `text`.

    Falls back to the surface token (lowercased) if zeyrek can't parse a word
    or raises IndexError mid-parse. Preserves order; emits one stem per word.
    """
    if not text or not text.strip():
        return []
    analyzer = _get_analyzer()
    out: list[str] = []
    for token in _TOKEN_RE.findall(text):
        lowered = normalize_turkish(token)
        try:
            parses = analyzer.analyze(lowered)
        except Exception:
            out.append(lowered)
            continue
        if not parses or not parses[0]:
            out.append(lowered)
            continue
        try:
            lemma = parses[0][0].lemma  # zeyrek Parse.lemma — newest API
            if not lemma or lemma == "Unk":
                out.append(lowered)
            else:
                out.append(lemma.casefold())
        except (AttributeError, IndexError):
            out.append(lowered)
    return out
