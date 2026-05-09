"""AUTSL SignList CSV → bidirectional lookup, with Turkish-correct normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Turkish letter → ASCII fold map for diacritic-insensitive fallback lookup.
# Used only when the strict (preserve-i/ı) match fails.
_DIACRITIC_FOLD = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
    }
)


def normalize_turkish(s: str) -> str:
    """Turkish-correct case-fold.

    Maps İ→i and I→ı before casefolding so the dotted/dotless distinction
    is preserved while case is normalized.
    """
    return s.replace("İ", "i").replace("I", "ı").casefold().strip()


def fold_diacritics(s: str) -> str:
    """ASCII-fold Turkish diacritics. Used as a fallback when strict match fails.

    Some upstream label CSVs ship with diacritics already stripped
    (e.g., `hayir` for `hayır`); folding the query lets those still match.
    """
    return s.translate(_DIACRITIC_FOLD)


class Vocabulary:
    def __init__(
        self,
        by_label: dict[str, int],
        by_id: dict[int, str],
        by_folded: dict[str, int] | None = None,
    ):
        self._by_label = by_label
        self._by_id = by_id
        self._by_folded = by_folded or {}

    @classmethod
    def load(cls, csv_path: Path) -> Vocabulary:
        # AUTSL CSV is latin5-encoded in older releases; try utf-8 first.
        for enc in ("utf-8", "latin5", "latin-1"):
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Cannot decode {csv_path} with utf-8/latin5/latin-1")
        by_label: dict[str, int] = {}
        by_id: dict[int, str] = {}
        by_folded: dict[str, int] = {}
        for row in df.itertuples():
            label = str(row.TR)
            cid = int(row.ClassId)
            normalized = normalize_turkish(label)
            by_label[normalized] = cid
            by_id[cid] = label
            # Fallback index: diacritic-folded form. First-write-wins on collision.
            folded = fold_diacritics(normalized)
            by_folded.setdefault(folded, cid)
        return cls(by_label=by_label, by_id=by_id, by_folded=by_folded)

    def lookup(self, lemma: str) -> int | None:
        normalized = normalize_turkish(lemma)
        cid = self._by_label.get(normalized)
        if cid is not None:
            return cid
        # Fallback: drop Turkish diacritics so e.g. "hayır" matches "hayir" in
        # CSVs that have already been ASCII-stripped.
        return self._by_folded.get(fold_diacritics(normalized))

    def label(self, class_id: int) -> str:
        return self._by_id.get(class_id, f"class_{class_id}")
