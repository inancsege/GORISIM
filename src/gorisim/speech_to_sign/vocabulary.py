"""AUTSL SignList CSV → bidirectional lookup, with Turkish-correct normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_turkish(s: str) -> str:
    """Turkish-correct case-fold.

    Maps İ→i and I→ı before casefolding so the dotted/dotless distinction
    is preserved while case is normalized.
    """
    return s.replace("İ", "i").replace("I", "ı").casefold().strip()


class Vocabulary:
    def __init__(self, by_label: dict[str, int], by_id: dict[int, str]):
        self._by_label = by_label
        self._by_id = by_id

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
        for row in df.itertuples():
            label = str(row.TR)
            cid = int(row.ClassId)
            by_label[normalize_turkish(label)] = cid
            by_id[cid] = label
        return cls(by_label=by_label, by_id=by_id)

    def lookup(self, lemma: str) -> int | None:
        return self._by_label.get(normalize_turkish(lemma))

    def label(self, class_id: int) -> str:
        return self._by_id.get(class_id, f"class_{class_id}")
