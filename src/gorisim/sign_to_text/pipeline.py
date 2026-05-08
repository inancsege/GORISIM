"""End-to-end Sign → Text pipeline."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel

from gorisim.config import get_settings
from gorisim.sign_to_text.classify import SignClassifier
from gorisim.sign_to_text.crop import compute_crop_window, crop_frame
from gorisim.sign_to_text.pose import PoseExtractor


class Alternative(BaseModel):
    text: str
    confidence: float


class SignToTextResult(BaseModel):
    text: str
    confidence: float
    alternatives: list[Alternative]
    duration_ms: int


class SignToTextPipeline:
    def __init__(
        self,
        pose: PoseExtractor | None = None,
        classifier: SignClassifier | None = None,
        signlist_csv: Path | None = None,
    ):
        s = get_settings()
        self.pose = pose or PoseExtractor()
        self.classifier = classifier or SignClassifier()
        csv_path = signlist_csv or (s.data_dir / "SignList_ClassId_TR_EN.csv")
        df = pd.read_csv(csv_path, encoding="latin5")
        # Expected columns: ClassId, TR, EN — map class_id (int) → Turkish label
        self._labels: dict[int, str] = {int(row.ClassId): str(row.TR) for row in df.itertuples()}

    @classmethod
    def load(cls) -> SignToTextPipeline:
        return cls()

    def run(self, video_path: Path) -> SignToTextResult:
        t0 = time.perf_counter()
        keypoints = self.pose.extract(video_path)
        center, radius = compute_crop_window(keypoints, frame_size=512)

        cap = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                cropped = crop_frame(frame, center, radius)
                cropped = cv2.resize(cropped, (256, 256))
                frames.append(cropped)
        finally:
            cap.release()

        logits = self.classifier.classify(frames)
        topk = self.classifier.top_k(logits, k=5)
        top1_id, top1_score = topk[0]

        return SignToTextResult(
            text=self._labels.get(top1_id, f"class_{top1_id}"),
            confidence=top1_score,
            alternatives=[
                Alternative(text=self._labels.get(cid, f"class_{cid}"), confidence=score)
                for cid, score in topk[1:]
            ],
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
