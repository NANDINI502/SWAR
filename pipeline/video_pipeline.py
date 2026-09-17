"""
Video Pipeline — Real-time video capture, face detection, and deepfake classification.
Runs in a separate thread for non-blocking UI.
"""

import cv2
import time
import numpy as np
import threading
from typing import Optional

from config import (
    VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT, FRAME_SKIP,
    REAL_COLOR, FAKE_COLOR, FONT_SCALE, BOX_THICKNESS,
)
from models.face_extractor import FaceExtractor
from models.video_detector import VideoDetectorInference


class VideoPipeline:
    """
    Real-time video deepfake detection pipeline.
    Captures webcam, detects faces, and classifies them as real/fake.
    """

    def __init__(self, device: str = "cuda", use_onnx: bool = False):
        self.face_extractor = FaceExtractor(device=device)
        self.detector = VideoDetectorInference(device=device)
        self.frame_count = 0
        self.last_result = None
        self.last_boxes = []
        self.fps = 0
        self._fps_counter = 0
        self._fps_time = time.time()

        # Optional third signal (roadmap item 1) — set externally, e.g.
        # `video_pipeline.lipsync = LipSyncConsistency()`. When unset,
        # process_frame() behaves exactly as before (extra landmark data
        # from MTCNN is simply not requested).
        self.lipsync = None

        # Optional forensic-cue fusion (roadmap item 2) — set externally,
        # e.g. `video_pipeline.forensics = ForensicFeatureExtractor()`.
        # When unset, process_frame() returns the ViT's own result
        # unchanged, exactly as before.
        self.forensics = None

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Process a single video frame.

        Args:
            frame: BGR numpy array from webcam/video

        Returns:
            (annotated_frame, result_dict)
        """
        self.frame_count += 1
        self._update_fps()

        # Only run detection every N frames for performance
        if self.frame_count % FRAME_SKIP == 0:
            if self.lipsync is not None:
                # Same MTCNN forward pass as extract_faces() — landmarks are
                # a free byproduct, not a second detection call.
                faces = self.face_extractor.extract_faces_with_landmarks(frame)
            else:
                faces = self.face_extractor.extract_faces(frame)
            if faces:
                # Classify the most prominent face
                best_face = max(faces, key=lambda f: f["confidence"])
                result = self.detector.predict(best_face["face"])
                if self.forensics is not None:
                    result = self.forensics.fuse(result, best_face["face"])
                self.last_result = result
                self.last_boxes = [(f["box"], f["confidence"]) for f in faces]
                if self.lipsync is not None:
                    self.lipsync.update_video(frame, best_face.get("landmarks"))
            else:
                self.last_result = {"label": "NO FACE", "confidence": 0.0, "fake_probability": 0.0}
                self.last_boxes = []
                if self.lipsync is not None:
                    self.lipsync.update_video(frame, None)

        # Draw overlays using cached results
        annotated = self._draw_overlays(frame.copy())

        return annotated, self.last_result or {"label": "DETECTING...", "confidence": 0.0, "fake_probability": 0.0}

    def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes, labels, and FPS on frame."""
        if self.last_result and self.last_boxes:
            for box, det_conf in self.last_boxes:
                x1, y1, x2, y2 = box
                label = self.last_result["label"]
                confidence = self.last_result["confidence"]

                # Color based on prediction
                color = FAKE_COLOR if label == "FAKE" else REAL_COLOR

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

                # Draw label background
                text = f"{label} {confidence:.1%}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w + 10, y1),
                    color, -1
                )
                cv2.putText(
                    frame, text,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                    (255, 255, 255), 2,
                )

        # FPS counter
        cv2.putText(
            frame, f"FPS: {self.fps:.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2,
        )

        return frame

    def _update_fps(self):
        """Calculate rolling FPS."""
        self._fps_counter += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self.fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = time.time()

    def process_uploaded_video(self, video_path: str):
        """
        Process an uploaded video file frame by frame.

        Args:
            video_path: Path to video file

        Yields:
            (annotated_frame, result_dict) for each processed frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.frame_count = 0
        self.last_result = None
        self.last_boxes = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))
                annotated, result = self.process_frame(frame)
                yield annotated, result
        finally:
            cap.release()
