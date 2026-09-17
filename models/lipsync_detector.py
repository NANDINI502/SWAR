"""
Audio-Visual Lip-Sync Consistency Check — roadmap item 1 (see
CLAUDE_CODE_HANDOFF.md §6.1).

Cross-correlates a cheap per-frame "mouth motion" signal (frame-differenced
grayscale energy in the mouth region, located via MTCNN's 5-point
landmarks) against a per-chunk audio RMS envelope, over a short rolling
window. A real speaker's mouth motion and voice energy track each other
tightly; a face-swap + voice-clone combo (or dubbed/out-of-sync audio)
decorrelates.

This is a *third*, independent signal. It does NOT touch
models/video_detector.py or models/audio_detector.py's predict() contract,
and does NOT change FaceExtractor.extract_faces()'s existing return shape —
it consumes raw video frames + MTCNN mouth landmarks (from the new
FaceExtractor.extract_faces_with_landmarks()) and raw audio chunks
directly, both cheap byproducts of pipelines that already exist.

Design note (2026-09-17, see CLAUDE_CODE_HANDOFF.md for the durable copy of
this note): MTCNN's 5-point landmarks give mouth *corners* only
(mouth_left, mouth_right) — no top/bottom lip points — so there's no cheap
geometric "vertical lip opening" distance available without adding a
heavier landmark model (mediapipe/dlib). The roadmap text itself offers an
alternative: "optical flow in the mouth region". Frame-differencing energy
inside the mouth bounding box captures opening/closing motion as a
texture-change signal instead of a geometric one, at effectively zero extra
cost, since MTCNN already computes these landmarks as part of its normal
forward pass (no second model, no second detection pass).
"""

import time
from collections import deque

import cv2
import numpy as np


class LipSyncConsistency:
    """
    Tracks rolling mouth-motion and audio-energy time series and reports how
    well they correlate. Call update_video()/update_audio() as frames/audio
    chunks arrive (any order, any rate, timestamps optional), then
    get_sync_score() whenever a reading is needed (e.g. once per video
    frame).
    """

    def __init__(
        self,
        window_seconds: float = 2.0,
        mouth_crop_size: int = 32,
        max_lag_seconds: float = 0.3,
        resample_hz: float = 20.0,
        min_samples: int = 8,
        synced_threshold: float = 0.3,
    ):
        self.window_seconds = window_seconds
        self.mouth_crop_size = mouth_crop_size
        self.max_lag_seconds = max_lag_seconds
        self.resample_hz = resample_hz
        self.min_samples = min_samples
        self.synced_threshold = synced_threshold

        self._video_series = deque()  # (timestamp, motion_energy)
        self._audio_series = deque()  # (timestamp, rms)
        self._prev_mouth_gray = None
        self._latest_ts = None

    def update_video(self, frame: np.ndarray, landmarks, timestamp: float = None):
        """
        frame: BGR numpy array (H, W, 3) — the *original* frame the
               landmarks were detected on, NOT a resized 224x224 face crop
               (matches what FaceExtractor.extract_faces_with_landmarks()
               was given).
        landmarks: dict with at least "mouth_left"/"mouth_right" (x, y)
                   tuples in `frame` pixel coordinates, as returned by
                   FaceExtractor.extract_faces_with_landmarks(). None/empty
                   -> no-op (no face this frame, nothing to correlate; also
                   resets frame-differencing state so a re-appearing face
                   doesn't diff against a stale crop).
        """
        if not landmarks:
            self._prev_mouth_gray = None
            return
        if timestamp is None:
            timestamp = time.time()

        mouth_crop = self._extract_mouth_region(frame, landmarks)
        if mouth_crop is None:
            self._prev_mouth_gray = None
            return

        gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY) if mouth_crop.ndim == 3 else mouth_crop
        gray = cv2.resize(gray, (self.mouth_crop_size, self.mouth_crop_size)).astype(np.float32)

        if self._prev_mouth_gray is not None:
            motion = float(np.mean(np.abs(gray - self._prev_mouth_gray)))
            self._video_series.append((timestamp, motion))

        self._prev_mouth_gray = gray
        self._touch(timestamp)

    def update_audio(self, chunk: np.ndarray, sr: int = None, timestamp: float = None):
        """
        chunk: 1D float32 numpy array, any length > 0 (e.g. a ~100ms block
               from a mic callback, or whatever arrives per audio WebSocket
               message). sr is accepted for API symmetry with
               AudioDetectorInference.predict() but isn't needed for an RMS
               envelope.
        """
        if chunk is None or len(chunk) == 0:
            return
        if timestamp is None:
            timestamp = time.time()

        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        self._audio_series.append((timestamp, rms))
        self._touch(timestamp)

    def get_sync_score(self) -> dict:
        """
        Returns:
            {"label": "SYNCED" | "OUT_OF_SYNC" | "INSUFFICIENT_DATA",
             "sync_score": float, roughly [-1, 1] (higher = more in sync),
             "samples": int}
        """
        if len(self._video_series) < self.min_samples or len(self._audio_series) < self.min_samples:
            return {
                "label": "INSUFFICIENT_DATA",
                "sync_score": 0.0,
                "samples": min(len(self._video_series), len(self._audio_series)),
            }

        t_start = max(self._video_series[0][0], self._audio_series[0][0])
        t_end = min(self._video_series[-1][0], self._audio_series[-1][0])
        if t_end - t_start < 0.5:
            return {"label": "INSUFFICIENT_DATA", "sync_score": 0.0, "samples": 0}

        grid = np.arange(t_start, t_end, 1.0 / self.resample_hz)
        if len(grid) < self.min_samples:
            return {"label": "INSUFFICIENT_DATA", "sync_score": 0.0, "samples": len(grid)}

        v_t, v_y = zip(*self._video_series)
        a_t, a_y = zip(*self._audio_series)
        v_resampled = np.interp(grid, v_t, v_y)
        a_resampled = np.interp(grid, a_t, a_y)

        max_lag = max(1, int(self.max_lag_seconds * self.resample_hz))
        best_corr = -1.0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                v_seg, a_seg = v_resampled[-lag:], a_resampled[: len(a_resampled) + lag]
            elif lag > 0:
                v_seg, a_seg = v_resampled[: len(v_resampled) - lag], a_resampled[lag:]
            else:
                v_seg, a_seg = v_resampled, a_resampled

            if len(v_seg) < self.min_samples:
                continue
            if np.std(v_seg) < 1e-8 or np.std(a_seg) < 1e-8:
                continue

            corr = float(np.corrcoef(v_seg, a_seg)[0, 1])
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr

        if best_corr == -1.0:
            # Every lag failed the variance check (e.g. silent audio, no
            # mouth motion at all) — not enough signal to say anything.
            return {"label": "INSUFFICIENT_DATA", "sync_score": 0.0, "samples": len(grid)}

        label = "SYNCED" if best_corr >= self.synced_threshold else "OUT_OF_SYNC"
        return {"label": label, "sync_score": best_corr, "samples": len(grid)}

    def _extract_mouth_region(self, frame: np.ndarray, landmarks):
        mx1, my1 = landmarks["mouth_left"]
        mx2, my2 = landmarks["mouth_right"]
        cx, cy = (mx1 + mx2) / 2.0, (my1 + my2) / 2.0
        mouth_width = abs(mx2 - mx1)
        half = max(mouth_width * 0.9, 12)  # generous box around both corners
        x1 = int(max(0, cx - half))
        x2 = int(min(frame.shape[1], cx + half))
        y1 = int(max(0, cy - half * 0.8))
        y2 = int(min(frame.shape[0], cy + half * 0.8))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _touch(self, timestamp: float):
        """Advance the shared 'now' and trim both series against it.

        Deliberately NOT time.time() — trimming is relative to the latest
        timestamp actually seen (real time.time() values when callers don't
        pass one, but arbitrary caller-supplied timestamps otherwise), so
        this class behaves identically fed live or fed a synthetic replay
        (see test_lipsync.py).
        """
        if self._latest_ts is None or timestamp > self._latest_ts:
            self._latest_ts = timestamp
        cutoff = self._latest_ts - self.window_seconds
        while self._video_series and self._video_series[0][0] < cutoff:
            self._video_series.popleft()
        while self._audio_series and self._audio_series[0][0] < cutoff:
            self._audio_series.popleft()

    def reset(self):
        self._video_series.clear()
        self._audio_series.clear()
        self._prev_mouth_gray = None
        self._latest_ts = None
