"""
Frequency/residual-domain forensic cues — roadmap item 2 (see
CLAUDE_CODE_HANDOFF.md §6b).

Two cheap, model-free cues computed directly on a face crop, fused with the
ViT classifier's own fake_probability via a simple weighted average (per
the roadmap: "not a full retrain"):

- Residual energy: crop minus a Gaussian-blurred copy of itself. Real
  camera photos carry natural sensor noise / skin micro-texture that
  survives this high-pass; over-smoothed GAN/diffusion output tends not
  to. LOW residual energy -> flagged more fake-like.
- LBP texture entropy: entropy of a Local Binary Pattern histogram.
  Natural skin/hair texture spreads across many LBP bins (higher entropy);
  an unnaturally flat/synthetic texture concentrates into a few bins
  (lower entropy). LOW entropy -> flagged more fake-like.

This does NOT touch models/video_detector.py's predict() contract — it
reads a face crop (the same raw RGB uint8 array FaceExtractor.extract_faces()
already produces and VideoDetectorInference.predict() already accepts) and
that detector's own output, and returns a new dict in the *same shape*
({"label", "confidence", "fake_probability"}) plus an extra "components"
key for debugging/tuning.

Both cues are hand-tuned heuristics, not trained classifiers — there's no
labeled deepfake dataset in this repo to fit a proper fusion model against
(see training/README.md), so the weighted-average starting point here is
meant to be tuned once real fake/real face samples are available, not
treated as final.
"""

import cv2
import numpy as np

from config import FAKE_THRESHOLD

try:
    from skimage.feature import local_binary_pattern
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
N_LBP_BINS = LBP_POINTS + 2  # uniform method produces P+2 distinct bins
# LBP cost scales with pixel count, not information content for a texture
# *histogram* (we only need distribution statistics, not per-pixel detail).
# Benchmarked at ~8-30ms at 224x224 (machine-load dependent) vs <1ms at
# 96x96 with no meaningful change in the resulting entropy — downsize
# before running it so this stays cheap enough for the ~5fps hot path.
LBP_DOWNSCALE_SIZE = 96

DEFAULT_WEIGHTS = {"vit": 0.7, "residual": 0.15, "lbp": 0.15}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class ForensicFeatureExtractor:
    """Computes and fuses the residual + LBP forensic cues for one face crop."""

    def __init__(self):
        self.ready = _HAS_SKIMAGE
        if not _HAS_SKIMAGE:
            print("[ForensicFeatures] `scikit-image` not installed — "
                  "LBP texture cue disabled (residual cue still runs). "
                  "Run `pip install scikit-image` (see requirements.txt).")

    def compute(self, face_rgb: np.ndarray) -> dict:
        """
        face_rgb: raw RGB uint8 crop (H, W, 3) — same format
                  FaceExtractor.extract_faces() returns.

        Returns: {"residual_score": float 0-1, "lbp_score": float 0-1,
                  "residual_energy": float, "lbp_entropy": float}
        residual_score/lbp_score: higher = more fake-like by this heuristic.
        lbp_score is a neutral 0.5 if scikit-image isn't installed.
        """
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual_energy = float(np.mean(np.abs(gray - blurred)))
        # Natural 224x224 face crops typically land in roughly a 1-11
        # (8-bit intensity units) residual-energy band; map that band to
        # [0, 1] and flip so LOW residual -> HIGH fake_score. Hand-tuned —
        # revisit once real fake/real samples are available to calibrate.
        residual_score = 1.0 - _clamp01((residual_energy - 1.0) / 10.0)

        if self.ready:
            small_gray = cv2.resize(
                gray.astype(np.uint8), (LBP_DOWNSCALE_SIZE, LBP_DOWNSCALE_SIZE)
            )
            lbp = local_binary_pattern(small_gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
            hist, _ = np.histogram(lbp.ravel(), bins=int(N_LBP_BINS), range=(0, N_LBP_BINS), density=True)
            hist = hist[hist > 0]
            entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) else 0.0
            max_entropy = float(np.log2(N_LBP_BINS))
            lbp_entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
            lbp_score = 1.0 - _clamp01(lbp_entropy_norm)
        else:
            entropy = 0.0
            lbp_score = 0.5  # neutral — no opinion without skimage

        return {
            "residual_score": residual_score,
            "lbp_score": lbp_score,
            "residual_energy": residual_energy,
            "lbp_entropy": entropy,
        }

    def fuse(self, vit_result: dict, face_rgb: np.ndarray, weights: dict = None) -> dict:
        """
        vit_result: whatever VideoDetectorInference.predict() returned for
                    this same face_rgb crop (read-only — only its
                    fake_probability is used; vit_result itself is
                    untouched).

        Returns a NEW dict with the same shape as predict()'s output
        ({"label", "confidence", "fake_probability"}), recomputed from the
        fused score, plus a "components" key with the raw per-cue scores.
        """
        weights = weights or DEFAULT_WEIGHTS
        cues = self.compute(face_rgb)

        vit_prob = float(vit_result.get("fake_probability", 0.5))
        fused_prob = _clamp01(
            weights["vit"] * vit_prob
            + weights["residual"] * cues["residual_score"]
            + weights["lbp"] * cues["lbp_score"]
        )

        is_fake = fused_prob > FAKE_THRESHOLD
        label = "FAKE" if is_fake else "REAL"
        confidence = fused_prob if is_fake else (1.0 - fused_prob)

        return {
            "label": label,
            "confidence": float(confidence),
            "fake_probability": float(fused_prob),
            "components": cues,
        }
