"""
Smoke test for models/forensic_features.py (roadmap item 2).

Checks the two cheap forensic cues (residual energy, LBP texture entropy)
move in the direction the module's docstring claims:
  - A flat/over-smoothed crop (no natural sensor noise/texture) should
    score MORE fake-like on both cues than a crop with natural-looking
    per-pixel noise/texture.
  - fuse() should blend a cue's fake-like leaning into the ViT's own
    fake_probability (not ignore it, not let it swamp a confident ViT
    prediction on its own) and keep the same output shape predict()
    callers already rely on ({"label", "confidence", "fake_probability"}).

Like test_lipsync.py, this doesn't validate against real deepfakes — it
validates that the code computes a sane *direction*, which is the class of
bug (wrong shape, wrong sign, always-constant output) that bit this project
before (see CLAUDE_CODE_HANDOFF.md §3).
"""
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.forensic_features import ForensicFeatureExtractor

RNG = np.random.default_rng(0)


def make_smooth_crop() -> np.ndarray:
    """Flat color, no texture — the over-smoothed / synthetic-looking case."""
    crop = np.full((224, 224, 3), 128, dtype=np.uint8)
    return crop


def make_textured_crop() -> np.ndarray:
    """Per-pixel noise on top of a base value — the natural-camera-photo case."""
    base = np.full((224, 224, 3), 128, dtype=np.float32)
    noise = RNG.normal(0, 18, size=base.shape)
    crop = np.clip(base + noise, 0, 255).astype(np.uint8)
    return crop


print("=" * 60)
print("Test: ForensicFeatureExtractor.compute() — direction sanity check")
print("=" * 60)
try:
    extractor = ForensicFeatureExtractor()
    smooth_cues = extractor.compute(make_smooth_crop())
    textured_cues = extractor.compute(make_textured_crop())
    print(f"Smooth crop cues:   {smooth_cues}")
    print(f"Textured crop cues: {textured_cues}")

    assert smooth_cues["residual_score"] > textured_cues["residual_score"], (
        "a flat/smooth crop should score more fake-like on residual energy "
        "than a naturally textured one"
    )
    if extractor.ready:
        assert smooth_cues["lbp_score"] > textured_cues["lbp_score"], (
            "a flat/smooth crop should score more fake-like on LBP texture "
            "entropy than a naturally textured one"
        )
    print("\n[OK] Direction sanity check passed: smooth > textured on both cues")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Test: ForensicFeatureExtractor.fuse() — shape + blending sanity check")
print("=" * 60)
try:
    extractor = ForensicFeatureExtractor()
    confident_real_vit = {"label": "REAL", "confidence": 0.99, "fake_probability": 0.01}

    fused_smooth = extractor.fuse(confident_real_vit, make_smooth_crop())
    fused_textured = extractor.fuse(confident_real_vit, make_textured_crop())
    print(f"Fused (smooth crop):   {fused_smooth}")
    print(f"Fused (textured crop): {fused_textured}")

    for out in (fused_smooth, fused_textured):
        assert set(["label", "confidence", "fake_probability"]).issubset(out.keys())
        assert 0.0 <= out["fake_probability"] <= 1.0
        assert out["label"] in ("REAL", "FAKE")

    # A confident-REAL ViT score should still end up MORE fake-leaning when
    # paired with a suspiciously flat/smooth crop than with a naturally
    # textured one — the forensic cues should nudge the fused score, not be
    # ignored, while not overriding a confident ViT call on their own
    # (weights default to vit=0.7 in DEFAULT_WEIGHTS).
    assert fused_smooth["fake_probability"] > fused_textured["fake_probability"], (
        "fusing in forensic cues should raise fake_probability for the "
        "suspiciously smooth crop relative to the naturally textured one"
    )
    assert fused_textured["fake_probability"] < confident_real_vit["fake_probability"] + 0.15, (
        "forensic cues on a naturally textured crop shouldn't swamp a "
        "confident REAL call from the ViT"
    )
    print("\n[OK] fuse() keeps predict()'s output shape and blends cues in the right direction")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print("\nDONE")
