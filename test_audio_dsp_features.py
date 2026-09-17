"""
Smoke test for models/audio_dsp_features.py (roadmap item 3).

Checks the jitter/shimmer voice-quality cues move in the direction the
module's docstring claims: a perfectly steady, mathematically pure tone
(no cycle-to-cycle pitch/amplitude perturbation — the "over-smoothed
synthetic" extreme) should score MORE fake-like than the same tone with
realistic frame-scale pitch/amplitude perturbation added (a crude proxy for
the micro-instability a real vocal tract produces).

Like test_lipsync.py / test_forensic_features.py, this doesn't validate
against real deepfake audio — it validates that the code computes a sane
*direction*, which is the class of bug (wrong shape, wrong sign,
always-constant output, or — as with the original audio_detector.py bug in
this repo — silently running on the wrong thing entirely) that a
shape-only check would miss.
"""
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.audio_dsp_features import AudioDSPFeatureExtractor

SR = 16000
DURATION_SECONDS = 3.0
N_SAMPLES = int(SR * DURATION_SECONDS)
HOP = 512
RNG = np.random.default_rng(0)


def make_steady_tone() -> np.ndarray:
    """A pure, perfectly periodic sine — zero pitch/amplitude perturbation."""
    t = np.arange(N_SAMPLES) / SR
    return (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)


def make_jittery_tone() -> np.ndarray:
    """Same base pitch, with frame-scale random pitch+amplitude wobble —
    a crude proxy for a real vocal tract's natural micro-instability."""
    n_hops = N_SAMPLES // HOP + 2
    freqs = 150 + RNG.normal(0, 15.0, n_hops)   # ~10% relative pitch jitter
    amps = 0.3 + RNG.normal(0, 0.06, n_hops)    # ~20% relative amplitude shimmer
    freq_per_sample = np.repeat(freqs, HOP)[:N_SAMPLES]
    amp_per_sample = np.repeat(amps, HOP)[:N_SAMPLES]
    phase = 2 * np.pi * np.cumsum(freq_per_sample) / SR
    return (amp_per_sample * np.sin(phase)).astype(np.float32)


print("=" * 60)
print("Test: AudioDSPFeatureExtractor.compute() — direction sanity check")
print("=" * 60)
try:
    extractor = AudioDSPFeatureExtractor()
    steady_cues = extractor.compute(make_steady_tone(), sr=SR)
    jittery_cues = extractor.compute(make_jittery_tone(), sr=SR)
    print(f"Steady tone cues:  {steady_cues}")
    print(f"Jittery tone cues: {jittery_cues}")

    assert steady_cues["voiced_frames"] > 0 and jittery_cues["voiced_frames"] > 0, (
        "the RMS voicing gate should treat a continuous tone as voiced"
    )
    assert jittery_cues["jitter"] > steady_cues["jitter"], (
        "the perturbed tone should measure higher raw jitter"
    )
    assert jittery_cues["shimmer"] > steady_cues["shimmer"], (
        "the perturbed tone should measure higher raw shimmer"
    )
    assert steady_cues["jitter_score"] > jittery_cues["jitter_score"], (
        "a perfectly steady (more synthetic-like) tone should score MORE "
        "fake-like on jitter than a naturally-perturbed one"
    )
    assert steady_cues["shimmer_score"] > jittery_cues["shimmer_score"], (
        "a perfectly steady (more synthetic-like) tone should score MORE "
        "fake-like on shimmer than a naturally-perturbed one"
    )
    print("\n[OK] Direction sanity check passed: steady > jittery on both fake-like scores")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Test: AudioDSPFeatureExtractor.fuse() — shape + blending sanity check")
print("=" * 60)
try:
    extractor = AudioDSPFeatureExtractor()
    confident_real_wav2vec = {"label": "REAL", "confidence": 0.99, "fake_probability": 0.01}

    fused_steady = extractor.fuse(confident_real_wav2vec, make_steady_tone(), sr=SR)
    fused_jittery = extractor.fuse(confident_real_wav2vec, make_jittery_tone(), sr=SR)
    print(f"Fused (steady tone):  {fused_steady}")
    print(f"Fused (jittery tone): {fused_jittery}")

    for out in (fused_steady, fused_jittery):
        assert set(["label", "confidence", "fake_probability"]).issubset(out.keys())
        assert 0.0 <= out["fake_probability"] <= 1.0
        assert out["label"] in ("REAL", "FAKE")

    assert fused_steady["fake_probability"] > fused_jittery["fake_probability"], (
        "fusing in the DSP cues should raise fake_probability for the "
        "steady/synthetic-like tone relative to the naturally-perturbed one"
    )
    assert fused_jittery["fake_probability"] < confident_real_wav2vec["fake_probability"] + 0.15, (
        "DSP cues on a naturally-perturbed tone shouldn't swamp a "
        "confident REAL call from Wav2Vec2"
    )
    print("\n[OK] fuse() keeps predict()'s output shape and blends cues in the right direction")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Test: silence / too-short input doesn't crash")
print("=" * 60)
try:
    extractor = AudioDSPFeatureExtractor()
    silent = np.zeros(N_SAMPLES, dtype=np.float32)
    result = extractor.compute(silent, sr=SR)
    print(f"Silent-window result: {result}")
    assert result["jitter_score"] == 0.5 and result["shimmer_score"] == 0.5, (
        "silence should fall back to the neutral score, not crash or fabricate a reading"
    )
    print("[OK] Silence handled without crashing")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print("\nDONE")
