"""
Audio DSP handcrafted features — roadmap item 3 (see
CLAUDE_CODE_HANDOFF.md §6c).

Extracts classic voice-quality DSP cues (jitter, shimmer) from a raw
waveform and fuses a heuristic score derived from them with Wav2Vec2's own
fake_probability — the same "cheap independent signal, weighted average,
not a full retrain" pattern as models/forensic_features.py (roadmap item 2).

Why jitter/shimmer specifically (not the full cue list the roadmap
sketches, e.g. raw pitch/spectral centroid): pitch (F0) and spectral shape
are already deeply baked into what Wav2Vec2 was fine-tuned on — it's a
self-supervised speech model, so those are exactly the kind of low-level
acoustic properties it already encodes. Reusing them risks *repeating* what
the end-to-end model already sees rather than adding something orthogonal.
Micro-perturbation measures — jitter (cycle-to-cycle F0 instability) and
shimmer (cycle-to-cycle amplitude instability) — are different: a
well-documented tell in synthetic-speech literature, because vocoders tend
to smooth out exactly this kind of natural, "noisy" micro-variation a real
vocal tract produces. Same reasoning as forensic_features.py's residual
cue (real capture leaves a noise floor; synthesis over-smooths it) in the
audio domain.

**Perf decision — yin, not pyin:** the roadmap and most voice-quality
literature point at Praat/parselmouth or librosa.pyin for pitch tracking.
Benchmarked on this machine for a 3s/16kHz window (this repo's
AUDIO_WINDOW_SECONDS/SAMPLE_RATE): librosa.pyin ~190-350ms depending on
hop_length — far too slow for a window that needs to be re-classified
every ~1s stride (AUDIO_STRIDE_SECONDS). librosa.yin (deterministic, no
voiced-probability output) measured ~10ms for the same window — used here
instead, with voiced/unvoiced decided separately via a simple RMS-energy
threshold. This trades some robustness (yin always returns an F0 guess,
even for pure noise) for being ~30x cheaper; the RMS gate is what keeps
jitter/shimmer from being computed over silence.

**No trained classifier here** (the honest gap vs. the roadmap text): the
roadmap suggests training "a small classical ML model (logistic regression
/ gradient boosted trees) on these features," but there's no labeled
real/fake voice dataset checked into this repo to fit one against (see
training/README.md) — weights/audio_model/ was fine-tuned on ASVspoof2019
externally (training/train_audio_kaggle.py), not on data available here.
So `fuse()` below is a hand-tuned heuristic weighted average, same caveat
as models/forensic_features.py — revisit the thresholds once a labeled
dataset is available to actually fit the acoustic-ML stage the roadmap
describes.
"""

import numpy as np
import librosa

from config import SAMPLE_RATE, AUDIO_FAKE_THRESHOLD

FMIN_HZ = 65.0    # ~C2, below typical adult male voice floor
FMAX_HZ = 400.0   # comfortably above typical adult female voice ceiling
FRAME_LENGTH = 2048
HOP_LENGTH = 512
MIN_VOICED_FRAMES = 4

DEFAULT_WEIGHTS = {"wav2vec": 0.75, "dsp": 0.25}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class AudioDSPFeatureExtractor:
    """Computes jitter/shimmer voice-quality cues and fuses them with Wav2Vec2's score."""

    def compute(self, waveform: np.ndarray, sr: int = None) -> dict:
        """
        waveform: 1D float32 numpy array (same format
                  AudioDetectorInference.predict() accepts).
        sr: sample rate of `waveform`; defaults to config.SAMPLE_RATE.

        Returns: {"jitter_score": float 0-1, "shimmer_score": float 0-1,
                  "jitter": float|None, "shimmer": float|None,
                  "voiced_frames": int}
        jitter_score/shimmer_score: higher = more fake-like by this
        heuristic (0.5 neutral when too few voiced frames to measure).
        """
        if sr is None:
            sr = SAMPLE_RATE
        wav = waveform.astype(np.float32)

        f0 = librosa.yin(wav, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=sr,
                          frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=wav, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

        n = min(len(f0), len(rms))
        f0, rms = f0[:n], rms[:n]

        # yin has no voiced/unvoiced output — gate on frame energy instead:
        # a frame is "voiced" only if it's meaningfully loud relative to
        # this window's own peak (not an absolute threshold, so it adapts
        # to mic gain / distance from mic — and works whether the window
        # is continuous speech or speech-with-pauses).
        voiced_flag = rms > max(0.3 * float(np.max(rms)), 1e-4)

        jitter = shimmer = None
        voiced_count = int(np.sum(voiced_flag))

        if voiced_count >= MIN_VOICED_FRAMES:
            v_idx = np.where(voiced_flag)[0]
            # Only consecutive voiced-frame pairs (skip across gaps) so a
            # jump from one voiced segment to another doesn't get counted
            # as a pitch/amplitude "perturbation".
            is_consecutive = np.diff(v_idx) == 1
            cur = v_idx[1:][is_consecutive]
            prev = v_idx[:-1][is_consecutive]

            if len(cur) >= 3:
                f0_mean = float(np.mean(f0[voiced_flag]))
                if f0_mean > 0:
                    jitter = float(np.mean(np.abs(f0[cur] - f0[prev])) / f0_mean)

                rms_mean = float(np.mean(rms[voiced_flag]))
                if rms_mean > 0:
                    shimmer = float(np.mean(np.abs(rms[cur] - rms[prev])) / rms_mean)

        # Hand-tuned bands, not fit on labeled data — see module docstring.
        jitter_score = 0.5 if jitter is None else 1.0 - _clamp01((jitter - 0.01) / 0.05)
        shimmer_score = 0.5 if shimmer is None else 1.0 - _clamp01((shimmer - 0.03) / 0.15)

        return {
            "jitter_score": jitter_score,
            "shimmer_score": shimmer_score,
            "jitter": jitter,
            "shimmer": shimmer,
            "voiced_frames": voiced_count,
        }

    def fuse(self, wav2vec_result: dict, waveform: np.ndarray, sr: int = None, weights: dict = None) -> dict:
        """
        wav2vec_result: whatever AudioDetectorInference.predict() returned
                        for this same waveform (read-only — only its
                        fake_probability is used).

        Returns a NEW dict with the same shape as predict()'s output
        ({"label", "confidence", "fake_probability"}), recomputed from the
        fused score, plus a "components" key with the raw cues.
        """
        weights = weights or DEFAULT_WEIGHTS
        cues = self.compute(waveform, sr)
        dsp_score = (cues["jitter_score"] + cues["shimmer_score"]) / 2.0

        wav2vec_prob = float(wav2vec_result.get("fake_probability", 0.5))
        fused_prob = _clamp01(
            weights["wav2vec"] * wav2vec_prob + weights["dsp"] * dsp_score
        )

        is_fake = fused_prob > AUDIO_FAKE_THRESHOLD
        label = "FAKE" if is_fake else "REAL"
        confidence = fused_prob if is_fake else (1.0 - fused_prob)

        return {
            "label": label,
            "confidence": float(confidence),
            "fake_probability": float(fused_prob),
            "components": cues,
        }
