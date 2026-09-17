"""
Smoke test for models/lipsync_detector.py (roadmap item 1).

Feeds two synthetic scenarios and checks the score goes the right
direction:
  1. "synced": mouth-motion bursts line up with audio-energy bursts (like a
     real talking segment followed by silence, repeated).
  2. "desynced": audio bursts happen while the mouth stays still, and
     mouth-motion bursts happen during audio silence (like dubbed/mistimed
     audio, or a face-swap+voice-clone pair that don't actually match).

This doesn't validate on real video/audio — it validates that
LipSyncConsistency.get_sync_score() actually goes up when the two signals
move together and down when they don't. That's exactly the class of bug
that bit this project before (models/video_detector.py silently returning
"UNKNOWN FORMAT" for every input, models/audio_detector.py silently running
on random weights) — a shape/type check alone wouldn't have caught either;
checking output *direction* against a known-good and a known-bad input did.
"""
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.lipsync_detector import LipSyncConsistency

FRAME_SIZE = 128
LANDMARKS = {
    "mouth_left": (50.0, 90.0),
    "mouth_right": (78.0, 90.0),
}
FPS = 5.0
AUDIO_CHUNK_SECONDS = 0.1
DURATION_SECONDS = 9.0
# Anti-phase correlation crosses zero at a lag of BURST_PERIOD/2 — keep this
# comfortably above the detector's max_lag_seconds (0.3s default) so the
# desynced scenario's lag search can't accidentally slide into the
# next burst and read as "synced". Real speech isn't this periodic; this
# constraint is a synthetic-test-construction detail, not a production one.
BURST_PERIOD = 1.5  # alternate "talking"/"silent" every this many seconds


def make_frame(bright: bool) -> np.ndarray:
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    value = 200 if bright else 30
    frame[70:110, 40:88] = value
    return frame


def run_scenario(synced: bool) -> dict:
    detector = LipSyncConsistency(window_seconds=DURATION_SECONDS + 1, min_samples=5)
    t = 0.0
    next_video_t = 0.0
    next_audio_t = 0.0
    toggle = False

    while t < DURATION_SECONDS:
        talking = int(t / BURST_PERIOD) % 2 == 0

        if t >= next_video_t:
            if talking:
                toggle = not toggle
            frame = make_frame(bright=toggle)
            detector.update_video(frame, LANDMARKS, timestamp=t)
            next_video_t += 1.0 / FPS

        if t >= next_audio_t:
            mouth_active = talking if synced else (not talking)
            amplitude = 0.5 if mouth_active else 0.0
            chunk = (np.random.randn(int(16000 * AUDIO_CHUNK_SECONDS)).astype(np.float32) * amplitude)
            detector.update_audio(chunk, sr=16000, timestamp=t)
            next_audio_t += AUDIO_CHUNK_SECONDS

        t += 0.02

    return detector.get_sync_score()


print("=" * 60)
print("Test: LipSyncConsistency — synced vs desynced synthetic streams")
print("=" * 60)
try:
    np.random.seed(0)
    synced_result = run_scenario(synced=True)
    desynced_result = run_scenario(synced=False)
    print(f"Synced scenario:   {synced_result}")
    print(f"Desynced scenario: {desynced_result}")

    assert synced_result["label"] != "INSUFFICIENT_DATA", "synced scenario should have enough samples"
    assert desynced_result["label"] != "INSUFFICIENT_DATA", "desynced scenario should have enough samples"
    assert synced_result["sync_score"] > desynced_result["sync_score"], (
        "synced streams should score higher than desynced streams"
    )
    assert synced_result["label"] == "SYNCED", "synced scenario should be labeled SYNCED"
    assert desynced_result["label"] == "OUT_OF_SYNC", "desynced scenario should be labeled OUT_OF_SYNC"

    print("\n[OK] Direction sanity check passed: synced > desynced, synced == SYNCED")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Test: no-face / silent-audio edge cases don't crash")
print("=" * 60)
try:
    detector = LipSyncConsistency(min_samples=5)
    detector.update_video(make_frame(True), None, timestamp=0.0)  # no landmarks
    detector.update_audio(np.zeros(0, dtype=np.float32), sr=16000, timestamp=0.0)  # empty chunk
    result = detector.get_sync_score()
    print(f"Empty-input result: {result}")
    assert result["label"] == "INSUFFICIENT_DATA"
    print("[OK] Empty/no-face inputs handled without crashing")
except Exception as e:
    print(f"[FAIL] {e}")
    traceback.print_exc()

print("\nDONE")
