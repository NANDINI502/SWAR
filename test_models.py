"""
Quick test to check that the trained checkpoints in weights/ actually load
and produce output. This exercises the exact same code path as the app
(models/video_detector.py, models/audio_detector.py) instead of a
hand-rolled HuggingFace call, so it catches path/format mismatches too.
"""
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Test 1: Video deepfake model (weights/video_model/)")
print("=" * 60)
try:
    from models.video_detector import VideoDetectorInference

    detector = VideoDetectorInference(device="cpu")
    if not detector.ready:
        print("Video detector did not become ready — see [VideoDetector] warnings above.")
    else:
        # A flat red image — not a real face, just checking the model runs end to end.
        red_img = np.zeros((224, 224, 3), dtype=np.uint8)
        red_img[:, :, 0] = 255
        result = detector.predict(red_img)
        print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Test 2: Audio deepfake model (weights/audio_model/)")
print("=" * 60)
try:
    from models.audio_detector import AudioDetectorInference

    detector = AudioDetectorInference(device="cpu")
    if not detector.ready:
        print("Audio detector did not become ready — see [AudioDetector] warnings above.")
    else:
        # 3 seconds of random noise — just checking the model runs end to end.
        audio = np.random.randn(16000 * 3).astype(np.float32) * 0.05
        result = detector.predict(audio, sr=16000)
        print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

print("\nDONE")
