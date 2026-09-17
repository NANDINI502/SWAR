"""Quick test script to verify all imports and basic model inference."""
import sys
import os
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("Testing imports and models...")
print("=" * 50)

# Test 1: Config
try:
    from config import DEVICE
    print(f"[OK] config.py — Device: {DEVICE}")
except Exception as e:
    print(f"[FAIL] config.py — {e}")
    traceback.print_exc()

# Test 2: Face extractor
try:
    from models.face_extractor import FaceExtractor
    print("[OK] face_extractor.py imported")
except Exception as e:
    print(f"[FAIL] face_extractor.py — {e}")
    traceback.print_exc()

# Test 3: Video detector
try:
    from models.video_detector import VideoDetectorInference
    print("[OK] video_detector.py imported")
except Exception as e:
    print(f"[FAIL] video_detector.py — {e}")
    traceback.print_exc()

# Test 4: Audio detector
try:
    from models.audio_detector import AudioDetectorInference
    print("[OK] audio_detector.py imported")
except Exception as e:
    print(f"[FAIL] audio_detector.py — {e}")
    traceback.print_exc()

# Test 5: Video model — load weights/video_model/ and classify a dummy face crop
try:
    detector_v = VideoDetectorInference(device="cpu")
    if not detector_v.ready:
        print("[WARN] Video model — initialized but not ready (see [VideoDetector] "
              "log lines above — likely missing weights/video_model/ or `transformers`)")
    else:
        dummy_face = np.zeros((224, 224, 3), dtype=np.uint8)  # raw RGB crop, like FaceExtractor returns
        out = detector_v.predict(dummy_face)
        print(f"[OK] Video model — output: {out}")
except Exception as e:
    print(f"[FAIL] Video model — {e}")
    traceback.print_exc()

# Test 6: Audio model — load weights/audio_model/ and classify a dummy waveform
try:
    detector_a = AudioDetectorInference(device="cpu")
    if not detector_a.ready:
        print("[WARN] Audio model — initialized but not ready (see [AudioDetector] "
              "log lines above — likely missing weights/audio_model/ or `transformers`)")
    else:
        dummy_audio = np.zeros(16000 * 3, dtype=np.float32)  # 3s of silence @ 16kHz
        out = detector_a.predict(dummy_audio, sr=16000)
        print(f"[OK] Audio model — output: {out}")
except Exception as e:
    print(f"[FAIL] Audio model — {e}")
    traceback.print_exc()

# Test 7: Pipelines
try:
    from pipeline.video_pipeline import VideoPipeline
    from pipeline.audio_pipeline import AudioPipeline
    print("[OK] Pipeline imports")
except Exception as e:
    print(f"[FAIL] Pipeline imports — {e}")
    traceback.print_exc()

print("=" * 50)
print("Tests complete!")
